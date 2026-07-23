"""Agentic hierarchy retriever — topic-based navigation.

At query time:
  1. Select the relevant document(s) from their descriptions.
  2a. Short documents (≤ DIRECT_SELECT_THRESHOLD pages): ask the VLM to rank
      page indices directly using per-page summaries and visual descriptions.
      One LLM call, no topic indirection, correct ordering by relevance.
  2b. Long documents: navigate by topic IDs, then collect their page indices.
  3. Return full-page images first (up to top_k), then fill remaining slots
     with crops.  Distances are position-based (0.05 · rank) so hindex results
     compete fairly with vector-search cosine distances in overlay modes instead
     of always winning with the previous hardcoded 0.0.

Return format is compatible with RAGImgRetriever so results merge cleanly
into the hybrid context.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .hierarchy_builder import COLLECTION_THRESHOLD, CollectionIndex, DocumentCluster, DocumentHierarchy, PageRecord
from .hierarchy_store import HierarchyStore

# Navigation calls produce short JSON lists (["T01", "T02"] or ["doc.pdf"]).
# 256 tokens is more than enough and avoids wasting budget on thinking tokens.
_NAV_TOKENS = 256

# Documents with at most this many pages use direct page-index selection instead
# of the two-step topic → pages navigation.  Topic indirection helps for long
# documents where the full page list overflows the prompt; for short ones it
# just adds noise and an extra LLM call.
_DIRECT_SELECT_THRESHOLD = 30


class HierarchyRetriever:
    """Retrieves page images by navigating the document topic structure.

    Parameters
    ----------
    store:
        HierarchyStore with persisted DocumentHierarchy objects.
    kvstore:
        Colette HDF5ImageStorage for fetching images by key.
    vlm_fn:
        Callable(images, prompt) -> str — same as used by HierarchyBuilder.
    top_k:
        Maximum number of result items to return.
    logger:
        Standard logger or duck-typed equivalent.
    """

    def __init__(self, store: HierarchyStore, kvstore, vlm_fn, top_k: int = 4, logger=None):
        self.store = store
        self.kvstore = kvstore
        self.vlm_fn = vlm_fn
        self.top_k = top_k
        self.logger = logger or _NullLogger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def invoke(self, question: str) -> dict[str, Any]:
        """Return page / crop images most relevant to *question*.

        Return value is compatible with the ``docs`` dict consumed by
        ``RAGImgRetriever`` downstream (ids, distances, metadatas keys).

        Distance assignment
        -------------------
        Distances are position-based: ``0.05 * (rank + 1)`` for full-page
        images and ``+0.025`` for their crops.  This keeps hindex results in
        the same ballpark as typical cosine distances from ChromaDB (0.1–0.6),
        so in overlay modes the two result lists compete on merit rather than
        hindex always winning with the previous hardcoded 0.0.

        Budget allocation
        -----------------
        Full-page images are added first (up to top_k), then any remaining
        slots are filled with crops.  Previously crops consumed slots alongside
        full pages, which could exhaust the budget before all selected pages
        were included.
        """
        hierarchies = self.store.load_all()
        if not hierarchies:
            self.logger.warning("[hindex] No document hierarchies found in store")
            return _empty_docs()

        candidate_pages = self._navigate(question, hierarchies)
        if not candidate_pages:
            return _empty_docs()

        hier_by_name = {h.doc_name: h for h in hierarchies}

        # Collect full-page and crop items separately so we can fill full pages
        # first, then use remaining budget for crops.
        full_items: list[tuple[str, float, dict]] = []
        crop_items: list[tuple[str, float, dict]] = []

        for rank, (doc_name, page_idx) in enumerate(candidate_pages):
            hier = hier_by_name.get(doc_name)
            if hier is None:
                continue

            record = hier.record_for_page(page_idx)
            page_key = (
                record.page_key
                if record and record.page_key
                else (hier.page_keys[page_idx] if page_idx < len(hier.page_keys) else "")
            )
            if not page_key:
                continue

            resolved_key, crop_label = self._resolve_key(page_key)
            if resolved_key is None:
                continue

            # Position-based distance: most-relevant page gets the lowest value.
            page_dist = 0.05 * (rank + 1)

            full_items.append((
                resolved_key,
                page_dist,
                self._page_metadata(doc_name, page_idx, crop_label, record),
            ))

            if record and record.crop_keys:
                crop_dist = page_dist + 0.025
                for crop_key in record.crop_keys:
                    if self.kvstore.has_key(crop_key):
                        crop_items.append((
                            crop_key,
                            crop_dist,
                            self._page_metadata(doc_name, page_idx, "crop", record,
                                                extra={"crop_key": crop_key}),
                        ))

        ids, distances, metadatas = [], [], []

        # Phase 1: full-page images (guaranteed slots before any crops)
        for key, dist, meta in full_items:
            if len(ids) >= self.top_k:
                break
            ids.append(key)
            distances.append(dist)
            metadatas.append(meta)

        # Phase 2: crops fill remaining budget
        for key, dist, meta in crop_items:
            if len(ids) >= self.top_k:
                break
            ids.append(key)
            distances.append(dist)
            metadatas.append(meta)

        return {"ids": [ids], "distances": [distances], "metadatas": [metadatas]}

    # ------------------------------------------------------------------
    # Navigation steps
    # ------------------------------------------------------------------

    def _navigate(
        self, question: str, hierarchies: list[DocumentHierarchy]
    ) -> list[tuple[str, int]]:
        """Document + topic navigation → deduplicated (doc_name, page_idx) pairs."""
        relevant_docs = self._select_documents(question, hierarchies)
        if not relevant_docs:
            self.logger.warning("[hindex] Navigator found no relevant documents")
            return []

        selected: list[tuple[str, int]] = []
        for hierarchy in relevant_docs:
            for page_idx in self._select_pages(question, hierarchy):
                selected.append((hierarchy.doc_name, page_idx))

        seen: set[tuple[str, int]] = set()
        unique: list[tuple[str, int]] = []
        for item in selected:
            if item not in seen:
                seen.add(item)
                unique.append(item)
        return unique

    def _select_documents(
        self, question: str, hierarchies: list[DocumentHierarchy]
    ) -> list[DocumentHierarchy]:
        """VLM picks which document(s) to search.

        At small scale (< COLLECTION_THRESHOLD docs): single-stage flat selection
        — all document descriptions fit comfortably in one prompt.

        At large scale (≥ COLLECTION_THRESHOLD docs): two-stage selection using
        the pre-built collection index:
          Stage 1 — VLM picks 1-2 clusters from a short cluster list.
          Stage 2 — VLM picks 1-2 documents from within those clusters.

        Falls back to flat selection if no collection index is available.
        """
        if len(hierarchies) == 1:
            return hierarchies

        if len(hierarchies) >= COLLECTION_THRESHOLD:
            collection_idx = self.store.load_collection_index()
            if collection_idx:
                return self._select_documents_via_collection(question, hierarchies, collection_idx)

        return self._select_documents_flat(question, hierarchies)

    def _select_documents_flat(
        self, question: str, hierarchies: list[DocumentHierarchy]
    ) -> list[DocumentHierarchy]:
        """Single-stage document selection — works well for small collections."""
        doc_list = "\n".join(
            f"- {h.doc_name}: {h.doc_description} ({h.page_count} pages)"
            for h in hierarchies
        )
        prompt = (
            "Select the document(s) most likely to answer the question.\n"
            "Output a JSON array of document names — no explanation.\n\n"
            f"Question: {question}\n\n"
            f"Documents:\n{doc_list}"
        )
        raw = self.vlm_fn([], prompt, _NAV_TOKENS)
        selected_names = set(_parse_json_list(raw, self.logger))
        matched = [h for h in hierarchies if h.doc_name in selected_names]
        return matched or hierarchies[:1]

    def _select_documents_via_collection(
        self,
        question: str,
        hierarchies: list[DocumentHierarchy],
        collection_idx: CollectionIndex,
    ) -> list[DocumentHierarchy]:
        """Two-stage document selection for large corpora.

        Stage 1: VLM picks 1-2 thematic clusters (short list, fast).
        Stage 2: VLM picks 1-2 documents within those clusters (flat selection).
        """
        hier_by_name = {h.doc_name: h for h in hierarchies}

        # Stage 1 — cluster selection
        cluster_lines = "\n".join(
            f"[{c.cluster_id}] {c.label}: {c.description} ({len(c.doc_names)} docs)"
            for c in collection_idx.clusters
        )
        prompt = (
            "Select the document cluster(s) most likely to contain the answer.\n"
            "Output a JSON array of cluster IDs (e.g. [\"C01\", \"C03\"]) — no explanation.\n\n"
            f"Question: {question}\n\n"
            f"Clusters:\n{cluster_lines}"
        )
        raw = self.vlm_fn([], prompt, _NAV_TOKENS)
        selected_cluster_ids = set(_parse_json_list(raw, self.logger))
        self.logger.debug("[hindex] Selected clusters: %s", selected_cluster_ids)

        # Gather candidate documents from selected clusters
        candidate_names: list[str] = []
        for cluster in collection_idx.clusters:
            if cluster.cluster_id in selected_cluster_ids:
                candidate_names.extend(cluster.doc_names)

        if not candidate_names:
            self.logger.warning(
                "[hindex] No clusters matched for question; falling back to largest cluster"
            )
            largest = max(collection_idx.clusters, key=lambda c: len(c.doc_names))
            candidate_names = largest.doc_names

        candidate_hierarchies = [
            hier_by_name[n] for n in candidate_names if n in hier_by_name
        ]
        if not candidate_hierarchies:
            return hierarchies[:1]

        # Stage 2 — flat document selection within candidates
        return self._select_documents_flat(question, candidate_hierarchies)

    def _select_pages(self, question: str, hierarchy: DocumentHierarchy) -> list[int]:
        """VLM picks relevant pages — strategy depends on document length.

        Short documents (≤ _DIRECT_SELECT_THRESHOLD pages)
            Direct page selection: the VLM ranks page indices using per-page
            summaries and visual descriptions.  One LLM call, no topic
            indirection, pages returned in relevance order.

        Long documents
            Two-step navigation: topic IDs → page indices.  Keeps prompt size
            manageable when a full page list would overflow context.
        """
        if hierarchy.page_count <= _DIRECT_SELECT_THRESHOLD and hierarchy.page_records:
            return self._select_pages_direct(question, hierarchy)
        if hierarchy.topics:
            return self._select_pages_via_topics(question, hierarchy)
        if hierarchy.nodes:
            return self._select_pages_via_nodes(question, hierarchy)
        return list(range(min(hierarchy.page_count, self.top_k)))

    def _select_pages_direct(
        self, question: str, hierarchy: DocumentHierarchy
    ) -> list[int]:
        """Direct page selection for short documents.

        Shows the VLM every page's summary and visual description, then asks it
        to return the most relevant page indices ranked by relevance.  Skips the
        topic abstraction layer entirely, which avoids two failure modes:
        - All pages in a selected topic receiving equal (arbitrary) priority.
        - An irrelevant page winning just because it belongs to the right topic.

        Visual descriptions are highlighted so chart/figure pages are properly
        identified even when their text content is sparse.
        """
        page_lines: list[str] = []
        for r in hierarchy.page_records:
            parts: list[str] = []
            if r.summary:
                parts.append(r.summary)
            if r.visual_description:
                parts.append(f"[visual: {r.visual_description}]")
            if r.keywords:
                parts.append(f"({', '.join(r.keywords)})")
            page_lines.append(f"  p{r.page_idx}: {' '.join(parts) or '(no description)'}")

        page_context = "\n".join(page_lines)[:4000]

        prompt = (
            f"Document: {hierarchy.doc_name}\n\n"
            f"Question: {question}\n\n"
            "Which pages are most relevant to answer this question?\n"
            f"Return a JSON array of 0-based page indices, most relevant first "
            f"(at most {self.top_k}).\n"
            "Output only the JSON array — no explanation.\n\n"
            f"Pages:\n{page_context}"
        )
        raw = self.vlm_fn([], prompt, _NAV_TOKENS)
        raw_list = _parse_json_list(raw, self.logger)

        pages: list[int] = []
        seen: set[int] = set()
        for s in raw_list:
            try:
                idx = int(s)
                if 0 <= idx < hierarchy.page_count and idx not in seen:
                    seen.add(idx)
                    pages.append(idx)
            except (ValueError, TypeError):
                pass

        if not pages:
            self.logger.warning(
                f"[hindex] Direct page selection returned no valid indices for "
                f"'{hierarchy.doc_name}'; falling back to topic navigation"
            )
            if hierarchy.topics:
                return self._select_pages_via_topics(question, hierarchy)
            return list(range(min(hierarchy.page_count, self.top_k)))

        return pages

    def _select_pages_via_topics(
        self, question: str, hierarchy: DocumentHierarchy
    ) -> list[int]:
        """Navigate by topics — used for long documents."""
        topic_list = "\n".join(
            f"[{t.topic_id}] {t.theme}: {t.description}  "
            f"(pages: {', '.join(str(p) for p in t.page_indices)})"
            for t in hierarchy.topics
        )
        # Include per-page summaries as additional context
        page_summaries = "\n".join(
            f"  p{r.page_idx:02d}: {r.summary}  [{', '.join(r.keywords)}]"
            for r in hierarchy.page_records
            if r.summary
        )[:3000]

        prompt = (
            f"Document: {hierarchy.doc_name}\n"
            f"Description: {hierarchy.doc_description}\n\n"
            "Select the topic ID(s) most relevant to the question.\n"
            "Output a JSON array of topic ID strings (e.g. [\"T01\", \"T03\"]).\n"
            "Output only the JSON array.\n\n"
            f"Question: {question}\n\n"
            f"Topics:\n{topic_list}\n\n"
            f"Page summaries:\n{page_summaries}"
        )
        raw = self.vlm_fn([], prompt, _NAV_TOKENS)
        selected_ids = set(_parse_json_list(raw, self.logger))

        pages: list[int] = []
        seen: set[int] = set()
        for topic in hierarchy.topics:
            if topic.topic_id in selected_ids:
                for p in topic.page_indices:
                    if p not in seen:
                        seen.add(p)
                        pages.append(p)

        if not pages:
            self.logger.warning(
                f"[hindex] No topics matched for '{hierarchy.doc_name}'; "
                "falling back to first topic"
            )
            if hierarchy.topics:
                pages = list(hierarchy.topics[0].page_indices)

        return pages

    def _select_pages_via_nodes(
        self, question: str, hierarchy: DocumentHierarchy
    ) -> list[int]:
        """Legacy navigation via HierarchyNodes."""
        tree_text = hierarchy.as_text_tree()
        prompt = (
            f"Navigate '{hierarchy.doc_name}' to find the answer.\n"
            "Return the node IDs (e.g. [\"0001\", \"0003\"]) of relevant sections.\n"
            "Output only the JSON array.\n\n"
            f"Question: {question}\n\n"
            f"Structure:\n{tree_text}"
        )
        raw = self.vlm_fn([], prompt, _NAV_TOKENS)
        selected_ids = set(_parse_json_list(raw, self.logger))

        pages: list[int] = []
        seen: set[int] = set()
        for node in hierarchy.flat_nodes():
            if node.node_id in selected_ids:
                for p in range(node.start_page, node.end_page + 1):
                    if p not in seen:
                        seen.add(p)
                        pages.append(p)

        if not pages and hierarchy.nodes:
            n = hierarchy.nodes[0]
            pages = list(range(n.start_page, n.end_page + 1))
        return pages

    # ------------------------------------------------------------------
    # Key resolution + metadata helpers
    # ------------------------------------------------------------------

    def _resolve_key(self, page_key: str) -> tuple[str | None, str]:
        """Return (resolved_kvstore_key, crop_label), falling back to crop_0000."""
        try:
            if self.kvstore.has_key(page_key):
                return page_key, "full_page"
            fallback = f"{page_key}_crop_0000"
            if self.kvstore.has_key(fallback):
                self.logger.debug(
                    f"[hindex] Full-page key '{page_key}' absent; using crop fallback"
                )
                return fallback, "crop"
        except Exception as exc:
            self.logger.warning(f"[hindex] kvstore check failed for '{page_key}': {exc}")
        self.logger.warning(f"[hindex] Key '{page_key}' not found in kvstore")
        return None, ""

    @staticmethod
    def _page_metadata(
        doc_name: str,
        page_idx: int,
        crop_label: str,
        record: PageRecord | None,
        extra: dict | None = None,
    ) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "source": doc_name,
            "page_number": page_idx + 1,  # 1-indexed, consistent with ChromaDB convention
            "crop_label": crop_label,
            "retrieval_source": "hindex",
        }
        if record:
            meta["text_content"] = record.text_content
            meta["visual_description"] = record.visual_description
            meta["summary"] = record.summary
            meta["keywords"] = record.keywords
        if extra:
            meta.update(extra)
        return meta


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page_key(uuid5_val: str, page_idx: int) -> str:
    return f"{uuid5_val}_{page_idx:04d}"


def _parse_json_list(raw: str, logger) -> list[str]:
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        result = json.loads(raw)
        if isinstance(result, list):
            return [str(x) for x in result]
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                result = json.loads(m.group(0))
                if isinstance(result, list):
                    return [str(x) for x in result]
            except json.JSONDecodeError:
                pass
    logger.warning(f"[hindex] Could not parse JSON list from: {raw[:200]}")
    return []


def _empty_docs() -> dict[str, Any]:
    return {"ids": [[]], "distances": [[]], "metadatas": [[]]}


class _NullLogger:
    def info(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass
    def debug(self, *_a, **_k): pass
