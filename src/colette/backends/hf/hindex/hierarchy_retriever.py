"""Agentic hierarchy retriever — topic-based navigation.

At query time:
  1. Select the relevant document(s) from their descriptions.
  2. Select the most relevant topic(s) from the topic list.
  3. Collect the page indices for those topics.
  4. For each page, return the full-page image key AND all crop keys,
     enriched with the pre-extracted per-page metadata (OCR, summary, etc.)
     so the answering VLM gets richer context without re-reading from scratch.

Return format is compatible with RAGImgRetriever so results merge cleanly
into the hybrid context.
"""

from __future__ import annotations

import json
import re
from typing import Any

from .hierarchy_builder import DocumentHierarchy, PageRecord
from .hierarchy_store import HierarchyStore


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
        """
        hierarchies = self.store.load_all()
        if not hierarchies:
            self.logger.warning("[hindex] No document hierarchies found in store")
            return _empty_docs()

        # Step 1 — Select relevant document(s)
        relevant_docs = self._select_documents(question, hierarchies)
        if not relevant_docs:
            self.logger.warning("[hindex] Navigator found no relevant documents")
            return _empty_docs()

        # Step 2 — For each doc, select relevant topics → collect page indices
        selected_pages: list[tuple[str, int]] = []  # (doc_name, page_idx)
        for hierarchy in relevant_docs:
            page_indices = self._select_pages(question, hierarchy)
            for page_idx in page_indices:
                selected_pages.append((hierarchy.doc_name, page_idx))

        if not selected_pages:
            return _empty_docs()

        # Deduplicate, preserve order
        seen: set[tuple[str, int]] = set()
        unique: list[tuple[str, int]] = []
        for item in selected_pages:
            if item not in seen:
                seen.add(item)
                unique.append(item)

        # Step 3 — Build result items (page image + crops, enriched metadata)
        hier_by_name = {h.doc_name: h for h in hierarchies}
        ids, distances, metadatas = [], [], []

        for doc_name, page_idx in unique:
            if len(ids) >= self.top_k:
                break
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

            # Resolve full-page key (fall back to first crop if full-page absent)
            resolved_key, crop_label = self._resolve_key(page_key)
            if resolved_key is None:
                continue

            # Base metadata enriched with pre-extracted page information
            base_meta = self._page_metadata(doc_name, page_idx, crop_label, record)
            ids.append(resolved_key)
            distances.append(0.0)
            metadatas.append(base_meta)

            # Also include individual crop keys for this page
            if record and record.crop_keys:
                for crop_key in record.crop_keys:
                    if len(ids) >= self.top_k:
                        break
                    if self.kvstore.has_key(crop_key):
                        ids.append(crop_key)
                        distances.append(0.0)
                        metadatas.append(
                            self._page_metadata(doc_name, page_idx, "crop", record,
                                                extra={"crop_key": crop_key})
                        )

        return {"ids": [ids], "distances": [distances], "metadatas": [metadatas]}

    # ------------------------------------------------------------------
    # Navigation steps
    # ------------------------------------------------------------------

    def _select_documents(
        self, question: str, hierarchies: list[DocumentHierarchy]
    ) -> list[DocumentHierarchy]:
        """VLM picks which document(s) to search."""
        if len(hierarchies) == 1:
            return hierarchies

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
        raw = self.vlm_fn([], prompt)
        selected_names = set(_parse_json_list(raw, self.logger))
        matched = [h for h in hierarchies if h.doc_name in selected_names]
        return matched or hierarchies[:1]

    def _select_pages(self, question: str, hierarchy: DocumentHierarchy) -> list[int]:
        """VLM picks relevant topic IDs → returns their page indices."""
        # New model: navigate by topics
        if hierarchy.topics:
            return self._select_pages_via_topics(question, hierarchy)
        # Legacy fallback: navigate by flat nodes
        if hierarchy.nodes:
            return self._select_pages_via_nodes(question, hierarchy)
        # Last resort: return all pages up to top_k
        return list(range(min(hierarchy.page_count, self.top_k)))

    def _select_pages_via_topics(
        self, question: str, hierarchy: DocumentHierarchy
    ) -> list[int]:
        """Navigate by topics — the new model."""
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
        raw = self.vlm_fn([], prompt)
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
        raw = self.vlm_fn([], prompt)
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
            "page_number": page_idx,
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
        m = re.search(r"\[.*?\]", raw, re.DOTALL)
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
