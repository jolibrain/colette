"""Agentic hierarchy retriever.

At query time the VLM acts as a navigator:
  1. It receives the document tree as a compact text summary.
  2. It selects the node IDs whose page ranges are most relevant to the question.
  3. Page images for those ranges are fetched from the HDF5 kvstore.
  4. Retrieved images are returned in the same format used by RAGImgRetriever so
     they can be merged into the hybrid context without any downstream changes.

The three logical operations mirror PageIndex's tool pattern:
  - get_documents()           → list of available docs with metadata
  - get_document_structure()  → compact text tree (no images)
  - get_page_images()         → PIL images for a page range

These are not exposed as external tools here; they are called internally in the
agentic loop.  A future refactor could expose them as MCP tools.
"""

from __future__ import annotations

import json
import re
from typing import Any

from PIL import Image

from .hierarchy_builder import DocumentHierarchy, HierarchyNode
from .hierarchy_store import HierarchyStore


# Maximum nodes the navigator may select in a single query.
_MAX_SELECTED_NODES = 6


class HierarchyRetriever:
    """Retrieves page images by navigating the document hierarchy.

    Parameters
    ----------
    store:
        HierarchyStore used to look up document trees.
    kvstore:
        Colette's HDF5ImageStorage (or compatible) for fetching page images.
        Keys follow the convention used by the preprocessing pipeline:
        ``<source_slug>_page_<N>`` where N is the 0-based physical page index.
    vlm_fn:
        Same callable used by HierarchyBuilder — Callable(images, prompt) -> str.
    top_k:
        Maximum number of page images to return.
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
    # Public API — mirrors RAGImgRetriever.invoke() return format
    # ------------------------------------------------------------------

    def invoke(self, question: str) -> dict[str, Any]:
        """Return page images most relevant to *question*.

        Return value is compatible with the ``docs`` dict consumed by
        ``RAGImgRetriever`` downstream (ids, distances, metadatas keys).
        """
        hierarchies = self.store.load_all()
        if not hierarchies:
            self.logger.warning("[hindex] No document hierarchies found in store")
            return _empty_docs()

        # Step 1 — Ask the VLM which document(s) to look at
        relevant_docs = self._select_documents(question, hierarchies)
        if not relevant_docs:
            self.logger.warning("[hindex] Navigator found no relevant documents")
            return _empty_docs()

        # Step 2 — For each selected doc, navigate the tree to select nodes
        selected_pages: list[tuple[str, int]] = []  # (doc_name, page_idx)
        for hierarchy in relevant_docs:
            nodes = self._navigate_tree(question, hierarchy)
            for node in nodes:
                for page in range(node.start_page, node.end_page + 1):
                    selected_pages.append((hierarchy.doc_name, page))

        if not selected_pages:
            return _empty_docs()

        # Deduplicate while preserving order
        seen: set[tuple[str, int]] = set()
        unique_pages: list[tuple[str, int]] = []
        for item in selected_pages:
            if item not in seen:
                seen.add(item)
                unique_pages.append(item)

        # Step 3 — Fetch images from kvstore (up to top_k)
        unique_pages = unique_pages[: self.top_k]
        ids, distances, metadatas = [], [], []
        for doc_name, page_idx in unique_pages:
            key = _page_key(doc_name, page_idx)
            try:
                # We don't load the PIL image here — downstream code fetches it
                # from the kvstore using the key stored in metadatas.
                if self.kvstore.has_key(key):
                    ids.append(key)
                    distances.append(0.0)  # hierarchy retrieval has no distance score
                    metadatas.append(
                        {
                            "source": doc_name,
                            "page_number": page_idx,
                            "crop_label": "full_page",
                            "retrieval_source": "hindex",
                        }
                    )
            except Exception as exc:
                self.logger.warning(f"[hindex] Could not check kvstore key '{key}': {exc}")

        return {
            "ids": [ids],
            "distances": [distances],
            "metadatas": [metadatas],
        }

    # ------------------------------------------------------------------
    # Navigation steps
    # ------------------------------------------------------------------

    def _select_documents(
        self, question: str, hierarchies: list[DocumentHierarchy]
    ) -> list[DocumentHierarchy]:
        """Ask the VLM to select the most relevant document(s) for the question."""
        if len(hierarchies) == 1:
            return hierarchies

        doc_list = "\n".join(
            f"- {h.doc_name}: {h.doc_description} ({h.page_count} pages)"
            for h in hierarchies
        )
        prompt = (
            f"You are given a list of documents and a question.\n"
            f"Select the document names most likely to contain the answer.\n"
            f"Output a JSON array of document names (strings) — no explanation.\n\n"
            f"Question: {question}\n\n"
            f"Documents:\n{doc_list}"
        )
        raw = self.vlm_fn([], prompt)
        selected_names = _parse_json_list(raw, self.logger)
        if not selected_names:
            return hierarchies[:1]
        name_set = set(selected_names)
        return [h for h in hierarchies if h.doc_name in name_set] or hierarchies[:1]

    def _navigate_tree(
        self, question: str, hierarchy: DocumentHierarchy
    ) -> list[HierarchyNode]:
        """Ask the VLM to pick the most relevant node IDs from the tree."""
        tree_text = hierarchy.as_text_tree()
        prompt = (
            f"You are navigating the structure of '{hierarchy.doc_name}'.\n"
            f"Select the node IDs (the bracketed codes like [0001]) that are most "
            f"likely to contain the answer to the question below.\n"
            f"Return a JSON array of node ID strings — at most {_MAX_SELECTED_NODES}.\n"
            f"Output only the JSON array.\n\n"
            f"Question: {question}\n\n"
            f"Document structure:\n{tree_text}"
        )
        raw = self.vlm_fn([], prompt)
        selected_ids = set(_parse_json_list(raw, self.logger))

        flat = hierarchy.flat_nodes()
        matched = [n for n in flat if n.node_id in selected_ids]
        if not matched:
            self.logger.warning(
                f"[hindex] Navigator returned no matching nodes for '{hierarchy.doc_name}'; "
                "falling back to top-level nodes"
            )
            matched = hierarchy.nodes[:2]
        return matched


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _page_key(doc_name: str, page_idx: int) -> str:
    slug = re.sub(r"[^A-Za-z0-9_\-]", "_", doc_name)
    return f"{slug}_page_{page_idx}"


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
