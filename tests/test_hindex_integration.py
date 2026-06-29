"""Integration test: builder → store → retriever pipeline.

Exercises the full hindex pipeline end-to-end without any real VLM or GPU.
Tests that the three components wire together correctly and that the output
of the retriever is compatible with the format expected by RAGImgRetriever.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from colette.backends.hf.hindex import HierarchyBuilder, HierarchyRetriever, HierarchyStore, _page_key


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _blank_page() -> Image.Image:
    return Image.new("RGB", (100, 100), color=255)


class _SequentialVLM:
    """Returns a pre-configured sequence of responses."""

    def __init__(self, responses: list[str]):
        self._q = list(responses)
        self._idx = 0

    def __call__(self, images, prompt: str) -> str:
        if self._idx < len(self._q):
            resp = self._q[self._idx]
            self._idx += 1
            return resp
        return ""


class _FakeKVStore:
    def __init__(self, keys: set[str]):
        self._keys = keys

    def has_key(self, key: str) -> bool:
        return key in self._keys


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_vlm_responses(doc_name: str, n_pages: int) -> list[str]:
    """Produce a consistent fake VLM response sequence for a 2-chapter document."""
    toc_scan = 3
    # OCR responses for toc scan
    ocr = [f"Page {i} text" for i in range(min(toc_scan, n_pages))]
    # TOC detection
    toc = json.dumps({
        "has_toc": True,
        "entries": [
            {"title": "Introduction", "level": 1, "page_label": "1"},
            {"title": "Core Concepts", "level": 1, "page_label": "4"},
            {"title": "Overview", "level": 2, "page_label": "4"},
            {"title": "Details", "level": 2, "page_label": "6"},
            {"title": "Conclusion", "level": 1, "page_label": "8"},
        ],
    })
    description = f"Comprehensive guide to {doc_name}."
    # Summaries: one per top-level node (3 top-level)
    summaries = ["Intro summary.", "Core summary.", "Conclusion summary."]
    # Level-2 summaries (2 children under Core Concepts)
    child_summaries = ["Overview summary.", "Details summary."]
    return ocr + [toc, description] + summaries + child_summaries


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


def test_full_pipeline_build_store_retrieve(tmp_path):
    doc_name = "technical_manual.pdf"
    n_pages = 10
    pages = [_blank_page() for _ in range(n_pages)]

    # Page label map: physical index → printed label (offset: preface uses i/ii/iii)
    page_label_map = {
        0: "i", 1: "ii", 2: "iii",
        3: "1",  # chapter starts at physical page 3
        4: "2",
        5: "3",
        6: "4",
        7: "5",
        8: "6",
        9: "7",
    }

    # Adjust toc entries to match this label map
    responses = [f"Page {i} text" for i in range(3)] + [
        json.dumps({
            "has_toc": True,
            "entries": [
                {"title": "Introduction", "level": 1, "page_label": "1"},
                {"title": "Methods", "level": 1, "page_label": "4"},
                {"title": "Results", "level": 1, "page_label": "6"},
            ],
        }),
        "A technical manual covering procedures.",
        "Intro summary.",
        "Methods summary.",
        "Results summary.",
    ]

    vlm_build = _SequentialVLM(responses)
    builder = HierarchyBuilder(
        vlm_fn=vlm_build,
        page_label_map=page_label_map,
        toc_scan_pages=3,
        add_node_summaries=True,
    )

    # 1. Build
    hier = builder.build(doc_name, pages)
    assert hier.page_count == n_pages
    assert len(hier.nodes) == 3

    # 2. Store — populate page_keys (normally done by RAGImg._build_hindex_for_doc)
    _FAKE_UUID5 = "deadbeef-0000-0000-0000-000000000001"
    hier.page_keys = [_page_key(_FAKE_UUID5, i) for i in range(n_pages)]

    store = HierarchyStore(tmp_path)
    store.save(hier)
    assert store.exists(doc_name)

    # Verify round-trip fidelity
    loaded = store.load(doc_name)
    assert loaded.doc_name == doc_name
    assert len(loaded.nodes) == 3
    assert loaded.nodes[0].title == "Introduction"
    assert loaded.nodes[1].title == "Methods"

    # 3. Retrieve
    all_keys = set(hier.page_keys)
    kvstore = _FakeKVStore(all_keys)
    vlm_retrieve = _SequentialVLM(['["0002"]'])  # select Methods node

    retriever = HierarchyRetriever(store, kvstore, vlm_retrieve, top_k=5)
    result = retriever.invoke("How were the experiments conducted?")

    # Output must be compatible with RAGImgRetriever format
    assert "ids" in result
    assert "distances" in result
    assert "metadatas" in result
    assert isinstance(result["ids"], list)
    assert isinstance(result["ids"][0], list)

    ids = result["ids"][0]
    assert len(ids) > 0
    for meta in result["metadatas"][0]:
        assert meta["retrieval_source"] == "hindex"
        assert meta["source"] == doc_name


def test_pipeline_output_is_compatible_with_ragimg_format(tmp_path):
    """The dict keys and value shapes must match what RAGImgRetriever returns."""
    store = HierarchyStore(tmp_path)
    from colette.backends.hf.hindex.hierarchy_builder import DocumentHierarchy, HierarchyNode

    _UUID = "cafebabe-0000-0000-0000-000000000001"
    node = HierarchyNode("Chapter 1", "0001", 0, 2)
    hier = DocumentHierarchy("compat.pdf", "Compatibility test", 3, nodes=[node],
                             page_keys=[_page_key(_UUID, i) for i in range(3)])
    store.save(hier)

    kvstore = _FakeKVStore(set(hier.page_keys))
    vlm = _SequentialVLM(['["0001"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=3)

    result = retriever.invoke("Anything")

    # Must have exactly these keys (subset of what RAGImgRetriever may return)
    assert set(result.keys()) >= {"ids", "distances", "metadatas"}
    assert len(result["ids"]) == 1
    assert len(result["distances"]) == 1
    assert len(result["metadatas"]) == 1
    assert len(result["ids"][0]) == len(result["metadatas"][0])
    assert len(result["ids"][0]) == len(result["distances"][0])
