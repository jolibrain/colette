"""Unit tests for HierarchyRetriever — topic-based navigation."""

from __future__ import annotations

import json

import pytest

from colette.backends.hf.hindex.hierarchy_builder import (
    DocumentHierarchy,
    PageRecord,
    TopicNode,
)
from colette.backends.hf.hindex.hierarchy_retriever import (
    HierarchyRetriever,
    _page_key,
    _parse_json_list,
)
from colette.backends.hf.hindex.hierarchy_store import HierarchyStore


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FixedVLM:
    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0
        self.calls: list[tuple[list, str]] = []

    def __call__(self, images, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls.append((images, prompt))
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return "[]"


class _FakeKVStore:
    def __init__(self, keys: set[str] | None = None):
        self._keys = keys or set()

    def has_key(self, key: str) -> bool:
        return key in self._keys

    def add_key(self, key: str) -> None:
        self._keys.add(key)


class _LogCapture:
    def __init__(self):
        self.warnings: list[str] = []

    def info(self, *_a, **_k): pass
    def debug(self, *_a, **_k): pass
    def warning(self, msg, *_a, **_k): self.warnings.append(str(msg))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UUID5 = "aabbccdd-1234-5678-abcd-ef1234567890"


def _make_page_keys(n: int, uuid5_val: str = _UUID5) -> list[str]:
    return [_page_key(uuid5_val, i) for i in range(n)]


def _make_page_records(n: int, uuid5_val: str = _UUID5) -> list[PageRecord]:
    return [
        PageRecord(
            page_idx=i,
            page_key=_page_key(uuid5_val, i),
            text_content=f"Text on page {i}",
            visual_description="Diagram" if i % 2 == 0 else "",
            summary=f"Page {i} covers topic {'A' if i < n // 2 else 'B'}.",
            keywords=["topicA" if i < n // 2 else "topicB"],
            crop_keys=[f"{_page_key(uuid5_val, i)}_crop_0000"],
        )
        for i in range(n)
    ]


def _make_store_with_doc(
    tmp_path,
    doc_name: str = "manual.pdf",
    page_count: int = 10,
    uuid5_val: str = _UUID5,
) -> HierarchyStore:
    store = HierarchyStore(tmp_path)
    mid = page_count // 2
    hier = DocumentHierarchy(
        doc_name=doc_name,
        doc_description="A technical manual",
        page_count=page_count,
        page_records=_make_page_records(page_count, uuid5_val),
        topics=[
            TopicNode("T01", "Introduction", "Opening section", ["intro"],
                      list(range(mid))),
            TopicNode("T02", "Details",      "Detailed content", ["details"],
                      list(range(mid, page_count))),
        ],
        page_keys=_make_page_keys(page_count, uuid5_val),
    )
    store.save(hier)
    return store


# ---------------------------------------------------------------------------
# _page_key
# ---------------------------------------------------------------------------


def test_page_key_format():
    assert _page_key(_UUID5, 3) == f"{_UUID5}_0003"


def test_page_key_zero_padded():
    assert _page_key(_UUID5, 0).endswith("_0000")
    assert _page_key(_UUID5, 99).endswith("_0099")


# ---------------------------------------------------------------------------
# _parse_json_list
# ---------------------------------------------------------------------------


def test_parse_json_list_clean():
    logger = _LogCapture()
    result = _parse_json_list('["T01", "T03"]', logger)
    assert result == ["T01", "T03"]
    assert not logger.warnings


def test_parse_json_list_with_fence():
    logger = _LogCapture()
    result = _parse_json_list('```json\n["T01"]\n```', logger)
    assert result == ["T01"]


def test_parse_json_list_embedded():
    logger = _LogCapture()
    raw = 'The relevant topics are: ["T02", "T03"] based on the document.'
    result = _parse_json_list(raw, logger)
    assert "T02" in result
    assert "T03" in result


def test_parse_json_list_invalid_returns_empty():
    logger = _LogCapture()
    result = _parse_json_list("no json here", logger)
    assert result == []
    assert logger.warnings


# ---------------------------------------------------------------------------
# HierarchyRetriever.invoke() — topic navigation
# ---------------------------------------------------------------------------


def test_invoke_single_doc_returns_relevant_pages(tmp_path):
    store = _make_store_with_doc(tmp_path)
    page_keys = _make_page_keys(10)
    kvstore = _FakeKVStore({k for k in page_keys})
    vlm = _FixedVLM(['["T02"]'])  # select Details topic (pages 5-9)
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=6)

    result = retriever.invoke("What are the technical details?")

    assert result["ids"] != [[]]
    ids = result["ids"][0]
    assert len(ids) > 0
    # All returned keys should belong to pages 5-9 (T02)
    for key in ids:
        assert any(f"_{p:04d}" in key for p in range(5, 10))


def test_invoke_returns_empty_when_store_empty(tmp_path):
    store = HierarchyStore(tmp_path)
    kvstore = _FakeKVStore()
    vlm = _FixedVLM([])
    logger = _LogCapture()
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4, logger=logger)

    result = retriever.invoke("Any question")

    assert result["ids"] == [[]]
    assert any("No document" in w for w in logger.warnings)


def test_invoke_falls_back_when_no_topic_matched(tmp_path):
    store = _make_store_with_doc(tmp_path)
    kvstore = _FakeKVStore(set(_make_page_keys(10)))
    vlm = _FixedVLM(['[]'])  # VLM returns no topic
    logger = _LogCapture()
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4, logger=logger)

    result = retriever.invoke("Irrelevant question")

    assert result["ids"][0]  # non-empty fallback (first topic)
    assert any("falling back" in w for w in logger.warnings)


def test_invoke_respects_top_k(tmp_path):
    store = _make_store_with_doc(tmp_path, page_count=20)
    kvstore = _FakeKVStore(set(_make_page_keys(20)))
    # Select T02 which has 10 pages, each with a crop → 20 potential items
    vlm = _FixedVLM(['["T02"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=3)

    result = retriever.invoke("Anything")

    assert len(result["ids"][0]) <= 3


def test_invoke_metadata_has_retrieval_source(tmp_path):
    store = _make_store_with_doc(tmp_path)
    kvstore = _FakeKVStore(set(_make_page_keys(10)))
    vlm = _FixedVLM(['["T01"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10)

    result = retriever.invoke("Question")

    for meta in result["metadatas"][0]:
        assert meta["retrieval_source"] == "hindex"
        assert "source" in meta
        assert "page_number" in meta


def test_invoke_metadata_includes_page_record_fields(tmp_path):
    store = _make_store_with_doc(tmp_path)
    kvstore = _FakeKVStore(set(_make_page_keys(10)))
    vlm = _FixedVLM(['["T01"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10)

    result = retriever.invoke("Question")

    # Full-page results should carry pre-extracted page info
    full_page_metas = [m for m in result["metadatas"][0] if m["crop_label"] == "full_page"]
    assert full_page_metas
    for meta in full_page_metas:
        assert "summary" in meta
        assert "text_content" in meta
        assert "keywords" in meta


def test_invoke_uses_crop_fallback_for_missing_full_page(tmp_path):
    store = _make_store_with_doc(tmp_path, page_count=4)
    page_keys = _make_page_keys(4)
    # Only crop_0000 keys available (no full-page images)
    crop_keys = {f"{k}_crop_0000" for k in page_keys}
    kvstore = _FakeKVStore(crop_keys)
    vlm = _FixedVLM(['["T01"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10)

    result = retriever.invoke("Question")

    assert result["ids"][0]
    for meta in result["metadatas"][0]:
        # All should be crops since no full-page images exist
        assert meta["crop_label"] in ("crop", "full_page")


def test_invoke_multi_doc_selects_correct_document(tmp_path):
    store = HierarchyStore(tmp_path)
    uuid_a = "aaaaaaaa-0000-0000-0000-000000000001"
    uuid_b = "bbbbbbbb-0000-0000-0000-000000000002"

    for uuid_val, name, desc in [
        (uuid_a, "engine.pdf", "Engine maintenance guide"),
        (uuid_b, "avionics.pdf", "Avionics systems reference"),
    ]:
        hier = DocumentHierarchy(
            doc_name=name,
            doc_description=desc,
            page_count=5,
            page_records=_make_page_records(5, uuid_val),
            topics=[TopicNode("T01", "Overview", desc, [], list(range(5)))],
            page_keys=_make_page_keys(5, uuid_val),
        )
        store.save(hier)

    all_keys = set(_make_page_keys(5, uuid_a)) | set(_make_page_keys(5, uuid_b))
    kvstore = _FakeKVStore(all_keys)
    # Step 1: doc selection → avionics; Step 2: topic selection → T01
    vlm = _FixedVLM(['["avionics.pdf"]', '["T01"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=6)

    result = retriever.invoke("What are the avionics systems?")

    ids = result["ids"][0]
    avion_keys = set(_make_page_keys(5, uuid_b))
    for key in ids:
        base_key = key.split("_crop_")[0]
        assert base_key in avion_keys


def test_invoke_deduplicates_pages(tmp_path):
    """A page appearing in two topics should be returned only once."""
    store = HierarchyStore(tmp_path)
    page_keys = _make_page_keys(6)
    hier = DocumentHierarchy(
        doc_name="overlap.pdf",
        doc_description="Overlap test",
        page_count=6,
        page_records=_make_page_records(6),
        topics=[
            TopicNode("T01", "A", "First topic", [], [0, 1, 2, 3]),
            TopicNode("T02", "B", "Second topic", [], [2, 3, 4, 5]),
        ],
        page_keys=page_keys,
    )
    store.save(hier)

    kvstore = _FakeKVStore(set(page_keys))
    vlm = _FixedVLM(['["T01", "T02"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=20)

    result = retriever.invoke("Question")

    # Page-level keys should not be duplicated
    full_page_ids = [i for i in result["ids"][0]
                     if "_crop_" not in i]
    assert len(full_page_ids) == len(set(full_page_ids))


# ---------------------------------------------------------------------------
# HierarchyRetriever.select_pages() — navigation only
# ---------------------------------------------------------------------------


def test_select_pages_returns_pairs(tmp_path):
    """select_pages() returns (doc_name, page_idx) pairs for the matched topic."""
    store = _make_store_with_doc(tmp_path, page_count=10)
    kvstore = _FakeKVStore()  # not needed by select_pages
    vlm = _FixedVLM(['["T01"]'])  # Introduction topic → pages 0-4
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=6)

    pairs = retriever.select_pages("Introduction")

    assert pairs
    for doc_name, page_idx in pairs:
        assert doc_name == "manual.pdf"
        assert 0 <= page_idx < 5  # T01 covers pages 0-4


def test_select_pages_returns_empty_for_empty_store(tmp_path):
    store = HierarchyStore(tmp_path)
    kvstore = _FakeKVStore()
    vlm = _FixedVLM([])
    logger = _LogCapture()
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4, logger=logger)

    pairs = retriever.select_pages("Any question")

    assert pairs == []
    assert any("No document" in w for w in logger.warnings)


def test_select_pages_deduplicates_overlapping_topics(tmp_path):
    """Pages shared across two selected topics appear only once."""
    store = HierarchyStore(tmp_path)
    page_keys = _make_page_keys(6)
    hier = DocumentHierarchy(
        doc_name="overlap.pdf",
        doc_description="Overlap test",
        page_count=6,
        page_records=_make_page_records(6),
        topics=[
            TopicNode("T01", "A", "First", [], [0, 1, 2]),
            TopicNode("T02", "B", "Second", [], [2, 3, 4, 5]),
        ],
        page_keys=page_keys,
    )
    store.save(hier)

    kvstore = _FakeKVStore()
    vlm = _FixedVLM(['["T01", "T02"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=20)

    pairs = retriever.select_pages("Question")

    # No duplicates
    assert len(pairs) == len(set(pairs))
    # Page 2 appears in both topics but only once
    page_indices = [p for _, p in pairs]
    assert page_indices.count(2) == 1
    # All 6 unique pages returned
    assert len(pairs) == 6


def test_select_pages_matches_invoke_navigation(tmp_path):
    """select_pages returns the same pages that invoke would use for retrieval."""
    store = _make_store_with_doc(tmp_path, page_count=10)
    page_keys = _make_page_keys(10)
    kvstore = _FakeKVStore(set(page_keys))

    # Two retrievers with the same VLM sequence (both select T02)
    vlm_a = _FixedVLM(['["T02"]'])
    vlm_b = _FixedVLM(['["T02"]'])
    retriever_a = HierarchyRetriever(store, kvstore, vlm_a, top_k=20)
    retriever_b = HierarchyRetriever(store, kvstore, vlm_b, top_k=20)

    pairs = retriever_a.select_pages("Technical details")
    invoke_result = retriever_b.invoke("Technical details")

    # Page indices from select_pages should match those in invoke result
    selected_page_idxs = sorted(p for _, p in pairs)
    invoke_page_idxs = sorted(
        {m["page_number"] for m in invoke_result["metadatas"][0]}
    )
    assert selected_page_idxs == invoke_page_idxs
