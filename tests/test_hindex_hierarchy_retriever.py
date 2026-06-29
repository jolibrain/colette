"""Unit tests for HierarchyRetriever.

All VLM calls and kvstore access are replaced with deterministic fakes.
"""

from __future__ import annotations

import json

import pytest

from colette.backends.hf.hindex.hierarchy_builder import DocumentHierarchy, HierarchyNode
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

    def __call__(self, images, prompt: str) -> str:
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
    def warning(self, msg, *_a, **_k): self.warnings.append(msg)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_UUID5 = "aabbccdd-1234-5678-abcd-ef1234567890"


def _make_page_keys(n: int, uuid5_val: str = _UUID5) -> list[str]:
    return [_page_key(uuid5_val, i) for i in range(n)]


def _make_store_with_doc(
    tmp_path,
    doc_name="manual.pdf",
    page_count=10,
    uuid5_val: str = _UUID5,
) -> HierarchyStore:
    store = HierarchyStore(tmp_path)
    root = HierarchyNode("Chapter 1", "0001", 0, 4, summary="Introduction")
    root.children = [HierarchyNode("Section 1.1", "0002", 1, 2)]
    section2 = HierarchyNode("Chapter 2", "0003", 5, 9, summary="Details")
    hier = DocumentHierarchy(
        doc_name,
        "A technical manual",
        page_count,
        nodes=[root, section2],
        page_keys=_make_page_keys(page_count, uuid5_val),
    )
    store.save(hier)
    return store


# ---------------------------------------------------------------------------
# _page_key
# ---------------------------------------------------------------------------


def test_page_key_format():
    key = _page_key(_UUID5, 3)
    assert key == f"{_UUID5}_0003"


def test_page_key_zero_padded():
    assert _page_key(_UUID5, 0).endswith("_0000")
    assert _page_key(_UUID5, 99).endswith("_0099")


# ---------------------------------------------------------------------------
# _parse_json_list
# ---------------------------------------------------------------------------


def test_parse_json_list_clean():
    logger = _LogCapture()
    result = _parse_json_list('["0001", "0003"]', logger)
    assert result == ["0001", "0003"]
    assert not logger.warnings


def test_parse_json_list_with_fence():
    logger = _LogCapture()
    result = _parse_json_list('```json\n["0001"]\n```', logger)
    assert result == ["0001"]


def test_parse_json_list_embedded():
    logger = _LogCapture()
    raw = 'The relevant sections are: ["0002", "0003"] based on the document.'
    result = _parse_json_list(raw, logger)
    assert "0002" in result
    assert "0003" in result


def test_parse_json_list_invalid_returns_empty():
    logger = _LogCapture()
    result = _parse_json_list("no json here", logger)
    assert result == []
    assert logger.warnings


# ---------------------------------------------------------------------------
# HierarchyRetriever.invoke() — single document
# ---------------------------------------------------------------------------


def test_invoke_single_doc_returns_relevant_pages(tmp_path):
    store = _make_store_with_doc(tmp_path)
    page_keys = _make_page_keys(10)
    kvstore = _FakeKVStore(set(page_keys[5:10]))  # pages 5-9 available
    vlm = _FixedVLM(['["0003"]'])  # select Chapter 2 (pages 5-9)
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4)

    result = retriever.invoke("What are the technical details?")

    assert result["ids"] != [[]]
    ids = result["ids"][0]
    assert len(ids) > 0
    assert len(ids) <= 4


def test_invoke_returns_empty_when_store_empty(tmp_path):
    store = HierarchyStore(tmp_path)
    kvstore = _FakeKVStore()
    vlm = _FixedVLM([])
    logger = _LogCapture()
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4, logger=logger)

    result = retriever.invoke("Any question")

    assert result["ids"] == [[]]
    assert any("No document" in w for w in logger.warnings)


def test_invoke_falls_back_when_navigator_returns_no_match(tmp_path):
    store = _make_store_with_doc(tmp_path)
    page_keys = _make_page_keys(10)
    kvstore = _FakeKVStore(set(page_keys))
    # VLM returns an empty list — no nodes selected
    vlm = _FixedVLM(['[]'])
    logger = _LogCapture()
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4, logger=logger)

    result = retriever.invoke("Irrelevant question")

    # Falls back to top-level nodes (Chapter 1 pages 0-4)
    assert result["ids"][0]  # non-empty fallback
    assert any("falling back" in w for w in logger.warnings)


def test_invoke_respects_top_k(tmp_path):
    store = HierarchyStore(tmp_path)
    page_keys = _make_page_keys(20)
    big = HierarchyNode("Big Chapter", "0001", 0, 19)
    hier = DocumentHierarchy("big.pdf", "A very long document", 20,
                             nodes=[big], page_keys=page_keys)
    store.save(hier)

    kvstore = _FakeKVStore(set(page_keys))
    vlm = _FixedVLM(['["0001"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=3)

    result = retriever.invoke("Anything")

    assert len(result["ids"][0]) <= 3


def test_invoke_deduplicates_pages(tmp_path):
    store = HierarchyStore(tmp_path)
    page_keys = _make_page_keys(6)
    # Two overlapping nodes
    n1 = HierarchyNode("A", "0001", 0, 3)
    n2 = HierarchyNode("B", "0002", 2, 5)
    hier = DocumentHierarchy("overlap.pdf", "Overlap test", 6,
                             nodes=[n1, n2], page_keys=page_keys)
    store.save(hier)

    kvstore = _FakeKVStore(set(page_keys))
    vlm = _FixedVLM(['["0001", "0002"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10)

    result = retriever.invoke("Question")

    ids = result["ids"][0]
    assert len(ids) == len(set(ids)), "Duplicate page keys returned"


def test_invoke_skips_page_missing_from_kvstore(tmp_path):
    store = HierarchyStore(tmp_path)
    page_keys = _make_page_keys(5)
    n = HierarchyNode("Chapter", "0001", 0, 4)
    hier = DocumentHierarchy("doc.pdf", "Test", 5, nodes=[n], page_keys=page_keys)
    store.save(hier)

    # Only pages 0 and 2 are in kvstore
    kvstore = _FakeKVStore({page_keys[0], page_keys[2]})
    vlm = _FixedVLM(['["0001"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10)

    result = retriever.invoke("Question")

    returned_ids = set(result["ids"][0])
    assert page_keys[0] in returned_ids
    assert page_keys[2] in returned_ids
    assert page_keys[1] not in returned_ids  # not in kvstore


# ---------------------------------------------------------------------------
# HierarchyRetriever.invoke() — multi-document selection
# ---------------------------------------------------------------------------


def test_invoke_multi_doc_selects_correct_document(tmp_path):
    store = HierarchyStore(tmp_path)
    uuid_engine = "aaaaaaaa-0000-0000-0000-000000000001"
    uuid_avion  = "bbbbbbbb-0000-0000-0000-000000000002"
    h1 = DocumentHierarchy("engine.pdf", "Engine maintenance guide", 5,
                           nodes=[HierarchyNode("Engine Overview", "0001", 0, 4)],
                           page_keys=_make_page_keys(5, uuid_engine))
    h2 = DocumentHierarchy("avionics.pdf", "Avionics systems reference", 8,
                           nodes=[HierarchyNode("Avionics Overview", "0001", 0, 7)],
                           page_keys=_make_page_keys(8, uuid_avion))
    store.save(h1)
    store.save(h2)

    kvstore = _FakeKVStore(
        set(_make_page_keys(8, uuid_avion)) | set(_make_page_keys(5, uuid_engine))
    )
    # VLM: doc selection → avionics.pdf; tree navigation → node 0001
    vlm = _FixedVLM(['["avionics.pdf"]', '["0001"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4)

    result = retriever.invoke("What are the avionics systems?")

    ids = result["ids"][0]
    avion_keys = set(_make_page_keys(8, uuid_avion))
    assert all(k in avion_keys for k in ids)


def test_invoke_metadata_has_retrieval_source(tmp_path):
    store = _make_store_with_doc(tmp_path)
    kvstore = _FakeKVStore(set(_make_page_keys(10)))
    vlm = _FixedVLM(['["0001"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4)

    result = retriever.invoke("Question")

    for meta in result["metadatas"][0]:
        assert meta["retrieval_source"] == "hindex"
        assert "source" in meta
        assert "page_number" in meta


def test_invoke_warns_when_page_key_out_of_range(tmp_path):
    store = HierarchyStore(tmp_path)
    # page_keys only has 3 entries but node claims pages 0-5
    page_keys = _make_page_keys(3)
    n = HierarchyNode("Chapter", "0001", 0, 5)
    hier = DocumentHierarchy("short.pdf", "Short doc", 3,
                             nodes=[n], page_keys=page_keys)
    store.save(hier)

    kvstore = _FakeKVStore(set(page_keys))
    logger = _LogCapture()
    vlm = _FixedVLM(['["0001"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10, logger=logger)

    result = retriever.invoke("Question")

    # Pages 0-2 returned, pages 3-5 warned and skipped
    assert len(result["ids"][0]) == 3
    assert any("No kvstore key" in w for w in logger.warnings)
