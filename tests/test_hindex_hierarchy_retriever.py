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


def _make_store_with_doc(tmp_path, doc_name="manual.pdf", page_count=10) -> HierarchyStore:
    store = HierarchyStore(tmp_path)
    root = HierarchyNode("Chapter 1", "0001", 0, 4, summary="Introduction")
    root.children = [HierarchyNode("Section 1.1", "0002", 1, 2)]
    section2 = HierarchyNode("Chapter 2", "0003", 5, 9, summary="Details")
    hier = DocumentHierarchy(doc_name, "A technical manual", page_count, nodes=[root, section2])
    store.save(hier)
    return store


# ---------------------------------------------------------------------------
# _page_key
# ---------------------------------------------------------------------------


def test_page_key_basic():
    assert _page_key("manual.pdf", 3) == "manual_pdf_page_3"


def test_page_key_with_spaces():
    key = _page_key("my document.pdf", 0)
    assert " " not in key
    assert "page_0" in key


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
    # kvstore has pages 5–9 for Chapter 2
    kvstore = _FakeKVStore({_page_key("manual.pdf", i) for i in range(5, 10)})
    # VLM: navigate → select Chapter 2 (node 0003)
    vlm = _FixedVLM(['["0003"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4)

    result = retriever.invoke("What are the technical details?")

    assert result["ids"] != [[]]
    ids = result["ids"][0]
    assert all("page_" in i for i in ids)
    assert all("manual_pdf" in i for i in ids)
    # Should have returned pages 5–8 (top_k=4) from chapter 2 (pages 5-9)
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
    kvstore = _FakeKVStore({_page_key("manual.pdf", i) for i in range(10)})
    # VLM returns an empty list — no nodes selected
    vlm = _FixedVLM(['[]'])
    logger = _LogCapture()
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4, logger=logger)

    result = retriever.invoke("Irrelevant question")

    # Falls back to top-level nodes (Chapter 1 pages 0-4)
    assert result["ids"][0]  # non-empty fallback
    assert any("falling back" in w for w in logger.warnings)


def test_invoke_respects_top_k(tmp_path):
    store = _make_store_with_doc(tmp_path, page_count=20)
    # Add a large chapter spanning pages 0-19
    store2 = HierarchyStore(tmp_path)
    big = HierarchyNode("Big Chapter", "0001", 0, 19)
    hier = DocumentHierarchy("big.pdf", "A very long document", 20, nodes=[big])
    store2.save(hier)

    kvstore = _FakeKVStore({_page_key("big.pdf", i) for i in range(20)})
    vlm = _FixedVLM(['["0001"]'])
    retriever = HierarchyRetriever(store2, kvstore, vlm, top_k=3)

    result = retriever.invoke("Anything")

    assert len(result["ids"][0]) <= 3


def test_invoke_deduplicates_pages(tmp_path):
    store = HierarchyStore(tmp_path)
    # Two overlapping nodes
    n1 = HierarchyNode("A", "0001", 0, 3)
    n2 = HierarchyNode("B", "0002", 2, 5)
    hier = DocumentHierarchy("overlap.pdf", "Overlap test", 6, nodes=[n1, n2])
    store.save(hier)

    kvstore = _FakeKVStore({_page_key("overlap.pdf", i) for i in range(6)})
    vlm = _FixedVLM(['["0001", "0002"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10)

    result = retriever.invoke("Question")

    ids = result["ids"][0]
    assert len(ids) == len(set(ids)), "Duplicate page keys returned"


# ---------------------------------------------------------------------------
# HierarchyRetriever.invoke() — multi-document selection
# ---------------------------------------------------------------------------


def test_invoke_multi_doc_selects_correct_document(tmp_path):
    store = HierarchyStore(tmp_path)
    # Two documents
    h1 = DocumentHierarchy("engine.pdf", "Engine maintenance guide", 5, nodes=[
        HierarchyNode("Engine Overview", "0001", 0, 4),
    ])
    h2 = DocumentHierarchy("avionics.pdf", "Avionics systems reference", 8, nodes=[
        HierarchyNode("Avionics Overview", "0001", 0, 7),
    ])
    store.save(h1)
    store.save(h2)

    kvstore = _FakeKVStore(
        {_page_key("avionics.pdf", i) for i in range(8)}
        | {_page_key("engine.pdf", i) for i in range(5)}
    )
    # VLM: doc selection → avionics.pdf; tree navigation → node 0001
    vlm = _FixedVLM(['["avionics.pdf"]', '["0001"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4)

    result = retriever.invoke("What are the avionics systems?")

    ids = result["ids"][0]
    assert all("avionics_pdf" in i for i in ids)


def test_invoke_metadata_has_retrieval_source(tmp_path):
    store = _make_store_with_doc(tmp_path)
    kvstore = _FakeKVStore({_page_key("manual.pdf", i) for i in range(10)})
    vlm = _FixedVLM(['["0001"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=4)

    result = retriever.invoke("Question")

    for meta in result["metadatas"][0]:
        assert meta["retrieval_source"] == "hindex"
        assert "source" in meta
        assert "page_number" in meta
