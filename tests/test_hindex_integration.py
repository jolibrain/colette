"""Integration test: builder → store → retriever pipeline.

Exercises the full hindex pipeline end-to-end without any real VLM or GPU.
Verifies that the three components wire together correctly and that the
retriever output format is compatible with RAGImgRetriever.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from colette.backends.hf.hindex import (
    HierarchyBuilder,
    HierarchyRetriever,
    HierarchyStore,
    _page_key,
)
from colette.backends.hf.hindex.hierarchy_builder import PageRecord, TopicNode


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


def _blank_page() -> Image.Image:
    return Image.new("RGB", (100, 100), color=255)


class _SequentialVLM:
    """Returns a pre-configured sequence of responses."""

    def __init__(self, responses: list[str]):
        self._responses = list(responses)
        self._idx = 0

    def __call__(self, images, prompt: str) -> str:
        if self._idx < len(self._responses):
            resp = self._responses[self._idx]
            self._idx += 1
            return resp
        return ""


class _FakeKVStore:
    def __init__(self, keys: set[str]):
        self._keys = keys

    def has_key(self, key: str) -> bool:
        return key in self._keys


# ---------------------------------------------------------------------------
# Full pipeline test
# ---------------------------------------------------------------------------


def _make_page_record_responses(n_pages: int) -> list[str]:
    """Produce sequential per-page extraction JSON responses."""
    responses = []
    for i in range(n_pages):
        data = {
            "text_content": f"Page {i} text content here.",
            "visual_description": f"Diagram on page {i}" if i % 3 == 0 else "",
            "summary": f"This page covers topic {'A' if i < n_pages // 2 else 'B'}.",
            "keywords": ["topicA" if i < n_pages // 2 else "topicB", f"page{i}"],
        }
        responses.append(json.dumps(data))
    return responses


def test_full_pipeline_build_store_retrieve(tmp_path):
    doc_name = "technical_manual.pdf"
    n_pages = 6
    pages = [_blank_page() for _ in range(n_pages)]
    _UUID5 = "deadbeef-0000-0000-0000-000000000001"

    # Pre-computed page keys
    page_keys = [_page_key(_UUID5, i) for i in range(n_pages)]

    # Build VLM responses: n_pages extraction + 1 doc description + 1 topic clustering
    page_responses = _make_page_record_responses(n_pages)
    doc_description_response = "A technical manual covering procedures A and B."
    topic_response = json.dumps([
        {
            "theme": "Topic A",
            "description": "Pages about topic A",
            "keywords": ["topicA"],
            "page_indices": list(range(n_pages // 2)),
        },
        {
            "theme": "Topic B",
            "description": "Pages about topic B",
            "keywords": ["topicB"],
            "page_indices": list(range(n_pages // 2, n_pages)),
        },
    ])

    vlm_build = _SequentialVLM(page_responses + [doc_description_response, topic_response])
    builder = HierarchyBuilder(vlm_fn=vlm_build)

    # 1. Build
    hier = builder.build(doc_name, pages, page_keys=page_keys)
    assert hier.page_count == n_pages
    assert len(hier.page_records) == n_pages
    assert len(hier.topics) == 2
    assert hier.doc_description == doc_description_response

    # Verify per-page records
    assert hier.page_records[0].text_content == "Page 0 text content here."
    assert hier.page_records[0].keywords == ["topicA", "page0"]

    # 2. Store
    store = HierarchyStore(tmp_path)
    store.save(hier)
    assert store.exists(doc_name)

    # Verify round-trip fidelity
    loaded = store.load(doc_name)
    assert loaded.doc_name == doc_name
    assert len(loaded.page_records) == n_pages
    assert loaded.page_records[2].summary == hier.page_records[2].summary
    assert loaded.topics[0].theme == "Topic A"
    assert loaded.topics[1].page_indices == list(range(n_pages // 2, n_pages))

    # 3. Retrieve
    kvstore = _FakeKVStore(set(page_keys))
    vlm_retrieve = _SequentialVLM(['["T02"]'])  # select Topic B (pages 3-5)

    retriever = HierarchyRetriever(store, kvstore, vlm_retrieve, top_k=8)
    result = retriever.invoke("How were the procedures B conducted?")

    # Output format must match RAGImgRetriever
    assert "ids" in result
    assert "distances" in result
    assert "metadatas" in result

    ids = result["ids"][0]
    assert len(ids) > 0
    # All retrieved keys must be from Topic B pages (3-5)
    for key in ids:
        base = key.split("_crop_")[0]
        assert any(base == _page_key(_UUID5, p) for p in range(3, 6))

    # Metadata should include pre-extracted fields
    for meta in result["metadatas"][0]:
        assert meta["retrieval_source"] == "hindex"
        if meta["crop_label"] == "full_page":
            assert "summary" in meta
            assert "text_content" in meta


def test_pipeline_output_is_compatible_with_ragimg_format(tmp_path):
    """The dict keys and value shapes must match what RAGImgRetriever returns."""
    store = HierarchyStore(tmp_path)
    _UUID = "cafebabe-0000-0000-0000-000000000001"
    page_keys = [_page_key(_UUID, i) for i in range(3)]

    hier_data = {
        "doc_name": "compat.pdf",
        "doc_description": "Compatibility test",
        "page_count": 3,
        "page_keys": page_keys,
        "page_records": [
            {
                "page_idx": i,
                "page_key": page_keys[i],
                "text_content": f"text {i}",
                "visual_description": "",
                "summary": f"summary {i}",
                "keywords": ["test"],
                "crop_keys": [],
            }
            for i in range(3)
        ],
        "topics": [
            {
                "topic_id": "T01",
                "theme": "All",
                "description": "All pages",
                "keywords": ["test"],
                "page_indices": [0, 1, 2],
            }
        ],
        "nodes": [],
    }
    from colette.backends.hf.hindex.hierarchy_builder import DocumentHierarchy
    store.save(DocumentHierarchy.from_dict(hier_data))

    kvstore = _FakeKVStore(set(page_keys))
    vlm = _SequentialVLM(['["T01"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=5)

    result = retriever.invoke("Anything")

    assert set(result.keys()) >= {"ids", "distances", "metadatas"}
    assert len(result["ids"]) == 1
    assert len(result["distances"]) == 1
    assert len(result["metadatas"]) == 1
    assert len(result["ids"][0]) == len(result["metadatas"][0])
    assert len(result["ids"][0]) == len(result["distances"][0])
