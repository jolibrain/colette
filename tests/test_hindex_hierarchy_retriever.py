"""Unit tests for HierarchyRetriever — topic-based navigation."""

from __future__ import annotations

import json

import pytest

from colette.backends.hf.hindex import retrieval_mode_uses_hindex
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
from colette.backends.tantivy.tantivy_index import (
    retrieval_mode_uses_embedding_retrieval,
    retrieval_mode_uses_text_search_engine,
)
from colette.backends.hf.rag.rag_img import _merge_hindex_results


# ---------------------------------------------------------------------------
# retrieval_mode_uses_hindex routing tests
# ---------------------------------------------------------------------------


class _HIndexCfg:
    def __init__(self, enable: bool):
        self.enable_hindex = enable


def test_routing_vision_hierarchical_always_uses_hindex():
    assert retrieval_mode_uses_hindex("vision_hierarchical") is True
    assert retrieval_mode_uses_hindex("vision_hierarchical", None) is True
    assert retrieval_mode_uses_hindex("vision_hierarchical", _HIndexCfg(False)) is True


def test_routing_embedding_with_enable_hindex():
    assert retrieval_mode_uses_hindex("embedding_retrieval", _HIndexCfg(True)) is True


def test_routing_text_search_with_enable_hindex():
    assert retrieval_mode_uses_hindex("text_search_retrieval", _HIndexCfg(True)) is True


def test_routing_hybrid_with_enable_hindex():
    assert retrieval_mode_uses_hindex("hybrid", _HIndexCfg(True)) is True


def test_routing_embedding_without_enable_hindex_is_false():
    assert retrieval_mode_uses_hindex("embedding_retrieval") is False
    assert retrieval_mode_uses_hindex("embedding_retrieval", _HIndexCfg(False)) is False
    assert retrieval_mode_uses_hindex("embedding_retrieval", None) is False


def test_routing_text_search_without_enable_hindex_is_false():
    assert retrieval_mode_uses_hindex("text_search_retrieval") is False
    assert retrieval_mode_uses_hindex("text_search_retrieval", _HIndexCfg(False)) is False


def test_routing_hybrid_without_enable_hindex_is_false():
    assert retrieval_mode_uses_hindex("hybrid") is False
    assert retrieval_mode_uses_hindex("hybrid", _HIndexCfg(False)) is False


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
# HierarchyRetriever.invoke() — page selection (direct for short docs, topic for long)
# ---------------------------------------------------------------------------


def test_invoke_single_doc_returns_relevant_pages(tmp_path):
    """Short doc (10 pages ≤ 30 threshold) uses direct page selection."""
    store = _make_store_with_doc(tmp_path)
    page_keys = _make_page_keys(10)
    kvstore = _FakeKVStore({k for k in page_keys})
    # Direct selection: VLM returns 0-based page indices, most relevant first
    vlm = _FixedVLM(['["5", "6", "7", "8", "9"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=6)

    result = retriever.invoke("What are the technical details?")

    assert result["ids"] != [[]]
    ids = result["ids"][0]
    assert len(ids) > 0
    # All returned keys should belong to the pages selected (5-9)
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
    """top_k caps the total number of returned items; long doc uses topic navigation."""
    # 40 pages > 30 threshold → topic navigation path (not direct selection)
    store = _make_store_with_doc(tmp_path, page_count=40)
    kvstore = _FakeKVStore(set(_make_page_keys(40)))
    # T02 covers pages 20-39 (20 pages × full + crop = 40 potential items)
    vlm = _FixedVLM(['["T02"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=3)

    result = retriever.invoke("Anything")

    assert len(result["ids"][0]) <= 3


def test_invoke_metadata_has_retrieval_source(tmp_path):
    store = _make_store_with_doc(tmp_path)
    kvstore = _FakeKVStore(set(_make_page_keys(10)))
    # Direct selection for 10-page doc: return page indices
    vlm = _FixedVLM(['["0", "1", "2", "3", "4"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10)

    result = retriever.invoke("Question")

    for meta in result["metadatas"][0]:
        assert meta["retrieval_source"] == "hindex"
        assert "source" in meta
        assert "page_number" in meta


def test_invoke_metadata_includes_page_record_fields(tmp_path):
    store = _make_store_with_doc(tmp_path)
    kvstore = _FakeKVStore(set(_make_page_keys(10)))
    # Direct selection for 10-page doc: return page indices
    vlm = _FixedVLM(['["0", "1", "2", "3", "4"]'])
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
    # Direct selection for 4-page doc: return all page indices
    vlm = _FixedVLM(['["0", "1", "2", "3"]'])
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
    # Step 1: doc selection → avionics; Step 2: direct page selection (5 pages ≤ 30)
    vlm = _FixedVLM(['["avionics.pdf"]', '["0", "1", "2", "3", "4"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=6)

    result = retriever.invoke("What are the avionics systems?")

    ids = result["ids"][0]
    avion_keys = set(_make_page_keys(5, uuid_b))
    for key in ids:
        base_key = key.split("_crop_")[0]
        assert base_key in avion_keys


def test_invoke_deduplicates_pages(tmp_path):
    """Duplicate page indices from direct selection should be returned only once."""
    store = _make_store_with_doc(tmp_path, page_count=6)
    page_keys = _make_page_keys(6)
    kvstore = _FakeKVStore(set(page_keys))
    # Direct selection (6 pages ≤ 30): VLM returns duplicates — must be deduped
    vlm = _FixedVLM(['["0", "1", "2", "0", "3", "4", "5", "1"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=20)

    result = retriever.invoke("Question")

    # Page-level keys should not be duplicated
    full_page_ids = [i for i in result["ids"][0] if "_crop_" not in i]
    assert len(full_page_ids) == len(set(full_page_ids))


def test_invoke_long_doc_uses_topic_navigation(tmp_path):
    """Documents with more than _DIRECT_SELECT_THRESHOLD pages use topic navigation."""
    # 40 pages > 30 threshold → topic navigation (VLM returns topic IDs, not page indices)
    store = _make_store_with_doc(tmp_path, page_count=40)
    page_keys = _make_page_keys(40)
    kvstore = _FakeKVStore(set(page_keys))
    vlm = _FixedVLM(['["T02"]'])  # T02 covers pages 20-39
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=10)

    result = retriever.invoke("What are the technical details?")

    ids = result["ids"][0]
    assert ids
    # All returned keys should belong to pages 20-39 (T02)
    for key in ids:
        assert any(f"_{p:04d}" in key for p in range(20, 40))


def test_invoke_distances_are_position_based(tmp_path):
    """Returned distances must increase by rank so the most-relevant page scores lowest."""
    store = _make_store_with_doc(tmp_path)
    page_keys = _make_page_keys(10)
    kvstore = _FakeKVStore(set(page_keys))
    # Return three pages in relevance order; direct selection (10 ≤ 30)
    vlm = _FixedVLM(['["3", "7", "1"]'])
    retriever = HierarchyRetriever(store, kvstore, vlm, top_k=3)

    result = retriever.invoke("Question")

    distances = result["distances"][0]
    assert len(distances) == 3
    # Each successive full-page distance should be larger than the previous
    assert distances[0] < distances[1] < distances[2]
    # And all distances must be positive (not zero)
    assert all(d > 0 for d in distances)


# ---------------------------------------------------------------------------
# retrieval_mode_uses_embedding_retrieval — vision_hierarchical must be excluded
# ---------------------------------------------------------------------------


def test_vision_hierarchical_does_not_use_embedding_retrieval():
    """vision_hierarchical is hindex-only; embedding retrieval must NOT run for it."""
    assert retrieval_mode_uses_embedding_retrieval("vision_hierarchical") is False


def test_embedding_retrieval_uses_embedding():
    assert retrieval_mode_uses_embedding_retrieval("embedding_retrieval") is True


def test_hybrid_uses_embedding():
    assert retrieval_mode_uses_embedding_retrieval("hybrid") is True


def test_text_search_does_not_use_embedding():
    assert retrieval_mode_uses_embedding_retrieval("text_search_retrieval") is False


def test_retrieval_mode_uses_text_search_engine():
    assert retrieval_mode_uses_text_search_engine("text_search_retrieval") is True
    assert retrieval_mode_uses_text_search_engine("hybrid") is True
    assert retrieval_mode_uses_text_search_engine("embedding_retrieval") is False
    assert retrieval_mode_uses_text_search_engine("vision_hierarchical") is False


# ---------------------------------------------------------------------------
# _merge_hindex_results — result must be sorted by distance ascending
# ---------------------------------------------------------------------------


def _make_docs(ids, distances, sources=None):
    """Build a minimal docs dict compatible with _merge_hindex_results."""
    n = len(ids)
    metadatas = [{"retrieval_source": s} for s in (sources or ["embedding"] * n)]
    return {"ids": [ids], "distances": [distances], "metadatas": [metadatas]}


def test_merge_hindex_results_sorts_by_distance():
    """After merge (no top_k), combined list must be sorted by distance ascending."""
    docs = _make_docs(["emb_p1", "emb_p2", "emb_p3"], [0.68, 0.57, 0.51],
                      sources=["embedding", "embedding", "embedding"])
    hindex_docs = _make_docs(["hindex_p1", "hindex_p2"], [0.05, 0.10],
                              sources=["hindex", "hindex"])

    merged = _merge_hindex_results(docs, hindex_docs)

    dists = merged["distances"][0]
    assert dists == sorted(dists), "Merged distances must be sorted ascending"
    assert merged["metadatas"][0][0]["retrieval_source"] == "hindex"
    assert merged["metadatas"][0][1]["retrieval_source"] == "hindex"


def test_merge_hindex_results_empty_hindex_unchanged():
    """When hindex returns no results, original docs are returned unchanged."""
    docs = _make_docs(["emb_p1"], [0.60])
    hindex_docs = {"ids": [[]], "distances": [[]], "metadatas": [[]]}

    merged = _merge_hindex_results(docs, hindex_docs)

    assert merged["ids"][0] == ["emb_p1"]
    assert merged["distances"][0] == [0.60]


def test_merge_hindex_results_sets_disable_thinking():
    """Merged result must set _disable_thinking to suppress VLM thinking mode."""
    docs = _make_docs(["emb_p1"], [0.60])
    hindex_docs = _make_docs(["hindex_p1"], [0.05])

    merged = _merge_hindex_results(docs, hindex_docs)

    assert merged.get("_disable_thinking") is True


def test_merge_hindex_results_interleaves_correctly():
    """Without top_k, results from both sources are combined and sorted by distance."""
    docs = _make_docs(["emb_p1", "emb_p2"], [0.30, 0.80],
                      sources=["embedding", "embedding"])
    hindex_docs = _make_docs(["h_p1", "h_p2"], [0.05, 0.50],
                              sources=["hindex", "hindex"])

    merged = _merge_hindex_results(docs, hindex_docs)

    ids   = merged["ids"][0]
    dists = merged["distances"][0]
    assert dists == sorted(dists)
    # Expected order by distance: h_p1(0.05), emb_p1(0.30), h_p2(0.50), emb_p2(0.80)
    assert ids == ["h_p1", "emb_p1", "h_p2", "emb_p2"]


def test_merge_hindex_results_balanced_slots_with_top_k():
    """With top_k, each source gets ceil(top_k/2) slots — embedding is not crowded out."""
    # 4 hindex results (low dist) + 4 embedding results (high dist)
    docs = _make_docs(
        ["e1", "e2", "e3", "e4"], [0.50, 0.55, 0.60, 0.65],
        sources=["embedding"] * 4,
    )
    hindex_docs = _make_docs(
        ["h1", "h2", "h3", "h4"], [0.05, 0.10, 0.15, 0.20],
        sources=["hindex"] * 4,
    )

    merged = _merge_hindex_results(docs, hindex_docs, top_k=4)

    ids = merged["ids"][0]
    srcs = [m["retrieval_source"] for m in merged["metadatas"][0]]

    # Exactly 2 hindex + 2 embedding (ceil(4/2) = 2 each)
    assert srcs.count("hindex") == 2
    assert srcs.count("embedding") == 2
    assert len(ids) == 4
    # Still sorted by distance within the balanced set
    dists = merged["distances"][0]
    assert dists == sorted(dists)
    # Best from each source: h1(0.05), h2(0.10), e1(0.50), e2(0.55)
    assert ids == ["h1", "h2", "e1", "e2"]


def test_merge_hindex_results_top_k_odd():
    """With odd top_k, hindex gets the extra slot (ceil rounds up)."""
    docs = _make_docs(["e1", "e2", "e3"], [0.50, 0.55, 0.60], sources=["embedding"] * 3)
    hindex_docs = _make_docs(["h1", "h2", "h3"], [0.05, 0.10, 0.15], sources=["hindex"] * 3)

    merged = _merge_hindex_results(docs, hindex_docs, top_k=3)

    srcs = [m["retrieval_source"] for m in merged["metadatas"][0]]
    # ceil(3/2) = 2 hindex, 3-2 = 1 embedding
    assert srcs.count("hindex") == 2
    assert srcs.count("embedding") == 1
    assert len(merged["ids"][0]) == 3


def test_merge_hindex_results_no_top_k_uses_default():
    """When top_k is omitted, the default (4) is used and results are balanced."""
    docs = _make_docs(["e1", "e2", "e3", "e4"], [0.50, 0.55, 0.60, 0.65],
                      sources=["embedding"] * 4)
    hindex_docs = _make_docs(["h1", "h2", "h3", "h4"], [0.05, 0.10, 0.15, 0.20],
                              sources=["hindex"] * 4)

    merged = _merge_hindex_results(docs, hindex_docs)  # uses default top_k=4

    # Balanced: 2 hindex + 2 embedding = 4 total
    assert len(merged["ids"][0]) == 4
    result_ids = merged["ids"][0]
    assert result_ids[:2] == ["h1", "h2"]   # hindex first (lower distances)
    assert result_ids[2:] == ["e1", "e2"]   # embedding fills remaining slots


# ---------------------------------------------------------------------------
# Collection index tests
# ---------------------------------------------------------------------------

def _make_hierarchy(doc_name: str, page_count: int = 5) -> DocumentHierarchy:
    from colette.backends.hf.hindex.hierarchy_builder import DocumentHierarchy
    return DocumentHierarchy(
        doc_name=doc_name,
        doc_description=f"A document about {doc_name}",
        page_count=page_count,
    )


def test_build_collection_index_below_threshold_returns_none():
    """build_collection_index returns None when doc count < COLLECTION_THRESHOLD."""
    from colette.backends.hf.hindex import build_collection_index, COLLECTION_THRESHOLD
    hierarchies = [_make_hierarchy(f"doc{i}.pdf") for i in range(COLLECTION_THRESHOLD - 1)]
    result = build_collection_index(hierarchies, vlm_fn=lambda *a, **kw: "[]",
                                    logger=_LogCapture())
    assert result is None


def test_build_collection_index_uses_alphabetical_fallback_on_bad_json():
    """When VLM output is unparseable, alphabetical clustering is used."""
    from colette.backends.hf.hindex import build_collection_index, COLLECTION_THRESHOLD
    hierarchies = [_make_hierarchy(f"doc{i:02d}.pdf") for i in range(COLLECTION_THRESHOLD + 5)]
    # VLM returns garbage
    idx = build_collection_index(hierarchies, vlm_fn=lambda *a, **kw: "not json at all",
                                 logger=_LogCapture(), max_clusters=4)
    assert idx is not None
    assert len(idx.clusters) > 0
    # All docs must be covered
    covered = {n for c in idx.clusters for n in c.doc_names}
    assert covered == {h.doc_name for h in hierarchies}


def test_build_collection_index_parses_vlm_json():
    """When VLM output is valid JSON, clusters are built from it."""
    from colette.backends.hf.hindex import build_collection_index, COLLECTION_THRESHOLD, CollectionIndex
    hierarchies = [_make_hierarchy(f"doc{i:02d}.pdf") for i in range(COLLECTION_THRESHOLD + 5)]
    doc_names = [h.doc_name for h in hierarchies]

    vlm_output = json.dumps([
        {"id": "C01", "label": "Group A", "desc": "First half", "docs": doc_names[:10]},
        {"id": "C02", "label": "Group B", "desc": "Second half", "docs": doc_names[10:]},
    ])
    idx = build_collection_index(hierarchies, vlm_fn=lambda *a, **kw: vlm_output,
                                 logger=_LogCapture())
    assert idx is not None
    assert len(idx.clusters) == 2
    assert idx.clusters[0].cluster_id == "C01"
    assert set(idx.clusters[0].doc_names) == set(doc_names[:10])


def test_collection_index_roundtrip():
    """CollectionIndex serialises and deserialises correctly."""
    from colette.backends.hf.hindex import CollectionIndex, DocumentCluster
    idx = CollectionIndex(clusters=[
        DocumentCluster("C01", "Reports", "Annual reports", ["a.pdf", "b.pdf"]),
        DocumentCluster("C02", "Manuals", "Product manuals", ["c.pdf"]),
    ])
    recovered = CollectionIndex.from_dict(idx.to_dict())
    assert len(recovered.clusters) == 2
    assert recovered.clusters[1].doc_names == ["c.pdf"]


def test_retriever_uses_collection_index_two_stage(tmp_path):
    """HierarchyRetriever uses two-stage selection when collection index exists."""
    from colette.backends.hf.hindex import (
        HierarchyRetriever, HierarchyStore, CollectionIndex, DocumentCluster,
        COLLECTION_THRESHOLD,
    )
    from colette.backends.hf.hindex.hierarchy_builder import DocumentHierarchy, TopicNode

    # Build enough hierarchies to exceed the threshold
    n = COLLECTION_THRESHOLD + 5
    store = HierarchyStore(tmp_path)

    # Group A: docs 0-9, Group B: docs 10-n
    group_a = [f"docA{i:02d}.pdf" for i in range(10)]
    group_b = [f"docB{i:02d}.pdf" for i in range(n - 10)]
    target_doc = group_a[0]

    for name in group_a + group_b:
        h = DocumentHierarchy(doc_name=name, doc_description=f"About {name}", page_count=3)
        h.topics = [TopicNode("T01", "Topic", "desc", [], [0, 1, 2])]
        store.save(h)

    # Save collection index with two clusters
    idx = CollectionIndex(clusters=[
        DocumentCluster("C01", "Group A", "Group A documents", group_a),
        DocumentCluster("C02", "Group B", "Group B documents", group_b),
    ])
    store.save_collection_index(idx)

    calls = []

    def vlm_fn(images, prompt, max_tokens):
        calls.append(prompt)
        if "Clusters:" in prompt:
            return '["C01"]'   # pick Group A
        if "Select the document" in prompt:
            return json.dumps([target_doc])
        return '["T01"]'   # topic selection

    retriever = HierarchyRetriever(store=store, kvstore=_FakeKVStore(), vlm_fn=vlm_fn, top_k=2)
    retriever.invoke("What does docA00 say?")

    # Must have called cluster selection first
    assert any("Clusters:" in c for c in calls), "Expected cluster-selection call"
    # Must have called flat doc selection within cluster
    assert any("Select the document" in c for c in calls), "Expected doc-selection call"
