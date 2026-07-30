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
    # Use a fixed count large enough to exceed any reasonable threshold and split evenly.
    n = max(20, COLLECTION_THRESHOLD + 10)
    hierarchies = [_make_hierarchy(f"doc{i:02d}.pdf") for i in range(n)]
    doc_names = [h.doc_name for h in hierarchies]

    vlm_output = json.dumps([
        {"id": "C01", "label": "Group A", "desc": "First half", "docs": doc_names[:n // 2]},
        {"id": "C02", "label": "Group B", "desc": "Second half", "docs": doc_names[n // 2:]},
    ])
    idx = build_collection_index(hierarchies, vlm_fn=lambda *a, **kw: vlm_output,
                                 logger=_LogCapture())
    assert idx is not None
    assert len(idx.clusters) == 2
    assert idx.clusters[0].cluster_id == "C01"
    assert set(idx.clusters[0].doc_names) == set(doc_names[:n // 2])


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


# ---------------------------------------------------------------------------
# vision_semantic routing
# ---------------------------------------------------------------------------

from colette.backends.tantivy.tantivy_index import retrieval_mode_uses_vision_semantic
from colette.backends.hf.rag.rag_img import (
    _hierarchy_to_vlm_tantivy_docs,
    _vlm_hits_to_page_docs,
)


def test_vision_semantic_routing_true():
    assert retrieval_mode_uses_vision_semantic("vision_semantic") is True


def test_vision_semantic_routing_false_for_all_other_modes():
    for mode in ("embedding_retrieval", "text_search_retrieval", "hybrid", "vision_hierarchical"):
        assert retrieval_mode_uses_vision_semantic(mode) is False, f"Expected False for {mode}"


def test_vision_semantic_does_not_use_embedding_retrieval():
    assert retrieval_mode_uses_embedding_retrieval("vision_semantic") is False


def test_vision_semantic_does_not_use_text_search_engine():
    assert retrieval_mode_uses_text_search_engine("vision_semantic") is False


def test_vision_semantic_does_not_use_hindex():
    assert retrieval_mode_uses_hindex("vision_semantic") is False


# ---------------------------------------------------------------------------
# _hierarchy_to_vlm_tantivy_docs
# ---------------------------------------------------------------------------

def _make_full_hierarchy(doc_name: str = "/data/doc.pdf") -> DocumentHierarchy:
    return DocumentHierarchy(
        doc_name=doc_name,
        doc_description="A test document",
        page_count=3,
        page_records=[
            PageRecord(
                page_idx=0,
                page_key="uuid_0000",
                text_content="Revenue grew 12% in Q3.",
                visual_description="Bar chart showing quarterly revenue.",
                summary="Q3 financial results show strong growth.",
                keywords=["revenue", "Q3", "growth"],
            ),
            PageRecord(
                page_idx=1,
                page_key="uuid_0001",
                text_content="",
                visual_description="",
                summary="",
                keywords=[],
            ),
            PageRecord(
                page_idx=2,
                page_key="uuid_0002",
                text_content="Operating expenses decreased.",
                visual_description="",
                summary="Cost reduction summary.",
                keywords=["expenses", "cost"],
            ),
        ],
    )


def test_vlm_tantivy_docs_one_per_non_empty_page():
    hierarchy = _make_full_hierarchy()
    docs = _hierarchy_to_vlm_tantivy_docs(hierarchy)
    # Page 1 (idx=1) is all-empty — should be skipped
    assert len(docs) == 2


def test_vlm_tantivy_docs_page_number_is_one_indexed():
    hierarchy = _make_full_hierarchy()
    docs = _hierarchy_to_vlm_tantivy_docs(hierarchy)
    assert docs[0]["page_number"] == 1   # page_idx=0 → page_number=1
    assert docs[1]["page_number"] == 3   # page_idx=2 → page_number=3


def test_vlm_tantivy_docs_crop_label_is_pagedesc():
    docs = _hierarchy_to_vlm_tantivy_docs(_make_full_hierarchy())
    assert all(d["crop_label"] == "pagedesc" for d in docs)


def test_vlm_tantivy_docs_doc_id_format():
    hierarchy = _make_full_hierarchy("/data/doc.pdf")
    docs = _hierarchy_to_vlm_tantivy_docs(hierarchy)
    assert docs[0]["doc_id"] == "/data/doc.pdf:page:1"
    assert docs[0]["source"] == "/data/doc.pdf"


def test_vlm_tantivy_docs_content_includes_all_fields():
    hierarchy = _make_full_hierarchy()
    doc = _hierarchy_to_vlm_tantivy_docs(hierarchy)[0]
    content = doc["content"]
    assert "Q3 financial results" in content   # from summary
    assert "Revenue grew 12%" in content       # from text_content
    assert "Bar chart" in content              # from visual_description
    assert "revenue" in content                # from keywords


def test_vlm_tantivy_docs_empty_hierarchy():
    h = DocumentHierarchy(doc_name="empty.pdf", doc_description="", page_count=0)
    assert _hierarchy_to_vlm_tantivy_docs(h) == []


# ---------------------------------------------------------------------------
# _vlm_hits_to_page_docs
# ---------------------------------------------------------------------------

def _make_hits(source: str = "/data/doc.pdf") -> list[dict]:
    return [
        {"source": source, "page_number": 3, "score": 8.5, "crop_label": "pagedesc"},
        {"source": source, "page_number": 1, "score": 5.2, "crop_label": "pagedesc"},
        {"source": source, "page_number": 7, "score": 2.1, "crop_label": "pagedesc"},
    ]


def test_vlm_hits_to_page_docs_structure():
    docs = _vlm_hits_to_page_docs(_make_hits(), top_k=3)
    assert "ids" in docs and "distances" in docs and "metadatas" in docs
    assert len(docs["ids"][0]) == 3


def test_vlm_hits_to_page_docs_respects_top_k():
    docs = _vlm_hits_to_page_docs(_make_hits(), top_k=2)
    assert len(docs["ids"][0]) == 2


def test_vlm_hits_to_page_docs_distances_ascend_with_lower_score():
    docs = _vlm_hits_to_page_docs(_make_hits(), top_k=3)
    dists = docs["distances"][0]
    # Hits are in score-descending order; distances should reflect that
    # (higher BM25 score → lower distance)
    assert dists[0] < dists[1] < dists[2]


def test_vlm_hits_to_page_docs_deduplicates():
    hits = [
        {"source": "/data/doc.pdf", "page_number": 2, "score": 9.0},
        {"source": "/data/doc.pdf", "page_number": 2, "score": 7.0},  # duplicate
        {"source": "/data/doc.pdf", "page_number": 4, "score": 3.0},
    ]
    docs = _vlm_hits_to_page_docs(hits, top_k=5)
    assert len(docs["ids"][0]) == 2


def test_vlm_hits_to_page_docs_skips_invalid_hits():
    hits = [
        {"source": "", "page_number": 1, "score": 5.0},    # empty source
        {"source": "/doc.pdf", "page_number": None, "score": 5.0},  # missing page
        {"source": "/doc.pdf", "page_number": 2, "score": 4.0},     # valid
    ]
    docs = _vlm_hits_to_page_docs(hits, top_k=5)
    assert len(docs["ids"][0]) == 1


def test_vlm_hits_to_page_docs_page_key_derivable():
    from uuid import uuid5, NAMESPACE_URL
    source = "/data/reports/annual.pdf"
    hits = [{"source": source, "page_number": 5, "score": 3.0}]
    docs = _vlm_hits_to_page_docs(hits, top_k=1)
    expected_uuid = str(uuid5(NAMESPACE_URL, source))
    expected_key = f"{expected_uuid}_0004"   # page_idx = 5 - 1 = 4
    assert docs["ids"][0][0] == expected_key


def test_vlm_hits_to_page_docs_metadata_fields():
    source = "/doc.pdf"
    hits = [{"source": source, "page_number": 3, "score": 6.0}]
    docs = _vlm_hits_to_page_docs(hits, top_k=1)
    meta = docs["metadatas"][0][0]
    assert meta["source"] == source
    assert meta["page_number"] == 3
    assert meta["crop_label"] == "pagedesc"


# ---------------------------------------------------------------------------
# New vision retrieval mode routing helpers
# ---------------------------------------------------------------------------

from colette.backends.tantivy.tantivy_index import (
    retrieval_mode_uses_vision_text_embed,
    retrieval_mode_uses_vision_hybrid_bm25,
    retrieval_mode_uses_vision_topic_embed,
    retrieval_mode_needs_vlm_descriptions,
)
from colette.backends.hf.rag.rag_img import (
    _hindex_page_text,
    _hindex_topic_text,
    _bm25_hits_to_page_list,
    _reciprocal_rank_fusion,
)


def test_vision_text_embed_routing():
    assert retrieval_mode_uses_vision_text_embed("vision_text_embed") is True
    assert retrieval_mode_uses_vision_text_embed("vision_bm25_rerank") is True
    assert retrieval_mode_uses_vision_text_embed("vision_hybrid_bm25") is False
    assert retrieval_mode_uses_vision_text_embed("vision_topic_embed") is False
    assert retrieval_mode_uses_vision_text_embed("vision_semantic") is False
    assert retrieval_mode_uses_vision_text_embed("embedding_retrieval") is False


def test_vision_hybrid_bm25_routing():
    assert retrieval_mode_uses_vision_hybrid_bm25("vision_hybrid_bm25") is True
    assert retrieval_mode_uses_vision_hybrid_bm25("vision_bm25_rerank") is True
    assert retrieval_mode_uses_vision_hybrid_bm25("vision_text_embed") is False
    assert retrieval_mode_uses_vision_hybrid_bm25("vision_topic_embed") is False
    assert retrieval_mode_uses_vision_hybrid_bm25("hybrid") is False


def test_vision_topic_embed_routing():
    assert retrieval_mode_uses_vision_topic_embed("vision_topic_embed") is True
    assert retrieval_mode_uses_vision_topic_embed("vision_text_embed") is False
    assert retrieval_mode_uses_vision_topic_embed("vision_bm25_rerank") is False
    assert retrieval_mode_uses_vision_topic_embed("vision_hybrid_bm25") is False


def test_retrieval_mode_needs_vlm_descriptions():
    for mode in ("vision_semantic", "vision_text_embed", "vision_hybrid_bm25", "vision_topic_embed", "vision_bm25_rerank"):
        assert retrieval_mode_needs_vlm_descriptions(mode) is True, f"Expected True for {mode}"
    for mode in ("embedding_retrieval", "text_search_retrieval", "hybrid", "vision_hierarchical"):
        assert retrieval_mode_needs_vlm_descriptions(mode) is False, f"Expected False for {mode}"


# ---------------------------------------------------------------------------
# _hindex_page_text and _hindex_topic_text helpers
# ---------------------------------------------------------------------------


def _make_page_record(**kwargs):
    defaults = dict(
        page_idx=0, page_key="uuid_0000",
        text_content="", visual_description="", summary="", keywords=[], crop_keys=[]
    )
    defaults.update(kwargs)
    return PageRecord(**defaults)


def test_hindex_page_text_combines_fields():
    rec = _make_page_record(
        summary="Revenue summary",
        keywords=["revenue", "growth"],
        text_content="Revenue grew 12%",
        visual_description="Bar chart",
    )
    text = _hindex_page_text(rec)
    assert "Revenue summary" in text
    assert "revenue" in text
    assert "Revenue grew 12%" in text
    assert "Bar chart" in text
    assert " | " in text


def test_hindex_page_text_empty_record():
    rec = _make_page_record()
    assert _hindex_page_text(rec) == ""


def test_hindex_page_text_truncates_text_content():
    rec = _make_page_record(text_content="x" * 2000)
    text = _hindex_page_text(rec)
    assert len(text) <= 900  # 800 chars truncated + separator overhead


def test_hindex_topic_text_combines_fields():
    topic = TopicNode(
        topic_id="t0",
        theme="Financial Performance",
        description="Covers Q4 revenue and profitability",
        keywords=["revenue", "Q4"],
        page_indices=[0, 1, 2],
    )
    text = _hindex_topic_text(topic)
    assert "Financial Performance" in text
    assert "Q4 revenue" in text
    assert "revenue" in text


def test_hindex_topic_text_empty():
    topic = TopicNode(topic_id="t0", theme="", description="", keywords=[], page_indices=[])
    assert _hindex_topic_text(topic) == ""


# ---------------------------------------------------------------------------
# _bm25_hits_to_page_list
# ---------------------------------------------------------------------------


def test_bm25_hits_to_page_list_basic():
    hits = [
        {"source": "/doc.pdf", "page_number": 3, "score": 9.0},
        {"source": "/doc.pdf", "page_number": 1, "score": 4.0},
    ]
    result = _bm25_hits_to_page_list(hits)
    assert len(result) == 2
    assert result[0][0] == "/doc.pdf"
    assert result[0][1] == 3
    assert 0.0 < result[0][2] <= 1.0  # distance in (0, 1]


def test_bm25_hits_to_page_list_deduplicates():
    hits = [
        {"source": "/doc.pdf", "page_number": 2, "score": 8.0},
        {"source": "/doc.pdf", "page_number": 2, "score": 5.0},  # duplicate
    ]
    result = _bm25_hits_to_page_list(hits)
    assert len(result) == 1


def test_bm25_hits_to_page_list_skips_invalid():
    hits = [
        {"source": "", "page_number": 1, "score": 5.0},    # empty source
        {"source": "/doc.pdf", "page_number": None, "score": 5.0},  # no page
        {"source": "/doc.pdf", "page_number": 3, "score": 3.0},  # valid
    ]
    result = _bm25_hits_to_page_list(hits)
    assert len(result) == 1
    assert result[0][1] == 3


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion
# ---------------------------------------------------------------------------


def test_reciprocal_rank_fusion_merges_two_lists():
    bm25 = [("/a.pdf", 1), ("/a.pdf", 2), ("/b.pdf", 3)]
    embed = [("/b.pdf", 3), ("/a.pdf", 1), ("/c.pdf", 5)]
    result = _reciprocal_rank_fusion([bm25, embed])
    pages = [(s, p) for s, p, _ in result]
    # ("/a.pdf", 1) appears rank-1 in bm25, rank-2 in embed → highest RRF score
    assert ("/a.pdf", 1) in pages
    assert ("/b.pdf", 3) in pages
    assert ("/c.pdf", 5) in pages


def test_reciprocal_rank_fusion_single_list_preserves_order():
    ranked = [("/doc.pdf", 1), ("/doc.pdf", 2), ("/doc.pdf", 3)]
    result = _reciprocal_rank_fusion([ranked])
    # RRF score is monotonically decreasing with rank → order preserved
    out = [(s, p) for s, p, _ in result]
    assert out == ranked


def test_reciprocal_rank_fusion_scores_decrease():
    ranked = [("/doc.pdf", i) for i in range(1, 6)]
    result = _reciprocal_rank_fusion([ranked])
    scores = [sc for _, _, sc in result]
    assert scores == sorted(scores, reverse=True)


def test_reciprocal_rank_fusion_deduplicates_across_lists():
    lst_a = [("/doc.pdf", 1), ("/doc.pdf", 2)]
    lst_b = [("/doc.pdf", 1), ("/doc.pdf", 3)]
    result = _reciprocal_rank_fusion([lst_a, lst_b])
    pages = [(s, p) for s, p, _ in result]
    assert len(pages) == len(set(pages))  # no duplicates
