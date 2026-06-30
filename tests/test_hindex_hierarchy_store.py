"""Unit tests for HierarchyStore."""

from __future__ import annotations

import json

import pytest

from colette.backends.hf.hindex.hierarchy_builder import (
    DocumentHierarchy,
    HierarchyNode,
    PageRecord,
    TopicNode,
)
from colette.backends.hf.hindex.hierarchy_store import HierarchyStore, _slug


# ---------------------------------------------------------------------------
# _slug
# ---------------------------------------------------------------------------


def test_slug_replaces_special_chars():
    assert _slug("my doc (2024).pdf") == "my_doc__2024__pdf"


def test_slug_preserves_safe_chars():
    assert _slug("report-2024_v1") == "report-2024_v1"


def test_slug_truncates_long_names():
    long = "a" * 300
    assert len(_slug(long)) == 200


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_FAKE_UUID = "deadbeef-1234-5678-abcd-000000000001"


def _make_page_keys(n: int) -> list[str]:
    return [f"{_FAKE_UUID}_{i:04d}" for i in range(n)]


def _make_page_records(n: int) -> list[PageRecord]:
    return [
        PageRecord(
            page_idx=i,
            page_key=f"{_FAKE_UUID}_{i:04d}",
            text_content=f"Text of page {i}",
            visual_description="A diagram",
            summary=f"Summary of page {i}.",
            keywords=["keyword", f"page{i}"],
            crop_keys=[f"{_FAKE_UUID}_{i:04d}_crop_0000"],
        )
        for i in range(n)
    ]


def _make_topics(page_count: int) -> list[TopicNode]:
    return [
        TopicNode(
            topic_id="T01",
            theme="Introduction",
            description="Opening pages",
            keywords=["intro"],
            page_indices=list(range(min(3, page_count))),
        ),
        TopicNode(
            topic_id="T02",
            theme="Details",
            description="Detail pages",
            keywords=["details"],
            page_indices=list(range(min(3, page_count), page_count)),
        ),
    ]


def _make_hierarchy(name: str = "test.pdf", n_pages: int = 10) -> DocumentHierarchy:
    h = DocumentHierarchy(
        doc_name=name,
        doc_description="A test document",
        page_count=n_pages,
        page_records=_make_page_records(n_pages),
        topics=_make_topics(n_pages),
        page_keys=_make_page_keys(n_pages),
    )
    return h


# ---------------------------------------------------------------------------
# HierarchyStore
# ---------------------------------------------------------------------------


def test_store_save_and_load(tmp_path):
    store = HierarchyStore(tmp_path)
    hier = _make_hierarchy()
    path = store.save(hier)

    assert path.exists()
    loaded = store.load("test.pdf")
    assert loaded is not None
    assert loaded.doc_name == "test.pdf"
    assert loaded.page_count == 10
    assert len(loaded.page_records) == 10
    assert loaded.page_records[0].text_content == "Text of page 0"
    assert len(loaded.topics) == 2
    assert loaded.topics[0].theme == "Introduction"


def test_store_load_nonexistent_returns_none(tmp_path):
    store = HierarchyStore(tmp_path)
    assert store.load("no_such_doc.pdf") is None


def test_store_exists(tmp_path):
    store = HierarchyStore(tmp_path)
    assert not store.exists("test.pdf")
    store.save(_make_hierarchy())
    assert store.exists("test.pdf")


def test_store_delete(tmp_path):
    store = HierarchyStore(tmp_path)
    store.save(_make_hierarchy())
    assert store.exists("test.pdf")
    store.delete("test.pdf")
    assert not store.exists("test.pdf")


def test_store_delete_nonexistent_is_safe(tmp_path):
    store = HierarchyStore(tmp_path)
    store.delete("ghost.pdf")


def test_store_list_doc_names(tmp_path):
    store = HierarchyStore(tmp_path)
    store.save(_make_hierarchy("alpha.pdf"))
    store.save(_make_hierarchy("beta.pdf"))
    names = store.list_doc_names()
    assert "alpha_pdf" in names
    assert "beta_pdf" in names


def test_store_load_all(tmp_path):
    store = HierarchyStore(tmp_path)
    store.save(_make_hierarchy("doc1.pdf", n_pages=5))
    store.save(_make_hierarchy("doc2.pdf", n_pages=15))
    all_h = store.load_all()
    assert len(all_h) == 2
    page_counts = {h.page_count for h in all_h}
    assert page_counts == {5, 15}


def test_store_load_all_skips_corrupt_files(tmp_path):
    store = HierarchyStore(tmp_path)
    store.save(_make_hierarchy("good.pdf"))
    bad_path = store.hindex_dir / "bad_file.json"
    bad_path.write_text("not valid json", encoding="utf-8")
    all_h = store.load_all()
    assert len(all_h) == 1
    assert all_h[0].doc_name == "good.pdf"


def test_store_overwrite(tmp_path):
    store = HierarchyStore(tmp_path)
    store.save(_make_hierarchy("doc.pdf", n_pages=5))
    store.save(_make_hierarchy("doc.pdf", n_pages=20))
    loaded = store.load("doc.pdf")
    assert loaded.page_count == 20


def test_store_round_trip_preserves_page_records(tmp_path):
    store = HierarchyStore(tmp_path)
    hier = _make_hierarchy()
    store.save(hier)
    loaded = store.load("test.pdf")
    assert len(loaded.page_records) == len(hier.page_records)
    assert loaded.page_records[3].summary == "Summary of page 3."
    assert loaded.page_records[3].crop_keys == [f"{_FAKE_UUID}_0003_crop_0000"]


def test_store_round_trip_preserves_topics(tmp_path):
    store = HierarchyStore(tmp_path)
    hier = _make_hierarchy()
    store.save(hier)
    loaded = store.load("test.pdf")
    assert loaded.topics[0].topic_id == "T01"
    assert loaded.topics[0].theme == "Introduction"
    assert 0 in loaded.topics[0].page_indices


def test_store_round_trip_preserves_page_keys(tmp_path):
    store = HierarchyStore(tmp_path)
    hier = _make_hierarchy()
    store.save(hier)
    loaded = store.load("test.pdf")
    assert loaded.page_keys == hier.page_keys
    assert len(loaded.page_keys) == 10


def test_store_creates_hindex_subdir(tmp_path):
    index_dir = tmp_path / "mm_index"
    store = HierarchyStore(index_dir)
    assert store.hindex_dir.exists()
    assert store.hindex_dir == index_dir / "hindex"


def test_store_backward_compat_with_legacy_nodes(tmp_path):
    """Old JSON files with 'nodes' but no page_records/topics still load."""
    store = HierarchyStore(tmp_path)
    legacy_data = {
        "doc_name": "legacy.pdf",
        "doc_description": "Old format",
        "page_count": 5,
        "page_keys": [f"{_FAKE_UUID}_{i:04d}" for i in range(5)],
        "nodes": [
            {
                "title": "Chapter 1",
                "node_id": "0001",
                "start_page": 0,
                "end_page": 4,
                "summary": "All pages",
                "children": [],
            }
        ],
    }
    slug = _slug("legacy.pdf")
    (store.hindex_dir / f"{slug}.json").write_text(
        json.dumps(legacy_data), encoding="utf-8"
    )
    loaded = store.load("legacy.pdf")
    assert loaded is not None
    assert loaded.page_count == 5
    assert len(loaded.nodes) == 1
    assert loaded.nodes[0].title == "Chapter 1"
    assert loaded.page_records == []
    assert loaded.topics == []
