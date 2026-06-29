"""Unit tests for HierarchyStore."""

from __future__ import annotations

import json

import pytest

from colette.backends.hf.hindex.hierarchy_builder import DocumentHierarchy, HierarchyNode
from colette.backends.hf.hindex.hierarchy_store import HierarchyStore, _slug


# ---------------------------------------------------------------------------
# _slug
# ---------------------------------------------------------------------------


def test_slug_replaces_special_chars():
    # dots and parens are all replaced with underscore
    assert _slug("my doc (2024).pdf") == "my_doc__2024__pdf"


def test_slug_preserves_safe_chars():
    assert _slug("report-2024_v1") == "report-2024_v1"


def test_slug_truncates_long_names():
    long = "a" * 300
    assert len(_slug(long)) == 200


# ---------------------------------------------------------------------------
# HierarchyStore
# ---------------------------------------------------------------------------


def _make_hierarchy(name: str = "test.pdf", n_pages: int = 10) -> DocumentHierarchy:
    node = HierarchyNode("Chapter 1", "0001", 0, 4, summary="Intro")
    node.children = [HierarchyNode("Section 1.1", "0002", 1, 2)]
    return DocumentHierarchy(name, "A test document", n_pages, nodes=[node])


def test_store_save_and_load(tmp_path):
    store = HierarchyStore(tmp_path)
    hier = _make_hierarchy()
    path = store.save(hier)

    assert path.exists()
    loaded = store.load("test.pdf")
    assert loaded is not None
    assert loaded.doc_name == "test.pdf"
    assert loaded.page_count == 10
    assert len(loaded.nodes) == 1
    assert loaded.nodes[0].title == "Chapter 1"
    assert len(loaded.nodes[0].children) == 1


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
    store.delete("ghost.pdf")  # should not raise


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
    # Write a corrupt JSON file directly
    bad_path = store.hindex_dir / "bad_file.json"
    bad_path.write_text("not valid json", encoding="utf-8")
    all_h = store.load_all()
    assert len(all_h) == 1
    assert all_h[0].doc_name == "good.pdf"


def test_store_overwrite(tmp_path):
    store = HierarchyStore(tmp_path)
    store.save(_make_hierarchy("doc.pdf", n_pages=5))
    store.save(_make_hierarchy("doc.pdf", n_pages=20))  # overwrite
    loaded = store.load("doc.pdf")
    assert loaded.page_count == 20


def test_store_round_trip_preserves_tree(tmp_path):
    store = HierarchyStore(tmp_path)
    hier = _make_hierarchy()
    store.save(hier)
    loaded = store.load("test.pdf")
    assert loaded.nodes[0].children[0].node_id == "0002"
    assert loaded.nodes[0].children[0].title == "Section 1.1"


def test_store_creates_hindex_subdir(tmp_path):
    index_dir = tmp_path / "mm_index"
    # Do not create mm_index ahead of time
    store = HierarchyStore(index_dir)
    assert store.hindex_dir.exists()
    assert store.hindex_dir == index_dir / "hindex"
