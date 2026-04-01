from pathlib import Path

import pytest

tantivy = pytest.importorskip("tantivy")

from colette.backends.tantivy import TantivyIndex, build_text_context


class _Logger:
    def info(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None

    def debug(self, *_args, **_kwargs):
        return None


def test_text_search_engine_index_add_and_search(tmp_path):
    index = TantivyIndex(Path(tmp_path) / "text_search_engine", _Logger())
    index.add_documents(
        [
            {
                "doc_id": "doc-1",
                "source": "manual.pdf",
                "page_number": 1,
                "crop_label": "full_page",
                "label": "overview",
                "content": "The fuel pump relay is described in chapter one.",
            },
            {
                "doc_id": "doc-2",
                "source": "manual.pdf",
                "page_number": 2,
                "crop_label": "table",
                "label": "crop",
                "content": "Torque values for the engine cover are listed here.",
            },
        ],
        recreate=True,
    )

    hits = index.search("fuel pump relay", limit=5, fields=["content"], crop_label="full_page")

    assert hits
    assert hits[0]["doc_id"] == "doc-1"
    assert hits[0]["source"] == "manual.pdf"


def test_build_text_context_applies_budgets():
    hits, context = build_text_context(
        [
            {"source": "a.pdf", "page_number": 1, "content": "A" * 50},
            {"source": "b.pdf", "page_number": 2, "content": "B" * 50},
        ],
        max_chars_per_doc=20,
        max_total_chars=30,
    )

    assert len(hits) == 2
    assert hits[0]["content"] == "A" * 20
    assert hits[1]["content"] == "B" * 10
    assert "Text source 1: a.pdf page 1" in context
