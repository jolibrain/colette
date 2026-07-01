"""Unit tests for HierarchyBuilder (bottom-up PageRecord + TopicNode model).

All VLM calls are replaced with deterministic fakes — no GPU required.
"""

from __future__ import annotations

import json

import pytest
from PIL import Image

from colette.backends.hf.hindex.hierarchy_builder import (
    DocumentHierarchy,
    HierarchyBuilder,
    HierarchyNode,
    PageRecord,
    TopicNode,
    _parse_json_response,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _blank_image(w: int = 100, h: int = 100) -> Image.Image:
    return Image.new("RGB", (w, h), color=(255, 255, 255))


def _make_pages(n: int) -> list[Image.Image]:
    return [_blank_image() for _ in range(n)]


class _FixedVLM:
    """Returns a pre-configured sequence of string responses."""

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
        return ""


class _LogCapture:
    def __init__(self):
        self.warnings: list[str] = []

    def info(self, msg, *_a, **_k):
        pass

    def debug(self, msg, *_a, **_k):
        pass

    def warning(self, msg, *_a, **_k):
        self.warnings.append(msg)


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


def test_parse_json_response_clean():
    logger = _LogCapture()
    result = _parse_json_response('{"visual_description": "a chart", "summary": "page 1"}', logger)
    assert result == {"visual_description": "a chart", "summary": "page 1"}
    assert not logger.warnings


def test_parse_json_response_with_markdown_fence():
    logger = _LogCapture()
    raw = '```json\n{"keywords": ["rag", "llm"]}\n```'
    result = _parse_json_response(raw, logger)
    assert result == {"keywords": ["rag", "llm"]}


def test_parse_json_response_embedded_in_prose():
    logger = _LogCapture()
    raw = 'Here is the JSON: {"summary": "intro", "keywords": ["a", "b"]}'
    result = _parse_json_response(raw, logger)
    assert result is not None
    assert result["summary"] == "intro"


def test_parse_json_response_invalid_returns_none():
    logger = _LogCapture()
    result = _parse_json_response("not json at all", logger)
    assert result is None
    assert logger.warnings


def test_parse_json_response_salvages_truncated_object():
    """Model output truncated mid text_content — summary and keywords already written."""
    logger = _LogCapture()
    # Simulate model writing fields in the new order, truncated during text_content
    raw = (
        '{\n'
        '  "visual_description": "A bar chart showing metrics",\n'
        '  "summary": "This page presents performance results.",\n'
        '  "keywords": ["performance", "metrics", "results"],\n'
        '  "text_content": "Performance Q1 2025 — '  # truncated here
    )
    result = _parse_json_response(raw, logger)
    assert result is not None
    assert result["summary"] == "This page presents performance results."
    assert result["visual_description"] == "A bar chart showing metrics"
    assert result["keywords"] == ["performance", "metrics", "results"]
    assert "text_content" in result  # partial value captured
    assert logger.warnings  # truncation warning emitted


def test_parse_json_response_salvages_truncated_text_content_only():
    """Worst case: truncated before summary/keywords even appear."""
    logger = _LogCapture()
    raw = '{\n  "visual_description": "pie chart",\n  "summary": "Summary of slide'
    result = _parse_json_response(raw, logger)
    assert result is not None
    assert result["visual_description"] == "pie chart"
    assert "summary" in result
    assert logger.warnings


# ---------------------------------------------------------------------------
# PageRecord
# ---------------------------------------------------------------------------


def test_page_record_round_trip():
    rec = PageRecord(
        page_idx=3,
        page_key="doc::p003",
        text_content="some text",
        visual_description="a figure",
        summary="This page shows a figure.",
        keywords=["figure", "data"],
        crop_keys=["doc::p003_crop_0000"],
    )
    restored = PageRecord.from_dict(rec.to_dict())
    assert restored.page_idx == 3
    assert restored.page_key == "doc::p003"
    assert restored.text_content == "some text"
    assert restored.visual_description == "a figure"
    assert restored.summary == "This page shows a figure."
    assert restored.keywords == ["figure", "data"]
    assert restored.crop_keys == ["doc::p003_crop_0000"]


def test_page_record_from_dict_defaults():
    rec = PageRecord.from_dict({"page_idx": 0, "page_key": ""})
    assert rec.text_content == ""
    assert rec.visual_description == ""
    assert rec.summary == ""
    assert rec.keywords == []
    assert rec.crop_keys == []


# ---------------------------------------------------------------------------
# TopicNode
# ---------------------------------------------------------------------------


def test_topic_node_round_trip():
    t = TopicNode(
        topic_id="T02",
        theme="System Architecture",
        description="Core components of the system.",
        keywords=["architecture", "RAG"],
        page_indices=[1, 3, 5],
    )
    restored = TopicNode.from_dict(t.to_dict())
    assert restored.topic_id == "T02"
    assert restored.theme == "System Architecture"
    assert restored.description == "Core components of the system."
    assert restored.keywords == ["architecture", "RAG"]
    assert restored.page_indices == [1, 3, 5]


# ---------------------------------------------------------------------------
# DocumentHierarchy
# ---------------------------------------------------------------------------


def _make_hierarchy(n_pages: int = 4) -> DocumentHierarchy:
    records = [
        PageRecord(
            page_idx=i,
            page_key=f"doc::p{i:03d}",
            text_content=f"text {i}",
            visual_description="",
            summary=f"Summary of page {i}.",
            keywords=[f"kw{i}"],
        )
        for i in range(n_pages)
    ]
    topics = [
        TopicNode("T01", "Intro", "Introductory pages.", ["intro"], [0, 1]),
        TopicNode("T02", "Results", "Results and conclusions.", ["results"], [2, 3]),
    ]
    return DocumentHierarchy(
        doc_name="test.pdf",
        doc_description="A test document.",
        page_count=n_pages,
        page_records=records,
        topics=topics,
        page_keys=[f"doc::p{i:03d}" for i in range(n_pages)],
    )


def test_document_hierarchy_round_trip():
    hier = _make_hierarchy(4)
    restored = DocumentHierarchy.from_dict(hier.to_dict())
    assert restored.doc_name == "test.pdf"
    assert restored.page_count == 4
    assert len(restored.page_records) == 4
    assert restored.page_records[2].summary == "Summary of page 2."
    assert len(restored.topics) == 2
    assert restored.topics[0].theme == "Intro"
    assert restored.topics[1].page_indices == [2, 3]


def test_document_hierarchy_record_for_page():
    hier = _make_hierarchy(3)
    rec = hier.record_for_page(1)
    assert rec is not None
    assert rec.page_idx == 1
    assert hier.record_for_page(99) is None


def test_document_hierarchy_pages_for_topic():
    hier = _make_hierarchy(4)
    assert hier.pages_for_topic("T01") == [0, 1]
    assert hier.pages_for_topic("T02") == [2, 3]
    assert hier.pages_for_topic("T99") == []


def test_document_hierarchy_as_text_tree():
    hier = _make_hierarchy(4)
    text = hier.as_text_tree()
    assert "test.pdf" in text
    assert "T01" in text
    assert "Intro" in text
    assert "T02" in text
    assert "Results" in text
    assert "p00" in text or "p0" in text


def test_document_hierarchy_preserves_legacy_nodes():
    """DocumentHierarchy still round-trips old-format HierarchyNode data."""
    node = HierarchyNode("Chapter 1", "0001", 0, 5, summary="Intro")
    node.children = [HierarchyNode("Section 1.1", "0002", 1, 3)]
    hier = DocumentHierarchy("old.pdf", "Legacy doc", 10, nodes=[node])
    restored = DocumentHierarchy.from_dict(hier.to_dict())
    assert len(restored.nodes) == 1
    assert restored.nodes[0].title == "Chapter 1"
    assert len(restored.nodes[0].children) == 1


# ---------------------------------------------------------------------------
# HierarchyBuilder.build() — happy path
# ---------------------------------------------------------------------------


def _page_json(idx: int, long_text: bool = False) -> str:
    text = ("word " * 60).strip() if long_text else f"Key text on page {idx}"
    return json.dumps({
        "visual_description": "" if idx % 2 == 0 else f"Chart on page {idx}",
        "summary": f"Page {idx} covers topic {idx % 2}.",
        "keywords": [f"kw{idx}", "colette"],
        "text_content": text,
    })


def _topics_json(page_count: int) -> str:
    all_pages = list(range(page_count))
    return json.dumps([
        {
            "theme": "Topic A",
            "description": "First half of the document.",
            "keywords": ["intro", "setup"],
            "page_indices": all_pages[: page_count // 2],
        },
        {
            "theme": "Topic B",
            "description": "Second half of the document.",
            "keywords": ["results", "conclusion"],
            "page_indices": all_pages[page_count // 2 :],
        },
    ])


def test_build_returns_correct_structure():
    n = 4
    page_responses = [_page_json(i) for i in range(n)]
    doc_desc = "A concise four-page test document."
    topics_resp = _topics_json(n)
    vlm = _FixedVLM(page_responses + [doc_desc, topics_resp])

    builder = HierarchyBuilder(vlm_fn=vlm)
    hier = builder.build("test.pdf", _make_pages(n))

    assert hier.doc_name == "test.pdf"
    assert hier.page_count == n
    assert len(hier.page_records) == n
    assert hier.doc_description == doc_desc
    assert len(hier.topics) == 2
    assert hier.topics[0].theme == "Topic A"
    assert hier.topics[1].theme == "Topic B"


def test_build_page_records_populated():
    n = 3
    page_responses = [_page_json(i) for i in range(n)]
    vlm = _FixedVLM(page_responses + ["A three-page document.", _topics_json(n)])
    builder = HierarchyBuilder(vlm_fn=vlm)
    hier = builder.build("doc.pdf", _make_pages(n))

    for i, rec in enumerate(hier.page_records):
        assert rec.page_idx == i
        assert f"Page {i}" in rec.summary or rec.summary != ""
        assert len(rec.keywords) >= 1


def test_build_field_order_salvage():
    """If text_content is truncated, summary and keywords must already be present."""
    n = 2
    # Page 0: normal response (new field order)
    page0 = json.dumps({
        "visual_description": "a diagram",
        "summary": "First page intro.",
        "keywords": ["intro"],
        "text_content": "Intro text.",
    })
    # Page 1: truncated during text_content (summary+keywords already written)
    page1_truncated = (
        '{\n'
        '  "visual_description": "",\n'
        '  "summary": "Dense slide with many items.",\n'
        '  "keywords": ["dense", "slide"],\n'
        '  "text_content": "Item 1 Item 2 Item 3 Item 4'  # cut off
    )
    vlm = _FixedVLM([page0, page1_truncated, "A two-page document.", _topics_json(n)])
    logger = _LogCapture()
    builder = HierarchyBuilder(vlm_fn=vlm, logger=logger)
    hier = builder.build("dense.pdf", _make_pages(n))

    rec1 = hier.record_for_page(1)
    assert rec1 is not None
    assert rec1.summary == "Dense slide with many items."
    assert rec1.keywords == ["dense", "slide"]
    assert any("[hindex] JSON truncated" in w for w in logger.warnings)


def test_build_with_page_keys():
    n = 2
    keys = ["doc::p000", "doc::p001"]
    responses = [_page_json(i) for i in range(n)] + ["Doc desc.", _topics_json(n)]
    vlm = _FixedVLM(responses)
    builder = HierarchyBuilder(vlm_fn=vlm)
    hier = builder.build("keyed.pdf", _make_pages(n), page_keys=keys)

    assert hier.page_keys == keys
    assert hier.page_records[0].page_key == "doc::p000"
    assert hier.page_records[1].page_key == "doc::p001"


def test_build_single_page():
    responses = [
        _page_json(0),
        "A single-page document.",
        json.dumps([{
            "theme": "Only Topic",
            "description": "Everything.",
            "keywords": ["all"],
            "page_indices": [0],
        }]),
    ]
    vlm = _FixedVLM(responses)
    builder = HierarchyBuilder(vlm_fn=vlm)
    hier = builder.build("single.pdf", _make_pages(1))

    assert hier.page_count == 1
    assert len(hier.page_records) == 1
    assert len(hier.topics) == 1
