"""Unit tests for HierarchyBuilder.

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
    _nest_by_level,
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
    """A fake VLM that returns a pre-configured sequence of responses."""

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

    def info(self, msg, *_a, **_k): pass
    def debug(self, msg, *_a, **_k): pass
    def warning(self, msg, *_a, **_k): self.warnings.append(msg)


# ---------------------------------------------------------------------------
# _parse_json_response
# ---------------------------------------------------------------------------


def test_parse_json_response_clean():
    logger = _LogCapture()
    result = _parse_json_response('{"has_toc": true, "entries": []}', logger)
    assert result == {"has_toc": True, "entries": []}
    assert not logger.warnings


def test_parse_json_response_with_markdown_fence():
    logger = _LogCapture()
    raw = "```json\n{\"has_toc\": false, \"entries\": []}\n```"
    result = _parse_json_response(raw, logger)
    assert result == {"has_toc": False, "entries": []}


def test_parse_json_response_embedded_in_prose():
    logger = _LogCapture()
    raw = 'Here is the result: {"has_toc": true, "entries": [{"title": "Intro", "level": 1, "page_label": "1"}]}'
    result = _parse_json_response(raw, logger)
    assert result is not None
    assert result["has_toc"] is True


def test_parse_json_response_invalid_returns_none():
    logger = _LogCapture()
    result = _parse_json_response("not json at all", logger)
    assert result is None
    assert logger.warnings


# ---------------------------------------------------------------------------
# _nest_by_level
# ---------------------------------------------------------------------------


def test_nest_by_level_flat():
    entries = [
        {"level": 1, "page_label": "1"},
        {"level": 1, "page_label": "5"},
    ]
    flat = [
        HierarchyNode("A", "0001", 0, 4),
        HierarchyNode("B", "0002", 4, 9),
    ]
    roots = _nest_by_level(entries, flat)
    assert len(roots) == 2
    assert roots[0].title == "A"
    assert roots[1].title == "B"
    assert not roots[0].children


def test_nest_by_level_with_children():
    entries = [
        {"level": 1, "page_label": "1"},
        {"level": 2, "page_label": "3"},
        {"level": 2, "page_label": "5"},
        {"level": 1, "page_label": "7"},
    ]
    flat = [
        HierarchyNode("Chapter 1", "0001", 0, 6),
        HierarchyNode("Section 1.1", "0002", 2, 3),
        HierarchyNode("Section 1.2", "0003", 4, 5),
        HierarchyNode("Chapter 2", "0004", 6, 9),
    ]
    roots = _nest_by_level(entries, flat)
    assert len(roots) == 2
    assert roots[0].title == "Chapter 1"
    assert len(roots[0].children) == 2
    assert roots[0].children[0].title == "Section 1.1"
    assert roots[0].children[1].title == "Section 1.2"
    assert roots[1].title == "Chapter 2"
    assert not roots[1].children


# ---------------------------------------------------------------------------
# HierarchyNode serialization round-trip
# ---------------------------------------------------------------------------


def test_hierarchy_node_round_trip():
    node = HierarchyNode("Chapter 1", "0001", 0, 5, summary="An intro chapter")
    node.children = [HierarchyNode("Section 1.1", "0002", 1, 3)]
    restored = HierarchyNode.from_dict(node.to_dict())
    assert restored.title == node.title
    assert restored.node_id == node.node_id
    assert restored.start_page == node.start_page
    assert restored.end_page == node.end_page
    assert restored.summary == node.summary
    assert len(restored.children) == 1
    assert restored.children[0].title == "Section 1.1"


# ---------------------------------------------------------------------------
# DocumentHierarchy
# ---------------------------------------------------------------------------


def test_document_hierarchy_flat_nodes():
    root = HierarchyNode("Chap 1", "0001", 0, 5)
    child = HierarchyNode("Sec 1.1", "0002", 1, 3)
    root.children = [child]
    hier = DocumentHierarchy("doc.pdf", "A test document", 10, nodes=[root])
    flat = hier.flat_nodes()
    assert len(flat) == 2
    assert flat[0].node_id == "0001"
    assert flat[1].node_id == "0002"


def test_document_hierarchy_as_text_tree():
    root = HierarchyNode("Introduction", "0001", 0, 2)
    root.children = [HierarchyNode("Background", "0002", 1, 2)]
    hier = DocumentHierarchy("report.pdf", "Annual report", 10, nodes=[root])
    text = hier.as_text_tree()
    assert "report.pdf" in text
    assert "[0001]" in text
    assert "Introduction" in text
    assert "[0002]" in text
    assert "Background" in text


def test_document_hierarchy_round_trip():
    root = HierarchyNode("Part I", "0001", 0, 9, summary="First part")
    root.children = [HierarchyNode("Chapter 1", "0002", 0, 4)]
    hier = DocumentHierarchy("book.pdf", "A great book", 20, nodes=[root])
    restored = DocumentHierarchy.from_dict(hier.to_dict())
    assert restored.doc_name == "book.pdf"
    assert restored.page_count == 20
    assert len(restored.nodes) == 1
    assert restored.nodes[0].title == "Part I"
    assert len(restored.nodes[0].children) == 1


# ---------------------------------------------------------------------------
# HierarchyBuilder.build() — with TOC
# ---------------------------------------------------------------------------


def _toc_vlm_responses(n_entries: int = 2) -> list[str]:
    """Build a minimal sequence of VLM responses that gets through build()."""
    # Response 1-N: OCR of first pages (one per page, toc_scan_pages=2)
    ocr_responses = ["Page text " + str(i) for i in range(2)]
    # Response for TOC detection
    toc_json = json.dumps(
        {
            "has_toc": True,
            "entries": [
                {"title": "Introduction", "level": 1, "page_label": "1"},
                {"title": "Methods", "level": 1, "page_label": "3"},
            ][:n_entries],
        }
    )
    # Response for doc description
    description = "This document explains test procedures."
    # Responses for per-node summaries (one per node, no children)
    summaries = [f"Summary of section {i+1}" for i in range(n_entries)]
    return ocr_responses + [toc_json, description] + summaries


def test_build_with_toc():
    pages = _make_pages(5)
    page_label_map = {0: "i", 1: "1", 2: "2", 3: "3", 4: "4"}
    vlm = _FixedVLM(_toc_vlm_responses(n_entries=2))
    builder = HierarchyBuilder(
        vlm_fn=vlm,
        page_label_map=page_label_map,
        toc_scan_pages=2,
        add_node_summaries=True,
    )
    hier = builder.build("test.pdf", pages)

    assert hier.doc_name == "test.pdf"
    assert hier.page_count == 5
    assert len(hier.nodes) == 2
    assert hier.nodes[0].title == "Introduction"
    assert hier.nodes[1].title == "Methods"
    assert hier.nodes[0].start_page == 1  # page_label "1" → physical index 1
    assert hier.nodes[1].start_page == 3  # page_label "3" → physical index 3
    # end_page of first = start of second - 1
    assert hier.nodes[0].end_page == 2


# ---------------------------------------------------------------------------
# HierarchyBuilder.build() — no TOC fallback
# ---------------------------------------------------------------------------


def _no_toc_vlm_responses() -> list[str]:
    ocr_responses = ["Page text " + str(i) for i in range(2)]
    no_toc_json = json.dumps({"has_toc": False, "entries": []})
    # Infer structure responses (per page OCR + structure JSON)
    page_ocr = ["Page text " + str(i) for i in range(5)]
    structure_json = json.dumps(
        [
            {"title": "Part One", "start_page": 0, "end_page": 2},
            {"title": "Part Two", "start_page": 3, "end_page": 4},
        ]
    )
    description = "A document without a table of contents."
    summaries = ["Summary one", "Summary two"]
    return ocr_responses + [no_toc_json] + page_ocr + [structure_json, description] + summaries


def test_build_without_toc_falls_back_to_inference():
    pages = _make_pages(5)
    vlm = _FixedVLM(_no_toc_vlm_responses())
    builder = HierarchyBuilder(
        vlm_fn=vlm,
        toc_scan_pages=2,
        add_node_summaries=True,
    )
    hier = builder.build("noindex.pdf", pages)

    assert hier.page_count == 5
    assert len(hier.nodes) == 2
    assert hier.nodes[0].title == "Part One"
    assert hier.nodes[1].title == "Part Two"


# ---------------------------------------------------------------------------
# HierarchyBuilder — no summaries
# ---------------------------------------------------------------------------


def test_build_without_summaries():
    pages = _make_pages(4)
    page_label_map = {0: "1", 1: "2", 2: "3", 3: "4"}
    responses = (
        ["OCR page " + str(i) for i in range(2)]
        + [
            json.dumps(
                {
                    "has_toc": True,
                    "entries": [{"title": "Only Section", "level": 1, "page_label": "1"}],
                }
            ),
            "A short document.",
        ]
    )
    vlm = _FixedVLM(responses)
    builder = HierarchyBuilder(
        vlm_fn=vlm,
        page_label_map=page_label_map,
        toc_scan_pages=2,
        add_node_summaries=False,
    )
    hier = builder.build("single.pdf", pages)
    assert len(hier.nodes) == 1
    assert hier.nodes[0].summary == ""
