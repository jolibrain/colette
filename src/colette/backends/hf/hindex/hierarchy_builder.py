"""Vision-first hierarchical document index builder.

Inspired by the PageIndex concept (VectifyAI/PageIndex) but implemented
from scratch using Colette's existing VLM infrastructure:

  1. Page images are already rendered by the preprocessing pipeline (300 DPI).
  2. The VLM (Qwen2-VL / Qwen3-VL via vLLM) performs OCR on each page image
     to extract clean text — no separate OCR library needed.
  3. Extracted text is processed to detect and parse the document's table of
     contents (TOC) into a hierarchical JSON tree.
  4. Page-number-to-physical-index mapping is resolved from PDF metadata at
     preprocessing time, so the VLM never has to read printed page numbers.

The output is a JSON tree stored by HierarchyStore and consumed by
HierarchyRetriever at query time.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class HierarchyNode:
    """One node in the document hierarchy tree."""

    title: str
    node_id: str
    start_page: int  # 0-based physical page index
    end_page: int    # 0-based physical page index (inclusive)
    summary: str = ""
    children: list["HierarchyNode"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "node_id": self.node_id,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "summary": self.summary,
            "children": [c.to_dict() for c in self.children],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "HierarchyNode":
        node = cls(
            title=d["title"],
            node_id=d["node_id"],
            start_page=d["start_page"],
            end_page=d["end_page"],
            summary=d.get("summary", ""),
        )
        node.children = [cls.from_dict(c) for c in d.get("children", [])]
        return node


@dataclass
class DocumentHierarchy:
    """Root of the hierarchy tree for a single document."""

    doc_name: str
    doc_description: str
    page_count: int
    nodes: list[HierarchyNode] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "doc_description": self.doc_description,
            "page_count": self.page_count,
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DocumentHierarchy":
        obj = cls(
            doc_name=d["doc_name"],
            doc_description=d.get("doc_description", ""),
            page_count=d["page_count"],
        )
        obj.nodes = [HierarchyNode.from_dict(n) for n in d.get("nodes", [])]
        return obj

    def flat_nodes(self) -> list[HierarchyNode]:
        """Return all nodes in document order (pre-order traversal)."""
        result: list[HierarchyNode] = []

        def _walk(nodes: list[HierarchyNode]) -> None:
            for n in nodes:
                result.append(n)
                _walk(n.children)

        _walk(self.nodes)
        return result

    def as_text_tree(self) -> str:
        """Compact text representation of the tree for LLM context."""
        lines: list[str] = [f"Document: {self.doc_name} ({self.page_count} pages)"]

        def _fmt(nodes: list[HierarchyNode], indent: int = 0) -> None:
            for n in nodes:
                prefix = "  " * indent
                lines.append(
                    f"{prefix}[{n.node_id}] {n.title} (pages {n.start_page}–{n.end_page})"
                )
                _fmt(n.children, indent + 1)

        _fmt(self.nodes)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

# Number of pages to scan when looking for a TOC.
_DEFAULT_TOC_SCAN_PAGES = 20

# JSON schema hint injected into prompts so the VLM emits structured output.
_TOC_JSON_SCHEMA = """
{
  "has_toc": true | false,
  "entries": [
    {"title": "Section title", "level": 1, "page_label": "1"},
    ...
  ]
}
""".strip()

_SUMMARY_MAX_WORDS = 50


class HierarchyBuilder:
    """Builds a DocumentHierarchy from page images using a VLM as OCR + reasoner.

    Parameters
    ----------
    vlm_fn:
        Callable(images: list[PIL.Image], prompt: str) -> str.
        Wraps whatever VLM backend is active (vLLM, HF, …).  The caller is
        responsible for wiring the correct model; HierarchyBuilder is backend-
        agnostic.
    page_label_map:
        Mapping from physical page index (0-based) to the printed page label
        (e.g. {"0": "i", "4": "1"}).  Built from PDF metadata at preprocessing
        time by DocumentProcessor — avoids asking the VLM to read page numbers.
    toc_scan_pages:
        How many pages to inspect when searching for a TOC.
    add_node_summaries:
        Whether to generate a one-line summary for each node via the VLM.
    logger:
        Standard Python logger or any object with .info / .warning / .debug.
    """

    def __init__(
        self,
        vlm_fn,
        page_label_map: dict[int, str] | None = None,
        toc_scan_pages: int = _DEFAULT_TOC_SCAN_PAGES,
        add_node_summaries: bool = True,
        logger=None,
    ):
        self.vlm_fn = vlm_fn
        self.page_label_map: dict[int, str] = page_label_map or {}
        self.toc_scan_pages = toc_scan_pages
        self.add_node_summaries = add_node_summaries
        self.logger = logger or _NullLogger()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        doc_name: str,
        page_images: list[Image.Image],
    ) -> DocumentHierarchy:
        """Build the full hierarchy for *doc_name* from its rendered page images.

        Parameters
        ----------
        doc_name:
            Display name / filename of the document.
        page_images:
            Ordered list of PIL images — one per physical page (0-indexed).
        """
        page_count = len(page_images)
        self.logger.info(f"[hindex] Building hierarchy for '{doc_name}' ({page_count} pages)")

        # Step 1 — OCR first N pages and look for a TOC
        scan_images = page_images[: min(self.toc_scan_pages, page_count)]
        toc_text = self._ocr_pages(scan_images)
        toc_data = self._detect_and_parse_toc(toc_text, scan_images)

        # Step 2 — Map TOC entries to physical page indices
        if toc_data and toc_data.get("has_toc") and toc_data.get("entries"):
            self.logger.info(f"[hindex] TOC detected with {len(toc_data['entries'])} entries")
            nodes = self._map_toc_to_pages(toc_data["entries"], page_count)
        else:
            self.logger.info("[hindex] No TOC found — inferring structure from content")
            nodes = self._infer_structure(page_images)

        # Step 3 — Generate a document-level description
        doc_description = self._generate_doc_description(page_images[:3], doc_name)

        # Step 4 — Optionally generate per-node summaries
        if self.add_node_summaries:
            self._add_summaries(nodes, page_images)

        return DocumentHierarchy(
            doc_name=doc_name,
            doc_description=doc_description,
            page_count=page_count,
            nodes=nodes,
        )

    # ------------------------------------------------------------------
    # Internal steps
    # ------------------------------------------------------------------

    def _ocr_pages(self, images: list[Image.Image]) -> str:
        """Use the VLM to OCR a list of page images and return joined text."""
        prompt = (
            "Read and transcribe all text visible in this document page exactly as it appears. "
            "Preserve line breaks and indentation. Output only the transcribed text."
        )
        texts: list[str] = []
        for i, img in enumerate(images):
            self.logger.debug(f"[hindex] OCR page {i}")
            texts.append(self.vlm_fn([img], prompt))
        return "\n\n".join(texts)

    def _detect_and_parse_toc(
        self, ocr_text: str, scan_images: list[Image.Image]
    ) -> dict[str, Any] | None:
        """Ask the VLM to find and extract a TOC from the OCR'd text."""
        prompt = (
            "You are analyzing the text extracted from the first pages of a document.\n"
            "Determine if there is a Table of Contents (TOC).\n"
            "If yes, extract all entries as a JSON object matching this schema:\n"
            f"{_TOC_JSON_SCHEMA}\n\n"
            "Rules:\n"
            "- 'level' is the heading depth (1 = top-level chapter, 2 = section, 3 = subsection).\n"
            "- 'page_label' is the printed page number/label string exactly as it appears (may be Roman).\n"
            "- If no TOC exists set has_toc to false and entries to [].\n"
            "Output ONLY the JSON object, no markdown fences.\n\n"
            f"Extracted text:\n{ocr_text[:8000]}"
        )
        raw = self.vlm_fn(scan_images[:1], prompt)
        return _parse_json_response(raw, self.logger)

    def _map_toc_to_pages(
        self, entries: list[dict[str, Any]], page_count: int
    ) -> list[HierarchyNode]:
        """Convert flat TOC entries (with printed page labels) to HierarchyNodes."""
        # Invert page_label_map: printed label → physical index
        label_to_phys: dict[str, int] = {v: k for k, v in self.page_label_map.items()}

        nodes_flat: list[HierarchyNode] = []
        for i, entry in enumerate(entries):
            label = str(entry.get("page_label", "")).strip()
            phys = label_to_phys.get(label)
            if phys is None:
                # Fallback: try numeric parse
                try:
                    phys = int(label) - 1
                except (ValueError, TypeError):
                    phys = 0
            node_id = f"{i + 1:04d}"
            nodes_flat.append(
                HierarchyNode(
                    title=entry.get("title", f"Section {node_id}"),
                    node_id=node_id,
                    start_page=max(0, phys),
                    end_page=max(0, phys),  # end_page filled in below
                )
            )

        # Assign end_page: each node ends where the next one begins
        for j in range(len(nodes_flat) - 1):
            nodes_flat[j].end_page = max(nodes_flat[j].start_page, nodes_flat[j + 1].start_page - 1)
        if nodes_flat:
            nodes_flat[-1].end_page = page_count - 1

        return _nest_by_level(entries, nodes_flat)

    def _infer_structure(self, page_images: list[Image.Image]) -> list[HierarchyNode]:
        """For documents without a TOC, ask the VLM to infer a high-level structure."""
        # OCR all pages (batched in chunks to avoid context overflow)
        all_texts: list[str] = []
        for i, img in enumerate(page_images):
            all_texts.append(f"--- Page {i} ---\n" + self.vlm_fn([img], "Transcribe all text on this page."))

        combined = "\n\n".join(all_texts)[:12000]
        prompt = (
            "Given the following document text (page-by-page), identify the main sections.\n"
            "Output a JSON array of section objects:\n"
            '[{"title": "...", "start_page": 0, "end_page": 3}, ...]\n'
            "Use 0-based page indices. Output ONLY the JSON array.\n\n"
            f"{combined}"
        )
        raw = self.vlm_fn(page_images[:1], prompt)
        data = _parse_json_response(raw, self.logger)
        if not isinstance(data, list):
            self.logger.warning("[hindex] Could not infer structure; falling back to single root node")
            return [
                HierarchyNode(
                    title="Document",
                    node_id="0001",
                    start_page=0,
                    end_page=len(page_images) - 1,
                )
            ]

        nodes: list[HierarchyNode] = []
        for i, item in enumerate(data):
            nodes.append(
                HierarchyNode(
                    title=item.get("title", f"Section {i + 1}"),
                    node_id=f"{i + 1:04d}",
                    start_page=int(item.get("start_page", 0)),
                    end_page=int(item.get("end_page", len(page_images) - 1)),
                )
            )
        return nodes

    def _generate_doc_description(self, first_pages: list[Image.Image], doc_name: str) -> str:
        prompt = (
            f"You are reading the first pages of a document named '{doc_name}'.\n"
            "Write a single sentence that describes what this document is about.\n"
            "Output only the sentence."
        )
        return self.vlm_fn(first_pages, prompt).strip()

    def _add_summaries(
        self, nodes: list[HierarchyNode], page_images: list[Image.Image]
    ) -> None:
        """Generate a short summary for each leaf/top-level node."""

        def _summarise(node: HierarchyNode) -> None:
            imgs = page_images[node.start_page : node.end_page + 1]
            if not imgs:
                return
            # Use only first image to keep cost low; enough for a short summary
            prompt = (
                f"You are reading pages {node.start_page}–{node.end_page} of a document "
                f"(section: '{node.title}').\n"
                f"Write a summary in at most {_SUMMARY_MAX_WORDS} words.\n"
                "Output only the summary."
            )
            node.summary = self.vlm_fn(imgs[:1], prompt).strip()
            for child in node.children:
                _summarise(child)

        for n in nodes:
            _summarise(n)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_response(raw: str, logger) -> Any:
    """Extract JSON from a VLM response that may contain surrounding prose."""
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # Try to extract the first {...} or [...] block
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    logger.warning(f"[hindex] Could not parse JSON from VLM response: {raw[:200]}")
    return None


def _nest_by_level(
    entries: list[dict[str, Any]], flat_nodes: list[HierarchyNode]
) -> list[HierarchyNode]:
    """Reconstruct a tree from flat nodes using the 'level' field in entries."""
    if not flat_nodes:
        return []

    levels = [int(e.get("level", 1)) for e in entries]
    stack: list[tuple[int, HierarchyNode]] = []  # (level, node)
    roots: list[HierarchyNode] = []

    for level, node in zip(levels, flat_nodes):
        while stack and stack[-1][0] >= level:
            stack.pop()
        if not stack:
            roots.append(node)
        else:
            stack[-1][1].children.append(node)
        stack.append((level, node))

    return roots


class _NullLogger:
    def info(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass
    def debug(self, *_a, **_k): pass
