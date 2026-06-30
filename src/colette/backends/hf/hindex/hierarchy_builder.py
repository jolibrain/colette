"""Vision-first hierarchical document index builder.

Build strategy (bottom-up):

  Phase 1 — Per-page extraction (one rich VLM call per page)
    For each page image the VLM extracts:
      • text_content      : verbatim OCR of all text
      • visual_description: description of diagrams / charts / figures
      • summary           : 2-3 sentence condensed meaning
      • keywords          : 3-6 thematic tags

  Phase 2 — Document-level synthesis (one VLM call over all summaries)
    • doc_description: what the whole document is about
    • topics: thematic clusters of pages (pages may belong to multiple topics)

  Phase 3 — Crop key assignment (no VLM, after preprocessing)
    Caller passes the parts list from preprocessing; each PageRecord gets the
    kvstore keys of all crops belonging to its page.

At query time the retriever can use:
  • Topic navigation: VLM picks a topic → returns pages in that topic
  • Summary search : query matched against pre-extracted summaries / keywords
  • Both together  : search to rank, topics to expand / re-rank

The pre-extracted per-page data is also returned alongside page images so the
answering LLM gets richer context without re-reading from scratch.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from PIL import Image


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class PageRecord:
    """Rich per-page extracted information."""

    page_idx: int            # 0-based physical page index
    page_key: str            # kvstore key for the full-page image
    text_content: str        # verbatim OCR
    visual_description: str  # description of diagrams / figures / charts
    summary: str             # 2-3 sentence condensed meaning
    keywords: list[str]      # thematic tags (3-6)
    crop_keys: list[str] = field(default_factory=list)
    """Kvstore keys of all layout-detected crops for this page.
    Populated by the caller (RAGImg._build_hindex_for_doc) after
    preprocessing so the key format matches the stored crops exactly."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_idx": self.page_idx,
            "page_key": self.page_key,
            "text_content": self.text_content,
            "visual_description": self.visual_description,
            "summary": self.summary,
            "keywords": self.keywords,
            "crop_keys": self.crop_keys,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PageRecord":
        return cls(
            page_idx=d["page_idx"],
            page_key=d.get("page_key", ""),
            text_content=d.get("text_content", ""),
            visual_description=d.get("visual_description", ""),
            summary=d.get("summary", ""),
            keywords=d.get("keywords", []),
            crop_keys=d.get("crop_keys", []),
        )


@dataclass
class TopicNode:
    """A thematic cluster of pages.

    Pages may belong to multiple topics (this is a DAG, not a strict tree).
    """

    topic_id: str
    theme: str               # short label, e.g. "RAG Architecture"
    description: str         # what this topic covers
    keywords: list[str]
    page_indices: list[int]  # 0-based; a page can appear in multiple topics

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_id": self.topic_id,
            "theme": self.theme,
            "description": self.description,
            "keywords": self.keywords,
            "page_indices": self.page_indices,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TopicNode":
        return cls(
            topic_id=d["topic_id"],
            theme=d["theme"],
            description=d.get("description", ""),
            keywords=d.get("keywords", []),
            page_indices=d.get("page_indices", []),
        )


@dataclass
class HierarchyNode:
    """Legacy node kept for backward compatibility with existing JSON files."""

    title: str
    node_id: str
    start_page: int
    end_page: int
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
    """Root of the hierarchy for a single document."""

    doc_name: str
    doc_description: str
    page_count: int

    # Rich per-page records (new model)
    page_records: list[PageRecord] = field(default_factory=list)
    # Thematic topic clusters (new model)
    topics: list[TopicNode] = field(default_factory=list)

    # Flat list of kvstore keys (one per physical page, 0-indexed)
    page_keys: list[str] = field(default_factory=list)

    # Legacy TOC-based nodes (kept for backward compat with old JSON files)
    nodes: list[HierarchyNode] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        return {
            "doc_name": self.doc_name,
            "doc_description": self.doc_description,
            "page_count": self.page_count,
            "page_keys": self.page_keys,
            "page_records": [r.to_dict() for r in self.page_records],
            "topics": [t.to_dict() for t in self.topics],
            "nodes": [n.to_dict() for n in self.nodes],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DocumentHierarchy":
        obj = cls(
            doc_name=d["doc_name"],
            doc_description=d.get("doc_description", ""),
            page_count=d["page_count"],
            page_keys=d.get("page_keys", []),
        )
        obj.page_records = [PageRecord.from_dict(r) for r in d.get("page_records", [])]
        obj.topics = [TopicNode.from_dict(t) for t in d.get("topics", [])]
        obj.nodes = [HierarchyNode.from_dict(n) for n in d.get("nodes", [])]
        return obj

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def flat_nodes(self) -> list[HierarchyNode]:
        """Legacy: all HierarchyNodes in pre-order (for backward compat)."""
        result: list[HierarchyNode] = []

        def _walk(nodes: list[HierarchyNode]) -> None:
            for n in nodes:
                result.append(n)
                _walk(n.children)

        _walk(self.nodes)
        return result

    def as_text_tree(self) -> str:
        """Compact text representation for LLM context and debugging."""
        lines: list[str] = [f"Document: {self.doc_name} ({self.page_count} pages)"]
        if self.doc_description:
            lines.append(f"  {self.doc_description}")

        if self.topics:
            lines.append("")
            lines.append("Topics:")
            for t in self.topics:
                pages_str = ", ".join(str(p) for p in t.page_indices)
                lines.append(f"  [{t.topic_id}] {t.theme}  (pages: {pages_str})")
                if t.description:
                    lines.append(f"       {t.description}")
        elif self.nodes:
            # Legacy fallback
            def _fmt(nodes: list[HierarchyNode], indent: int = 0) -> None:
                for n in nodes:
                    prefix = "  " * indent
                    lines.append(
                        f"{prefix}[{n.node_id}] {n.title} "
                        f"(pages {n.start_page}–{n.end_page})"
                    )
                    _fmt(n.children, indent + 1)

            lines.append("")
            _fmt(self.nodes)

        if self.page_records:
            lines.append("")
            lines.append("Pages:")
            for r in self.page_records:
                kw = ", ".join(r.keywords)
                lines.append(f"  p{r.page_idx:02d}: {r.summary[:80]}…  [{kw}]")

        return "\n".join(lines)

    def pages_for_topic(self, topic_id: str) -> list[int]:
        """Return page indices belonging to a topic."""
        for t in self.topics:
            if t.topic_id == topic_id:
                return list(t.page_indices)
        return []

    def record_for_page(self, page_idx: int) -> PageRecord | None:
        """Return the PageRecord for a given 0-based page index."""
        for r in self.page_records:
            if r.page_idx == page_idx:
                return r
        return None


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------

_DEFAULT_TOC_SCAN_PAGES = 20  # kept for API compatibility
_SUMMARY_MAX_WORDS = 50       # kept for legacy _add_summaries

_PAGE_EXTRACT_SCHEMA = """{
  "text_content": "all text visible on the page, verbatim",
  "visual_description": "describe any diagrams, charts, figures, images (empty string if none)",
  "summary": "2-3 sentence summary of what this page is about",
  "keywords": ["keyword1", "keyword2", "keyword3"]
}"""

_TOPIC_SCHEMA = """[
  {
    "theme": "Short theme label (2-4 words)",
    "description": "One sentence describing what this topic covers",
    "keywords": ["kw1", "kw2"],
    "page_indices": [0, 2, 5]
  }
]"""


class HierarchyBuilder:
    """Builds a DocumentHierarchy from page images using a VLM.

    Parameters
    ----------
    vlm_fn:
        Callable(images: list[PIL.Image], prompt: str) -> str.
    page_label_map:
        Mapping from physical page index to printed page label (unused in the
        new bottom-up build; kept for API compatibility).
    toc_scan_pages:
        Unused in the new build (kept for API compatibility).
    add_node_summaries:
        Unused in the new build (kept for API compatibility).
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
        page_keys: list[str] | None = None,
    ) -> DocumentHierarchy:
        """Build the full hierarchy for *doc_name* from its rendered page images.

        Parameters
        ----------
        doc_name:
            Display name / filename of the document.
        page_images:
            Ordered list of PIL images — one per physical page (0-indexed).
        page_keys:
            Optional pre-computed kvstore keys for each page.  If omitted,
            keys are derived from the document uuid5 by the caller and set
            on the returned hierarchy afterwards.
        """
        page_count = len(page_images)
        self.logger.info(
            f"[hindex] Building hierarchy for '{doc_name}' ({page_count} pages)"
        )

        # Phase 1 — Rich per-page extraction
        page_records: list[PageRecord] = []
        for idx, img in enumerate(page_images):
            self.logger.info(f"[hindex] Extracting page {idx}/{page_count - 1}")
            record = self._extract_page_record(img, idx, page_keys[idx] if page_keys else "")
            page_records.append(record)

        # Phase 2 — Document description + topic clustering
        doc_description = self._generate_doc_description(page_records)
        topics = self._cluster_into_topics(page_records)

        return DocumentHierarchy(
            doc_name=doc_name,
            doc_description=doc_description,
            page_count=page_count,
            page_records=page_records,
            topics=topics,
            page_keys=page_keys or [""] * page_count,
        )

    # ------------------------------------------------------------------
    # Phase 1 — Per-page extraction
    # ------------------------------------------------------------------

    def _extract_page_record(
        self, image: Image.Image, page_idx: int, page_key: str = ""
    ) -> PageRecord:
        """One rich VLM call per page → structured PageRecord."""
        prompt = (
            "Analyze this document page carefully and output a JSON object with "
            "exactly these keys:\n"
            f"{_PAGE_EXTRACT_SCHEMA}\n\n"
            "Rules:\n"
            "- text_content: transcribe every word visible, preserve structure\n"
            "- visual_description: if no visual elements (diagrams/charts/figures), "
            "write an empty string\n"
            "- summary: 2-3 sentences on the meaning/purpose of this page\n"
            "- keywords: 3-6 thematic terms (not generic words like 'page' or 'slide')\n"
            "Output ONLY the JSON object, no markdown fences, no other text."
        )
        raw = self.vlm_fn([image], prompt)
        data = _parse_json_response(raw, self.logger) or {}

        return PageRecord(
            page_idx=page_idx,
            page_key=page_key,
            text_content=str(data.get("text_content", "")),
            visual_description=str(data.get("visual_description", "")),
            summary=str(data.get("summary", "")),
            keywords=list(data.get("keywords", [])),
        )

    # ------------------------------------------------------------------
    # Phase 2 — Document synthesis
    # ------------------------------------------------------------------

    def _generate_doc_description(self, page_records: list[PageRecord]) -> str:
        """One VLM call over all summaries → single-sentence document description."""
        combined = "\n".join(
            f"Page {r.page_idx}: {r.summary}  [{', '.join(r.keywords)}]"
            for r in page_records
            if r.summary
        )[:6000]
        prompt = (
            "Below are per-page summaries of a document.\n"
            "Write a single sentence that describes the overall document topic and purpose.\n"
            "Output only that sentence.\n\n"
            f"{combined}"
        )
        return self.vlm_fn([], prompt).strip()

    def _cluster_into_topics(self, page_records: list[PageRecord]) -> list[TopicNode]:
        """One VLM call over all summaries → thematic topic clusters."""
        page_count = len(page_records)
        combined = "\n".join(
            f"Page {r.page_idx}: {r.summary}  Keywords: {', '.join(r.keywords)}"
            for r in page_records
        )[:8000]
        last = page_count - 1
        prompt = (
            f"A document has {page_count} pages (indices 0 to {last}).\n"
            "Group them into 3-7 thematic topics. A page may belong to multiple topics.\n"
            f"Every page index (0 to {last}) must appear in at least one topic.\n"
            "Output ONLY a JSON array matching this schema:\n"
            f"{_TOPIC_SCHEMA}\n\n"
            "Rules:\n"
            "- theme: 2-4 words, descriptive, not generic ('Introduction', 'Conclusion' are ok)\n"
            "- page_indices: use 0-based integer indices\n"
            "- Output ONLY the JSON array, no other text\n\n"
            f"Page summaries:\n{combined}"
        )
        raw = self.vlm_fn([], prompt)
        data = _parse_json_response(raw, self.logger)

        topics: list[TopicNode] = []
        if isinstance(data, list):
            for i, item in enumerate(data):
                try:
                    indices = [
                        int(p) for p in item.get("page_indices", [])
                        if 0 <= int(p) < page_count
                    ]
                    if not indices:
                        continue
                    topics.append(TopicNode(
                        topic_id=f"T{i + 1:02d}",
                        theme=str(item.get("theme", f"Topic {i + 1}")),
                        description=str(item.get("description", "")),
                        keywords=list(item.get("keywords", [])),
                        page_indices=indices,
                    ))
                except (TypeError, ValueError):
                    continue

        if not topics:
            self.logger.warning(
                "[hindex] Topic clustering failed; falling back to single topic"
            )
            topics = [TopicNode(
                topic_id="T01",
                theme="Document",
                description="All pages of the document",
                keywords=[],
                page_indices=list(range(page_count)),
            )]

        # Ensure every page is covered by at least one topic
        covered = {p for t in topics for p in t.page_indices}
        missing = [p for p in range(page_count) if p not in covered]
        if missing:
            self.logger.warning(
                f"[hindex] {len(missing)} page(s) not assigned to any topic; "
                "adding catch-all topic"
            )
            topics.append(TopicNode(
                topic_id=f"T{len(topics) + 1:02d}",
                theme="Additional Pages",
                description="Pages not grouped into other topics",
                keywords=[],
                page_indices=missing,
            ))

        return topics

    # ------------------------------------------------------------------
    # Legacy helpers (kept for backward compatibility / tests)
    # ------------------------------------------------------------------

    def _ocr_pages_list(self, images: list[Image.Image]) -> list[str]:
        """OCR each image and return one string per page (legacy)."""
        prompt = (
            "Read and transcribe all text visible in this document page exactly as it appears. "
            "Preserve line breaks and indentation. Output only the transcribed text."
        )
        return [self.vlm_fn([img], prompt) for img in images]

    def _ocr_pages(self, images: list[Image.Image]) -> str:
        """OCR and return joined text (legacy)."""
        return "\n\n".join(self._ocr_pages_list(images))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_json_response(raw: str, logger) -> Any:
    """Extract JSON from a VLM response that may contain surrounding prose."""
    if not raw:
        return None
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"(\{.*\}|\[.*\])", raw, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass
    logger.warning(f"[hindex] Could not parse JSON from VLM response: {raw[:200]}")
    return None


def _page_key(uuid5_val: str, page_idx: int) -> str:
    """Kvstore key for a full-page image."""
    return f"{uuid5_val}_{page_idx:04d}"


class _NullLogger:
    def info(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass
    def debug(self, *_a, **_k): pass
