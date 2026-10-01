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
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from time import time
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
    text_content: str        # key OCR text (metrics/values verbatim, max 300 chars)
    visual_description: str  # description of diagrams / figures / charts
    summary: str             # 2-3 sentence condensed meaning
    keywords: list[str]      # thematic tags (3-6)
    crop_keys: list[str] = field(default_factory=list)
    """Kvstore keys of all layout-detected crops for this page.
    Populated by the caller (RAGImg._build_hindex_for_doc) after
    preprocessing so the key format matches the stored crops exactly."""
    anticipated_queries: list[str] = field(default_factory=list)
    """Natural-language questions a user would ask to find this page.
    Indexed in BM25 with crop_label='queryhint' to bridge vocabulary gap.
    Must be diverse: visual questions, keyword phrases, conceptual questions,
    and optionally entity-anchored questions — NOT all entity-anchored."""
    entities: dict = field(default_factory=dict)
    """Typed named entities extracted from the page: persons, organizations,
    dates, identifiers. Amounts are excluded from entity BM25 (too common
    in homogeneous corpora — amounts like '31.0%' appear across many docs
    and have near-zero discriminating IDF)."""
    page_type: str = ""
    """Content-type classification: 'visual-dominant', 'table-dominant',
    'text-dominant', or 'mixed'. Auto-detected from VLM output — no extra
    VLM call. Used at query time to apply page-type-aware signal weighting."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "page_idx": self.page_idx,
            "page_key": self.page_key,
            "text_content": self.text_content,
            "visual_description": self.visual_description,
            "summary": self.summary,
            "keywords": self.keywords,
            "crop_keys": self.crop_keys,
            "anticipated_queries": self.anticipated_queries,
            "entities": self.entities,
            "page_type": self.page_type,
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
            anticipated_queries=d.get("anticipated_queries", []),
            entities=d.get("entities", {}),
            page_type=d.get("page_type", ""),
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
  "visual_description": "describe any diagrams, charts, figures, images (empty string if none)",
  "summary": "2-3 sentence summary of what this page is about",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "text_content": "key text from the page, MAX 300 characters; if the page contains structured data (tables, charts, lists, key-value pairs), capture at least 3 representative entries verbatim (e.g. 'Label: Value, Name: Result'); otherwise write the most important sentence or phrase visible; truncate if needed",
  "entities": {
    "persons": ["person name, optionally with role in parentheses e.g. 'John Smith (sender)'"],
    "organizations": ["company, agency, or institution name"],
    "dates": ["date as written on the page or in ISO format"],
    "identifiers": ["reference number, code, ID, product name, model name, or named system/algorithm"]
  },
  "anticipated_queries": [
    "keyword-only search phrase (3-5 exact terms from this page)",
    "plain-language question about the main topic of this page",
    "conceptual question approaching the topic from a different angle without entity names (text-dominant pages) OR paraphrase of the main finding in different vocabulary (visual/table pages)",
    "visual question: what the chart/figure on this page shows — only if visual content is present",
    "fact-lookup using ACTUAL data from this page: real row/column names and numbers, not placeholders — only if structured data is present",
    "cross-page question bridging this page with an adjacent page — only if content clearly spans two pages"
  ]
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
        max_workers: int = 1,
        batch_size: int = 1,
    ):
        self.vlm_fn = vlm_fn
        self.page_label_map: dict[int, str] = page_label_map or {}
        self.toc_scan_pages = toc_scan_pages
        self.add_node_summaries = add_node_summaries
        self.logger = logger or _NullLogger()
        # Workers > 1 enables parallel Phase 1 page extraction.
        # Safe for vllm_client (HTTP, stateless); for in-process HF/vLLM models
        # the ModelCache lock serialises calls anyway so this has no effect.
        self.max_workers = max(1, max_workers)
        # batch_size > 1 sends multiple page images in a single VLM call and
        # parses the JSON array response.  More efficient for HF/local models
        # where per-call overhead (prefill, KV-cache setup) dominates.
        self.batch_size = max(1, batch_size)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        doc_name: str,
        page_images: list[Image.Image],
        page_keys: list[str] | None = None,
        scanned_page_indices: set[int] | None = None,
        doc_language: str | None = None,
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
        t0 = time()
        self.logger.info(
            f"[hindex] Building hierarchy for '{doc_name}' "
            f"({page_count} pages, batch_size={self.batch_size}, workers={self.max_workers})"
        )

        # Phase 1 — Rich per-page extraction
        # Three modes:
        #   batch_size > 1  : N pages per VLM call (best for HF/local GPU models)
        #   max_workers > 1 : parallel single-page calls (best for vllm_client)
        #   default         : sequential single-page calls
        # Scanned pages always use single-page OCR mode regardless of batch_size.
        _scanned = scanned_page_indices or set()
        page_records: list[PageRecord | None] = [None] * page_count

        if self.batch_size > 1:
            for batch_start in range(0, page_count, self.batch_size):
                batch_end = min(batch_start + self.batch_size, page_count)
                idxs = list(range(batch_start, batch_end))

                # Scanned pages get individual OCR-focused extraction
                scanned_in_batch = [i for i in idxs if i in _scanned]
                batch_idxs = [i for i in idxs if i not in _scanned]

                for i in scanned_in_batch:
                    key = page_keys[i] if page_keys else ""
                    page_records[i] = self._extract_page_record(page_images[i], i, key, is_scanned=True, doc_language=doc_language)

                if batch_idxs:
                    imgs = [page_images[i] for i in batch_idxs]
                    keys_b = [page_keys[i] if page_keys else "" for i in batch_idxs]
                    self.logger.info(
                        f"[hindex] Phase 1 batch: pages {batch_start}–{batch_end - 1}"
                        + (f" ({len(scanned_in_batch)} scanned extracted individually)" if scanned_in_batch else "")
                    )
                    t_b = time()
                    batch_recs = self._extract_batch(imgs, batch_idxs, keys_b, doc_language=doc_language)
                    elapsed = time() - t_b
                    self.logger.info(
                        f"[hindex] Phase 1 batch done in {elapsed:.1f}s "
                        f"({elapsed / len(batch_idxs):.1f}s/page)"
                    )
                    for i, rec in zip(batch_idxs, batch_recs):
                        page_records[i] = rec
        else:
            def _extract_one(idx: int) -> tuple[int, PageRecord]:
                key = page_keys[idx] if page_keys else ""
                return idx, self._extract_page_record(page_images[idx], idx, key, is_scanned=idx in _scanned, doc_language=doc_language)

            n_workers = min(self.max_workers, page_count)
            if n_workers > 1:
                self.logger.info(
                    f"[hindex] Phase 1: {page_count} pages, {n_workers} workers"
                )
                with ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_extract_one, i): i for i in range(page_count)}
                    for fut in as_completed(futures):
                        idx, record = fut.result()
                        page_records[idx] = record
                        self.logger.info(
                            f"[hindex] Page {idx} done "
                            f"({sum(1 for r in page_records if r is not None)}/{page_count})"
                        )
            else:
                for i in range(page_count):
                    t_p = time()
                    self.logger.info(
                        f"[hindex] Phase 1: page {i}/{page_count - 1}"
                    )
                    _, page_records[i] = _extract_one(i)
                    self.logger.info(
                        f"[hindex] Page {i} done in {time() - t_p:.1f}s"
                    )

        self.logger.info(f"[hindex] Phase 1 done in {time() - t0:.1f}s")

        # Phase 2 — Document description + topic clustering
        t_p2 = time()
        doc_description = self._generate_doc_description(page_records)
        topics = self._cluster_into_topics(page_records)
        self.logger.info(f"[hindex] Phase 2 done in {time() - t_p2:.1f}s")

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
        self, image: Image.Image, page_idx: int, page_key: str = "", is_scanned: bool = False,
        doc_language: str | None = None,
    ) -> PageRecord:
        """One rich VLM call per page → structured PageRecord.

        Retries up to 2 times on parse failure; on the final attempt uses a
        simplified prompt that is easier for the model to answer correctly.
        When is_scanned=True, uses an OCR-focused prompt with a higher token
        budget to extract all visible text from image-only pages.
        """
        _lang_instr = f"in {doc_language}" if doc_language else "in the same language as the document content"
        _AQ_RULES = _build_aq_rules(_lang_instr)
        if is_scanned:
            prompt_full = (
                "This document page appears to be scanned (image-only, no embedded text). "
                "Analyze it carefully and output a JSON object with exactly these keys:\n"
                f"{_PAGE_EXTRACT_SCHEMA}\n\n"
                "Rules:\n"
                f"- Write ALL text fields {_lang_instr}; do not translate.\n"
                "- visual_description: describe any diagrams, charts, or figures\n"
                "- summary: 2-3 sentences on the meaning/purpose of this page\n"
                "- keywords: 3-6 thematic terms\n"
                "- text_content: transcribe ALL visible text verbatim up to 600 characters; "
                "do not summarize or paraphrase — copy the actual words from the page\n"
                "- entities: extract typed named entities (persons, organizations, dates, identifiers);\n"
                "  do NOT list numeric amounts or percentages — use empty lists for absent categories\n"
                f"{_AQ_RULES}"
                "Output ONLY the JSON object, no markdown fences, no other text."
            )
            token_budget = _SINGLE_PAGE_TOKENS * 2
        else:
            prompt_full = (
                "Analyze this document page carefully and output a JSON object with "
                "exactly these keys:\n"
                f"{_PAGE_EXTRACT_SCHEMA}\n\n"
                "Rules:\n"
                f"- Write ALL text fields {_lang_instr}; do not translate.\n"
                "- visual_description: if no visual elements (diagrams/charts/figures), "
                "write an empty string\n"
                "- summary: 2-3 sentences on the meaning/purpose of this page\n"
                "- keywords: 3-6 thematic terms (not generic words like 'page' or 'slide')\n"
                "- text_content: MAX 300 characters; if the page contains structured data "
                "(tables, charts, lists), capture ≥3 representative entries verbatim as "
                "'Label: Value' pairs; otherwise capture the most important text visible\n"
                "- entities: extract typed named entities (persons, organizations, dates, identifiers);\n"
                "  do NOT list numeric amounts or percentages — use empty lists for absent categories\n"
                f"{_AQ_RULES}"
                "Output ONLY the JSON object, no markdown fences, no other text."
            )
            token_budget = _SINGLE_PAGE_TOKENS

        # Simplified prompt used on the last retry attempt
        prompt_simple = (
            "Look at this document page and respond with a JSON object.\n"
            f"Write all text {_lang_instr} (do not translate).\n"
            '{"visual_description": "<describe any charts/diagrams, or empty string>", '
            '"summary": "<2-3 sentences about this page>", '
            '"keywords": ["<kw1>", "<kw2>", "<kw3>"], '
            '"text_content": "<key visible text or data entries, max 200 chars>", '
            '"entities": {"persons": [], "organizations": [], "dates": [], "identifiers": []}, '
            '"anticipated_queries": ["<visual question if chart/figure present>", '
            '"<keyword phrase 3-5 words>", "<plain-language question>", "<synonym/paraphrase question>"]}\n'
            "Output ONLY the JSON, nothing else."
        )

        image = _cap_image_size(image, max_pixels=_MAX_EXTRACT_PIXELS)

        max_retries = 2
        for attempt in range(max_retries + 1):
            prompt = prompt_simple if attempt == max_retries else prompt_full
            raw = self.vlm_fn([image], prompt, token_budget)
            data = _parse_json_response(raw, self.logger)
            if data and isinstance(data, dict):
                break
            self.logger.warning(
                f"[hindex] Page {page_idx} extraction attempt {attempt + 1}/{max_retries + 1} "
                f"failed — raw response: {raw[:300]!r}"
            )
        else:
            data = {}

        _entities = data.get("entities", {})
        record = PageRecord(
            page_idx=page_idx,
            page_key=page_key,
            text_content=str(data.get("text_content", "")),
            visual_description=str(data.get("visual_description", "")),
            summary=str(data.get("summary", "")),
            keywords=list(data.get("keywords", [])),
            anticipated_queries=list(data.get("anticipated_queries", [])),
            entities=_entities if isinstance(_entities, dict) else {},
        )
        record.page_type = _detect_page_type(record)
        return record

    def _extract_batch(
        self,
        images: list[Image.Image],
        idxs: list[int],
        keys: list[str],
        doc_language: str | None = None,
    ) -> list[PageRecord]:
        """Extract PageRecords for N pages in a single VLM call.

        Sends all N page images together with a prompt asking for a JSON array
        of N objects.  For any page whose slot parses as None, falls back to
        single-page extraction so no page is silently dropped.
        """
        n = len(images)
        capped = [_cap_image_size(img, max_pixels=_MAX_EXTRACT_PIXELS) for img in images]
        prompt = _batch_prompt(n, doc_language=doc_language)
        # Token budget: each page needs ~350 output tokens; cap at 2048.
        batch_tokens = min(2048, _BATCH_TOKENS_PER_PAGE * n)
        raw = self.vlm_fn(capped, prompt, batch_tokens)
        parsed = _parse_batch_response(raw, n, self.logger)

        records: list[PageRecord] = []
        for i, (data, idx, key) in enumerate(zip(parsed, idxs, keys)):
            if data and isinstance(data, dict):
                _entities = data.get("entities", {})
                _rec = PageRecord(
                    page_idx=idx,
                    page_key=key,
                    text_content=str(data.get("text_content", "")),
                    visual_description=str(data.get("visual_description", "")),
                    summary=str(data.get("summary", "")),
                    keywords=list(data.get("keywords", [])),
                    anticipated_queries=list(data.get("anticipated_queries", [])),
                    entities=_entities if isinstance(_entities, dict) else {},
                )
                _rec.page_type = _detect_page_type(_rec)
                records.append(_rec)
            else:
                self.logger.warning(
                    f"[hindex] Batch slot {i} (page {idx}) empty — "
                    "falling back to single-page extraction"
                )
                records.append(self._extract_page_record(images[i], idx, key, doc_language=doc_language))
        return records

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
        return self.vlm_fn([], prompt, _DOC_DESC_TOKENS).strip()

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
        raw = self.vlm_fn([], prompt, _TOPIC_CLUSTER_TOKENS)
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
    # Repair
    # ------------------------------------------------------------------

    def repair_empty_pages(
        self,
        hierarchy: "DocumentHierarchy",
        kvstore,
    ) -> tuple["DocumentHierarchy", list[int]]:
        """Re-extract PageRecords whose summary and text_content are both empty.

        Loads the page image for each failed page from *kvstore* using the
        stored ``page_key`` (falling back to the first crop if needed), then
        runs ``_extract_page_record`` with the new retry logic.

        Parameters
        ----------
        hierarchy:
            Existing DocumentHierarchy with (possibly) empty PageRecords.
        kvstore:
            An HDF5ImageStorage (or any object with ``has_key`` and
            ``retrieve_image`` methods) already opened for reading.

        Returns
        -------
        (updated_hierarchy, repaired_indices)
            The updated hierarchy and a list of page indices that were
            successfully re-extracted.
        """
        empty = [
            r for r in hierarchy.page_records
            if not r.summary and not r.text_content
        ]
        if not empty:
            self.logger.info("[hindex] repair_empty_pages: no empty records found")
            return hierarchy, []

        self.logger.info(
            f"[hindex] repair_empty_pages: re-extracting {len(empty)} pages: "
            f"{[r.page_idx for r in empty]}"
        )

        repaired: list[int] = []
        for rec in empty:
            key = rec.page_key
            resolved = key if kvstore.has_key(key) else f"{key}_crop_0000"
            try:
                img = kvstore.retrieve_image(resolved)
            except Exception as exc:
                self.logger.warning(
                    f"[hindex] repair: cannot load image for page {rec.page_idx} "
                    f"(key={resolved!r}): {exc}"
                )
                continue

            new_rec = self._extract_page_record(img, rec.page_idx, rec.page_key)
            if new_rec.summary or new_rec.text_content:
                new_rec.crop_keys = rec.crop_keys  # preserve existing crop keys
                # Replace in-place
                for i, r in enumerate(hierarchy.page_records):
                    if r.page_idx == rec.page_idx:
                        hierarchy.page_records[i] = new_rec
                        break
                repaired.append(rec.page_idx)
                self.logger.info(
                    f"[hindex] repair: page {rec.page_idx} OK — "
                    f"{len(new_rec.keywords)} keywords, "
                    f"{len(new_rec.summary)} char summary"
                )
            else:
                self.logger.warning(
                    f"[hindex] repair: page {rec.page_idx} still empty after retries"
                )

        return hierarchy, repaired

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


# Maximum pixel count for images passed to the per-page extraction VLM.
# Very large pages (e.g. 4001×2250 = ~9 MP) generate so many visual tokens
# that Qwen3-VL may truncate the response before the JSON is complete.
# 1280×720 = ~0.9 MP is plenty of detail for OCR + visual description.
_MAX_EXTRACT_PIXELS = 1280 * 720

# Per-call token budgets — sized to the actual output, not a single ceiling.
_SINGLE_PAGE_TOKENS = 1024   # one page JSON: ~500 tokens with entities + 4-8 anticipated queries
_BATCH_TOKENS_PER_PAGE = 640  # per-page budget when batching (tighter than single)
_DOC_DESC_TOKENS = 128        # one sentence
_TOPIC_CLUSTER_TOKENS = 1024  # 3-7 topics × ~120 tokens each


def _build_aq_rules(lang_instr: str, batch_mode: bool = False) -> str:
    """Build the anticipated_queries rule block for VLM extraction prompts.

    Centralised here so any change applies to both the single-page path
    (_extract_page_record) and the batch path (_batch_prompt) consistently.

    Parameters
    ----------
    lang_instr:
        Language instruction, e.g. "in English" or
        "in the same language as the document content".
    batch_mode:
        True when the VLM sees multiple pages simultaneously — the cross-page
        bullet is more permissive since the model can verify adjacency directly.
    """
    _target = (
        "per page that a user would type to find it"
        if batch_mode else
        "that a user would type to find this page"
    )
    _cross_page_rule = (
        "    • if a page is a continuation of a prior page, or its content clearly spans\n"
        "      two adjacent pages (split table, multi-page argument, figure on one page and\n"
        "      caption on next): add one CROSS-PAGE question whose answer requires reading\n"
        "      both pages together; assign it to the earlier of the two pages;\n"
        "      skip this bullet if the page is fully self-contained\n"
    ) if batch_mode else (
        "    • if this page is a continuation of a prior page, ends mid-sentence or mid-table,\n"
        "      or explicitly references adjacent content ('see previous page', 'continued',\n"
        "      split table, figure caption on a different page): add one CROSS-PAGE question\n"
        "      whose answer requires reading this page together with the adjacent page;\n"
        "      skip this bullet if the page is fully self-contained\n"
    )
    return (
        f"- anticipated_queries: 4-8 distinct questions {lang_instr} {_target};\n"
        "  IMPORTANT — include a DIVERSE MIX of question types; do NOT make all questions entity-anchored:\n"
        "    • if this page has charts/figures/visual data: at least one VISUAL question\n"
        "      (e.g. 'what does the bar chart show', 'which category is largest', 'what trend is visible')\n"
        "    • if this page has tables or specific numeric values: at least one FACT-LOOKUP question\n"
        "      using ACTUAL labels and numbers visible on this page — NOT placeholder names;\n"
        "      e.g. if the table shows 'BetaSearch recall@10: 0.82', ask 'What is BetaSearch's recall@10?';\n"
        "      anchor on real row names, column headers, and numeric results from this page\n"
        f"{_cross_page_rule}"
        "    • always include: one KEYWORD PHRASE (3-5 words), one PLAIN-LANGUAGE question\n"
        "    • if this page is mostly text/paragraphs/arguments (no dominant chart or table):\n"
        "      add one CONCEPTUAL question approaching the topic from a different angle WITHOUT\n"
        "      entity names — reframe the concept, not just the wording\n"
        "      (e.g. 'What does the paper argue about retrieval augmentation?' not 'What is ColPali?')\n"
        "    • if this page has visual or tabular content (chart, figure, table): add one\n"
        "      PARAPHRASE of the main finding using different vocabulary\n"
        "      (e.g. 'How does X change over time?' instead of 'What trend is visible in X?')\n"
        "    • entity-anchored questions (using a specific name/identifier) are appropriate ONLY\n"
        "      when the page actually contains named persons, organizations, or product names\n"
        "  Each query must be self-contained and meaningfully different from the others\n"
    )


def _batch_prompt(n: int, doc_language: str | None = None) -> str:
    """Prompt asking the VLM to extract structured info from N pages at once."""
    _lang_instr = f"in {doc_language}" if doc_language else "in the same language as the document content"
    _aq_rules = _build_aq_rules(_lang_instr, batch_mode=True)
    return (
        f"Analyze these {n} document pages (images are in order, one page each).\n"
        f"Output a JSON array of exactly {n} objects, one per page, in the same order.\n"
        "Each object must contain exactly these keys:\n"
        f"{_PAGE_EXTRACT_SCHEMA}\n\n"
        "Rules (apply independently to each page):\n"
        f"- Write ALL text fields {_lang_instr}; do not translate.\n"
        "- visual_description: if no visual elements (diagrams/charts/figures), "
        "write an empty string\n"
        "- summary: 2-3 sentences on the meaning/purpose of that page\n"
        "- keywords: 3-6 thematic terms per page\n"
        "- text_content: MAX 300 characters; if the page contains structured data "
        "(tables, charts, lists), capture ≥3 representative entries verbatim as "
        "'Label: Value' pairs; otherwise capture the most important text visible\n"
        "- entities: extract typed named entities (persons, organizations, dates, identifiers);\n"
        "  do NOT list numeric amounts or percentages — use empty lists for absent categories\n"
        f"{_aq_rules}"
        f"Output ONLY the JSON array of exactly {n} objects. "
        "No markdown fences, no extra text."
    )


def _detect_page_type(record: "PageRecord") -> str:
    """Classify page content type from VLM-extracted fields.

    No additional VLM call — uses visual_description richness and text_content
    structure. Stored in PageRecord.page_type and persisted in hindex JSON so
    the retriever can apply page-type-aware signal weighting at query time.

    Returns one of: 'visual-dominant', 'table-dominant', 'text-dominant', 'mixed'.
    """
    vd = (getattr(record, "visual_description", "") or "").lower()
    tc = (getattr(record, "text_content", "") or "").lower()

    # Visual: VLM produced a non-trivial description mentioning chart/figure elements
    _VISUAL_MARKERS = {
        "chart", "figure", "graph", "diagram", "plot", "bar chart", "pie chart",
        "scatter", "map", "heatmap", "histogram", "illustration", "donut",
        "treemap", "bubble", "waterfall", "gantt",
    }
    has_visual = len(vd) > 40 and any(m in vd for m in _VISUAL_MARKERS)

    # Table: multiple Label:Value patterns, pipe chars, or explicit table mentions
    has_table = (
        " | " in tc
        or "│" in tc
        or tc.count(":") >= 3
        or "table" in tc
        or "\t" in tc
    )

    if has_visual and has_table:
        return "mixed"
    if has_visual:
        return "visual-dominant"
    if has_table:
        return "table-dominant"
    return "text-dominant"


def _parse_batch_response(
    raw: str, expected_count: int, logger
) -> list[dict | None]:
    """Parse a JSON array of page-extraction dicts from a batch VLM call.

    Returns a list of length *expected_count*: each element is either a dict
    (successful parse for that page) or None (caller should fall back to
    single-page extraction for that slot).
    """
    if not raw:
        return [None] * expected_count

    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    def _to_results(lst: list) -> list[dict | None]:
        results: list[dict | None] = [
            d if isinstance(d, dict) else None for d in lst
        ]
        while len(results) < expected_count:
            results.append(None)
        return results[:expected_count]

    # Happy path: the whole output is a valid JSON array
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return _to_results(data)
    except json.JSONDecodeError:
        pass

    # Try extracting the outermost [...] block (handles leading/trailing prose)
    m = re.search(r"\[.*\]", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                results = _to_results(data)
                n_ok = sum(1 for r in results if r is not None)
                logger.warning(
                    f"[hindex] Batch parse: extracted {n_ok}/{expected_count} "
                    "objects from [...] block"
                )
                return results
        except json.JSONDecodeError:
            pass

    logger.warning(
        f"[hindex] Batch parse: could not extract JSON array "
        f"(response len={len(raw)}): {raw[:200]!r}"
    )
    return [None] * expected_count


def _cap_image_size(image: Image.Image, max_pixels: int = _MAX_EXTRACT_PIXELS) -> Image.Image:
    """Return *image* scaled down so its total pixel count is ≤ *max_pixels*."""
    w, h = image.size
    if w * h <= max_pixels:
        return image
    scale = (max_pixels / (w * h)) ** 0.5
    new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
    return image.resize(new_size, Image.LANCZOS)


def _parse_json_response(raw: str, logger) -> Any:
    """Extract JSON from a VLM response that may contain surrounding prose.

    Falls back to field-by-field salvage when the model's output is truncated
    mid-string (the most common failure mode: text_content is too long and
    max_new_tokens is exhausted before the JSON object closes).
    """
    if not raw:
        return None
    cleaned = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()

    # Happy path: valid JSON
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Try extracting the outermost {...} or [...] block.
    # When the string starts with '{', only match '{...}' — a truncated object may
    # have inner arrays (e.g. keywords) that would otherwise be matched and returned
    # as a list before the field-by-field salvage path has a chance to run.
    if cleaned.lstrip().startswith("{"):
        pattern = r"\{.*\}"
    else:
        pattern = r"(\{.*\}|\[.*\])"
    m = re.search(pattern, cleaned, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass

    # Salvage truncated JSON objects: pull out individual string fields and
    # the keywords array even when the closing brace was never generated.
    # Pattern: "key": "value"  — value may be cut off (no closing quote).
    if cleaned.lstrip().startswith("{"):
        salvaged: dict[str, Any] = {}
        for key in ("visual_description", "summary", "text_content"):
            m = re.search(
                rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)',
                cleaned,
                re.DOTALL,
            )
            if m:
                # Unescape the captured fragment (may be incomplete)
                salvaged[key] = (
                    m.group(1)
                    .replace('\\"', '"')
                    .replace("\\n", "\n")
                    .replace("\\t", "\t")
                )
        kw_m = re.search(r'"keywords"\s*:\s*(\[[^\]]*\]?)', cleaned, re.DOTALL)
        if kw_m:
            try:
                kw_raw = kw_m.group(1)
                if not kw_raw.endswith("]"):
                    kw_raw += "]"
                salvaged["keywords"] = json.loads(kw_raw)
            except json.JSONDecodeError:
                pass
        if salvaged:
            logger.warning(
                f"[hindex] JSON truncated — salvaged fields: {list(salvaged)}"
            )
            return salvaged

    logger.warning(f"[hindex] Could not parse JSON from VLM response: {raw[:200]}")
    return None


def _page_key(uuid5_val: str, page_idx: int) -> str:
    """Kvstore key for a full-page image."""
    return f"{uuid5_val}_{page_idx:04d}"


class _NullLogger:
    def info(self, *_a, **_k): pass
    def warning(self, *_a, **_k): pass
    def debug(self, *_a, **_k): pass


# ---------------------------------------------------------------------------
# Collection-level index (corpus scale)
# ---------------------------------------------------------------------------

# Minimum number of documents before collection-level clustering is worthwhile.
COLLECTION_THRESHOLD = 5

# Token budget for the single VLM call that clusters all documents.
# Short-name output (~10 tokens/doc) × 135 docs × 1 cluster assignment each ≈ 1350 tokens,
# plus JSON structure and cluster metadata ≈ 3000 tokens total.  Use 4096 to be safe.
_COLLECTION_BUILD_TOKENS = 4096


@dataclass
class DocumentCluster:
    """A thematic cluster of documents within the collection index."""

    cluster_id: str        # short ID, e.g. "C01"
    label: str             # human-readable category name
    description: str       # one-sentence description of what documents cover
    doc_names: list[str]   # document names belonging to this cluster

    def to_dict(self) -> dict[str, Any]:
        return {
            "cluster_id": self.cluster_id,
            "label": self.label,
            "description": self.description,
            "doc_names": self.doc_names,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DocumentCluster":
        return cls(
            cluster_id=d["cluster_id"],
            label=d["label"],
            description=d.get("description", ""),
            doc_names=d.get("doc_names", []),
        )


@dataclass
class CollectionIndex:
    """Top-level index grouping all document hierarchies into thematic clusters.

    Enables two-stage document selection at query time:
      Stage 1 — LLM picks 1–2 clusters from a short list (≤ 12 choices).
      Stage 2 — LLM picks 1–2 documents from within the selected clusters.

    This replaces a single flat selection over all N documents, which degrades
    when N > ~20 because the LLM prompt becomes too long and the choice set too
    large for reliable navigation.
    """

    clusters: list[DocumentCluster]

    def to_dict(self) -> dict[str, Any]:
        return {"clusters": [c.to_dict() for c in self.clusters]}

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "CollectionIndex":
        return cls(clusters=[DocumentCluster.from_dict(c) for c in d.get("clusters", [])])


def build_collection_index(
    hierarchies: list[DocumentHierarchy],
    vlm_fn,
    logger,
    max_clusters: int = 12,
) -> "CollectionIndex | None":
    """Group *hierarchies* into thematic clusters via one VLM call.

    Returns None when there are fewer than COLLECTION_THRESHOLD documents
    (flat single-stage selection is fine at that scale).
    """
    if len(hierarchies) < COLLECTION_THRESHOLD:
        return None

    # Use short names (basename without extension) in the prompt to keep
    # output tokens low and avoid copy errors on long absolute paths.
    from pathlib import Path as _Path
    short_names = [_Path(h.doc_name).stem for h in hierarchies]
    # Build reverse mapping short_name → full doc_name (handle collisions)
    short_to_full: dict[str, str] = {}
    for short, h in zip(short_names, hierarchies):
        short_to_full.setdefault(short, h.doc_name)

    doc_lines = "\n".join(
        f'- "{sn}": {h.doc_description[:120]}'
        for sn, h in zip(short_names, hierarchies)
    )
    prompt = (
        f"You have {len(hierarchies)} documents listed below. "
        f"Group them into at most {max_clusters} thematic clusters for retrieval.\n\n"
        "Output a JSON array. Each element must have exactly these keys:\n"
        '  "id"   : cluster ID string like "C01", "C02", ...\n'
        '  "label": short category name (3–5 words)\n'
        '  "desc" : one sentence describing what documents in this cluster cover\n'
        '  "docs" : JSON array of document name strings belonging to this cluster\n\n'
        "Rules:\n"
        "- Every document must appear in exactly one cluster.\n"
        "- Use the document names exactly as given (copy them verbatim).\n"
        "- Output only the JSON array, no explanation or markdown.\n\n"
        f"Documents:\n{doc_lines}"
    )

    raw = vlm_fn([], prompt, _COLLECTION_BUILD_TOKENS)
    clusters = _parse_cluster_output(raw, hierarchies, short_to_full, logger)
    if clusters is None:
        logger.warning(
            "[hindex] Collection index: VLM output could not be parsed; "
            "falling back to alphabetical grouping"
        )
        clusters = _alphabetical_clusters(hierarchies, max_clusters)

    index = CollectionIndex(clusters=clusters)
    logger.info(
        "[hindex] Collection index built: %d clusters for %d documents",
        len(clusters), len(hierarchies),
    )
    return index


# ---------------------------------------------------------------------------
# Internal helpers for collection index building
# ---------------------------------------------------------------------------


def _parse_cluster_output(
    raw: str,
    hierarchies: list[DocumentHierarchy],
    short_to_full: "dict[str, str] | None",
    logger,
) -> "list[DocumentCluster] | None":
    raw = re.sub(r"```(?:json)?", "", raw).strip().strip("`").strip()
    data = None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        m = re.search(r"\[.*\]", raw, re.DOTALL)
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                pass

    if not isinstance(data, list):
        return None

    all_names = {h.doc_name for h in hierarchies}
    # Also index by basename (with and without extension) for flexible matching.
    from pathlib import Path as _Path
    basename_to_full: dict[str, str] = {}
    for h in hierarchies:
        p = _Path(h.doc_name)
        basename_to_full[p.name] = h.doc_name        # e.g. "foo.pdf"
        basename_to_full[p.stem] = h.doc_name        # e.g. "foo"
    if short_to_full:
        basename_to_full.update(short_to_full)

    def _resolve(name: str) -> "str | None":
        s = str(name)
        if s in all_names:
            return s
        return basename_to_full.get(s)

    clusters: list[DocumentCluster] = []
    assigned: set[str] = set()

    for item in data:
        if not isinstance(item, dict):
            continue
        raw_docs = item.get("docs", [])
        doc_names = [full for d in raw_docs if (full := _resolve(d)) is not None]
        if not doc_names:
            continue
        clusters.append(DocumentCluster(
            cluster_id=str(item.get("id", f"C{len(clusters) + 1:02d}")),
            label=str(item.get("label", f"Group {len(clusters) + 1}")),
            description=str(item.get("desc", "")),
            doc_names=doc_names,
        ))
        assigned.update(doc_names)

    # Catch-all for any documents the VLM omitted
    unassigned = [h.doc_name for h in hierarchies if h.doc_name not in assigned]
    if unassigned:
        clusters.append(DocumentCluster(
            cluster_id=f"C{len(clusters) + 1:02d}",
            label="Other Documents",
            description="Documents not assigned to other clusters.",
            doc_names=unassigned,
        ))

    return clusters if clusters else None


def _alphabetical_clusters(
    hierarchies: list[DocumentHierarchy],
    max_clusters: int,
) -> list[DocumentCluster]:
    """Fallback: split alphabetically-sorted documents into equal-sized groups."""
    sorted_h = sorted(hierarchies, key=lambda h: h.doc_name.lower())
    n = len(sorted_h)
    size = max(1, (n + max_clusters - 1) // max_clusters)
    clusters: list[DocumentCluster] = []
    for i in range(0, n, size):
        chunk = sorted_h[i : i + size]
        cid = f"C{len(clusters) + 1:02d}"
        sample = ", ".join(h.doc_name[:25] for h in chunk[:2])
        clusters.append(DocumentCluster(
            cluster_id=cid,
            label=f"Documents {i + 1}–{min(i + size, n)}",
            description=f"e.g. {sample}{'…' if len(chunk) > 2 else ''}",
            doc_names=[h.doc_name for h in chunk],
        ))
    return clusters
