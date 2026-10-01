import base64
import hashlib
import json
import logging
import re
from io import BytesIO
from pathlib import Path
from time import time
from typing import Any, cast

import chromadb
import fitz
import torch

# chromadb's Rust backend uses a process-level singleton. Creating a second
# PersistentClient after the first has been GC'd fails with:
#   AttributeError: 'RustBindingsAPI' object has no attribute 'bindings'
# Keeping one client per path alive for the duration of the process avoids
# re-initialization and SQLite re-locking issues between sequential services.
_chroma_client_cache: dict = {}

# Pages with fewer than this many characters of PyMuPDF-extracted text are
# treated as scanned (image-only) and given an OCR-focused VLM prompt.
_SCANNED_TEXT_THRESHOLD = 50

from chromadb.api.types import (
    Embedding,
    EmbeddingFunction,
    Embeddings,
)
from chromadb.config import Settings
from PIL import Image
from qwen_vl_utils import process_vision_info
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel, Qwen2VLForConditionalGeneration
from colette.backends.hf.qwen3_vl_embedding import Qwen3VLEmbedder
from vllm import LLM as VLLM

from uuid import NAMESPACE_URL, uuid5 as _uuid5

from colette.apidata import InputConnectorObj, RAGObj, merge_rag_config
from colette.backends.coldb import ColDB
from colette.backends.tantivy import (
    TantivyIndex,
    build_text_context,
    retrieval_mode_uses_text_search_engine,
    retrieval_mode_uses_embedding_retrieval,
    retrieval_mode_uses_vision_semantic,
    retrieval_mode_uses_vision_text_embed,
    retrieval_mode_uses_vision_hybrid_bm25,
    retrieval_mode_uses_vision_topic_embed,
    retrieval_mode_needs_vlm_descriptions,
)
from colette.backends.hf.hindex import (
    HierarchyBuilder,
    HierarchyStore,
    HierarchyRetriever,
    CollectionIndex,
    build_collection_index,
    DocumentHierarchy,
    retrieval_mode_uses_hindex,
    _page_key,
)
from colette.inputconnector import InputConnectorBadParamException

from ..layout_detector import LayoutDetector
from ..attention import resolve_attn_implementation
from ..model_cache import ModelCache
from ..preprocessing import DocumentProcessor, ImageProcessor


# XXX: chromadb image handling is broken/embryonary
# we hack their image validation in order to pass arbitrary
# data to our custom embedding function
def is_image(target: Any) -> bool:
    return True


def _merge_hindex_results(docs: dict, hindex_docs: dict, top_k: int | None = 4) -> dict:
    """Merge hindex results into docs with balanced slot allocation.

    Each source (hindex, embedding) receives ceil(top_k / 2) slots so both
    contribute to the context the LLM sees.  hindex distances (0.05–0.20) are
    intentionally lower than embedding cosine distances (0.4–0.8); without
    capping, hindex would fill every slot and embedding would never appear.

    Within each source the best-ranked results are kept; the two halves are
    then sorted by distance so the LLM receives them in relevance order.
    """
    if not hindex_docs.get("ids", [[]])[0]:
        return docs

    import math
    # Slot budget: each source gets half; hindex rounds up when top_k is odd.
    if top_k is not None and top_k > 0:
        hindex_slots = math.ceil(top_k / 2)
        emb_slots = top_k - hindex_slots
    else:
        # No budget given: fall back to all results from each source
        hindex_slots = len(hindex_docs["ids"][0])
        emb_slots = len(docs["ids"][0])

    # Take the best results from each source (already sorted by distance)
    h_ids   = hindex_docs["ids"][0][:hindex_slots]
    h_dists = hindex_docs["distances"][0][:hindex_slots]
    h_metas = hindex_docs["metadatas"][0][:hindex_slots]

    e_ids   = docs["ids"][0][:emb_slots]
    e_dists = docs["distances"][0][:emb_slots]
    e_metas = docs["metadatas"][0][:emb_slots]

    combined = sorted(
        zip(h_ids + e_ids, h_dists + e_dists, h_metas + e_metas),
        key=lambda t: t[1],
    )
    if combined:
        ids, distances, metadatas = zip(*combined)
        docs["ids"][0] = list(ids)
        docs["distances"][0] = list(distances)
        docs["metadatas"][0] = list(metadatas)

    docs["_disable_thinking"] = True
    return docs


def _hierarchy_to_vlm_tantivy_docs(hierarchy: "DocumentHierarchy") -> list[dict]:
    """Build Tantivy document dicts from a DocumentHierarchy's VLM page descriptions.

    Each page produces one document with crop_label='pagedesc' so vision_semantic
    search can filter to these records without touching raw-PDF-text documents.
    """
    result = []
    source = hierarchy.doc_name
    for record in hierarchy.page_records:
        parts = [p for p in [
            record.summary,
            record.text_content,
            record.visual_description,
            " ".join(record.keywords) if record.keywords else "",
        ] if p and p.strip()]
        if not parts:
            continue
        result.append({
            "doc_id": f"{source}:page:{record.page_idx + 1}",
            "source": source,
            "page_number": record.page_idx + 1,
            "crop_label": "pagedesc",
            "label": None,
            "content": " | ".join(parts),
        })
    return result


def _hierarchy_to_doc_tantivy_doc(hierarchy: "DocumentHierarchy") -> dict | None:
    """Build a single doc-level Tantivy entry for cross-document BM25 retrieval.

    Uses crop_label='docdesc' and page_number=0 as a sentinel so retrieve()
    can distinguish doc-level hits from page-level hits and expand them to
    the document's constituent pages.
    """
    if not hierarchy.doc_description or not hierarchy.doc_description.strip():
        return None
    return {
        "doc_id": f"{hierarchy.doc_name}:doc",
        "source": hierarchy.doc_name,
        "page_number": 0,
        "crop_label": "docdesc",
        "label": None,
        "content": hierarchy.doc_description,
    }


def _hierarchy_to_queryhint_tantivy_docs(hierarchy: "DocumentHierarchy") -> list[dict]:
    """Build one Tantivy doc per anticipated query, tagged crop_label='queryhint'.

    Each doc carries the same source/page_number as its page so that
    _bm25_hits_to_page_list naturally resolves a queryhint hit back to the
    originating page without any special-casing in the retrieve path.
    """
    result = []
    source = hierarchy.doc_name
    for record in hierarchy.page_records:
        for i, query in enumerate(record.anticipated_queries or []):
            query = query.strip()
            if not query:
                continue
            result.append({
                "doc_id": f"{source}:page:{record.page_idx + 1}:queryhint:{i}",
                "source": source,
                "page_number": record.page_idx + 1,
                "crop_label": "queryhint",
                "label": None,
                "content": query,
            })
    return result


def _hierarchy_to_entity_tantivy_docs(hierarchy: "DocumentHierarchy") -> list[dict]:
    """Build one Tantivy doc per page containing extracted named entity values.

    Tagged crop_label='entity' for targeted BM25 search at query time.
    Only indexes high-specificity entity categories: persons, organizations,
    dates, and named identifiers. Amounts (numeric values, percentages) are
    excluded because in homogeneous corpora they coincidentally appear across
    many documents (e.g. '31.0%' in multiple VIRA reports) and have near-zero
    discriminating IDF — their inclusion causes false-positive page matches.
    Pages with no qualifying entities produce no doc (graceful degradation for
    hindex files built before entity extraction was added).
    """
    # Categories eligible for entity BM25 — named, specific, sparse.
    # 'amounts' excluded: numeric values / percentages are too common across docs.
    _ENTITY_CATEGORIES = {"persons", "organizations", "dates", "identifiers"}
    result = []
    source = hierarchy.doc_name
    for record in hierarchy.page_records:
        entities = getattr(record, "entities", {}) or {}
        parts = []
        for category, values in entities.items():
            if category not in _ENTITY_CATEGORIES:
                continue
            if isinstance(values, list):
                parts.extend(str(v).strip() for v in values if v and str(v).strip())
            elif values:
                parts.append(str(values).strip())
        if not parts:
            continue
        result.append({
            "doc_id": f"{source}:page:{record.page_idx + 1}:entity",
            "source": source,
            "page_number": record.page_idx + 1,
            "crop_label": "entity",
            "label": None,
            "content": " | ".join(parts),
        })
    return result


def _vlm_hits_to_page_docs(hits: list[dict], top_k: int) -> dict:
    """Convert Tantivy VLM-description hits to the standard docs format.

    Derives the kvstore page key from source path and page_number so the
    downstream VLM answering step can retrieve the actual page image.
    """
    from uuid import uuid5 as _u5, NAMESPACE_URL as _NS
    ids, distances, metadatas = [], [], []
    seen: set[tuple] = set()
    for hit in hits:
        source = hit.get("source", "")
        page_number = hit.get("page_number")
        if not source or page_number is None:
            continue
        key = (source, page_number)
        if key in seen:
            continue
        seen.add(key)
        page_idx = int(page_number) - 1
        uuid5_val = str(_u5(_NS, source))
        page_key = _page_key(uuid5_val, page_idx)
        score = max(hit.get("score", 0.0), 0.0)
        ids.append(page_key)
        distances.append(1.0 / (1.0 + score))  # invert BM25: high score → low distance
        metadatas.append({"source": source, "page_number": page_number, "crop_label": "pagedesc"})
        if len(ids) >= top_k:
            break
    return {"ids": [ids], "distances": [distances], "metadatas": [metadatas]}


def _page_hits_to_docs(
    hits: list[tuple[str, int, float]],
    hindex_store,
    kvstore,
    top_k: int,
) -> dict:
    """Convert (source, page_number, distance) tuples to docs dict via hindex crop_keys.

    Resolves each page to actual kvstore keys using PageRecord.crop_keys, so the
    downstream VLM receives real page images. Pages whose crop_keys are absent from
    the kvstore (e.g. old indexes) are silently skipped — text_context still covers them.
    """
    ids, distances, metadatas = [], [], []
    _loaded: dict = {}
    seen: set = set()

    for source, page_number, dist in hits:
        key = (source, page_number)
        if key in seen:
            continue
        seen.add(key)

        if source not in _loaded:
            _loaded[source] = hindex_store.load(source)
        hierarchy = _loaded[source]
        if hierarchy is None:
            continue

        page_idx = page_number - 1
        if not (0 <= page_idx < len(hierarchy.page_records)):
            continue

        record = hierarchy.page_records[page_idx]
        base = {"source": source, "page_number": page_number, "retrieval_source": "hindex"}

        valid_crops = [k for k in (record.crop_keys or [])[:2] if kvstore.has_key(k)]
        if valid_crops:
            for ck in valid_crops:
                ids.append(ck)
                distances.append(dist)
                metadatas.append({**base, "crop_label": "crop"})
        elif kvstore.has_key(record.page_key):
            ids.append(record.page_key)
            distances.append(dist)
            metadatas.append({**base, "crop_label": "full_page"})
        # No valid kvstore key → skip image; text_context carries the content

    return {"ids": [ids[:top_k * 2]], "distances": [distances[:top_k * 2]], "metadatas": [metadatas[:top_k * 2]]}


def _hindex_page_text(record) -> str:
    """Build an indexable text string from a PageRecord for dense text embedding.

    Includes anticipated_queries so that the text embedding signal captures
    semantic similarity to anticipated queries — not just the page description.
    This closes the vocabulary gap that BM25 queryhint misses: a user asking
    "which method is fastest?" can match "lowest average latency" via embedding
    even when BM25 keyword overlap is zero.
    """
    parts = [p for p in [
        record.summary,
        " ".join(record.keywords) if record.keywords else "",
        record.text_content[:800] if record.text_content else "",
        record.visual_description,
        " ".join(record.anticipated_queries) if getattr(record, "anticipated_queries", None) else "",
    ] if p and p.strip()]
    return " | ".join(parts)


def _hindex_topic_text(topic) -> str:
    """Build an indexable text string from a TopicNode for topic-level retrieval."""
    parts = [p for p in [
        topic.theme,
        topic.description,
        " ".join(topic.keywords) if topic.keywords else "",
    ] if p and p.strip()]
    return " | ".join(parts)


def _bm25_hits_to_page_list(hits: list[dict]) -> list[tuple[str, int, float]]:
    """Convert Tantivy hit dicts to (source, page_number, distance) tuples, deduped."""
    seen: set = set()
    result = []
    for hit in hits:
        source = hit.get("source", "")
        page_number = hit.get("page_number")
        if not source or page_number is None:
            continue
        key = (source, int(page_number))
        if key in seen:
            continue
        seen.add(key)
        score = max(hit.get("score", 0.0), 0.0)
        result.append((source, int(page_number), 1.0 / (1.0 + score)))
    return result


def _normalize_query_for_bm25(query: str) -> str:
    """Normalize a user query before BM25 search.

    Strips LaTeX math notation so that equation-heavy queries
    (e.g. arXiv figures) produce valid BM25 tokens instead of
    punctuation fragments that score zero.
    """
    # Unwrap inline math: \( ... \) → content
    q = re.sub(r'\\\(([^)]*?)\\\)', r'\1', query)
    # Remove display math blocks: \[ ... \]
    q = re.sub(r'\\\[.*?\\\]', ' ', q, flags=re.DOTALL)
    # Remove \begin{...} ... \end{...} blocks
    q = re.sub(r'\\begin\{[^}]+\}.*?\\end\{[^}]+\}', ' ', q, flags=re.DOTALL)
    # Expand remaining backslash commands: \delta → delta, \alpha → alpha, etc.
    q = re.sub(r'\\([a-zA-Z]+)', r'\1', q)
    return re.sub(r'\s+', ' ', q).strip() or query


def _reciprocal_rank_fusion(
    ranked_lists: list[list[tuple[str, int]]],
    k: int = 30,
    weights: list[float] | None = None,
) -> list[tuple[str, int, float]]:
    """Merge multiple (source, page_number) ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists:
        Each list is a ranking of (source, page_number) pairs, best-first.
    k:
        RRF smoothing constant (default 30).
    weights:
        Optional per-list multipliers. When provided, each list's contribution
        is scaled by its weight before summing. Used for intent-aware signal
        weighting (e.g. visual_embed gets 2× for visual queries).
        Must be the same length as ranked_lists; ignored if mismatched.
    """
    scores: dict[tuple[str, int], float] = {}
    _w = weights if weights and len(weights) == len(ranked_lists) else None
    for i, lst in enumerate(ranked_lists):
        w = _w[i] if _w else 1.0
        for rank, item in enumerate(lst):
            scores[item] = scores.get(item, 0.0) + w / (k + rank + 1)
    return [
        (s, p, score)
        for (s, p), score in sorted(scores.items(), key=lambda x: -x[1])
    ]


def _classify_query_intent(question: str) -> str:
    """Classify query as 'visual', 'entity', or 'semantic'.

    Used to gate entity BM25 (entity intent only) and to weight visual
    embedding higher for visual queries. Rule-based — fast, no model needed.

    visual   — asks about what a chart/figure shows, visual patterns, rankings,
               trends, proportions, spatial relationships
    entity   — asks for a specific value about a named entity (proper noun +
               value-lookup pattern); entity BM25 is activated
    semantic — conceptual, comparative, or interpretive question without a
               strong visual or entity signal; text embedding takes priority

    Priority order: visual > entity > semantic.
    """
    import re

    q_lower = question.lower()

    # ── Visual/analytical signals ──────────────────────────────────────────
    # Words describing visual elements or asking about patterns in data.
    # "figure" is intentionally absent — "sales figures" / "figures show" creates
    # false positives; "Figure N" references are caught by the named_ids regex below.
    _VISUAL_WORDS = {
        "chart", "graph", "diagram", "scatter plot", "bar chart", "pie chart",
        "line chart", "donut chart", "heatmap", "histogram", "treemap",
        "shows", "shown", "visual", "trend", "highlighted", "legend",
        "axis", "x-axis", "y-axis", "color", "colour", "darkest", "lightest",
        "tallest", "shortest", "stacked", "grouped", "on the left", "on the right",
    }
    _ANALYTICAL_PATTERNS = [
        # "which X is the most/least/highest/lowest..."
        r"\bwhich\b.{0,60}\b(?:most|least|highest|lowest|best|worst|largest|smallest|top|bottom)\b",
        # "which X shows/demonstrates/reveals..."
        r"\bwhich\b.{0,60}\b(?:shows?|demonstrates?|reveals?|indicates?|suggests?|appears?)\b",
        # "what is the most/least..."
        r"\bwhat (?:is|are) (?:the )?(?:most|least|best|worst|largest|smallest|highest|lowest)\b",
        # "how many/much..."
        r"\bhow (?:many|much)\b",
        # "do/does all/any/most/some..."
        r"\bdo(?:es)? (?:all|any|most|some)\b",
        # "what share/proportion/percentage..."
        r"\bwhat (?:share|proportion|percentage|fraction)\b",
        # trend/comparison words
        r"\b(?:compared? to|relative to|versus| vs )\b",
        r"\b(?:trend|dip|spike|peak|recovery|recovering|growth|decline|fluctuat)\b",
        r"\b(?:uninterrupted|consistent|steady|gradual|sudden)\b",
        r"\b(?:across all|overall|in general|generally speaking)\b",
        # comparison magnitude phrases ("is X larger/higher/greater than Y")
        r"\b(?:larger|smaller|higher|lower|greater|less|more|fewer) than\b",
        # standalone proportion/percentage questions (no "what" prefix needed)
        r"\b(?:proportion|percentage|fraction|share) of\b",
        r"\b(?:majority|minority|most of|fewer than half)\b",
    ]

    visual_score = sum(1 for w in _VISUAL_WORDS if w in q_lower)
    analytical_score = sum(1 for p in _ANALYTICAL_PATTERNS if re.search(p, q_lower))

    # ── Entity lookup signals ──────────────────────────────────────────────
    # TitleCase proper nouns (skip first word — always capitalized at start)
    _first_space = question.find(" ")
    _tail = question[_first_space + 1:] if _first_space >= 0 else ""
    proper_nouns = re.findall(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+)*\b", _tail)
    # Named identifiers: Figure 1, Table 2, AlphaNet, VIRA-v2, BenchA
    named_ids = re.findall(
        r"\b(?:Fig(?:ure)?|Table|Section|Appendix)\s*\d+\b"
        r"|\b[A-Z][a-zA-Z]*(?:\d+[a-zA-Z\d]*|-v?\d+)\b",
        question,
    )
    # Value-lookup pattern: "what is/are/was/were the X of Y" with a named thing
    value_lookup = bool(
        re.search(r"\bwhat (?:is|are|was|were) (?:the )?(?:\w+ ){0,5}(?:of|for)\b", q_lower)
        and (proper_nouns or named_ids)
    )
    entity_score = len(proper_nouns) + len(named_ids) * 2 + (3 if value_lookup else 0)

    # ── Classification (visual takes priority — it's the hardest to recover) ──
    # Interpretive shortcut: "why..." and "what does X suggest/imply" are
    # always conceptual — text embedding serves them best regardless of entity
    # signals (proper nouns in "Why does BetaSearch lead..." are not entity lookups).
    if re.search(r"^why\b", q_lower) or re.search(
        r"^what does\b.{0,80}\b(?:suggest|imply|tell us|indicate|mean)\b", q_lower
    ):
        return "semantic"
    if visual_score >= 2 or analytical_score >= 2:
        return "visual"
    if entity_score >= 4 and value_lookup:
        return "entity"
    if visual_score >= 1 or analytical_score >= 1:
        return "visual"
    if entity_score >= 2:
        return "entity"
    return "semantic"


def _filter_queryhint_hits(
    qh_all: list[tuple[str, int, float]],
    score_ratio_threshold: float = 0.5,
    per_doc_cap: int = 4,
) -> list[tuple[str, int, float]]:
    """Filter queryhint BM25 hits using a score-ratio threshold and per-doc cap.

    Two complementary filters prevent same-doc clustering while preserving coverage:

    - Score-ratio threshold: keep a page only if its BM25 score >=
      ``score_ratio_threshold * best_page_score_for_that_doc``.  Adaptive
      per-query: when one page clearly dominates (e.g. 100% vs 30%) only that
      page gets a vote; when several pages match similarly (e.g. 80%, 75%, 65%)
      all pass.

    - Per-doc cap: hard upper limit on pages from any single document.
      Setting this equal to ``top_k`` means a doc can at most fill every
      retrieval slot — the threshold does the real filtering; the cap is a
      safety valve against pathological cases.

    Args:
        qh_all: ``(source, page_number, inv_dist)`` tuples sorted by BM25
            score descending (``inv_dist = 1 / (1 + bm25_score)``; lower
            value means better match).  The sort order within each document
            is preserved in the output.
        score_ratio_threshold: fraction of the document's best BM25 score
            that a page must reach to be included.  0.5 means 50%.
        per_doc_cap: maximum pages from any single source document.

    Returns:
        Filtered list; relative order within each document is preserved.
    """
    qh_by_doc: dict[str, list[tuple[str, int, float]]] = {}
    for s, p, d in qh_all:
        qh_by_doc.setdefault(s, []).append((s, p, d))

    result: list[tuple[str, int, float]] = []
    for doc_pages in qh_by_doc.values():
        best_inv = doc_pages[0][2]       # smallest inv_dist = best BM25 match
        best_score = 1.0 / best_inv - 1.0
        threshold = score_ratio_threshold * best_score
        count = 0
        for s, p, d in doc_pages:
            if count >= per_doc_cap:
                break
            if (1.0 / d - 1.0) >= threshold:
                result.append((s, p, d))
                count += 1
    return result


def _theme_tokens(theme: str) -> set[str]:
    """Extract content-bearing tokens (>3 chars) from a topic theme string."""
    return {w.lower() for w in re.findall(r'[A-Za-z0-9]+', theme or '') if len(w) > 3}


def _expand_with_topic_siblings(
    page_list: list[tuple[str, int, float]],
    top_k: int,
    hindex_store,
    logger,
) -> list[tuple[str, int, float]]:
    """Inject cross-document pages that share a topic with any top-k anchor page.

    For each retrieved page, looks up its TopicNodes and finds pages from other
    documents whose topics overlap by theme vocabulary.  Siblings are merged into
    the candidate pool via RRF — this is the library-science 'See Also' reference.
    """
    if not page_list:
        return page_list

    try:
        all_hierarchies = hindex_store.load_all()
    except Exception as exc:
        logger.warning("[vision_hybrid_bm25] topic sibling load failed: %s", exc)
        return page_list

    if len(all_hierarchies) < 2:
        return page_list  # single-document corpus; nothing to cross-reference

    h_by_doc: dict[str, object] = {h.doc_name: h for h in all_hierarchies}
    sibling_seen: set[tuple[str, int]] = {(_s, _p) for _s, _p, _ in page_list}
    sibling_pages: list[tuple[str, int]] = []

    for _src, _pn, _ in page_list[:top_k]:
        h = h_by_doc.get(_src)
        if h is None or not getattr(h, 'topics', None):
            continue
        page_idx = _pn - 1  # topics store 0-indexed page indices

        anchor_themes: list[set[str]] = []
        for topic in h.topics:
            if page_idx in (topic.page_indices or []):
                toks = _theme_tokens(topic.theme)
                if toks:
                    anchor_themes.append(toks)
        if not anchor_themes:
            continue

        for other_h in all_hierarchies:
            if other_h.doc_name == _src:
                continue
            for topic in (getattr(other_h, 'topics', None) or []):
                other_toks = _theme_tokens(topic.theme)
                if not other_toks:
                    continue
                for a_toks in anchor_themes:
                    if len(a_toks & other_toks) >= 1:
                        for idx in (topic.page_indices or []):
                            _key = (other_h.doc_name, idx + 1)
                            if _key not in sibling_seen:
                                sibling_seen.add(_key)
                                sibling_pages.append(_key)
                        break  # this topic matched; move to next other topic

    if not sibling_pages:
        return page_list

    original_ranked = [(_s, _p) for _s, _p, _ in page_list]
    expanded = _reciprocal_rank_fusion([original_ranked, sibling_pages])
    logger.debug(
        "[vision_hybrid_bm25] topic siblings: +%d cross-doc pages injected",
        len(sibling_pages),
    )
    return expanded


def _hindex_text_context_for_pages(
    pages: list[tuple[str, int]],
    hindex_store,
    max_chars_per_doc: int,
    max_total_chars: int,
) -> tuple[list[dict], str]:
    """Build text context hits from hindex descriptions for given (source, page_number) pairs."""
    hits = []
    _loaded: dict = {}
    for source, page_number in pages:
        if source not in _loaded:
            _loaded[source] = hindex_store.load(source)
        hierarchy = _loaded[source]
        if hierarchy is None:
            continue
        page_idx = page_number - 1
        if not (0 <= page_idx < len(hierarchy.page_records)):
            continue
        record = hierarchy.page_records[page_idx]
        content = _hindex_page_text(record)
        if content.strip():
            hits.append({
                "source": source,
                "page_number": page_number,
                "content": content,
                "score": 0.0,
                "crop_label": "pagedesc",
            })
    return build_text_context(hits, max_chars_per_doc=max_chars_per_doc, max_total_chars=max_total_chars)


def compute_sha256_hash(file_path):
    sha256_hash = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(4096), b""):
            sha256_hash.update(chunk)

    return sha256_hash.hexdigest()


def transform_pil_image_to_base64(image):
    #
    # Transform PIL image to base64
    # @param image: PIL image
    # @return: str
    #
    buffered = BytesIO()
    # extract image format from image
    image_type = image.format
    image.save(buffered, format=image_type)
    encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image/{image_type.lower()};base64,{encoded_string}"


chromadb.api.types.is_image = is_image


def get_md5sum(file_path, kvstore):
    # Create an MD5 hash object
    md5_hash = hashlib.md5()

    # Open the file in binary read mode
    img = kvstore.retrieve_image(file_path.encode("utf-8", "replace").decode())
    md5_hash.update(img.tobytes())

    # Return the hexadecimal digest of the hash
    return md5_hash.hexdigest()


def take_top_k(data: dict[str, list], k: int) -> dict[str, list]:
    """
    Extracts the first k elements from each list in the data dictionary and reverses them.
    Each value in data is expected to be a list (or a list of lists).

    Args:
        data: dictionary with keys such as 'distances', 'documents', etc.
        k: number of elements to extract

    Returns:
        A new dictionary containing only the first k elements for each key, in reversed order.
    """
    result = {}
    for key, value in data.items():
        if value is None:
            result[key] = None
            continue

        if isinstance(value, list) and value and isinstance(value[0], list):
            result[key] = [value[0][:k][::-1]]
        elif isinstance(value, list):
            result[key] = value[:k][::-1]
        else:
            result[key] = value
    return result


def sort_and_select_top_k(
    data: dict[str, list[list[Any]]], k: int, remove_duplicates: bool, kvstore, logger
) -> dict[str, list[list[Any]]]:
    """
    Sorts the dictionary based on the 'distances' key and reorders all other keys accordingly.
    Returns a new dictionary containing only the top k elements.

    Args:
        data (dict[str, list[list[Any]]]): The input dictionary with lists of lists.
        k (int): The number of top elements to return.

    Returns:
        dict[str, list[list[Any]]]: A new dictionary containing the top k elements.
    """
    if "distances" not in data or not isinstance(data["distances"], list) or not data["distances"]:
        raise KeyError("'distances' key is missing or improperly formatted.")

    distances = data["distances"][0]  # Extract the first inner list
    if not isinstance(distances, list):
        raise ValueError("'distances' should be a list of lists.")

    # Ensure that distances contains valid numeric values
    if not all(isinstance(d, int | float) for d in distances):
        raise ValueError("All elements in 'distances' must be numeric.")

    # Sort indices based on distance values
    sorted_indices = sorted(range(len(distances)), key=lambda i: distances[i])

    # Gather sources
    sorted_sources = []
    for key, nested_list in data.items():
        if key != "ids":
            continue

        # Skip None values
        if nested_list is None:
            # sorted_data[key] = None
            continue

        # Check if the key holds a list of lists
        if isinstance(nested_list, list) and nested_list and isinstance(nested_list[0], list):
            flat_list = nested_list[0]
            for s in flat_list:
                sorted_sources.append(s)

    # check for successive duplicates and remove them
    if remove_duplicates:
        duplicate_indices = []
        for i in range(len(sorted_sources) - 1):
            if get_md5sum(sorted_sources[i], kvstore) == get_md5sum(sorted_sources[i + 1], kvstore):
                duplicate_indices.append(i)
        if len(duplicate_indices) > 0:
            logger.debug("Found %d duplicates", len(duplicate_indices))

        for i in sorted(duplicate_indices, reverse=True):
            del sorted_indices[i]

    top_k_indices = sorted_indices[:k]
    sorted_data = {}

    for key, nested_list in data.items():
        # Skip None values
        if nested_list is None:
            sorted_data[key] = None
            continue

        # Check if the key holds a list of lists
        if isinstance(nested_list, list) and nested_list and isinstance(nested_list[0], list):
            flat_list = nested_list[0]
            sorted_list = [flat_list[i] for i in top_k_indices if i < len(flat_list)]
            sorted_data[key] = [sorted_list]
        else:
            # Preserve non-list values as they are
            sorted_data[key] = nested_list

    return sorted_data


class ImageEmbeddingFunction(EmbeddingFunction):
    def __init__(self, ad: InputConnectorObj, models_repository, logger):
        
        self.device = ad.rag.gpu_id
        
        # embedder - embedding_model should be set by now from service config
        if ad.rag.embedding_model is None:
            msg = "rag.embedding_model must be specified in the service configuration"
            logger.error(msg)
            raise ValueError(msg)
        
        self.rag_embedding_model = ad.rag.embedding_model

        self.shared = ad.rag.shared_model
        self.rag_image_width = ad.rag.ragm.image_width
        self.rag_image_height = ad.rag.ragm.image_height
        self.rag_auto_scale_for_font = ad.rag.ragm.auto_scale_for_font
        self.logger = logger
        self.vllm = ad.rag.embedding_lib == "vllm"
        self.vllm_memory_utilization = ad.rag.vllm_rag_memory_utilization
        self.vllm_quantization = ad.rag.vllm_rag_quantization
        self.vllm_enforce_eager = ad.rag.vllm_rag_enforce_eager

        # min/max image size
        min_pixels = 1 * 28 * 28
        max_pixels = 2560 * 28 * 28

        # load model
        self.logger.debug(f"Embedding model: {self.rag_embedding_model}")

        ## qwen2vl and qwen3-vl embedder support
        expected_prefixes = ("Alibaba-NLP/gme-Qwen2-VL", "Qwen/Qwen3-VL-Embedding")
        if not self.vllm and not any(self.rag_embedding_model.startswith(p) for p in expected_prefixes):
            self.logger.warning("rag.embedding_model should start with one of: %s", ",".join(expected_prefixes))
        self.model = None
        if self.shared:
            cache_key = ("vllm_embed" if self.vllm else "huggingface", self.rag_embedding_model)
            cached_model = ModelCache.get(cache_key)
            if cached_model:
                self.model, self.processor, self.llm_lib = cached_model
                self.logger.info("Reusing cached LLM embedder for %s", self.rag_embedding_model)
                return

        # not cached or not shared, load the model up
        if not self.model:
            if self.vllm:
                if self.vllm_quantization == "bitsandbytes":
                    self.vllm_load_format = "bitsandbytes"
                else:
                    self.vllm_load_format = "auto"
                self.model = VLLM(
                    model=self.rag_embedding_model,
                    download_dir=models_repository,
                    load_format=self.vllm_load_format,
                    quantization=self.vllm_quantization,
                    max_model_len=4096,
                    max_num_seqs=5,
                    task="embed",
                    mm_processor_kwargs={
                        "min_pixels": 28 * 28,
                        "max_pixels": 1280 * 28 * 28,
                        "fps": 1,
                    },
                    disable_mm_preprocessor_cache=False,
                    gpu_memory_utilization=self.vllm_memory_utilization,
                    limit_mm_per_prompt={"image": 1},
                    enforce_eager=self.vllm_enforce_eager,
                )
            else:
                processor_kwargs = {
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    "cache_dir": str(models_repository),
                }
                # Keep Qwen2-VL processor behavior stable and silence fast-processor migration warning.
                if "Qwen2-VL" in self.rag_embedding_model:
                    processor_kwargs["use_fast"] = False

                self.processor = AutoProcessor.from_pretrained(
                    self.rag_embedding_model,
                    **processor_kwargs,
                )
                # Support both the legacy Alibaba Qwen2-VL and the newer Qwen3-VL-Embedding model.
                # Try to instantiate a model class that provides embeddings. We prefer the
                # Qwen2 VL-specific class when detected, otherwise fall back to a generic
                # AutoModel (the Qwen3 embedding model should expose hidden states).
                try:
                    attn_implementation = resolve_attn_implementation(self.rag_embedding_model)

                    if self.rag_embedding_model.startswith("Alibaba-NLP/gme-Qwen2-VL"):

                        self.model = (
                            Qwen2VLForConditionalGeneration.from_pretrained(
                                self.rag_embedding_model,
                                attn_implementation=attn_implementation,
                                torch_dtype=torch.bfloat16,
                                cache_dir=str(models_repository),
                            )
                            .to("cuda:" + str(self.device))
                            .eval()
                        )
                    elif "Qwen3-VL-Embedding" in self.rag_embedding_model:
                        embedder = Qwen3VLEmbedder(
                            model_name_or_path=self.rag_embedding_model,
                            cache_dir=str(models_repository),
                        )
                        self.model = embedder.model.to("cuda:" + str(self.device)).eval()
                        self.processor = embedder.processor
                    else:
                        # Generic HF model for embeddings (Qwen3-VL-Embedding or similar)
                        self.model = (
                            AutoModel.from_pretrained(
                                self.rag_embedding_model,
                                cache_dir=str(models_repository),
                                trust_remote_code=True,  # Load custom model code for embedding models
                            )
                            .to("cuda:" + str(self.device))
                            .eval()
                        )
                except Exception as e:
                    self.logger.error("Error loading embedding model %s: %s", self.rag_embedding_model, repr(e))
                    raise

        if not self.vllm:
            # Ensure tokenizer/model padding sides are set consistently for chat-style templates
            try:
                self.processor.tokenizer.padding_side = "right"
            except Exception:
                pass
            try:
                # some model wrappers expose padding_side on the model or processor
                self.model.padding_side = "left"
            except Exception:
                pass

        # Cache the embedder for future use
        if self.shared:
            if self.vllm:
                ModelCache.add(cache_key, self.model, None, self.rag_embedding_model)
            else:
                ModelCache.add(cache_key, self.model, self.processor, "qwen2vl")

    def get_embedding(self, last_hidden_state: torch.Tensor) -> torch.Tensor:
        reps = last_hidden_state[:, -1]
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
        return reps

    def get_embedding_vllm(self, embeddings: torch.Tensor) -> torch.Tensor:
        reps = torch.nn.functional.normalize(embeddings, p=2, dim=-1)
        return reps

    def __call_vllm__(self, input: dict) -> Embeddings:
        mm_inputs = []
        for item in input:
            if type(item) is dict:
                label = item.get("label", None)
                hindex_text = item.get("text", "")
                item = item.get("doc", None)
            else:
                label = None
                hindex_text = ""
            if "PIL" in str(type(item)):  # or "Png" in str(type(item)) or "Jpeg" in str(type(item)):
                if self.rag_image_width and self.rag_image_height:
                    width, height = item.size
                    if width > self.rag_image_width or height > self.rag_image_height:
                        item.thumbnail((self.rag_image_width, self.rag_image_height))
                if self.rag_auto_scale_for_font and (self.rag_image_width or self.rag_image_height):
                    self.logger.warn("Auto scaling for font is enabled. image_width and image_height are ignored")
                # hindex_text carries VLM-extracted page context (summary, keywords, OCR);
                # the embedding then captures semantic meaning, not just visual appearance.
                question = hindex_text if hindex_text else "What is the content of this image?"
                if label:
                    question = "This image has a " + label + ". " + question
                placeholder = "<|image_pad|>"
                prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )
                mm_inputs.append(
                    {
                        "data": item,
                        "prompt": prompt,
                    }
                )
            else:
                question = item
                placeholder = "<|image_pad|>"
                prompt = (
                    "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                    f"<|im_start|>user\n<|vision_start|>{placeholder}<|vision_end|>"
                    f"{question}<|im_end|>\n"
                    "<|im_start|>assistant\n"
                )

                mm_inputs.append({"data": Image.new("RGB", (28, 28)), "prompt": prompt})

        inputs = [{"prompt": mmi["prompt"], "multi_modal_data": {"image": mmi["data"]}} for mmi in mm_inputs]
        outputs = self.model.embed(inputs)
        embeddings = [o.outputs.embedding for o in outputs]
        doc_embeddings = self.get_embedding_vllm(torch.tensor(embeddings))
        db_embeddings = cast(Embedding, doc_embeddings.squeeze().tolist())  ##TODO: beware multiple docs...
        return db_embeddings

    def __call__(self, input: dict) -> Embeddings:
        if self.vllm:
            return self.__call_vllm__(input)
        # prepare data
        doc_messages = []

        for item in input:
            # if is_image(input): ##XXX: fails to detect PIL image, checks for np array...
            if type(item) is dict:
                label = item.get("label", None)
                hindex_text = item.get("text", "")
                item = item.get("doc", None)
            else:
                label = None
                hindex_text = ""
            if "PIL" in str(type(item)):  # or "Png" in str(type(item)) or "Jpeg" in str(type(item)):
                if self.rag_image_width and self.rag_image_height:
                    width, height = item.size
                    if width > self.rag_image_width or height > self.rag_image_height:
                        item.thumbnail((self.rag_image_width, self.rag_image_height))

                # hindex_text carries VLM-extracted page context (summary, keywords, OCR);
                # the embedding then captures semantic meaning, not just visual appearance.
                embed_prompt = hindex_text  # gme-Qwen2-VL: text follows the image token
                if label:
                    embed_prompt = "This image has a " + label + ". " + embed_prompt
                message = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": item,
                            },
                            #'resized_height':xxx , 'resized_width':xxx
                            # adjust the image size for efficiency trade-off
                            {
                                "type": "text",
                                "text": embed_prompt,
                            },
                        ],
                    }
                ]
            else:
                message = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": item},
                        ],
                    }
                ]
            doc_messages.append(message)

        doc_texts = [
            self.processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True) + "<|endoftext|>"
            for msg in doc_messages
        ]

        doc_image_inputs, doc_video_inputs = process_vision_info(doc_messages)
        doc_inputs = self.processor(
            text=doc_texts,
            images=doc_image_inputs,
            videos=doc_video_inputs,
            # https://huggingface.co/Alibaba-NLP/gme-Qwen2-VL-2B-Instruct/blob/main/gme_inference.py#L116
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to("cuda:" + str(self.device))

        # call on the model
        with torch.no_grad():
            # Reset rope_deltas before each call: Qwen2-VL stores the delta from the
            # previous batch on the model object.  When batch size changes between calls
            # (e.g. N-image batch during indexing → 1-item text query), the stale
            # rope_deltas produces a 0-row delta tensor whose broadcast collapses the
            # batch dimension to 0, causing a (seq×0 vs 1536×1536) matmul crash.
            if hasattr(self.model, "model") and hasattr(self.model.model, "rope_deltas"):
                self.model.model.rope_deltas = None

            # If the model implements an `embed` method, use it (some wrappers provide it).
            if hasattr(self.model, "embed"):
                outputs = self.model.embed(doc_inputs)
                # model.embed may return a list-like of outputs with .outputs.embedding
                try:
                    embeddings = [o.outputs.embedding for o in outputs]
                    doc_embeddings = self.get_embedding_vllm(torch.tensor(embeddings))
                except Exception:
                    # fallback: if `embed` returns tensor-like
                    doc_embeddings = torch.nn.functional.normalize(torch.tensor(outputs), p=2, dim=-1)
            else:
                output = self.model(**doc_inputs, use_cache=False, return_dict=True, output_hidden_states=True)
                # prefer last_hidden_state if available, otherwise use hidden_states
                if hasattr(output, "last_hidden_state") and output.last_hidden_state is not None:
                    src = output.last_hidden_state
                else:
                    src = output.hidden_states[-1]
                doc_embeddings = self.get_embedding(src)
                del output
        db_embeddings = cast(Embedding, doc_embeddings.squeeze().tolist())  ##TODO: beware multiple docs...

        return db_embeddings


class RAGImgRetriever:
    def __init__(
        self,
        indexlib,
        indexdb,
        top_k,
        remove_duplicates,
        filter_width,
        filter_height,
        app_repository,
        kvstore,
        text_search_engine_index,
        logger,
    ):
        self.indexlib = indexlib
        self.indexdb = indexdb
        self.top_k = top_k
        self.remove_duplicates = remove_duplicates
        self.filter_width = filter_width
        self.filter_height = filter_height
        self.query_depth_mult = (
            200  # XXX: chroma has inconsistent best results wrt depth, so we push depth artificially
        )
        self.app_repository = app_repository
        self.kvstore = kvstore
        self.text_search_engine_index = text_search_engine_index
        self.logger = logger
        if self.indexlib == "coldb":
            self.colretriever = self.indexdb.as_retriever(search_type="similarity", search_kwargs={"k": self.top_k})

    def invoke(
        self,
        question: str,
        query_depth_mult: int,
        crop_label: str | list[str] | None = None,
        retrieval_mode: str = "embedding_retrieval",
        text_search_engine_top_k: int = 4,
        text_search_engine_fields: list[str] | None = None,
        top_k: int | None = None,
    ):
        ##- call on DB
        if query_depth_mult is None:
            query_depth_mult = self.query_depth_mult
        effective_top_k = top_k if top_k is not None else self.top_k

        docs = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
        if retrieval_mode_uses_embedding_retrieval(retrieval_mode):
            if self.indexdb is None:
                self.logger.warning(
                    "Embedding retrieval requested but no vector index is available. "
                    "Return empty embedding context."
                )
                docs = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
            elif self.indexlib == "chromadb":
                where_filter = None
                if crop_label is not None:
                    if isinstance(crop_label, str):
                        where_filter = {"crop_label": crop_label}
                    elif isinstance(crop_label, list):
                        where_filter = {"crop_label": {"$in": crop_label}}
                    else:
                        self.logger.warning("crop_label should be either a string or a list of strings.")

                docs = self.indexdb.query(
                    query_texts=[question],
                    n_results=effective_top_k * query_depth_mult,
                    where=where_filter,
                )
            else:
                docs = self.colretriever.invoke(question, query_depth_mult)

            self.logger.debug(f"retrieved vector documents: {json.dumps(docs, indent=2)}")
            docs = self.filter(docs, effective_top_k)

        if retrieval_mode_uses_text_search_engine(retrieval_mode):
            if self.text_search_engine_index is None:
                self.logger.warning("Text search retrieval requested but no text search engine index is available")
                docs["text_context"] = []
            else:
                docs["text_context"] = self.text_search_engine_index.search(
                    question,
                    limit=text_search_engine_top_k,
                    fields=text_search_engine_fields,
                    crop_label=crop_label,
                )

        return docs

    def filter(self, docs, top_k: int | None = None):
        # filter only top_k docs based on distance
        # we do not filter based on the distance, as it is already done by the embedder
        k = top_k if top_k is not None else self.top_k
        if self.remove_duplicates:
            return sort_and_select_top_k(docs, k, self.remove_duplicates, self.kvstore, self.logger)
        else:
            return take_top_k(docs, k)


class RAGImg:
    def init(
        self,
        ad: InputConnectorObj,
        app_repository: Path,
        models_repository: Path,
        cpu: bool,
        logger: logging.Logger,
        kvstore,
    ):
        self.ad = ad
        self.kvstore = kvstore
        self.app_repository = app_repository
        self.models_repository = models_repository
        self.cpu = cpu
        self.logger = logger

        ##XXX: no preprocessing for now, only get images

        if ad.rag is not None:
            self.rag = True
            self.rag_embf = None
            self.rag_reindex = ad.rag.reindex
            self.rag_index_protection = ad.rag.index_protection
            self.rag_top_k = ad.rag.top_k
            self.rag_remove_duplicates = ad.rag.remove_duplicates
            self.rag_chunk_num = ad.rag.chunk_num
            self.rag_chunk_overlap = ad.rag.chunk_overlap
            self.rag_indexdb_lib = ad.rag.indexdb_lib
            self.rag_retrieval_mode = ad.rag.retrieval_mode
            self.rag_text_search_engine_top_k = ad.rag.text_search_engine_top_k
            self.rag_text_search_engine_fields = ad.rag.text_search_engine_fields
            self.rag_text_search_engine_max_chars_per_doc = ad.rag.text_search_engine_max_chars_per_doc
            self.rag_text_search_engine_max_total_chars = ad.rag.text_search_engine_max_total_chars
            self.rag_num_partitions = ad.rag.num_partitions
            self.gpu_id = ad.rag.gpu_id

            if ad.rag.ragm is not None:
                self.rag_layout_detection = ad.rag.ragm.layout_detection
                self.rag_layout_detector_gpu_id = ad.rag.ragm.layout_detector_gpu_id
                if self.rag_layout_detection and self.rag_layout_detector_gpu_id is None:
                    self.rag_layout_detector_gpu_id = ad.rag.gpu_id
                self.rag_layout_detector_model_path = ad.rag.ragm.layout_detector_model_path
                self.rag_filter_width = ad.rag.ragm.filter_width
                self.rag_filter_height = ad.rag.ragm.filter_height
                self.rag_index_overview = ad.rag.ragm.index_overview
                self.rag_auto_scale_for_font = ad.rag.ragm.auto_scale_for_font
                self.rag_min_font_size = ad.rag.ragm.min_font_size
                self.rag_word_detector_gpu_id = ad.rag.gpu_id

        # indexdb
        self.indexpath = self.app_repository / "mm_index"
        self.text_search_engine_index = TantivyIndex(self.indexpath / "text_search_engine", self.logger)
        self.hindex_store = HierarchyStore(self.indexpath)
        # vlm_fn is injected by HFLib after the inference model is initialized
        self.vlm_fn = None
        self.hindex_retriever = None
        # Store embedding configuration for reuse during indexing
        self.rag_embedding_model = ad.rag.embedding_model
        self.rag_embedding_lib = ad.rag.embedding_lib
        
        if self.rag_indexdb_lib == "chromadb":
            path_key = str(self.indexpath)
            if path_key not in _chroma_client_cache:
                _chroma_client_cache[path_key] = chromadb.PersistentClient(
                    path=path_key,
                    settings=Settings(anonymized_telemetry=False),
                )
            self.rag_indexdb_client = _chroma_client_cache[path_key]
        else:
            self.rag_indexdb_client = None

        self.logger.info(f"self.indexpath exists: {self.indexpath.exists()}")

        if self.rag_indexdb_lib == "chromadb":
            self.logger.info(f"# collections: {self.rag_indexdb_client.count_collections()}")

        self.rag_indexdb_collection = None
        self.rag_text_embed_collection = None   # mm_text_db: page text embeddings
        self.rag_topic_embed_collection = None  # mm_topic_db: topic text embeddings
        self.reload_index_if_any(ad)

        # layout detection
        # XXX: URL and models are hardcoded for now since this is a custom
        #      that accomodates 'smart' crops for the multimodal rag documents.

        # retriever
        self.rag_retriever = None

    def __del__(self):
        if hasattr(self, 'rag_retriever') and self.rag_retriever is not None:
            del self.rag_retriever
            self.rag_retriever = None
        if hasattr(self, 'rag_indexdb_collection') and self.rag_indexdb_collection is not None:
            del self.rag_indexdb_collection
        if hasattr(self, 'rag_indexdb_client') and self.rag_indexdb_client is not None:
            # Drop the instance reference only. The module-level _chroma_client_cache
            # keeps the PersistentClient alive so the Rust backend singleton is not
            # torn down between sequential service creations in the same process.
            # Calling _system.stop() here would permanently break the Rust backend
            # for any subsequent PersistentClient call in this process.
            del self.rag_indexdb_client

    def reload_index_if_any(self, ad):
        rag_config = self._get_effective_rag_config(ad)
        vector_requested = retrieval_mode_uses_embedding_retrieval(rag_config.retrieval_mode)
        text_search_engine_requested = retrieval_mode_uses_text_search_engine(rag_config.retrieval_mode)
        text_embed_requested = retrieval_mode_uses_vision_text_embed(rag_config.retrieval_mode)
        topic_embed_requested = retrieval_mode_uses_vision_topic_embed(rag_config.retrieval_mode)

        self.has_existing_index = False
        self.has_existing_vector_index = False
        self.has_existing_text_embed_index = False
        self.has_existing_topic_embed_index = False
        self.has_existing_text_search_engine_index = self.text_search_engine_index.exists()
        if self.rag_indexdb_lib == "chromadb":
            collection_names = self.rag_indexdb_client.list_collections()
            if collection_names and not isinstance(collection_names[0], str):
                collection_names = [col.name for col in collection_names]
            self.logger.debug(f"Existing collections: {collection_names}")
            sqlite_exists = (self.indexpath / "chroma.sqlite3").exists()
            self.has_existing_vector_index = (
                vector_requested and "mm_db" in collection_names and sqlite_exists
            )
            self.has_existing_text_embed_index = (
                text_embed_requested and "mm_text_db" in collection_names and sqlite_exists
            )
            self.has_existing_topic_embed_index = (
                topic_embed_requested and "mm_topic_db" in collection_names and sqlite_exists
            )
        else:
            self.has_existing_vector_index = vector_requested and self.indexpath.exists()

        self.has_existing_index = (
            self.has_existing_vector_index
            or self.has_existing_text_embed_index
            or self.has_existing_topic_embed_index
            or (text_search_engine_requested and self.has_existing_text_search_engine_index)
        )
        self.logger.info(f"has_existing_index: {self.has_existing_index}")

        if self.has_existing_vector_index:
            # Initialize embedding function if using chromadb
            if rag_config.gpu_id == -1:
                msg = "ad.rag.gpu_id is mandatory when reloading db at service creation or using coldb"
                self.logger.error(msg)
                raise InputConnectorBadParamException(msg)
            if self.rag_indexdb_lib == "chromadb":
                # Use stored embedding_model from service creation, not from partial config
                # This ensures consistency even when reloading with partial configs
                if rag_config.embedding_model is None and self.rag_embedding_model is not None:
                    rag_config.embedding_model = self.rag_embedding_model
                merged_ad = ad.model_copy(deep=True)
                merged_ad.rag = rag_config
                self.rag_embf = ImageEmbeddingFunction(merged_ad, self.models_repository, self.logger)
            else:
                self.rag_embf = None
            self.logger.info("Loading existing index")
            if self.rag_indexdb_lib == "chromadb":
                self.rag_indexdb_collection = self.rag_indexdb_client.get_collection(
                    name="mm_db", embedding_function=self.rag_embf
                )
            else:
                self.rag_indexdb_collection = ColDB(
                    persist_directory=self.indexpath,
                    embedding_model_path=self.models_repository,
                    embedding_function=None,
                    embedding_lib=self.rag_embedding_lib,
                    embedding_model=self.rag_embedding_model,
                    collection_name="mm_db",
                    logger=self.logger,
                    gpu_id=rag_config.gpu_id,
                    num_partitions=self.rag_num_partitions,
                    index_bsize=rag_config.index_bsize,
                    image_width=rag_config.ragm.image_width,
                    image_height=rag_config.ragm.image_height,
                    kvstore=self.kvstore,
                )
            self.logger.info("Existing index loaded successfully")

        if text_search_engine_requested and self.has_existing_text_search_engine_index:
            self.text_search_engine_index.open()

        # Load text/topic embed collections if they exist and the mode needs them
        if (self.has_existing_text_embed_index or self.has_existing_topic_embed_index):
            if not hasattr(self, "rag_embf") or self.rag_embf is None:
                if rag_config.gpu_id != -1:
                    if rag_config.embedding_model is None and self.rag_embedding_model is not None:
                        rag_config.embedding_model = self.rag_embedding_model
                    merged_ad = ad.model_copy(deep=True)
                    merged_ad.rag = rag_config
                    self.rag_embf = ImageEmbeddingFunction(merged_ad, self.models_repository, self.logger)
            if self.has_existing_text_embed_index:
                self.rag_text_embed_collection = self.rag_indexdb_client.get_collection(
                    name="mm_text_db", embedding_function=self.rag_embf
                )
                self.logger.info("Loaded existing mm_text_db (page text embeddings)")
            if self.has_existing_topic_embed_index:
                self.rag_topic_embed_collection = self.rag_indexdb_client.get_collection(
                    name="mm_topic_db", embedding_function=self.rag_embf
                )
                self.logger.info("Loaded existing mm_topic_db (topic text embeddings)")

    def index(self, ad: InputConnectorObj, sorted_documents: dict[str, list[str]]):
        rag_config = self._get_effective_rag_config(ad)
        # Persist index-time RAG config as the service default for later predict calls.
        self.ad.rag = rag_config
        # Write retrieval_mode back to config.json so CLI chat and cross-runtime
        # runs (e.g. notebook index then container chat) see the same mode.
        _config_path = self.app_repository / "config.json"
        if _config_path.exists():
            try:
                with open(_config_path) as _f:
                    _saved_cfg = json.load(_f)
                _retrieval_mode = (
                    rag_config.retrieval_mode.value
                    if hasattr(rag_config.retrieval_mode, "value")
                    else str(rag_config.retrieval_mode)
                )
                _saved_cfg.setdefault("parameters", {}).setdefault("input", {}).setdefault("rag", {})[
                    "retrieval_mode"
                ] = _retrieval_mode
                with open(_config_path, "w") as _f:
                    json.dump(_saved_cfg, _f, indent=4)
            except Exception as e:
                self.logger.warning(
                    "Failed to persist retrieval_mode to config.json: %s",
                    e,
                )  # non-fatal: in-memory state is still correct
        vector_requested = retrieval_mode_uses_embedding_retrieval(rag_config.retrieval_mode)
        text_search_engine_requested = retrieval_mode_uses_text_search_engine(rag_config.retrieval_mode)
        vision_semantic_requested = retrieval_mode_uses_vision_semantic(rag_config.retrieval_mode)
        text_embed_requested = retrieval_mode_uses_vision_text_embed(rag_config.retrieval_mode)
        hybrid_bm25_requested = retrieval_mode_uses_vision_hybrid_bm25(rag_config.retrieval_mode)
        topic_embed_requested = retrieval_mode_uses_vision_topic_embed(rag_config.retrieval_mode)
        vlm_desc_requested = retrieval_mode_needs_vlm_descriptions(rag_config.retrieval_mode)

        # Check whether this is an update to an existing index
        self.gpu_id = rag_config.gpu_id if rag_config.gpu_id != -1 else self.gpu_id
        self.preproc_dpi = ad.preprocessing.dpi
        self.rag_update_index = rag_config.update_index
        self.rag_reindex = rag_config.reindex
        self.rag_index_protection = rag_config.index_protection
        self.rag_layout_detector_gpu_id = rag_config.gpu_id

        if self.rag_layout_detection:
            self.rag_layout_detector = LayoutDetector(
                model_path=self.rag_layout_detector_model_path,
                resize_width=768,
                resize_height=1024,
                models_repository=self.models_repository,
                logger=self.logger,
                device=self.rag_layout_detector_gpu_id,
            )
        else:
            self.rag_layout_detector = None

        # Ensure index directory exists
        if not self.indexpath.exists():
            self.indexpath.mkdir(parents=True, exist_ok=True)
            self.logger.debug("Created app index dir %s", self.indexpath)
        # Recreate retriever on next request so it picks updated index handles/config.
        self.rag_retriever = None

        if self.has_existing_index and not self.rag_reindex:
            # index reload : already done at service creation
            pass
        else:
            if self.has_existing_index and self.rag_reindex:
                if self.rag_index_protection:
                    msg = "Index already exists and is protected. To reindex, disable index_protection."
                    self.logger.error(msg)
                    raise InputConnectorBadParamException(msg)
                # Delete the existing index if protection is off.
                if vector_requested and self.rag_indexdb_lib == "chromadb" and self.has_existing_vector_index:
                    self.rag_indexdb_client.delete_collection(name="mm_db")
                if text_embed_requested and self.rag_indexdb_lib == "chromadb" and self.has_existing_text_embed_index:
                    self.rag_indexdb_client.delete_collection(name="mm_text_db")
                    self.rag_text_embed_collection = None
                if topic_embed_requested and self.rag_indexdb_lib == "chromadb" and self.has_existing_topic_embed_index:
                    self.rag_indexdb_client.delete_collection(name="mm_topic_db")
                    self.rag_topic_embed_collection = None
                if (text_search_engine_requested or vision_semantic_requested or hybrid_bm25_requested) and self.has_existing_text_search_engine_index:
                    self.text_search_engine_index.reset()
                if retrieval_mode_uses_hindex(rag_config.retrieval_mode, rag_config.hindex):
                    for doc_name in self.hindex_store.list_doc_names():
                        self.hindex_store.delete(doc_name)

            self.logger.info("Creating new index")
            if vector_requested and self.rag_indexdb_lib == "chromadb":
                # Use stored embedding_model from service creation, not from partial config
                # This ensures consistency even when indexing with partial configs like vrag_default_index.json
                if rag_config.embedding_model is None and self.rag_embedding_model is not None:
                    rag_config.embedding_model = self.rag_embedding_model
                merged_ad = ad.model_copy(deep=True)
                merged_ad.rag = rag_config
                self.rag_embf = ImageEmbeddingFunction(merged_ad, self.models_repository, self.logger)
                self.rag_indexdb_collection = self.rag_indexdb_client.create_collection(
                    name="mm_db",
                    embedding_function=self.rag_embf,
                    metadata={"hnsw:space": "cosine"},
                )
            elif vector_requested:
                self.rag_embf = None
                self.rag_indexdb_collection = ColDB(
                    persist_directory=self.indexpath,
                    embedding_model_path=self.models_repository,
                    embedding_function=None,
                    embedding_lib=self.rag_embedding_lib,
                    embedding_model=self.rag_embedding_model,
                    collection_name="mm_db",
                    logger=self.logger,
                    num_partitions=self.rag_num_partitions,
                    gpu_id=rag_config.gpu_id,
                    index_bsize=rag_config.index_bsize,
                    image_width=rag_config.ragm.image_width,
                    image_height=rag_config.ragm.image_height,
                    kvstore=self.kvstore,
                )
            else:
                # vector_requested=False (e.g. hindex-only or text-only build).
                # Preserve an existing ChromaDB collection reference so a subsequent
                # hybrid query can still use it.  Only clear when there was never a
                # vector index to begin with.
                if not self.has_existing_vector_index:
                    self.rag_indexdb_collection = None

            # Create text/topic embed collections for new vision embedding modes.
            if (text_embed_requested or topic_embed_requested) and self.rag_indexdb_lib == "chromadb":
                if not hasattr(self, "rag_embf") or self.rag_embf is None:
                    if rag_config.embedding_model is None and self.rag_embedding_model is not None:
                        rag_config.embedding_model = self.rag_embedding_model
                    merged_ad = ad.model_copy(deep=True)
                    merged_ad.rag = rag_config
                    self.rag_embf = ImageEmbeddingFunction(merged_ad, self.models_repository, self.logger)
                if text_embed_requested and self.rag_text_embed_collection is None:
                    self.rag_text_embed_collection = self.rag_indexdb_client.get_or_create_collection(
                        name="mm_text_db",
                        embedding_function=self.rag_embf,
                        metadata={"hnsw:space": "cosine"},
                    )
                    self.logger.info("Created/opened mm_text_db (page text embeddings)")
                if topic_embed_requested and self.rag_topic_embed_collection is None:
                    self.rag_topic_embed_collection = self.rag_indexdb_client.get_or_create_collection(
                        name="mm_topic_db",
                        embedding_function=self.rag_embf,
                        metadata={"hnsw:space": "cosine"},
                    )
                    self.logger.info("Created/opened mm_topic_db (topic text embeddings)")

        # save state for for multiple index queries
        self.has_existing_index = True
        self.has_existing_vector_index = self.has_existing_vector_index or vector_requested
        self.has_existing_text_embed_index = self.has_existing_text_embed_index or text_embed_requested
        self.has_existing_topic_embed_index = self.has_existing_topic_embed_index or topic_embed_requested
        self.has_existing_text_search_engine_index = self.has_existing_text_search_engine_index or text_search_engine_requested or vision_semantic_requested or hybrid_bm25_requested

        if self.rag_update_index:
            files, offset = set(), 0
            if vector_requested and self.rag_indexdb_lib == "chromadb" and self.rag_indexdb_collection is not None:
                # Get all data from the collection by batch of 10_000
                while (result := self.rag_indexdb_collection.get(offset=offset, limit=10_000)) and (
                    len(result["ids"]) > 0
                ):
                    files.update([f["source"] for f in result["metadatas"]])
                    offset += len(result["ids"])
            self.logger.info(f"Existing index contains {len(files)} elements")
        elif self.rag_reindex:
            files = set()
        else:
            self.logger.info(f"{self.rag_indexdb_collection.count()} elements in index")

        if self.rag_reindex or self.rag_update_index:
            if self.rag_update_index:
                self.logger.info("Updating an existing index")

            # Process and add documents/images to the index
            processor = DocumentProcessor(
                app_repository=self.app_repository,
                logger=self.logger,
                dpi=self.preproc_dpi,
            )
            self.rag_word_detector_gpu_id = ad.rag.gpu_id

            image_processor = ImageProcessor(
                self.rag_layout_detector,
                self.rag_chunk_num,
                self.rag_chunk_overlap,
                self.rag_index_overview,
                self.rag_auto_scale_for_font,
                self.rag_min_font_size,
                self.rag_word_detector_gpu_id,
                self.rag_filter_width,
                self.rag_filter_height,
                self.logger,
            )

            doclist = []
            metadatalist = []
            tantivy_documents = []
            vlm_tantivy_documents = []
            doc_level_tantivy_docs: list[dict] = []
            queryhint_tantivy_docs: list[dict] = []
            entity_tantivy_docs: list[dict] = []
            text_embed_ids: list[str] = []
            text_embed_docs_list: list[str] = []
            text_embed_metas: list[dict] = []
            topic_embed_ids: list[str] = []
            topic_embed_docs_list: list[str] = []
            topic_embed_metas: list[dict] = []
            t1 = time()
            for fext, docs in sorted_documents.items():
                for doc in tqdm(docs, desc="Indexing documents"):
                    self.logger.info(f"Indexing document {doc}")
                    skip_vector_and_text = doc in files
                    if skip_vector_and_text:
                        # Doc is already in the vector store.  Check whether hindex
                        # or description-based modes still need to be built.
                        hindex_needed = (
                            retrieval_mode_uses_hindex(rag_config.retrieval_mode, rag_config.hindex)
                            and not self.hindex_store.exists(doc)
                        )
                        # Fast path for description-based modes: if hindex exists,
                        # collect docs from it without re-rendering images.
                        if vlm_desc_requested and self.hindex_store.exists(doc):
                            existing = self.hindex_store.load(doc)
                            if existing:
                                if vision_semantic_requested or hybrid_bm25_requested:
                                    vlm_tantivy_documents.extend(_hierarchy_to_vlm_tantivy_docs(existing))
                                if hybrid_bm25_requested:
                                    _ddoc = _hierarchy_to_doc_tantivy_doc(existing)
                                    if _ddoc:
                                        doc_level_tantivy_docs.append(_ddoc)
                                    queryhint_tantivy_docs.extend(_hierarchy_to_queryhint_tantivy_docs(existing))
                                    entity_tantivy_docs.extend(_hierarchy_to_entity_tantivy_docs(existing))
                                if text_embed_requested:
                                    for _pi, _rec in enumerate(existing.page_records):
                                        _txt = _hindex_page_text(_rec)
                                        if _txt.strip():
                                            _pn = _pi + 1
                                            text_embed_ids.append(f"{doc}:page:{_pn}")
                                            text_embed_docs_list.append(_txt)
                                            text_embed_metas.append({"source": doc, "page_number": _pn})
                                if topic_embed_requested:
                                    for _ti, _topic in enumerate(existing.topics):
                                        _txt = _hindex_topic_text(_topic)
                                        if _txt.strip():
                                            _pidx = ",".join(str(i) for i in (_topic.page_indices or []))
                                            topic_embed_ids.append(f"{doc}:topic:{_ti}")
                                            topic_embed_docs_list.append(_txt)
                                            topic_embed_metas.append({
                                                "source": doc,
                                                "page_indices": _pidx,
                                                "theme": _topic.theme or "",
                                            })
                            continue
                        if not hindex_needed and not vlm_desc_requested:
                            self.logger.info(f"Document {doc} already indexed")
                            continue
                        self.logger.info(
                            f"[hindex] Document {doc} already in vector store; "
                            "re-processing for hindex only"
                        )

                    document = dict(source=Path(doc), ext=fext.lower(), images=list())
                    # Augment document with images
                    processor.transform_documents_to_images([document])
                    self.logger.info(f"\t{len(document['images'])} images extracted")
                    document["npages"] = len(document["images"])
                    if not skip_vector_and_text:
                        page_texts = self._extract_page_texts(document["source"])

                    # Augment document with crops/chunks (must run before hindex so
                    # crop keys are available for per-page records)
                    image_processor.preprocess_images([document])
                    self.logger.info(f"\t{len(document['parts'])} parts generated")

                    # Hierarchical index — runs after preprocessing so crop keys are known.
                    # All vlm_desc modes build hindex files (source of VLM descriptions).
                    # preserve_existing_hindex=True so description modes never rebuild
                    # hindex even when reindex=True (reindex only controls tantivy/chromadb).
                    if retrieval_mode_uses_hindex(rag_config.retrieval_mode, rag_config.hindex) or vlm_desc_requested:
                        vlm_docs = self._build_hindex_for_doc(
                            document, rag_config,
                            preserve_existing_hindex=vlm_desc_requested,
                            page_texts=page_texts if not skip_vector_and_text else None,
                        )
                        if vision_semantic_requested or hybrid_bm25_requested:
                            vlm_tantivy_documents.extend(vlm_docs)
                        if hybrid_bm25_requested:
                            _h_built = self.hindex_store.load(str(document["source"]))
                            if _h_built:
                                _ddoc = _hierarchy_to_doc_tantivy_doc(_h_built)
                                if _ddoc:
                                    doc_level_tantivy_docs.append(_ddoc)
                                queryhint_tantivy_docs.extend(_hierarchy_to_queryhint_tantivy_docs(_h_built))
                                entity_tantivy_docs.extend(_hierarchy_to_entity_tantivy_docs(_h_built))

                    # Accumulate page/topic texts for dense embedding into ChromaDB.
                    if text_embed_requested or topic_embed_requested:
                        _h = self.hindex_store.load(str(document["source"]))
                        if _h is not None:
                            if text_embed_requested:
                                for _pi, _rec in enumerate(_h.page_records):
                                    _txt = _hindex_page_text(_rec)
                                    if _txt.strip():
                                        _pn = _pi + 1
                                        _src = str(document["source"])
                                        text_embed_ids.append(f"{_src}:page:{_pn}")
                                        text_embed_docs_list.append(_txt)
                                        text_embed_metas.append({"source": _src, "page_number": _pn})
                            if topic_embed_requested:
                                for _ti, _topic in enumerate(_h.topics):
                                    _txt = _hindex_topic_text(_topic)
                                    if _txt.strip():
                                        _src = str(document["source"])
                                        _pidx = ",".join(str(i) for i in (_topic.page_indices or []))
                                        topic_embed_ids.append(f"{_src}:topic:{_ti}")
                                        topic_embed_docs_list.append(_txt)
                                        topic_embed_metas.append({
                                            "source": _src,
                                            "page_indices": _pidx,
                                            "theme": _topic.theme or "",
                                        })

                    # Text search engine: one document per page (page-level text, not per-crop)
                    # Indexing per-crop would duplicate the same page text N times and
                    # return N identical-content hits for a single-page match.
                    if (text_search_engine_requested or hybrid_bm25_requested) and not skip_vector_and_text:
                        source_str = str(document["source"])
                        for page_number, page_text in page_texts.items():
                            page_text_stripped = page_text.strip()
                            if page_text_stripped:
                                tantivy_documents.append(
                                    {
                                        "doc_id": f"{source_str}:page:{page_number}",
                                        "source": source_str,
                                        "page_number": page_number,
                                        "crop_label": None,
                                        "label": None,
                                        "content": page_text_stripped,
                                    }
                                )

                    # store images in the index & vector store (skip if already indexed)
                    if skip_vector_and_text:
                        continue
                    for part in tqdm(document["parts"]):
                        metadatas = part["metadata"]

                        self.kvstore.store_image(
                            part["name"].encode("utf-8", "replace").decode(),
                            part["img"],
                        )

                        if vector_requested and self.rag_indexdb_lib == "chromadb":
                            # also pass the crop_label to the embedder
                            image_dict = {
                                "doc": part["img"],
                                "label": metadatas.get("crop_label"),
                            }
                            # store document in the vector store
                            self.rag_indexdb_collection.add(
                                images=[image_dict],
                                ids=[part["name"].encode("utf-8", "replace").decode()],
                                metadatas=[metadatas],
                            )
                        elif vector_requested:
                            doclist.append(part["name"])
                            metadatalist.append(metadatas)

            if vector_requested and self.rag_indexdb_lib == "coldb":
                self.rag_indexdb_collection.add_imgs(doclist, metadatalist, "mm_db")

            if text_search_engine_requested:
                self.text_search_engine_index.add_documents(tantivy_documents, recreate=self.rag_reindex and not self.rag_update_index)
            if vision_semantic_requested:
                self.text_search_engine_index.add_documents(vlm_tantivy_documents, recreate=not self.rag_update_index)
                self.logger.info(
                    "[vision_semantic] Indexed %d VLM page descriptions into text search engine [%.2fs]",
                    len(vlm_tantivy_documents), time() - t1,
                )
            if hybrid_bm25_requested:
                # Merge OCR text + VLM descriptions + doc-level descriptions into a
                # single Tantivy index so BM25 can match page content, semantic prose,
                # and whole-document context (for cross-document retrieval).
                all_hybrid_docs = tantivy_documents + vlm_tantivy_documents + doc_level_tantivy_docs + queryhint_tantivy_docs + entity_tantivy_docs
                self.text_search_engine_index.add_documents(
                    all_hybrid_docs, recreate=self.rag_reindex and not self.rag_update_index
                )
                self.logger.info(
                    "[vision_hybrid_bm25] Indexed %d docs (OCR=%d + VLM=%d + doc=%d + queryhint=%d + entity=%d) [%.2fs]",
                    len(all_hybrid_docs), len(tantivy_documents), len(vlm_tantivy_documents),
                    len(doc_level_tantivy_docs), len(queryhint_tantivy_docs), len(entity_tantivy_docs), time() - t1,
                )
            if text_embed_requested and self.rag_text_embed_collection is not None and text_embed_ids:
                # Batch upsert to avoid OOM: embedding all pages at once with padding=True
                # inflates KV-cache proportionally to batch_size × max_seq_len.
                _TEXT_EMBED_BATCH = 4
                for _i in range(0, len(text_embed_ids), _TEXT_EMBED_BATCH):
                    self.rag_text_embed_collection.upsert(
                        ids=text_embed_ids[_i:_i + _TEXT_EMBED_BATCH],
                        documents=text_embed_docs_list[_i:_i + _TEXT_EMBED_BATCH],
                        metadatas=text_embed_metas[_i:_i + _TEXT_EMBED_BATCH],
                    )
                self.logger.info(
                    "[vision_text_embed] Indexed %d page text embeddings [%.2fs]",
                    len(text_embed_ids), time() - t1,
                )
            if topic_embed_requested and self.rag_topic_embed_collection is not None and topic_embed_ids:
                _TOPIC_EMBED_BATCH = 4
                for _i in range(0, len(topic_embed_ids), _TOPIC_EMBED_BATCH):
                    self.rag_topic_embed_collection.upsert(
                        ids=topic_embed_ids[_i:_i + _TOPIC_EMBED_BATCH],
                        documents=topic_embed_docs_list[_i:_i + _TOPIC_EMBED_BATCH],
                        metadatas=topic_embed_metas[_i:_i + _TOPIC_EMBED_BATCH],
                    )
                self.logger.info(
                    "[vision_topic_embed] Indexed %d topic text embeddings [%.2fs]",
                    len(topic_embed_ids), time() - t1,
                )

            if self.rag_indexdb_collection is not None:
                self.logger.info(f"{self.rag_indexdb_collection.count()} elements in store [{time() - t1:.2f}]")
            elif text_search_engine_requested:
                self.logger.info(f"Text search engine indexing completed [{time() - t1:.2f}]")

        # Build / refresh collection index when the corpus is large enough.
        # Runs after all per-document hindex files are on disk so load_all() is complete.
        if retrieval_mode_uses_hindex(rag_config.retrieval_mode, rag_config.hindex):
            if self.vlm_fn is not None and not self.hindex_store.collection_index_exists():
                hierarchies = self.hindex_store.load_all()
                idx = build_collection_index(hierarchies, self.vlm_fn, self.logger)
                if idx is not None:
                    self.hindex_store.save_collection_index(idx)
                    self.logger.info(
                        "[hindex] Collection index saved: %d clusters for %d documents",
                        len(idx.clusters), len(hierarchies),
                    )

        # Release layout detector resources
        if self.rag_layout_detection:
            del self.rag_layout_detector.model

    # returns docs
    def retrieve(
        self,
        rag_question,
        query_depth_mult,
        crop_label: str | list[str] = None,
        request_rag: RAGObj | None = None,
    ):
        """
        Retrieve relevant documents for a question.
        
        Args:
            rag_question: The query text
            query_depth_mult: Depth multiplier for search
            crop_label: Optional label to filter results (e.g., 'text', 'figure', 'table')
        
        Returns:
            Retrieved documents
        """
        self.logger.debug("retrieving " + rag_question)
        if crop_label:
            self.logger.debug(f"with crop_label filter: {crop_label}")
        
        if self.rag_retriever is None:
            self.rag_retriever = RAGImgRetriever(
                self.rag_indexdb_lib,
                self.rag_indexdb_collection,
                self.rag_top_k,
                self.rag_remove_duplicates,
                self.rag_filter_width,
                self.rag_filter_height,
                self.app_repository,
                self.kvstore,
                self.text_search_engine_index,
                self.logger,
            )

        rag_config = self._get_effective_rag_config_for_request(request_rag)
        docs = self.rag_retriever.invoke(
            rag_question,
            query_depth_mult,
            crop_label,
            retrieval_mode=rag_config.retrieval_mode,
            text_search_engine_top_k=rag_config.text_search_engine_top_k,
            text_search_engine_fields=rag_config.text_search_engine_fields,
            top_k=rag_config.top_k,
        )

        if retrieval_mode_uses_hindex(rag_config.retrieval_mode, rag_config.hindex):
            if self.vlm_fn is None:
                self.logger.warning(
                    "[hindex] vision_hierarchical mode requested but vlm_fn is not set; "
                    "skipping hierarchy retrieval"
                )
            else:
                if self.hindex_retriever is None:
                    self.hindex_retriever = HierarchyRetriever(
                        self.hindex_store,
                        self.kvstore,
                        self.vlm_fn,
                        top_k=rag_config.top_k,
                        logger=self.logger,
                    )
                # hindex navigation: VLM selects relevant topics → page images and
                # crops (full-page views + per-page crops) are added to the context.
                # invoke() contributes full-page image keys from the kvstore, which
                # give the answering VLM holistic page context that crop-level
                # ChromaDB results cannot provide.
                hindex_docs = self.hindex_retriever.invoke(rag_question)
                docs = _merge_hindex_results(docs, hindex_docs, top_k=rag_config.top_k)

        if retrieval_mode_uses_vision_semantic(rag_config.retrieval_mode):
            if self.text_search_engine_index is None:
                self.logger.warning(
                    "[vision_semantic] text_search_engine_index not initialized; "
                    "index with retrieval_mode='vision_semantic' first."
                )
            else:
                hits = self.text_search_engine_index.search(
                    rag_question,
                    limit=rag_config.top_k * 4,
                    fields=["content"],
                    crop_label="pagedesc",
                )
                # vision_semantic is text-only at query time: no images loaded.
                docs = {"ids": [[]], "distances": [[]], "metadatas": [[]]}
                text_hits, text_prompt = build_text_context(
                    hits[:rag_config.top_k],
                    max_chars_per_doc=rag_config.text_search_engine_max_chars_per_doc,
                    max_total_chars=rag_config.text_search_engine_max_total_chars,
                )
                docs["text_context"] = text_hits
                docs["text_context_prompt"] = text_prompt
                self.logger.debug(
                    "[vision_semantic] %d hits for query '%s'", len(hits), rag_question[:80]
                )

        if retrieval_mode_uses_text_search_engine(rag_config.retrieval_mode):
            text_hits, text_prompt = build_text_context(
                docs.get("text_context", []),
                max_chars_per_doc=rag_config.text_search_engine_max_chars_per_doc,
                max_total_chars=rag_config.text_search_engine_max_total_chars,
            )
            docs["text_context"] = text_hits
            docs["text_context_prompt"] = text_prompt

        _mode = rag_config.retrieval_mode

        if _mode == "vision_text_embed":
            if self.rag_text_embed_collection is None:
                self.logger.warning("[vision_text_embed] mm_text_db not initialized; index with this mode first.")
            else:
                _results = self.rag_text_embed_collection.query(
                    query_texts=[rag_question],
                    n_results=min(rag_config.top_k * 4, self.rag_text_embed_collection.count() or 1),
                    include=["metadatas", "distances"],
                )
                _hits: list[tuple[str, int, float]] = []
                for _meta, _dist in zip(_results["metadatas"][0], _results["distances"][0]):
                    _src = _meta.get("source", "")
                    _pn = _meta.get("page_number")
                    if _src and _pn is not None:
                        _hits.append((_src, int(_pn), float(_dist)))
                docs = _page_hits_to_docs(_hits[: rag_config.top_k], self.hindex_store, self.kvstore, rag_config.top_k)
                _tc_hits, _tc_prompt = _hindex_text_context_for_pages(
                    [(_s, _p) for _s, _p, _ in _hits[: rag_config.top_k]],
                    self.hindex_store,
                    rag_config.text_search_engine_max_chars_per_doc,
                    rag_config.text_search_engine_max_total_chars,
                )
                docs["text_context"] = _tc_hits
                docs["text_context_prompt"] = _tc_prompt
                self.logger.debug("[vision_text_embed] %d page hits for '%s'", len(_hits), rag_question[:80])

        elif _mode == "vision_hybrid_bm25":
            if self.text_search_engine_index is None:
                self.logger.warning("[vision_hybrid_bm25] text_search_engine_index not initialized.")
            else:
                _bm25_query = _normalize_query_for_bm25(rag_question)
                _bm25_hits_all = self.text_search_engine_index.search(
                    _bm25_query,
                    limit=rag_config.top_k * 8,
                    fields=["content"],
                    crop_label=None,
                )
                # Split page-level from doc-level hits
                _page_bm25_hits = [h for h in _bm25_hits_all if h.get("crop_label") != "docdesc"]
                _doc_bm25_hits = [h for h in _bm25_hits_all if h.get("crop_label") == "docdesc"]
                _page_list = _bm25_hits_to_page_list(_page_bm25_hits)
                # Expand doc-level hits: add all pages of matching docs not already in page_list.
                # This improves cross-document retrieval where the query matches a whole document
                # rather than a specific page.
                _seen_pages = {(_s, _p) for _s, _p, _ in _page_list}
                _loaded_h: dict = {}
                for _dh in _doc_bm25_hits:
                    _src = _dh.get("source", "")
                    _score = max(_dh.get("score", 0.0), 0.0)
                    if not _src:
                        continue
                    if _src not in _loaded_h:
                        _loaded_h[_src] = self.hindex_store.load(_src)
                    _h = _loaded_h[_src]
                    if _h:
                        for _pi in range(len(_h.page_records)):
                            _pn = _pi + 1
                            if (_src, _pn) not in _seen_pages:
                                _seen_pages.add((_src, _pn))
                                _page_list.append((_src, _pn, 1.0 / (1.0 + _score)))
                # Queryhint-boosted BM25: search only over anticipated-query entries.
                # Pages whose anticipated queries match the user question get an extra RRF
                # vote, lifting them above pages that matched only via pagedesc or OCR.
                # This is critical when a specific inner page (e.g. a roadmap table) is
                # outranked by a cover page that incidentally contains the same keywords.
                # Score-ratio filter + per-doc cap via _filter_queryhint_hits:
                # threshold=0.5 (adaptive quality gate) and cap=top_k (safety valve).
                # See _filter_queryhint_hits docstring for rationale.
                _queryhint_page_list: list[tuple[str, int, float]] = []
                try:
                    _qh_hits = self.text_search_engine_index.search(
                        _bm25_query,
                        limit=rag_config.top_k * 8,
                        fields=["content"],
                        crop_label="queryhint",
                    )
                    _qh_all = _bm25_hits_to_page_list(_qh_hits)
                    # Group by doc; _qh_all is sorted by BM25 desc so first entry per
                    # doc is always the best-scoring page for that document.
                    _queryhint_page_list = _filter_queryhint_hits(
                        _qh_all,
                        score_ratio_threshold=0.5,
                        per_doc_cap=rag_config.top_k,
                    )
                except Exception as _e:
                    self.logger.warning("[vision_hybrid_bm25] queryhint BM25 failed: %s", _e)

                # Text embedding retrieval (semantic match over VLM-generated page text)
                _text_embed_page_list: list[tuple[str, int, float]] = []
                if self.rag_text_embed_collection is not None:
                    try:
                        _n_text = min(rag_config.top_k * 8, max(self.rag_text_embed_collection.count() or 1, 1))
                        _text_results = self.rag_text_embed_collection.query(
                            query_texts=[rag_question],
                            n_results=_n_text,
                            include=["metadatas", "distances"],
                        )
                        _text_seen: set[tuple[str, int]] = set()
                        for _meta, _dist in zip(_text_results["metadatas"][0], _text_results["distances"][0]):
                            _src = _meta.get("source", "")
                            _pn = _meta.get("page_number")
                            if _src and _pn is not None:
                                _key = (_src, int(_pn))
                                if _key not in _text_seen:
                                    _text_seen.add(_key)
                                    _text_embed_page_list.append((_src, int(_pn), float(_dist)))
                    except Exception as _e:
                        self.logger.warning("[vision_hybrid_bm25] text embed query failed: %s", _e)

                # Visual embedding retrieval (GME-Qwen2-VL, cross-modal) — gated by distance
                _embed_page_list: list[tuple[str, int, float]] = []
                if self.rag_indexdb_collection is not None:
                    try:
                        _n_embed = min(rag_config.top_k * 8, max(self.rag_indexdb_collection.count() or 1, 1))
                        _embed_results = self.rag_indexdb_collection.query(
                            query_texts=[rag_question],
                            n_results=_n_embed,
                            include=["metadatas", "distances"],
                        )
                        _embed_seen: set[tuple[str, int]] = set()
                        for _meta, _dist in zip(_embed_results["metadatas"][0], _embed_results["distances"][0]):
                            _src = _meta.get("source", "")
                            _pn = _meta.get("page_number")
                            if _src and _pn is not None:
                                _key = (_src, int(_pn))
                                if _key not in _embed_seen:
                                    _embed_seen.add(_key)
                                    _embed_page_list.append((_src, int(_pn), float(_dist)))
                    except Exception as _e:
                        self.logger.warning("[vision_hybrid_bm25] visual embed query failed: %s", _e)

                # ── Query intent classification ────────────────────────────────────
                # Determines which signals to activate and how to weight them.
                # visual   → entity BM25 off; visual_embed weight ×2
                # entity   → entity BM25 on; all weights equal
                # semantic → entity BM25 off; text_embed weight ×2
                _intent = _classify_query_intent(_bm25_query)
                self.logger.debug(
                    "[vision_hybrid_bm25] query intent=%s for '%s'",
                    _intent, _bm25_query[:80],
                )

                # ── Entity BM25 (intent-gated) ────────────────────────────────────
                # Only activated for entity-intent queries. For visual and semantic
                # queries the entity index adds noise: amounts and named identifiers
                # that appear across documents score without discriminating power.
                _entity_page_list: list[tuple[str, int, float]] = []
                if _intent == "entity":
                    try:
                        _ent_hits = self.text_search_engine_index.search(
                            _bm25_query,
                            limit=rag_config.top_k * 8,
                            fields=["content"],
                            crop_label="entity",
                        )
                        _entity_page_list = _bm25_hits_to_page_list(_ent_hits)
                    except Exception as _e:
                        self.logger.warning("[vision_hybrid_bm25] entity BM25 failed: %s", _e)

                # ── Intent-aware weighted RRF ─────────────────────────────────────
                # Signal weights by intent:
                #   visual   → visual_embed ×1.5 (pattern recognition primary)
                #   entity   → entity_bm25 ×1.5 (named entity lookup primary)
                #   semantic → text_embed ×1.5 (vocabulary bridge primary)
                # ×1.5 (not ×2) keeps boosted signal from crowding out queryhint
                # BM25, which is the primary cross-page fusion signal.
                # Base signals (BM25 + queryhint) always weight 1.0.
                _bm25_ranked     = [(_s, _p) for _s, _p, _ in _page_list]
                _queryhint_ranked= [(_s, _p) for _s, _p, _ in _queryhint_page_list]
                _text_ranked     = [(_s, _p) for _s, _p, _ in _text_embed_page_list]
                _embed_ranked    = [(_s, _p) for _s, _p, _ in _embed_page_list]
                _entity_ranked   = [(_s, _p) for _s, _p, _ in _entity_page_list]

                _rrf_lists: list[list[tuple[str, int]]] = [_bm25_ranked]
                _rrf_weights: list[float] = [1.0]

                if _queryhint_ranked:
                    _rrf_lists.append(_queryhint_ranked)
                    _rrf_weights.append(1.0)
                if _text_ranked:
                    _rrf_lists.append(_text_ranked)
                    _rrf_weights.append(1.5 if _intent == "semantic" else 1.0)
                if _embed_ranked:
                    _rrf_lists.append(_embed_ranked)
                    _rrf_weights.append(1.5 if _intent == "visual" else 1.0)
                if _entity_ranked:
                    _rrf_lists.append(_entity_ranked)
                    _rrf_weights.append(1.5)  # entity intent → entity BM25 is primary

                if len(_rrf_lists) > 1:
                    _page_list = _reciprocal_rank_fusion(_rrf_lists, weights=_rrf_weights)

                # ── Page-type boost for visual queries ────────────────────────────
                # After RRF, give a small relevance boost to visual-dominant pages
                # (charts, figures, diagrams) when the query is visual-intent.
                # Uses page_type stored in the hindex — no extra VLM call.
                # Hierarchies are cached per-source to avoid repeated disk reads.
                if _intent == "visual" and self.hindex_store is not None:
                    _pt_cache: dict[str, object] = {}
                    _boosted: list[tuple[str, int, float]] = []
                    for _src, _pn, _rrf_score in _page_list:
                        if _src not in _pt_cache:
                            _pt_cache[_src] = self.hindex_store.load(_src)
                        _pt_h = _pt_cache[_src]
                        _pt_rec = _pt_h.record_for_page(_pn - 1) if _pt_h else None
                        _pt = getattr(_pt_rec, "page_type", "") if _pt_rec else ""
                        _pt_boost = 1.2 if _pt == "visual-dominant" else (1.05 if _pt == "mixed" else 1.0)
                        _boosted.append((_src, _pn, _rrf_score * _pt_boost))
                    _page_list = sorted(_boosted, key=lambda x: -x[2])

                # Topic-sibling expansion: inject cross-document pages sharing a topic.
                # Disabled: token-overlap matching is too loose for thematically related
                # corpora (all docs share words like "performance", "metrics", "2025"),
                # causing irrelevant docs to dominate top-k slots. The existing doc-level
                # BM25 expansion already handles cross-doc discovery via docdesc hits.
                # _page_list = _expand_with_topic_siblings(
                #     _page_list, rag_config.top_k, self.hindex_store, self.logger
                # )

                # Per-doc cap: prevent a single document from filling all top-k slots.
                # Guarantees cross-document diversity in cross_doc queries where entity
                # BM25 can lock onto one document's entity matches exclusively.
                # Cap = max(1, top_k // 2) so each doc holds at most half the slots.
                _final_cap = max(1, rag_config.top_k // 2)
                _capped_list: list[tuple[str, int, float]] = []
                _doc_counts: dict[str, int] = {}
                for _s, _p, _sc in _page_list:
                    if _doc_counts.get(_s, 0) < _final_cap:
                        _capped_list.append((_s, _p, _sc))
                        _doc_counts[_s] = _doc_counts.get(_s, 0) + 1
                    if len(_capped_list) >= rag_config.top_k * 2:
                        break
                _page_list = _capped_list

                docs = _page_hits_to_docs(_page_list[: rag_config.top_k], self.hindex_store, self.kvstore, rag_config.top_k)
                _tc_hits, _tc_prompt = _hindex_text_context_for_pages(
                    [(_s, _p) for _s, _p, _ in _page_list[: rag_config.top_k]],
                    self.hindex_store,
                    rag_config.text_search_engine_max_chars_per_doc,
                    rag_config.text_search_engine_max_total_chars,
                )
                docs["text_context"] = _tc_hits
                docs["text_context_prompt"] = _tc_prompt
                self.logger.debug(
                    "[vision_hybrid_bm25] intent=%s bm25=%d queryhint=%d text_embed=%d visual_embed=%d entity=%d for '%s'",
                    _intent,
                    len(_page_bm25_hits), len(_queryhint_page_list),
                    len(_text_embed_page_list), len(_embed_page_list),
                    len(_entity_page_list), rag_question[:80],
                )

        elif _mode == "vision_topic_embed":
            if self.rag_topic_embed_collection is None:
                self.logger.warning("[vision_topic_embed] mm_topic_db not initialized; index with this mode first.")
            else:
                _n_topics = min(rag_config.top_k, max(self.rag_topic_embed_collection.count() or 1, 1))
                _results = self.rag_topic_embed_collection.query(
                    query_texts=[rag_question],
                    n_results=_n_topics,
                    include=["metadatas", "distances"],
                )
                _hits = []
                _seen: set[tuple[str, int]] = set()
                for _meta in _results["metadatas"][0]:
                    _src = _meta.get("source", "")
                    _pidx_str = _meta.get("page_indices", "")
                    if not _src or not _pidx_str:
                        continue
                    for _idx_s in _pidx_str.split(","):
                        try:
                            _pidx = int(_idx_s.strip())
                            _pn = _pidx + 1
                            _key = (_src, _pn)
                            if _key not in _seen:
                                _seen.add(_key)
                                _hits.append((_src, _pn, 0.0))
                        except ValueError:
                            pass
                docs = _page_hits_to_docs(_hits[: rag_config.top_k], self.hindex_store, self.kvstore, rag_config.top_k)
                _tc_hits, _tc_prompt = _hindex_text_context_for_pages(
                    [(_s, _p) for _s, _p, _ in _hits[: rag_config.top_k]],
                    self.hindex_store,
                    rag_config.text_search_engine_max_chars_per_doc,
                    rag_config.text_search_engine_max_total_chars,
                )
                docs["text_context"] = _tc_hits
                docs["text_context_prompt"] = _tc_prompt
                self.logger.debug(
                    "[vision_topic_embed] %d topics → %d unique pages for '%s'",
                    len(_results["metadatas"][0]), len(_hits), rag_question[:80],
                )

        elif _mode == "vision_bm25_rerank":
            # BM25 first pass + dense text-embedding pass → RRF merge → page images
            _bm25_page_list: list[tuple[str, int, float]] = []
            _embed_page_list: list[tuple[str, int, float]] = []
            if self.text_search_engine_index is not None:
                _bm25_raw = self.text_search_engine_index.search(
                    rag_question,
                    limit=rag_config.top_k * 8,
                    fields=["content"],
                    crop_label=None,
                )
                _bm25_raw_pages = [h for h in _bm25_raw if h.get("crop_label") != "docdesc"]
                _bm25_page_list = _bm25_hits_to_page_list(_bm25_raw_pages)
            if self.rag_text_embed_collection is not None:
                _n_embed = min(rag_config.top_k * 4, self.rag_text_embed_collection.count() or 1)
                _emb_res = self.rag_text_embed_collection.query(
                    query_texts=[rag_question],
                    n_results=_n_embed,
                    include=["metadatas", "distances"],
                )
                for _meta in _emb_res["metadatas"][0]:
                    _src = _meta.get("source", "")
                    _pn = _meta.get("page_number")
                    if _src and _pn is not None:
                        _embed_page_list.append((_src, int(_pn), 0.0))
            _bm25_keys = [(_s, _p) for _s, _p, _ in _bm25_page_list]
            _embed_keys = [(_s, _p) for _s, _p, _ in _embed_page_list]
            _rrf_hits = _reciprocal_rank_fusion([_bm25_keys, _embed_keys])
            docs = _page_hits_to_docs(_rrf_hits[: rag_config.top_k], self.hindex_store, self.kvstore, rag_config.top_k)
            _tc_hits, _tc_prompt = _hindex_text_context_for_pages(
                [(_s, _p) for _s, _p, _ in _rrf_hits[: rag_config.top_k]],
                self.hindex_store,
                rag_config.text_search_engine_max_chars_per_doc,
                rag_config.text_search_engine_max_total_chars,
            )
            docs["text_context"] = _tc_hits
            docs["text_context_prompt"] = _tc_prompt
            self.logger.debug(
                "[vision_bm25_rerank] bm25=%d embed=%d rrf=%d for '%s'",
                len(_bm25_page_list), len(_embed_page_list), len(_rrf_hits), rag_question[:80],
            )

        return docs

    def _get_effective_rag_config(self, ad: InputConnectorObj) -> RAGObj:
        return merge_rag_config(self.ad.rag, ad.rag)

    def _get_effective_rag_config_for_request(self, request_rag: RAGObj | None) -> RAGObj:
        return merge_rag_config(self.ad.rag, request_rag)

    def _build_hindex_for_doc(
        self,
        document: dict,
        rag_config: RAGObj,
        preserve_existing_hindex: bool = False,
        page_texts: dict[int, str] | None = None,
    ) -> list[dict]:
        """Build and persist the hierarchy for one document.

        Must be called AFTER image_processor.preprocess_images() so that
        document["parts"] is populated with crop keys.

        Returns VLM tantivy document dicts (one per page) so the caller can
        accumulate them for vision_semantic indexing.

        preserve_existing_hindex: when True, always reuse existing hindex files
        even if rag_reindex=True.  Used by vision_semantic, where reindex only
        controls the Tantivy index, not the hindex files.
        """
        doc_name = str(document["source"])
        if self.hindex_store.exists(doc_name) and (not self.rag_reindex or preserve_existing_hindex):
            self.logger.info(f"[hindex] Hierarchy already exists for '{doc_name}', skipping")
            existing = self.hindex_store.load(doc_name)
            return _hierarchy_to_vlm_tantivy_docs(existing) if existing else []
        if self.vlm_fn is None:
            self.logger.warning(
                f"[hindex] vlm_fn not set — cannot build hierarchy for '{doc_name}'. "
                "Set it via RAGImg.vlm_fn after the LLM model is initialized."
            )
            return []

        # Compute per-page kvstore keys (full-page images)
        uuid5_val = document.get("uuid5")
        if uuid5_val is None:
            uuid5_val = str(_uuid5(NAMESPACE_URL, str(document["source"])))

        page_count = len(document["images"])
        page_keys = [_page_key(uuid5_val, i) for i in range(page_count)]

        # Detect scanned (image-only) pages: PDF embedded text is too sparse to use.
        scanned_page_indices: set[int] = set()
        if page_texts is not None:
            for page_idx in range(page_count):
                pn = page_idx + 1  # _extract_page_texts keys are 1-indexed
                if len(page_texts.get(pn, "").strip()) < _SCANNED_TEXT_THRESHOLD:
                    scanned_page_indices.add(page_idx)
            if scanned_page_indices:
                self.logger.info(
                    "[hindex] Detected %d scanned page(s) in '%s' — using OCR-focused VLM prompt",
                    len(scanned_page_indices), doc_name,
                )

        # Detect document language from OCR text so the VLM can be instructed
        # explicitly (prevents defaulting to English for international docs).
        _doc_language: str | None = None
        if page_texts:
            try:
                from langdetect import detect as _langdetect
                _LANG_NAMES = {
                    "fr": "French", "en": "English", "de": "German", "es": "Spanish",
                    "it": "Italian", "nl": "Dutch", "pt": "Portuguese", "ja": "Japanese",
                    "zh-cn": "Chinese", "zh-tw": "Chinese", "ar": "Arabic",
                    "ko": "Korean", "ru": "Russian", "pl": "Polish",
                }
                _lang_sample = " ".join(
                    page_texts.get(pn, "").strip()
                    for pn in range(1, min(6, page_count + 1))
                    if len(page_texts.get(pn, "").strip()) > 20
                )
                if len(_lang_sample) > 50:
                    _detected = _langdetect(_lang_sample)
                    _doc_language = _LANG_NAMES.get(_detected)
                    if _doc_language:
                        self.logger.info("[hindex] Detected document language: %s", _doc_language)
            except Exception as _e:
                self.logger.debug("[hindex] Language detection failed: %s", _e)

        # Build the hierarchy (Phase 1 + 2 happen inside builder.build())
        page_images = [img_dict["image"] for img_dict in document["images"]]
        builder = HierarchyBuilder(
            vlm_fn=self.vlm_fn,
            logger=self.logger,
            max_workers=getattr(self, "hindex_max_workers", 1),
            batch_size=getattr(self, "hindex_batch_size", 4),
        )
        hierarchy = builder.build(
            doc_name, page_images, page_keys=page_keys,
            scanned_page_indices=scanned_page_indices or None,
            doc_language=_doc_language,
        )

        # Phase 3 — Populate crop_keys on each PageRecord from document["parts"].
        # parts[*]["metadata"]["page_number"] is 1-indexed, so subtract 1.
        from collections import defaultdict
        crops_by_page: dict[int, list[str]] = defaultdict(list)
        for part in document.get("parts", []):
            page_num_1idx = part["metadata"].get("page_number", 1)
            page_idx = page_num_1idx - 1
            crops_by_page[page_idx].append(
                part["name"].encode("utf-8", "replace").decode()
            )
        for record in hierarchy.page_records:
            record.crop_keys = crops_by_page.get(record.page_idx, [])

        self.hindex_store.save(hierarchy)
        self.logger.info(
            f"[hindex] Hierarchy saved for '{doc_name}' "
            f"({len(hierarchy.topics)} topics, {page_count} pages, "
            f"{sum(len(r.crop_keys) for r in hierarchy.page_records)} crops)"
        )

        # Re-embed crops with VLM-extracted text so that text queries match by
        # semantic meaning (summary / keywords / OCR) rather than visual appearance.
        if self.rag_indexdb_collection is not None:
            self._enrich_embeddings_for_doc(doc_name, hierarchy)

        return _hierarchy_to_vlm_tantivy_docs(hierarchy)

    def _enrich_embeddings_for_doc(
        self, doc_name: str, hierarchy: "DocumentHierarchy"
    ) -> None:
        """Re-embed all crops of *doc_name* with VLM-extracted page descriptions.

        Each crop's ChromaDB embedding is updated so that it encodes the semantic
        content of its parent page (summary + keywords + OCR text +
        visual_description) rather than just the raw pixel appearance.  After this
        step, ``query_texts=[question]`` finds crops by meaning, not by how they
        look.

        Must be called after the hindex hierarchy is built and saved, so that
        ``hierarchy.page_records`` is populated.
        """
        result = self.rag_indexdb_collection.get(
            where={"source": {"$eq": doc_name}},
            include=["metadatas"],
        )
        crop_ids: list[str] = result["ids"]
        metas: list[dict] = result["metadatas"]

        if not crop_ids:
            self.logger.info(
                "[hindex] No crops in ChromaDB for '%s'; skipping embedding enrichment",
                doc_name,
            )
            return

        self.logger.info(
            "[hindex] Enriching embeddings for %d crop(s) of '%s'",
            len(crop_ids),
            doc_name,
        )

        records_by_page = {r.page_idx: r for r in hierarchy.page_records}

        _BATCH = 8  # keep GPU memory pressure low during upsert
        n_done = 0
        for start in range(0, len(crop_ids), _BATCH):
            batch_ids = crop_ids[start : start + _BATCH]
            batch_metas = metas[start : start + _BATCH]

            image_dicts: list[dict] = []
            for crop_id, meta in zip(batch_ids, batch_metas):
                try:
                    img = self.kvstore.retrieve_image(crop_id)
                except Exception as exc:
                    self.logger.warning(
                        "[hindex] Cannot load crop '%s' for enrichment: %s", crop_id, exc
                    )
                    img = Image.new("RGB", (28, 28))

                # page_number in ChromaDB metadata is 1-indexed
                page_idx = meta.get("page_number", 1) - 1
                rec = records_by_page.get(page_idx)

                parts: list[str] = []
                if rec:
                    if rec.summary:
                        parts.append(rec.summary)
                    if rec.text_content:
                        parts.append(rec.text_content[:300])
                    if rec.keywords:
                        parts.append(", ".join(rec.keywords))
                    if rec.visual_description:
                        parts.append(rec.visual_description)

                image_dicts.append({
                    "doc": img,
                    "label": meta.get("crop_label"),
                    "text": " | ".join(parts),
                })

            self.rag_indexdb_collection.upsert(
                images=image_dicts,
                ids=batch_ids,
                metadatas=batch_metas,
            )
            n_done += len(batch_ids)

        self.logger.info(
            "[hindex] Embedding enrichment done: %d/%d crops updated for '%s'",
            n_done,
            len(crop_ids),
            doc_name,
        )

    def _extract_page_labels(self, source: Path | str) -> dict[int, str]:
        """Return physical-page-index → printed-label map from PDF metadata."""
        source_path = Path(source)
        if source_path.suffix.lower() != ".pdf":
            return {}
        label_map: dict[int, str] = {}
        with fitz.open(source_path) as pdf_doc:
            for i in range(pdf_doc.page_count):
                label = pdf_doc[i].get_label()
                label_map[i] = label if label else str(i + 1)
        return label_map

    def _extract_page_texts(self, source: Path | str) -> dict[int, str]:
        source_path = Path(source)
        if source_path.suffix.lower() != ".pdf":
            return {}

        page_texts: dict[int, str] = {}
        with fitz.open(source_path) as pdf_doc:
            for page_idx in range(pdf_doc.page_count):
                page_texts[page_idx + 1] = pdf_doc.load_page(page_idx).get_text("text")

        return page_texts

    def delete_embedder(self):
        if self.rag_embf:
            if self.rag_embf.vllm:
                cache_key = ("vllm_embed", self.rag_embf.rag_embedding_model)
            else:
                cache_key = ("huggingface", self.rag_embf.rag_embedding_model)
            if self.rag_embf.shared:
                ModelCache.acquire_lock(cache_key)
            if not self.rag_embf.shared or not ModelCache.is_in_use(cache_key, 1):
                del self.rag_embf.model
                self.rag_embf.model = None

            if self.rag_embf.shared:
                ModelCache.release_lock(cache_key)
                ModelCache.release(cache_key)

            self.rag_embf = None

            import gc

            gc.collect()
            torch.cuda.empty_cache()
