"""Unit tests for vision_hybrid_bm25 retrieval components.

Covers the three core helpers that drive retrieval quality in
``vision_hybrid_bm25`` mode and encode hard-won research decisions:

- ``_filter_queryhint_hits`` — score-ratio threshold + per-doc cap
- ``_hindex_page_text``      — text embedded per page (includes anticipated_queries)
- ``_reciprocal_rank_fusion``— RRF merge of multiple ranked lists

Each section documents the *intended behaviour* so that future changes that
break it show up as test failures rather than silent NDCG regressions.

Quality-regression section
--------------------------
``test_quality_regression_*`` tests assert NDCG@4 floors across question types
using saved benchmark results.  They are marked ``benchmark`` and skip
gracefully when the results file is absent (i.e. in plain CI without a GPU).
Run them after an eval cycle to catch logic regressions:

    pytest -m benchmark tests/test_vision_hybrid_bm25.py

To understand what's failing and why:

    python examples/compare_rounds.py --show-fails
    python examples/compare_rounds.py --show-zero --type visual_only
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pytest

from colette.backends.hf.hindex.hierarchy_builder import (
    DocumentHierarchy,
    PageRecord,
    TopicNode,
)
from colette.backends.hf.rag.rag_img import (
    _bm25_hits_to_page_list,
    _filter_queryhint_hits,
    _hierarchy_to_entity_tantivy_docs,
    _hindex_page_text,
    _normalize_query_for_bm25,
    _reciprocal_rank_fusion,
)

# ---------------------------------------------------------------------------
# Helpers shared across sections
# ---------------------------------------------------------------------------

_BENCH_RESULTS = Path("examples/new_vision_modes_eval_results_round16.json")
_BENCH_QA = Path("examples/benchmark/qa_dataset.jsonl")


def _make_rec(**kw) -> PageRecord:
    defaults = dict(
        page_idx=0,
        page_key="key_0",
        text_content="",
        visual_description="",
        summary="",
        keywords=[],
        crop_keys=[],
        anticipated_queries=[],
    )
    defaults.update(kw)
    return PageRecord(**defaults)


def _inv(score: float) -> float:
    """BM25 score → inv_dist (as stored in page lists)."""
    return 1.0 / (1.0 + score)


def _score(inv_dist: float) -> float:
    """inv_dist → original BM25 score."""
    return 1.0 / inv_dist - 1.0


# ---------------------------------------------------------------------------
# _filter_queryhint_hits
# ---------------------------------------------------------------------------
# Design contract (rounds 3-10):
#  - score_ratio_threshold=0.5 means a page must score >= 50% of the best
#    page for its document to earn a queryhint RRF vote.
#  - per_doc_cap=top_k prevents any one document from monopolising all slots.
#  - Both filters operate independently per source document.
# ---------------------------------------------------------------------------


class TestFilterQueryhintThreshold:
    """Score-ratio threshold: keep pages that score >= threshold * best_score."""

    def test_best_page_always_included(self):
        hits = [("a.pdf", 1, _inv(10.0))]
        result = _filter_queryhint_hits(hits)
        assert len(result) == 1
        assert result[0][:2] == ("a.pdf", 1)

    def test_page_above_threshold_included(self):
        # best=10.0, page=6.0 → 60% > 50% → included
        hits = [("a.pdf", 1, _inv(10.0)), ("a.pdf", 2, _inv(6.0))]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.5)
        pages = [p for _, p, _ in result]
        assert 1 in pages and 2 in pages

    def test_page_below_threshold_excluded(self):
        # best=10.0, page=4.0 → 40% < 50% → excluded
        hits = [("a.pdf", 1, _inv(10.0)), ("a.pdf", 2, _inv(4.0))]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.5)
        pages = [p for _, p, _ in result]
        assert 1 in pages
        assert 2 not in pages

    def test_page_exactly_at_threshold_included(self):
        # best=10.0, page=5.0 → exactly 50% → included (>= not >)
        hits = [("a.pdf", 1, _inv(10.0)), ("a.pdf", 2, _inv(5.0))]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.5)
        pages = [p for _, p, _ in result]
        assert 2 in pages

    def test_stricter_threshold_excludes_more(self):
        # 70% threshold: page at 65% is excluded even though 50% would keep it
        hits = [("a.pdf", 1, _inv(10.0)), ("a.pdf", 2, _inv(6.5))]
        at_50 = _filter_queryhint_hits(hits, score_ratio_threshold=0.5)
        at_70 = _filter_queryhint_hits(hits, score_ratio_threshold=0.70)
        assert len(at_50) == 2
        assert len(at_70) == 1

    def test_zero_threshold_keeps_all_pages(self):
        hits = [("a.pdf", i, _inv(float(10 - i))) for i in range(1, 6)]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.0, per_doc_cap=10)
        assert len(result) == 5

    def test_threshold_is_per_doc_not_global(self):
        # doc_a best=10, doc_b best=2; a page at score=3 passes for doc_b (150%)
        # but would fail for doc_a (30% < 50%)
        hits = [
            ("a.pdf", 1, _inv(10.0)),
            ("a.pdf", 2, _inv(3.0)),   # 30% of doc_a best → excluded
            ("b.pdf", 1, _inv(2.0)),
            ("b.pdf", 2, _inv(3.0)),   # 150% of doc_b (but it's best!); wait...
        ]
        # Actually b.pdf page 2 has higher score (3.0) than page 1 (2.0)
        # _qh_all is sorted desc by score so we must provide in order
        hits_sorted = [
            ("a.pdf", 1, _inv(10.0)),
            ("b.pdf", 2, _inv(3.0)),   # best for b
            ("a.pdf", 2, _inv(3.0)),
            ("b.pdf", 1, _inv(2.0)),   # 67% of b best → included at 50%
        ]
        result = _filter_queryhint_hits(hits_sorted, score_ratio_threshold=0.5, per_doc_cap=10)
        a_pages = [p for s, p, _ in result if s == "a.pdf"]
        b_pages = [p for s, p, _ in result if s == "b.pdf"]
        assert 1 in a_pages
        assert 2 not in a_pages   # 3/10 = 30% < 50%
        assert 2 in b_pages
        assert 1 in b_pages       # 2/3 = 67% > 50%


class TestFilterQueryhintCap:
    """Per-doc cap: hard upper limit regardless of score."""

    def test_cap_limits_output_per_doc(self):
        # 7 pages all with high scores, cap=4
        hits = [("doc.pdf", i, _inv(10.0 - i * 0.1)) for i in range(1, 8)]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.0, per_doc_cap=4)
        assert len(result) == 4

    def test_cap_does_not_cut_below_limit(self):
        hits = [("doc.pdf", i, _inv(10.0)) for i in range(1, 4)]  # 3 pages
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.0, per_doc_cap=4)
        assert len(result) == 3  # cap=4 but only 3 pages available

    def test_cap_applies_per_doc_independently(self):
        hits = [
            ("a.pdf", 1, _inv(10.0)), ("a.pdf", 2, _inv(9.0)),
            ("a.pdf", 3, _inv(8.0)), ("a.pdf", 4, _inv(7.0)),  # 4 pages for a
            ("b.pdf", 1, _inv(10.0)), ("b.pdf", 2, _inv(9.0)),  # 2 pages for b
        ]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.0, per_doc_cap=2)
        a_count = sum(1 for s, _, _ in result if s == "a.pdf")
        b_count = sum(1 for s, _, _ in result if s == "b.pdf")
        assert a_count == 2  # capped
        assert b_count == 2  # at limit naturally

    def test_cap_one_behaves_like_one_per_doc(self):
        hits = [("a.pdf", i, _inv(10.0 - i)) for i in range(1, 5)]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.0, per_doc_cap=1)
        a_pages = [p for s, p, _ in result if s == "a.pdf"]
        assert a_pages == [1]  # only top page

    def test_threshold_and_cap_both_apply(self):
        # 5 pages for a.pdf: best=10, rest=[6,5,4,3]
        # threshold=0.5: 6,5 pass (60%,50%); 4,3 excluded (40%,30%)
        # cap=3 would allow 3, but threshold only admits 3 (best=10, p2=6, p3=5)
        hits = [
            ("a.pdf", 1, _inv(10.0)),
            ("a.pdf", 2, _inv(6.0)),
            ("a.pdf", 3, _inv(5.0)),
            ("a.pdf", 4, _inv(4.0)),  # 40% → excluded by threshold
            ("a.pdf", 5, _inv(3.0)),  # 30% → excluded by threshold
        ]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.5, per_doc_cap=3)
        pages = [p for _, p, _ in result]
        assert sorted(pages) == [1, 2, 3]

    def test_cap_cut_wins_over_threshold(self):
        # All 5 pages pass threshold but cap=2
        hits = [("a.pdf", i, _inv(10.0)) for i in range(1, 6)]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.0, per_doc_cap=2)
        assert len(result) == 2
        assert result[0][1] == 1  # top page first
        assert result[1][1] == 2


class TestFilterQueryhintEdgeCases:
    """Boundary conditions for _filter_queryhint_hits."""

    def test_empty_input_returns_empty(self):
        assert _filter_queryhint_hits([]) == []

    def test_single_doc_single_page(self):
        hits = [("a.pdf", 1, _inv(5.0))]
        result = _filter_queryhint_hits(hits)
        assert result == hits

    def test_multiple_docs_processed_independently(self):
        # Threshold passes 1 page from each doc
        hits = [
            ("a.pdf", 1, _inv(10.0)),
            ("a.pdf", 2, _inv(2.0)),   # 20% of 10 → excluded
            ("b.pdf", 3, _inv(8.0)),
            ("b.pdf", 4, _inv(3.0)),   # 37.5% of 8 → excluded
        ]
        result = _filter_queryhint_hits(hits, score_ratio_threshold=0.5, per_doc_cap=4)
        sources = [(s, p) for s, p, _ in result]
        assert ("a.pdf", 1) in sources
        assert ("a.pdf", 2) not in sources
        assert ("b.pdf", 3) in sources
        assert ("b.pdf", 4) not in sources

    def test_output_preserves_inv_dist_values(self):
        hits = [("a.pdf", 1, _inv(9.0)), ("a.pdf", 2, _inv(5.0))]
        result = _filter_queryhint_hits(hits)
        assert abs(result[0][2] - _inv(9.0)) < 1e-9
        assert abs(result[1][2] - _inv(5.0)) < 1e-9

    def test_very_high_cap_acts_like_unlimited(self):
        hits = [("a.pdf", i, _inv(10.0)) for i in range(1, 8)]
        unlimited = _filter_queryhint_hits(hits, score_ratio_threshold=0.0, per_doc_cap=100)
        assert len(unlimited) == 7


# ---------------------------------------------------------------------------
# _hindex_page_text
# ---------------------------------------------------------------------------
# Design contract:
#  - Combines summary | keywords | text_content[:800] | visual_description
#    | anticipated_queries for dense text embedding.
#  - anticipated_queries bridges the vocabulary gap: embedding over anticipated
#    queries means semantic matching finds pages whose *expected questions*
#    resemble the user query even when BM25 keywords don't overlap.
# ---------------------------------------------------------------------------


class TestHindexPageText:

    def test_combines_all_fields(self):
        rec = _make_rec(
            summary="Revenue summary",
            keywords=["revenue", "growth"],
            text_content="Revenue grew 12%",
            visual_description="Bar chart of revenue",
            anticipated_queries=["How much did revenue grow?", "Q3 revenue number"],
        )
        text = _hindex_page_text(rec)
        assert "Revenue summary" in text
        assert "revenue" in text
        assert "Revenue grew 12%" in text
        assert "Bar chart of revenue" in text
        assert "How much did revenue grow?" in text
        assert "Q3 revenue number" in text

    def test_anticipated_queries_included(self):
        rec = _make_rec(anticipated_queries=["What is the P@4 score?", "best precision result"])
        text = _hindex_page_text(rec)
        assert "What is the P@4 score?" in text
        assert "best precision result" in text

    def test_empty_anticipated_queries_not_included(self):
        rec = _make_rec(summary="A summary", anticipated_queries=[])
        text = _hindex_page_text(rec)
        assert "A summary" in text
        # No extra separator from empty anticipated_queries
        assert text.count(" | ") == 0  # only one field → no separator

    def test_none_anticipated_queries_not_included(self):
        rec = _make_rec(summary="A summary")
        rec.anticipated_queries = None  # type: ignore[assignment]
        text = _hindex_page_text(rec)
        assert "A summary" in text  # no crash

    def test_empty_record_returns_empty_string(self):
        rec = _make_rec()
        assert _hindex_page_text(rec) == ""

    def test_text_content_truncated_at_800_chars(self):
        rec = _make_rec(text_content="x" * 2000)
        text = _hindex_page_text(rec)
        assert len(text) <= 820  # 800 + some separator overhead

    def test_separator_between_fields(self):
        rec = _make_rec(summary="Summary", visual_description="Chart")
        text = _hindex_page_text(rec)
        assert " | " in text

    def test_anticipated_queries_joined_with_space(self):
        rec = _make_rec(anticipated_queries=["query one", "query two"])
        text = _hindex_page_text(rec)
        assert "query one query two" in text


# ---------------------------------------------------------------------------
# _reciprocal_rank_fusion — gate and signal behaviour
# ---------------------------------------------------------------------------
# Design contract:
#  - Score = sum(1 / (k + rank + 1)) over all lists where the item appears.
#  - Default k=60; larger k dampens rank sensitivity.
#  - Items appearing in more lists score higher (multi-signal consensus).
# ---------------------------------------------------------------------------


class TestRRFProperties:

    def test_higher_rank_gives_higher_score(self):
        lst = [("doc.pdf", i) for i in range(1, 6)]
        result = _reciprocal_rank_fusion([lst])
        scores = [sc for _, _, sc in result]
        assert scores == sorted(scores, reverse=True)

    def test_item_in_two_lists_beats_item_in_one(self):
        lst_a = [("a.pdf", 1), ("b.pdf", 2)]
        lst_b = [("b.pdf", 2), ("a.pdf", 1)]
        result = _reciprocal_rank_fusion([lst_a, lst_b])
        score_map = {(s, p): sc for s, p, sc in result}
        # Both appear twice, but ("a.pdf", 1) is rank-1 in lst_a and rank-2 in lst_b
        # ("b.pdf", 2) is rank-2 in lst_a and rank-1 in lst_b → equal; both beat any
        # item in only one list
        assert score_map[("a.pdf", 1)] > 1.0 / (60 + 1 + 1)  # beats single-list rank-2
        assert score_map[("b.pdf", 2)] > 1.0 / (60 + 1 + 1)

    def test_no_duplicates_in_output(self):
        lst_a = [("doc.pdf", 1), ("doc.pdf", 2)]
        lst_b = [("doc.pdf", 1), ("doc.pdf", 3)]
        result = _reciprocal_rank_fusion([lst_a, lst_b])
        pages = [(s, p) for s, p, _ in result]
        assert len(pages) == len(set(pages))

    def test_single_empty_list_returns_empty(self):
        assert _reciprocal_rank_fusion([[]]) == []

    def test_k_parameter_changes_score_magnitude(self):
        lst = [("doc.pdf", 1)]
        r30 = _reciprocal_rank_fusion([lst], k=30)
        r60 = _reciprocal_rank_fusion([lst], k=60)
        assert r30[0][2] > r60[0][2]  # smaller k → higher score at same rank

    def test_visual_participates_in_stage1_rrf(self):
        """Visual embedding is a full RRF participant alongside text signals.
        A page that ranks high on visual but low on text should still surface
        via the 5-signal RRF — it cannot be suppressed by text-only Stage 1."""
        text_ranked = [("doc.pdf", 1), ("doc.pdf", 2), ("doc.pdf", 3)]
        # Page 3 ranks highest visually even though it is last on text
        visual_ranked = [("doc.pdf", 3), ("doc.pdf", 1), ("doc.pdf", 2)]

        merged = _reciprocal_rank_fusion([text_ranked, visual_ranked])
        scores = {p: sc for _, p, sc in merged}

        # Page 3: rank 2 on text (rrf ≈ 1/32) + rank 0 on visual (rrf ≈ 1/31) → boosted
        # Page 1: rank 0 on text + rank 1 on visual
        # With visual as a full signal, page 3 should score higher than it would
        # with text-only RRF (where it would be last).
        text_only = _reciprocal_rank_fusion([text_ranked])
        text_scores = {p: sc for _, p, sc in text_only}
        # Page 3's rank is improved by the visual signal
        rank_text_only = sorted(text_scores, key=text_scores.get, reverse=True).index(3)
        rank_combined = sorted(scores, key=scores.get, reverse=True).index(3)
        assert rank_combined < rank_text_only  # page 3 ranks higher with visual


# ---------------------------------------------------------------------------
# _hierarchy_to_entity_tantivy_docs — entity BM25 indexing
# ---------------------------------------------------------------------------


class TestEntityTantivyDocs:

    def _make_hierarchy(self, records: list[PageRecord]) -> DocumentHierarchy:
        return DocumentHierarchy(
            doc_name="test.pdf",
            doc_description="test document",
            page_count=len(records),
            page_records=records,
        )

    def test_page_with_entities_produces_one_doc(self):
        rec = _make_rec(page_idx=0, entities={
            "persons": ["John Smith (sender)"],
            "dates": ["1954-01-15"],
        })
        hierarchy = self._make_hierarchy([rec])
        docs = _hierarchy_to_entity_tantivy_docs(hierarchy)
        assert len(docs) == 1
        assert docs[0]["crop_label"] == "entity"
        assert docs[0]["page_number"] == 1
        assert "John Smith (sender)" in docs[0]["content"]
        assert "1954-01-15" in docs[0]["content"]

    def test_page_without_entities_produces_no_doc(self):
        rec = _make_rec(page_idx=0, entities={})
        hierarchy = self._make_hierarchy([rec])
        docs = _hierarchy_to_entity_tantivy_docs(hierarchy)
        assert docs == []

    def test_page_with_empty_entity_lists_produces_no_doc(self):
        rec = _make_rec(page_idx=0, entities={"persons": [], "dates": []})
        hierarchy = self._make_hierarchy([rec])
        docs = _hierarchy_to_entity_tantivy_docs(hierarchy)
        assert docs == []

    def test_all_entity_categories_concatenated(self):
        rec = _make_rec(page_idx=0, entities={
            "persons": ["Alice"],
            "organizations": ["ACME Corp"],
            "dates": ["2024-03-01"],
            "identifiers": ["REF-4821"],
            "amounts": ["$5,000"],
        })
        hierarchy = self._make_hierarchy([rec])
        docs = _hierarchy_to_entity_tantivy_docs(hierarchy)
        assert len(docs) == 1
        content = docs[0]["content"]
        assert "Alice" in content
        assert "ACME Corp" in content
        assert "REF-4821" in content
        assert "$5,000" in content

    def test_page_without_entities_attr_produces_no_doc(self):
        """Backward compat: existing hindex files may lack the entities field."""
        rec = _make_rec(page_idx=0)  # no entities kwarg → default empty dict
        hierarchy = self._make_hierarchy([rec])
        docs = _hierarchy_to_entity_tantivy_docs(hierarchy)
        assert docs == []

    def test_multiple_pages_independent(self):
        recs = [
            _make_rec(page_idx=0, entities={"identifiers": ["INV-001"]}),
            _make_rec(page_idx=1, entities={}),
            _make_rec(page_idx=2, entities={"persons": ["Bob"]}),
        ]
        hierarchy = self._make_hierarchy(recs)
        docs = _hierarchy_to_entity_tantivy_docs(hierarchy)
        assert len(docs) == 2  # pages 0 and 2 only
        page_numbers = {d["page_number"] for d in docs}
        assert page_numbers == {1, 3}  # 1-based

    def test_entity_doc_id_is_unique(self):
        recs = [
            _make_rec(page_idx=0, entities={"persons": ["Alice"]}),
            _make_rec(page_idx=1, entities={"persons": ["Bob"]}),
        ]
        hierarchy = self._make_hierarchy(recs)
        docs = _hierarchy_to_entity_tantivy_docs(hierarchy)
        ids = [d["doc_id"] for d in docs]
        assert len(ids) == len(set(ids))


# ---------------------------------------------------------------------------
# PageRecord entity round-trip
# ---------------------------------------------------------------------------


class TestPageRecordEntities:

    def test_entities_round_trip(self):
        entities = {
            "persons": ["John Smith (sender)"],
            "organizations": ["Philip Morris"],
            "dates": ["1954-01-15"],
            "identifiers": ["REQ-4821"],
            "amounts": ["$5,000"],
        }
        rec = _make_rec(page_idx=0, entities=entities)
        d = rec.to_dict()
        rec2 = PageRecord.from_dict(d)
        assert rec2.entities == entities

    def test_missing_entities_defaults_to_empty_dict(self):
        d = {
            "page_idx": 0, "page_key": "k", "text_content": "",
            "visual_description": "", "summary": "", "keywords": [],
            "crop_keys": [], "anticipated_queries": [],
            # no "entities" key — simulates old hindex files
        }
        rec = PageRecord.from_dict(d)
        assert rec.entities == {}

    def test_entities_in_to_dict(self):
        rec = _make_rec(page_idx=0, entities={"dates": ["2024-01-01"]})
        assert "entities" in rec.to_dict()


# ---------------------------------------------------------------------------
# _normalize_query_for_bm25 — LaTeX stripping
# ---------------------------------------------------------------------------


class TestNormalizeQueryForBm25:

    def test_plain_query_unchanged(self):
        q = "What is the emission rate in 2023?"
        assert _normalize_query_for_bm25(q) == q

    def test_inline_math_unwrapped(self):
        q = r"What is the value of \( P_f \) in the figure?"
        result = _normalize_query_for_bm25(q)
        assert r"\(" not in result
        assert r"\)" not in result
        assert "P_f" in result

    def test_backslash_command_expanded(self):
        q = r"At what value of \( \delta \) does convergence occur?"
        result = _normalize_query_for_bm25(q)
        assert "delta" in result
        assert r"\delta" not in result

    def test_display_math_removed(self):
        q = r"Interpret the formula \[ E = mc^2 \] shown in the slide."
        result = _normalize_query_for_bm25(q)
        assert r"\[" not in result

    def test_empty_query_unchanged(self):
        assert _normalize_query_for_bm25("") == ""

    def test_whitespace_collapsed(self):
        q = r"What   is  \( k \)  here?"
        result = _normalize_query_for_bm25(q)
        assert "  " not in result


# ---------------------------------------------------------------------------
# _bm25_hits_to_page_list — existing contract regression
# ---------------------------------------------------------------------------


class TestBm25HitsToPageList:

    def test_deduplicates_same_page_first_wins(self):
        # First occurrence wins (Tantivy returns hits sorted descending, so first = best)
        hits = [
            {"source": "doc.pdf", "page_number": 2, "score": 8.0},
            {"source": "doc.pdf", "page_number": 2, "score": 5.0},
        ]
        result = _bm25_hits_to_page_list(hits)
        assert len(result) == 1
        # inv_dist of first-seen score (8.0): 1/(1+8)
        assert abs(result[0][2] - 1.0 / 9.0) < 1e-6

    def test_skips_missing_source(self):
        hits = [{"source": "", "page_number": 1, "score": 5.0}]
        assert _bm25_hits_to_page_list(hits) == []

    def test_skips_missing_page(self):
        hits = [{"source": "doc.pdf", "page_number": None, "score": 5.0}]
        assert _bm25_hits_to_page_list(hits) == []

    def test_inv_dist_inversely_proportional_to_score(self):
        # Higher BM25 score → smaller inv_dist (more relevant)
        hits = [
            {"source": "doc.pdf", "page_number": 1, "score": 9.0},
            {"source": "doc.pdf", "page_number": 2, "score": 3.0},
        ]
        result = _bm25_hits_to_page_list(hits)
        p1_inv = next(d for _, p, d in result if p == 1)
        p2_inv = next(d for _, p, d in result if p == 2)
        assert p1_inv < p2_inv  # page 1 (score 9) has smaller inv_dist → more relevant


# ---------------------------------------------------------------------------
# NDCG@4 computation used throughout the eval pipeline
# ---------------------------------------------------------------------------


def _ndcg_at_k(retrieved: list[tuple], relevant: set, k: int = 4) -> float:
    """Reference NDCG@k — mirrors the benchmark evaluation logic."""
    hits = retrieved[:k]

    def dcg(lst):
        return sum(1.0 / (i + 1) for i, p in enumerate(lst) if p in relevant)

    ideal = min(len(relevant), k)
    return dcg(hits) / dcg(list(relevant)[:ideal]) if ideal else 0.0


class TestNdcgAtK:

    def test_perfect_retrieval_is_one(self):
        rel = {("a.pdf", 1), ("a.pdf", 2)}
        retrieved = [("a.pdf", 1), ("a.pdf", 2), ("b.pdf", 1), ("b.pdf", 2)]
        assert _ndcg_at_k(retrieved, rel) == pytest.approx(1.0)

    def test_no_hits_is_zero(self):
        rel = {("a.pdf", 1)}
        retrieved = [("b.pdf", 1), ("b.pdf", 2)]
        assert _ndcg_at_k(retrieved, rel) == pytest.approx(0.0)

    def test_rank_matters(self):
        rel = {("a.pdf", 1)}
        first = [("a.pdf", 1), ("b.pdf", 2)]
        second = [("b.pdf", 2), ("a.pdf", 1)]
        assert _ndcg_at_k(first, rel) > _ndcg_at_k(second, rel)

    def test_empty_relevant_is_zero(self):
        retrieved = [("a.pdf", 1)]
        assert _ndcg_at_k(retrieved, set()) == 0.0

    def test_partial_credit(self):
        rel = {("a.pdf", 1), ("a.pdf", 2)}
        retrieved = [("a.pdf", 1), ("b.pdf", 1), ("b.pdf", 2), ("b.pdf", 3)]
        score = _ndcg_at_k(retrieved, rel, k=4)
        assert 0.0 < score < 1.0

    def test_k_limits_window(self):
        rel = {("a.pdf", 5)}
        retrieved = [("b.pdf", i) for i in range(1, 5)] + [("a.pdf", 5)]
        assert _ndcg_at_k(retrieved, rel, k=4) == pytest.approx(0.0)
        assert _ndcg_at_k(retrieved, rel, k=5) > 0.0


# ---------------------------------------------------------------------------
# Quality-regression gate (requires saved benchmark results)
# ---------------------------------------------------------------------------
# These tests assert NDCG@4 floors from R12 results.  They serve as a
# regression gate: if pipeline logic changes cause a type's NDCG to drop
# below these floors, the test fails and investigation is needed.
#
# Floors are set conservatively below R12 actual values so that minor
# shuffles in LLM answers don't trigger false alarms.
#
# How to read a failure:
#   test_quality_*_visual_only fails  → chart page ranking regressed
#   test_quality_*_exact_lookup fails → BM25/queryhint keyword coverage dropped
#   test_quality_*_cross_doc fails    → same-doc clustering got worse
#   test_quality_*_table_cell fails   → queryhint filter too strict or too loose
#
# How to diagnose:
#   python examples/compare_rounds.py --show-fails --type visual_only
#   python examples/compare_rounds.py --show-zero
# ---------------------------------------------------------------------------

_FLOORS: dict[str, float] = {
    "overall":               0.760,  # R16=0.784 (cross_page AQ fixes)
    "cross_doc":             0.45,   # R16=0.521 — most volatile type (only 8 questions)
    "cross_page":            0.80,   # R16=0.848 — recovered from R11-R15 regression via AQ fixes
    "exact_lookup":          0.68,   # R16=0.706 (slight regression vs R12=0.750; AQ cross-contamination)
    "semantic_paraphrase":   0.58,   # R16=0.630
    "table_cell":            0.74,   # R16=0.786
    "visual_only":           0.84,   # R16=0.865
}

_COUNTS: dict[str, int] = {
    "cross_doc": 8, "cross_page": 3, "exact_lookup": 19,
    "semantic_paraphrase": 3, "table_cell": 35, "visual_only": 47,
}


def _load_benchmark_data():
    """Return (results_dict, qa_items) or skip the test."""
    if not _BENCH_RESULTS.exists() or not _BENCH_QA.exists():
        pytest.skip("Benchmark results not available — run eval first")
    raw = json.loads(_BENCH_RESULTS.read_text())
    # Combined files use "mode|qid" keys; per-mode round files use bare "qid" keys.
    if any(k.startswith("vision_hybrid_bm25|") for k in raw):
        results = {k.split("|", 1)[1]: v for k, v in raw.items()
                   if k.startswith("vision_hybrid_bm25|")}
    else:
        results = raw
    qa = [json.loads(l) for l in _BENCH_QA.read_text().splitlines() if l.strip()]
    qa = [q for q in qa if q.get("relevant_docs")]
    return results, qa


def _build_rel(q: dict[str, Any]) -> set[tuple[str, int]]:
    docs, pages = q.get("relevant_docs", []), q.get("relevant_pages", [])
    if not docs or not pages:
        return set()
    if len(docs) == 1:
        return {(docs[0], int(p)) for p in pages}
    return {(d, int(p)) for d, p in zip(docs, pages)}


def _retrieved_pages(result: dict) -> list[tuple[str, int]]:
    pages: list[tuple[str, int]] = []
    for h in result.get("image_hits", []) + result.get("text_hits", []):
        d, p = h.get("doc", ""), h.get("page")
        if d and p is not None:
            pages.append((d, int(p)))
    seen: list[tuple[str, int]] = []
    for x in pages:
        if x not in seen:
            seen.append(x)
    return seen[:4]


def _compute_per_type_ndcg(results: dict, qa: list) -> dict[str, float]:
    scores: dict[str, list[float]] = {t: [] for t in _COUNTS}
    for q in qa:
        r = results.get(q["id"])
        if not r:
            continue
        rel = _build_rel(q)
        if not rel:
            continue
        t = q.get("question_type", "")
        if t in scores:
            scores[t].append(_ndcg_at_k(_retrieved_pages(r), rel))
    avgs = {t: (sum(v) / len(v) if v else 0.0) for t, v in scores.items()}
    total = sum(avgs[t] * _COUNTS[t] for t in _COUNTS) / sum(_COUNTS.values())
    return {**avgs, "overall": total}


@pytest.mark.benchmark
class TestQualityRegression:
    """NDCG@4 regression gates against saved R16 benchmark results (k=30, cross_page AQ fixes).

    Requires ``examples/new_vision_modes_eval_results_round16.json``.
    Skip gracefully in plain CI.  Run after eval with ``-m benchmark``.
    """

    @pytest.fixture(scope="class")
    def per_type_ndcg(self):
        results, qa = _load_benchmark_data()
        return _compute_per_type_ndcg(results, qa)

    def test_overall_floor(self, per_type_ndcg):
        actual = per_type_ndcg["overall"]
        assert actual >= _FLOORS["overall"], (
            f"Overall NDCG@4 {actual:.3f} below floor {_FLOORS['overall']:.3f}.\n"
            "Run: python examples/compare_rounds.py --show-fails"
        )

    def test_table_cell_floor(self, per_type_ndcg):
        actual = per_type_ndcg["table_cell"]
        assert actual >= _FLOORS["table_cell"], (
            f"table_cell {actual:.3f} < {_FLOORS['table_cell']:.3f}.\n"
            "Likely cause: queryhint filter change — check _filter_queryhint_hits.\n"
            "Run: python examples/compare_rounds.py --type table_cell --show-fails"
        )

    def test_semantic_paraphrase_floor(self, per_type_ndcg):
        actual = per_type_ndcg["semantic_paraphrase"]
        assert actual >= _FLOORS["semantic_paraphrase"], (
            f"semantic_paraphrase {actual:.3f} < {_FLOORS['semantic_paraphrase']:.3f}.\n"
            "Likely cause: text embed or queryhint BM25 change.\n"
            "Run: python examples/compare_rounds.py --type semantic_paraphrase --show-fails"
        )

    def test_exact_lookup_floor(self, per_type_ndcg):
        actual = per_type_ndcg["exact_lookup"]
        assert actual >= _FLOORS["exact_lookup"], (
            f"exact_lookup {actual:.3f} < {_FLOORS['exact_lookup']:.3f}.\n"
            "Likely cause: vocabulary gap in anticipated_queries or BM25 config.\n"
            "Embed fix: ensure anticipated_queries included in _hindex_page_text.\n"
            "Run: python examples/compare_rounds.py --type exact_lookup --show-fails"
        )

    def test_visual_only_floor(self, per_type_ndcg):
        actual = per_type_ndcg["visual_only"]
        assert actual >= _FLOORS["visual_only"], (
            f"visual_only {actual:.3f} < {_FLOORS['visual_only']:.3f}.\n"
            "Likely cause: visual embed gate threshold or chart page descriptions.\n"
            "Persistent failures (need hindex regen): q030, q031, q059, q061, q020, q085, q086.\n"
            "Run: python examples/compare_rounds.py --type visual_only --show-zero"
        )

    def test_cross_doc_floor(self, per_type_ndcg):
        actual = per_type_ndcg["cross_doc"]
        assert actual >= _FLOORS["cross_doc"], (
            f"cross_doc {actual:.3f} < {_FLOORS['cross_doc']:.3f}.\n"
            "Likely cause: per_doc_cap or score_ratio_threshold too permissive.\n"
            "Trade-off: tighter cap helps cross_doc but hurts table_cell.\n"
            "Run: python examples/compare_rounds.py --type cross_doc --show-fails"
        )

    def test_cross_page_floor(self, per_type_ndcg):
        actual = per_type_ndcg["cross_page"]
        assert actual >= _FLOORS["cross_page"], (
            f"cross_page {actual:.3f} < {_FLOORS['cross_page']:.3f}.\n"
            "Run: python examples/compare_rounds.py --type cross_page --show-fails"
        )

    def test_score_summary(self, per_type_ndcg):
        """Print a diagnostic summary — always passes, useful with -s."""
        lines = ["\n=== NDCG@4 Quality Report ==="]
        for t in ["overall"] + list(_COUNTS):
            actual = per_type_ndcg.get(t, 0.0)
            floor = _FLOORS[t]
            gap = actual - floor
            status = "✓" if actual >= floor else "✗"
            lines.append(f"  {status} {t:<24} actual={actual:.3f}  floor={floor:.3f}  margin={gap:+.3f}")
        print("\n".join(lines))
        assert True  # diagnostic only
