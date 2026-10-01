#!/usr/bin/env python3
"""Compare two eval result snapshots for vision_hybrid_bm25.

Usage:
    python examples/compare_rounds.py                          # R3 vs current
    python examples/compare_rounds.py --r3 examples/new_vision_modes_eval_results_round4.json  # R4 vs current
    python examples/compare_rounds.py --show-fails             # only changed/failing questions
    python examples/compare_rounds.py --type cross_doc         # filter by question_type
    python examples/compare_rounds.py --show-zero              # show still-zero questions

Named round backups:
    new_vision_modes_eval_results_round3.json   — R3: unlimited queryhint, 0.664
    new_vision_modes_eval_results_round4.json   — R4: 1/doc cap, 0.571
    new_vision_modes_eval_results_round5.json   — R5: 2/doc cap, 0.626
    new_vision_modes_eval_results_round6.json   — R6: 50% threshold, no cap, 0.669
    new_vision_modes_eval_results_round7.json   — R7: 50% threshold + 2/doc cap, 0.631
    new_vision_modes_eval_results_round8.json   — R8: multi-doc-aware cap (=R7), 0.631
    new_vision_modes_eval_results_round9.json   — R9: 50% threshold + 3/doc cap, 0.663
    new_vision_modes_eval_results_round10.json  — R10: 50% threshold + 4/doc cap, 0.673
    new_vision_modes_eval_results_round11.json  — R11: AQ improvements + anticipated_queries in text embed
    new_vision_modes_eval_results_round12.json  — R12: RRF k=30
    new_vision_modes_eval_results_round13.json  — R13: RRF k=120
    new_vision_modes_eval_results_round14.json  — R14: visual_threshold=0.5
    new_vision_modes_eval_results_round15.json  — R15: visual_threshold=0.8
    new_vision_modes_eval_results.json          — current run (updated after each eval)
"""
import argparse
import json
import sys
from pathlib import Path

colette_root = Path(__file__).resolve().parent.parent
BENCHMARK_DIR = colette_root / "examples" / "benchmark"
TOP_K = 4


def build_relevant_set(item: dict) -> set:
    rel = set()
    docs = item.get("relevant_docs", [])
    pages = item.get("relevant_pages", [])
    if not docs or not pages:
        return rel
    if len(docs) == 1:
        for page in pages:
            rel.add((docs[0], int(page)))
    else:
        for doc, page in zip(docs, pages):
            rel.add((doc, int(page)))
    return rel


def _retrieved_pages(result: dict) -> list:
    pages = []
    for h in result.get("image_hits", []):
        doc = h.get("doc", "")
        page = h.get("page")
        if doc and page is not None:
            pages.append((doc, int(page)))
    for h in result.get("text_hits", []):
        doc = h.get("doc", "")
        page = h.get("page")
        if doc and page is not None:
            pages.append((doc, int(page)))
    seen, deduped = set(), []
    for p in pages:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped[:TOP_K]


def compute_ndcg(result: dict, relevant: set) -> float:
    retrieved = _retrieved_pages(result)
    k = TOP_K
    hits = retrieved[:k]

    def dcg(ranked):
        return sum(1.0 / (i + 1) for i, p in enumerate(ranked) if p in relevant)

    ideal = min(len(relevant), k)
    return dcg(hits) / dcg(list(relevant)[:ideal]) if ideal else 0.0


def short_doc(doc: str) -> str:
    return doc.replace("VTQ-", "").replace(".pdf", "")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--r3", default="examples/new_vision_modes_eval_results_round10.json",
                        help="Baseline results file (default: R10)")
    parser.add_argument("--r4", default="examples/new_vision_modes_eval_results.json",
                        help="Round 4 results file (written at eval end)")
    parser.add_argument("--r4-partial", default="app_vision_hybrid_bm25_eval/eval_results.json",
                        help="Round 4 partial results (incremental, during eval)")
    parser.add_argument("--show-fails", action="store_true",
                        help="Show only questions that changed or are still failing")
    parser.add_argument("--show-zero", action="store_true",
                        help="Show all questions still at NDCG=0 in the current round")
    parser.add_argument("--type", dest="qtype", default=None,
                        help="Filter by question_type")
    args = parser.parse_args()

    # Load QA dataset
    qa_items = []
    with open(BENCHMARK_DIR / "qa_dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                qa_items.append(json.loads(line))
    eval_items = [q for q in qa_items if q.get("relevant_docs")]
    print(f"QA items: {len(eval_items)}")

    # Load round 3
    r3_path = colette_root / args.r3
    if not r3_path.exists():
        print(f"ERROR: {r3_path} not found")
        sys.exit(1)
    r3_raw = json.loads(r3_path.read_text())
    r3 = {k.split("|", 1)[1]: v for k, v in r3_raw.items() if k.startswith("vision_hybrid_bm25|")}
    print(f"Round 3 entries: {len(r3)}")

    # Load round 4 — prefer partial (incremental) file during eval; use final only
    # when it has been updated (its mtime > partial mtime, or no partial exists).
    r4_path = colette_root / args.r4
    r4_partial_path = colette_root / args.r4_partial
    r4 = {}
    using_partial = False

    partial_exists = r4_partial_path.exists()
    final_exists = r4_path.exists()
    use_partial = partial_exists

    if partial_exists and final_exists:
        import os
        partial_mtime = os.path.getmtime(r4_partial_path)
        final_mtime = os.path.getmtime(r4_path)
        # Use final only if it was written AFTER the partial (eval completed)
        if final_mtime > partial_mtime:
            use_partial = False

    if use_partial and partial_exists:
        r4_raw = json.loads(r4_partial_path.read_text())
        r4 = r4_raw  # partial uses {qid: result} format directly
        using_partial = True
        print(f"Round 4 entries: {len(r4)} (from partial — eval still running)")
    elif final_exists:
        r4_raw = json.loads(r4_path.read_text())
        r4 = {k.split("|", 1)[1]: v for k, v in r4_raw.items() if k.startswith("vision_hybrid_bm25|")}
        print(f"Round 4 entries: {len(r4)} (from final file)")

    if not r4:
        print("ERROR: No round 4 results found. Eval may not have started yet.")
        sys.exit(1)

    # Compute per-question NDCG
    rows = []
    for item in eval_items:
        qid = item["id"]
        qtype = item.get("question_type", "?")
        question = item.get("question", "")[:60]
        relevant = build_relevant_set(item)
        if not relevant:
            continue
        if args.qtype and qtype != args.qtype:
            continue

        r3_result = r3.get(qid)
        r4_result = r4.get(qid)

        r3_ndcg = compute_ndcg(r3_result, relevant) if r3_result else None
        r4_ndcg = compute_ndcg(r4_result, relevant) if r4_result else None

        r3_pages = _retrieved_pages(r3_result) if r3_result else []
        r4_pages = _retrieved_pages(r4_result) if r4_result else []

        rows.append({
            "qid": qid,
            "qtype": qtype,
            "question": question,
            "relevant": relevant,
            "r3_ndcg": r3_ndcg,
            "r4_ndcg": r4_ndcg,
            "r3_pages": r3_pages,
            "r4_pages": r4_pages,
        })

    # Summary by type
    print()
    print("=" * 70)
    print(f"NDCG@4 by question_type  (R3 vs R4{'  partial' if using_partial else ''})")
    print("=" * 70)

    types = sorted(set(r["qtype"] for r in rows))
    type_stats = {}
    for qtype in types:
        type_rows = [r for r in rows if r["qtype"] == qtype]
        r3_scores = [r["r3_ndcg"] for r in type_rows if r["r3_ndcg"] is not None]
        r4_scores = [r["r4_ndcg"] for r in type_rows if r["r4_ndcg"] is not None]
        r3_avg = sum(r3_scores) / len(r3_scores) if r3_scores else 0.0
        r4_avg = sum(r4_scores) / len(r4_scores) if r4_scores else 0.0
        delta = r4_avg - r3_avg
        n = len(type_rows)
        print(f"  {qtype:<22s}  R3={r3_avg:.3f}  R4={r4_avg:.3f}  Δ={delta:+.3f}  (n={n})")
        type_stats[qtype] = {"r3": r3_avg, "r4": r4_avg, "delta": delta, "n": n}

    all_r3 = [r["r3_ndcg"] for r in rows if r["r3_ndcg"] is not None]
    all_r4 = [r["r4_ndcg"] for r in rows if r["r4_ndcg"] is not None]
    r3_overall = sum(all_r3) / len(all_r3) if all_r3 else 0.0
    r4_overall = sum(all_r4) / len(all_r4) if all_r4 else 0.0
    print(f"  {'OVERALL':<22s}  R3={r3_overall:.3f}  R4={r4_overall:.3f}  Δ={r4_overall-r3_overall:+.3f}  (n={len(all_r4)})")

    # Per-question diff — show changed questions
    improvements = [r for r in rows if r["r3_ndcg"] is not None and r["r4_ndcg"] is not None and r["r4_ndcg"] > r["r3_ndcg"] + 0.01]
    regressions  = [r for r in rows if r["r3_ndcg"] is not None and r["r4_ndcg"] is not None and r["r4_ndcg"] < r["r3_ndcg"] - 0.01]
    still_zero   = [r for r in rows if r["r4_ndcg"] is not None and r["r4_ndcg"] < 0.01]

    def fmt_pages(pages, relevant):
        parts = []
        for doc, pg in pages:
            marker = "✓" if (doc, pg) in relevant else " "
            parts.append(f"{marker}{short_doc(doc)}p{pg}")
        return "  ".join(parts)

    if improvements:
        print()
        print(f"IMPROVEMENTS ({len(improvements)} questions):")
        for r in sorted(improvements, key=lambda x: x["r4_ndcg"] - x["r3_ndcg"], reverse=True):
            delta = r["r4_ndcg"] - r["r3_ndcg"]
            print(f"  {r['qid']} [{r['qtype'][:12]}]  {r['r3_ndcg']:.2f}→{r['r4_ndcg']:.2f} (+{delta:.2f})")
            print(f"    Q: {r['question']}")
            if not args.show_fails:
                print(f"    R3: {fmt_pages(r['r3_pages'], r['relevant'])}")
                print(f"    R4: {fmt_pages(r['r4_pages'], r['relevant'])}")

    if regressions:
        print()
        print(f"REGRESSIONS ({len(regressions)} questions):")
        for r in sorted(regressions, key=lambda x: x["r3_ndcg"] - x["r4_ndcg"], reverse=True):
            delta = r["r4_ndcg"] - r["r3_ndcg"]
            print(f"  {r['qid']} [{r['qtype'][:12]}]  {r['r3_ndcg']:.2f}→{r['r4_ndcg']:.2f} ({delta:.2f})")
            print(f"    Q: {r['question']}")
            print(f"    R3: {fmt_pages(r['r3_pages'], r['relevant'])}")
            print(f"    R4: {fmt_pages(r['r4_pages'], r['relevant'])}")

    if (args.show_fails or args.show_zero) and still_zero:
        print()
        print(f"STILL ZERO in current round ({len(still_zero)} questions):")
        for r in sorted(still_zero, key=lambda x: (x["qtype"], x["qid"])):
            r3_str = f"R3={r['r3_ndcg']:.2f}" if r["r3_ndcg"] is not None else "R3=n/a"
            print(f"  {r['qid']} [{r['qtype'][:12]}]  {r3_str}")
            print(f"    Q: {r['question']}")
            print(f"    R4: {fmt_pages(r['r4_pages'], r['relevant'])}")


if __name__ == "__main__":
    main()
