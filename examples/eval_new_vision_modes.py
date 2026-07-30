"""Evaluate the 4 new vision retrieval modes on the synthetic 7-doc benchmark.

Modes evaluated (one run per mode, reusing existing hindex files):
  - vision_text_embed   : dense ANN on VLM page descriptions
  - vision_hybrid_bm25  : merged BM25 over OCR text + VLM descriptions
  - vision_topic_embed  : dense ANN on topic descriptions → page images
  - vision_bm25_rerank  : BM25 first pass + dense embed RRF

All modes reuse the hindex JSON files from app_eval_benchmark so no VLM calls
are needed at index time.

GPU allocation (per project rule):
    GPU 0 = embedding model (used here for text embedding)
    GPU 2 = LLM for indexing/eval
    GPU 3 = DO NOT USE
"""

import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

colette_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(colette_root / "src"))

EXISTING_APP_DIR = colette_root / "app_eval_benchmark"
BENCHMARK_DIR    = colette_root / "examples" / "benchmark"
TOP_K            = 4
LLM_GPU_IDS      = [2]

from colette.jsonapi import JSONApi
from colette.apidata import APIData

colette_api = JSONApi()


ALL_MODES = ["vision_text_embed", "vision_hybrid_bm25", "vision_topic_embed", "vision_bm25_rerank"]


def _app_dir(mode: str) -> Path:
    return colette_root / f"app_{mode}_eval"


def _copy_hindex_files(mode: str) -> int:
    """Copy per-doc hindex JSON files from app_eval_benchmark."""
    src = EXISTING_APP_DIR / "mm_index" / "hindex"
    dst = _app_dir(mode) / "mm_index" / "hindex"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.glob("*.json"):
        if f.name == "_collection_index.json":
            continue
        shutil.copy2(f, dst / f.name)
        n += 1
    print(f"  Copied {n} hindex files → {dst}")
    return n


def _copy_kvstore(mode: str):
    src = EXISTING_APP_DIR / "kvstore.db"
    if src.exists():
        dst = _app_dir(mode) / "kvstore.db"
        shutil.copy2(src, dst)
        print(f"  Copied kvstore.db ({src.stat().st_size // 1024}KB)")
    else:
        print("  WARNING: kvstore.db not found in app_eval_benchmark")


def run_query(app_name: str, mode: str, question: str) -> dict:
    payload = {"parameters": {"input": {
        "message": question,
        "rag": {"retrieval_mode": mode, "top_k": TOP_K},
    }}}
    t0 = time.time()
    response = colette_api.service_predict(app_name, APIData(**payload))
    elapsed = round(time.time() - t0, 2)

    sources = response.sources or {}
    image_hits = []
    for item in sources.get("context", []):
        doc = Path(item.get("source", "")).name or ""
        image_hits.append({
            "doc":        doc,
            "page":       item.get("page_number"),
            "distance":   item.get("distance"),
            "crop_label": item.get("crop_label", ""),
        })

    text_hits = []
    for item in sources.get("text_context", []):
        doc = Path(item.get("source", "")).name
        text_hits.append({
            "doc":     doc,
            "page":    item.get("page_number"),
            "score":   item.get("score"),
            "content": item.get("content", "")[:200],
        })

    return {
        "image_hits": image_hits,
        "text_hits":  text_hits,
        "output":     response.output or "",
        "latency_s":  elapsed,
    }


def build_relevant_set(item: dict) -> set:
    rel = set()
    docs  = item.get("relevant_docs", [])
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
        doc  = h.get("doc", "")
        page = h.get("page")
        if doc and page is not None:
            pages.append((doc, int(page)))
    for h in result.get("text_hits", []):
        doc  = h.get("doc", "")
        page = h.get("page")
        if doc and page is not None:
            pages.append((doc, int(page)))
    seen, deduped = set(), []
    for p in pages:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped[:TOP_K]


def compute_metrics(result: dict, relevant: set) -> dict:
    retrieved = _retrieved_pages(result)
    k = TOP_K

    hits_at_k = retrieved[:k]
    hit_k  = int(any(p in relevant for p in hits_at_k))
    p_k    = sum(1 for p in hits_at_k if p in relevant) / k if k else 0.0
    r_k    = sum(1 for p in hits_at_k if p in relevant) / len(relevant) if relevant else 0.0

    def dcg(ranked):
        return sum(1.0 / (i + 1) for i, p in enumerate(ranked) if p in relevant)

    ideal   = min(len(relevant), k)
    ndcg_k  = dcg(hits_at_k) / dcg(list(relevant)[:ideal]) if ideal else 0.0

    mrr = 0.0
    for rank, p in enumerate(retrieved, 1):
        if p in relevant:
            mrr = 1.0 / rank
            break

    return dict(hit_k=hit_k, p_k=p_k, r_k=r_k, ndcg_k=ndcg_k, mrr=mrr,
                ctx_size=len(retrieved))


async def eval_mode(mode: str, eval_items: list) -> dict:
    """Set up, index, query and return {qid: result} for one mode."""
    app_name = f"{mode.replace('_', '')}_eval"
    app_dir  = _app_dir(mode)

    print(f"\n{'='*65}")
    print(f"MODE: {mode}  (app={app_name})")
    print(f"{'='*65}")

    # ── app directory: reuse if index already built, otherwise start fresh
    partial_results = app_dir / "eval_results.json"
    resuming = partial_results.exists()
    if not resuming:
        if app_dir.exists():
            shutil.rmtree(app_dir)
        app_dir.mkdir(parents=True)
        print(f"Created: {app_dir}")
        n_hindex = _copy_hindex_files(mode)
        if n_hindex == 0:
            print("ERROR: no hindex files found. Run setup_eval_service.ipynb first.")
            return {}
        _copy_kvstore(mode)
    else:
        print(f"Resuming existing app dir: {app_dir}")

    # ── create service (reuse existing config, override mode + gpu)
    saved_cfg = json.loads((EXISTING_APP_DIR / "config.json").read_text())
    saved_cfg["parameters"]["llm"]["gpu_ids"] = LLM_GPU_IDS
    saved_cfg["parameters"]["input"]["rag"]["retrieval_mode"] = mode
    saved_cfg["app"]["repository"] = str(app_dir)
    # HyDE at query time: the hypothetical answer is generated by the LLM and used as
    # the retrieval query for both BM25 and visual embedding — bridges the vocabulary gap.
    if mode == "vision_hybrid_bm25":
        saved_cfg["parameters"]["llm"]["use_hyde"] = True

    try:
        await colette_api.service_delete(app_name)
    except Exception:
        pass
    colette_api.service_create(app_name, APIData(**saved_cfg))
    print(f"Service '{app_name}' created.")

    # ── index (build embed/BM25 collections from existing hindex files)
    if not resuming:
        print(f"Indexing…")
        index_cfg = {"parameters": {"input": {
            "data": [str(BENCHMARK_DIR)],
            "preprocessing": {"files": ["pdf"]},
            "rag": {
                "retrieval_mode": mode,
                "reindex": True,
                "update_index": False,
            },
        }}}
        t0 = time.time()
        colette_api.service_index(app_name, APIData(**index_cfg))
        print(f"Indexing done in {time.time() - t0:.1f}s")
    else:
        print("Skipping index (resuming from partial results)")

    # ── run queries (resume from partial save if present)
    out = app_dir / "eval_results.json"
    results: dict[str, dict] = {}
    if out.exists():
        results = json.loads(out.read_text())
        print(f"Resuming from {len(results)} saved results")

    for i, item in enumerate(eval_items, 1):
        qid = item["id"]
        if qid in results:
            print(f"[{i}/{len(eval_items)}] {qid}  (cached)")
            continue
        q = item["question"]
        print(f"[{i}/{len(eval_items)}] {qid}  {q[:55]}…", end=" ", flush=True)
        try:
            r = run_query(app_name, mode, q)
            results[qid] = r
            img = len(r["image_hits"])
            txt = len(r["text_hits"])
            print(f"✓ {r['latency_s']:.1f}s  img={img}  txt={txt}")
        except Exception as exc:
            print(f"ERROR: {exc}")
        # save after every query so a restart can resume
        out.write_text(json.dumps(results, default=str))

    print(f"Saved {len(results)} results → {out}")
    return results


async def main():
    parser = argparse.ArgumentParser(description="Evaluate new vision retrieval modes")
    parser.add_argument(
        "--modes", nargs="+", choices=ALL_MODES, default=ALL_MODES,
        help="Which modes to evaluate (default: all 4)",
    )
    args = parser.parse_args()

    if not EXISTING_APP_DIR.exists():
        print(f"ERROR: {EXISTING_APP_DIR} not found. Run setup_eval_service.ipynb first.")
        return

    # ── load QA items
    QA_ITEMS = []
    with open(BENCHMARK_DIR / "qa_dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                QA_ITEMS.append(json.loads(line))
    eval_items = [q for q in QA_ITEMS if q.get("relevant_docs")]
    print(f"Loaded {len(eval_items)} evaluable QA items\n")

    # ── run each mode
    all_results: dict[str, dict] = {}
    for mode in args.modes:
        all_results[mode] = await eval_mode(mode, eval_items)

    # ── load cached baseline results
    RESULTS_FILE = EXISTING_APP_DIR / "eval_results.json"
    cached: dict = {}
    if RESULTS_FILE.exists():
        raw = json.loads(RESULTS_FILE.read_text())
        cached = {tuple(k.split("|", 1)): v for k, v in raw.items()}
        print(f"\nLoaded {len(cached)} cached baseline results")

    COMPARE_METHODS = [
        {"name": "embedding",      "label": "Embedding"},
        {"name": "hybrid_hindex",  "label": "Hybrid+hindex"},
        {"name": "embed_hindex",   "label": "Embed+hindex"},
        {"name": "hybrid",         "label": "Hybrid"},
        {"name": "vision_hier",    "label": "hindex only"},
    ]
    VS_RESULTS_FILE = colette_root / "app_vision_semantic_eval" / "eval_results.json"
    vision_semantic_results: dict = {}
    if VS_RESULTS_FILE.exists():
        vision_semantic_results = json.loads(VS_RESULTS_FILE.read_text())

    # ── compute metrics
    try:
        import pandas as pd
    except ImportError:
        print("pandas not installed; skipping table output")
        return

    rows = []
    for item in eval_items:
        qid      = item["id"]
        qtype    = item["question_type"]
        diff     = item.get("difficulty", "")
        relevant = build_relevant_set(item)
        if not relevant:
            continue

        for mode, results in all_results.items():
            if qid in results:
                m = compute_metrics(results[qid], relevant)
                rows.append({"method": mode, **m, "question_type": qtype, "difficulty": diff})

        if qid in vision_semantic_results:
            m = compute_metrics(vision_semantic_results[qid], relevant)
            rows.append({"method": "vision_semantic", **m, "question_type": qtype, "difficulty": diff})

        for cm in COMPARE_METHODS:
            result = cached.get((cm["name"], qid))
            if result is None:
                continue
            m = compute_metrics(result, relevant)
            rows.append({"method": cm["label"], **m, "question_type": qtype, "difficulty": diff})

    df = pd.DataFrame(rows)
    summary = (
        df.groupby("method")[["hit_k", "ndcg_k", "mrr", "ctx_size"]]
        .mean()
        .rename(columns={
            "hit_k":    f"Hit@{TOP_K}",
            "ndcg_k":   f"NDCG@{TOP_K}",
            "mrr":      "MRR",
            "ctx_size": "avg_ctx",
        })
        .round(3)
        .sort_values(f"NDCG@{TOP_K}", ascending=False)
    )

    print("\n" + "=" * 65)
    print("RETRIEVAL COMPARISON — synthetic 7-doc benchmark")
    print("=" * 65)
    print(summary.to_string())
    print()

    # ── per question_type breakdown
    print("\n=== NDCG@4 by question_type ===")
    pivot = (
        df.groupby(["method", "question_type"])["ndcg_k"]
        .mean()
        .unstack(fill_value=0.0)
        .round(3)
    )
    print(pivot.to_string())

    # ── save combined results
    combined_out = colette_root / "examples" / "new_vision_modes_eval_results.json"
    combined = {}
    for mode, results in all_results.items():
        for qid, r in results.items():
            combined[f"{mode}|{qid}"] = r
    combined_out.write_text(json.dumps(combined, indent=2, default=str))
    print(f"\nSaved combined results → {combined_out}")


if __name__ == "__main__":
    asyncio.run(main())
