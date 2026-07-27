"""Evaluate vision_semantic retrieval on the synthetic 7-doc benchmark.

Compares against the cached results for all other retrieval methods stored in
app_eval_benchmark/eval_results.json.

How vision_semantic indexing works here
---------------------------------------
1. Copies existing hindex JSON files from app_eval_benchmark so no VLM calls
   are needed at index time (page descriptions are already on disk).
2. Runs service_index with vision_semantic + reindex=True to build a clean
   Tantivy index from those descriptions (BM25 over summary / text_content /
   visual_description / keywords, crop_label='pagedesc').
3. Because preserve_existing_hindex=True is used internally, hindex files are
   never rebuilt even when reindex=True.

GPU allocation (per project rule):
    GPU 0 = embedding model (not used here at retrieval time)
    GPU 1 = LLM (needed to load service; not called during retrieval)
    GPU 3 = DO NOT USE
"""

import asyncio
import json
import shutil
import sys
import time
from pathlib import Path

colette_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(colette_root / "src"))

EXISTING_APP_DIR = colette_root / "app_eval_benchmark"
NEW_APP_DIR      = colette_root / "app_vision_semantic_eval"
BENCHMARK_DIR    = colette_root / "examples" / "benchmark"
APP_NAME         = "vs_eval"
TOP_K            = 4
LLM_GPU_IDS      = [1]

from colette.jsonapi import JSONApi
from colette.apidata import APIData

colette_api = JSONApi()


def _copy_hindex_files():
    """Copy per-doc hindex JSON files (skip _collection_index.json)."""
    src = EXISTING_APP_DIR / "mm_index" / "hindex"
    dst = NEW_APP_DIR / "mm_index" / "hindex"
    dst.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in src.glob("*.json"):
        if f.name == "_collection_index.json":
            continue
        shutil.copy2(f, dst / f.name)
        n += 1
    print(f"Copied {n} hindex files → {dst}")
    return n


def run_query(question: str) -> dict:
    payload = {"parameters": {"input": {
        "message": question,
        "rag": {"retrieval_mode": "vision_semantic", "top_k": TOP_K},
    }}}
    t0 = time.time()
    response = colette_api.service_predict(APP_NAME, APIData(**payload))
    elapsed = round(time.time() - t0, 2)

    sources = response.sources or {}
    image_hits = []
    for item in sources.get("context", []):
        doc = Path(item.get("source", "")).name or ""
        image_hits.append({
            "doc":      doc,
            "page":     item.get("page_number"),
            "distance": item.get("distance"),
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
    for rd in item.get("relevant_docs", []):
        doc  = Path(rd["doc"]).name
        page = rd.get("page")
        if page is not None:
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


async def main():
    # ── 0. Fresh app directory ───────────────────────────────────────────────
    if NEW_APP_DIR.exists():
        shutil.rmtree(NEW_APP_DIR)
    NEW_APP_DIR.mkdir(parents=True)
    print(f"Created fresh app dir: {NEW_APP_DIR}")
    n_hindex = _copy_hindex_files()
    if n_hindex == 0:
        print("ERROR: no hindex files found in app_eval_benchmark. "
              "Run setup_eval_service.ipynb first.")
        return

    # ── 1. Create service ────────────────────────────────────────────────────
    saved_cfg = json.loads((EXISTING_APP_DIR / "config.json").read_text())
    saved_cfg["parameters"]["llm"]["gpu_ids"] = LLM_GPU_IDS
    saved_cfg["parameters"]["input"]["rag"]["retrieval_mode"] = "vision_semantic"
    saved_cfg["app"]["repository"] = str(NEW_APP_DIR)

    try:
        await colette_api.service_delete(APP_NAME)
    except Exception:
        pass
    colette_api.service_create(APP_NAME, APIData(**saved_cfg))
    print(f"Service '{APP_NAME}' created.\n")

    # ── 2. Index: build tantivy from existing hindex descriptions ────────────
    print("Indexing with vision_semantic (reading existing hindex files)…")
    index_cfg = {"parameters": {"input": {
        "data": [str(BENCHMARK_DIR)],
        "preprocessing": {"files": ["pdf"]},
        "rag": {
            "retrieval_mode": "vision_semantic",
            "reindex": True,
            "update_index": False,
        },
    }}}
    t0 = time.time()
    colette_api.service_index(APP_NAME, APIData(**index_cfg))
    print(f"Indexing done in {time.time() - t0:.1f}s\n")

    # ── 3. Load QA items ─────────────────────────────────────────────────────
    QA_ITEMS = []
    with open(BENCHMARK_DIR / "qa_dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                QA_ITEMS.append(json.loads(line))
    eval_items = [q for q in QA_ITEMS if q.get("relevant_docs")]
    print(f"Loaded {len(eval_items)} evaluable QA items\n")

    # ── 4. Run vision_semantic queries ────────────────────────────────────────
    vs_results: dict[str, dict] = {}
    for i, item in enumerate(eval_items, 1):
        qid = item["id"]
        q   = item["question"]
        print(f"[{i}/{len(eval_items)}] {qid}  {q[:60]}…", end=" ", flush=True)
        try:
            r = run_query(q)
            vs_results[qid] = r
            img = len(r["image_hits"])
            txt = len(r["text_hits"])
            print(f"✓ {r['latency_s']:.1f}s  img={img}  txt={txt}")
        except Exception as exc:
            print(f"ERROR: {exc}")

    # ── 5. Load cached results for comparison ─────────────────────────────────
    RESULTS_FILE = EXISTING_APP_DIR / "eval_results.json"
    ALL_CACHED: dict = {}
    if RESULTS_FILE.exists():
        raw = json.loads(RESULTS_FILE.read_text())
        ALL_CACHED = {
            tuple(k.split("|", 1)): v for k, v in raw.items()
        }
        print(f"\nLoaded {len(ALL_CACHED)} cached results from {RESULTS_FILE.name}")

    # ── 6. Compute metrics ────────────────────────────────────────────────────
    COMPARE_METHODS = [
        {"name": "embedding",     "label": "Embedding"},
        {"name": "text_search",   "label": "Text Search"},
        {"name": "hybrid",        "label": "Hybrid"},
        {"name": "vision_hier",   "label": "hindex only"},
        {"name": "embed_hindex",  "label": "Embed+hindex"},
        {"name": "text_hindex",   "label": "Text+hindex"},
        {"name": "hybrid_hindex", "label": "Hybrid+hindex"},
    ]

    import pandas as pd

    rows = []
    for item in eval_items:
        qid      = item["id"]
        qtype    = item["question_type"]
        diff     = item["difficulty"]
        relevant = build_relevant_set(item)
        if not relevant:
            continue

        # vision_semantic (just run)
        if qid in vs_results:
            m = compute_metrics(vs_results[qid], relevant)
            rows.append({"method": "vision_semantic", **m,
                         "question_type": qtype, "difficulty": diff})

        # cached comparison methods
        for cm in COMPARE_METHODS:
            result = ALL_CACHED.get((cm["name"], qid))
            if result is None:
                continue
            m = compute_metrics(result, relevant)
            rows.append({"method": cm["label"], **m,
                         "question_type": qtype, "difficulty": diff})

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

    # Save results
    OUT = NEW_APP_DIR / "eval_results.json"
    OUT.write_text(json.dumps(
        {qid: v for qid, v in vs_results.items()},
        indent=2, default=str,
    ))
    print(f"\nvision_semantic results saved → {OUT}")

    # Breakdown by question type
    print("\n" + "=" * 65)
    print(f"NDCG@{TOP_K} by question_type (vision_semantic vs Hybrid+hindex)")
    print("=" * 65)
    for qt in sorted(df["question_type"].unique()):
        sub = df[df["question_type"] == qt]
        vs = sub[sub["method"] == "vision_semantic"]["ndcg_k"].mean()
        hh = sub[sub["method"] == "Hybrid+hindex"]["ndcg_k"].mean()
        n  = len(sub[sub["method"] == "vision_semantic"])
        print(f"  {qt:<25}  vision_semantic={vs:.3f}  Hybrid+hindex={hh:.3f}  (n={n})")

    await colette_api.service_delete(APP_NAME)
    print("\nDone.")


asyncio.run(main())
