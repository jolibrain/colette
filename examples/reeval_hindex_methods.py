"""Re-run only the hindex-based retrieval methods on the synthetic benchmark.

The collection index is now active (two-stage cluster→doc→page navigation).
This script checks whether retrieval quality is unchanged / improved vs. the
prior flat single-stage baseline stored in eval_results.json.

Only the 4 hindex methods are re-run; the 3 pure methods (Embedding, Text
Search, Hybrid) are unaffected and kept from cache.

GPU allocation (per project rule):
    GPU 0 = embedding model
    GPU 1 = LLM
    GPU 2 = spare
    GPU 3 = DO NOT USE
"""

import asyncio
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

colette_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(colette_root / "src"))

app_name      = "eval_benchmark"
app_dir       = colette_root / "app_eval_benchmark"
BENCHMARK_DIR = colette_root / "examples" / "benchmark"
TOP_K         = 4
LLM_GPU_IDS   = [1]   # GPU 1 free; GPU 0 holds embedding, GPU 3 = DO NOT USE

from colette.jsonapi import JSONApi
from colette.apidata import APIData

colette_api = JSONApi()

# ── Methods to re-run (hindex-affected only) ─────────────────────────────────
RERUN_METHODS = [
    {
        "name": "vision_hier", "label": "hindex only",
        "config": {"retrieval_mode": "vision_hierarchical"},
    },
    {
        "name": "embed_hindex", "label": "Embed+hindex",
        "config": {"retrieval_mode": "embedding_retrieval",
                   "hindex": {"enable_hindex": True}},
    },
    {
        "name": "text_hindex", "label": "Text+hindex",
        "config": {"retrieval_mode": "text_search_retrieval",
                   "hindex": {"enable_hindex": True}},
    },
    {
        "name": "hybrid_hindex", "label": "Hybrid+hindex",
        "config": {"retrieval_mode": "hybrid",
                   "hindex": {"enable_hindex": True}},
    },
]

RERUN_NAMES = {m["name"] for m in RERUN_METHODS}

# ── UUID → doc mapping (same as notebook cell 3) ─────────────────────────────
HINDEX_DIR = app_dir / "mm_index" / "hindex"
UUID_TO_DOC: dict[str, str] = {}
if HINDEX_DIR.exists():
    for jf in sorted(HINDEX_DIR.glob("*.json")):
        if jf.name == "_collection_index.json":
            continue
        try:
            d = json.loads(jf.read_text())
            if d.get("page_keys") and d.get("doc_name"):
                doc_base = Path(d["doc_name"]).name
                for key in d["page_keys"]:
                    uuid_part = key[:36] if len(key) >= 36 else key
                    UUID_TO_DOC[uuid_part] = doc_base
        except Exception:
            pass

print(f"UUID→doc map: {len(UUID_TO_DOC)} entries for "
      f"{len(set(UUID_TO_DOC.values()))} documents")


def _doc_from_key(key: str) -> str:
    return UUID_TO_DOC.get(key[:36], "") if len(key) >= 36 else ""


def run_query(method: dict, question: str) -> dict:
    rag_params = dict(method["config"])
    rag_params["top_k"] = TOP_K
    payload = {"parameters": {"input": {"message": question, "rag": rag_params}}}
    t0 = time.time()
    response = colette_api.service_predict(app_name, APIData(**payload))
    elapsed = round(time.time() - t0, 2)

    sources = response.sources or {}
    image_hits, text_hits = [], []

    for item in sources.get("context", []):
        doc = item.get("source", "")
        doc = Path(doc).name if doc else _doc_from_key(item.get("key", ""))
        image_hits.append({
            "key":        item.get("key", ""),
            "doc":        doc,
            "page":       item.get("page_number"),
            "distance":   item.get("distance"),
            "crop_label": item.get("crop_label", ""),
            "retrieval_source": item.get("retrieval_source", "embedding"),
        })

    for item in sources.get("text_context", []):
        doc = Path(item.get("source", "")).name
        text_hits.append({
            "doc":   doc,
            "page":  item.get("page_number"),
            "score": item.get("score"),
            "content": item.get("content", "")[:200],
        })

    return {
        "image_hits": image_hits,
        "text_hits":  text_hits,
        "output":     response.output or "",
        "latency_s":  elapsed,
    }


def build_relevant_set(item: dict):
    """Return set of (doc_basename, page_number) ground-truth pairs."""
    rel = set()
    for rd in item.get("relevant_docs", []):
        doc  = Path(rd["doc"]).name
        page = rd.get("page")
        if page is not None:
            rel.add((doc, page))
    return rel


def _retrieved_pages(result: dict, source: str = "all") -> list[tuple[str, int]]:
    pages = []
    for h in result.get("image_hits", []):
        if source != "all" and h.get("retrieval_source", "embedding") != source:
            continue
        doc  = h.get("doc", "")
        page = h.get("page")
        if doc and page is not None:
            pages.append((doc, int(page)))
    for h in result.get("text_hits", []):
        doc  = h.get("doc", "")
        page = h.get("page")
        if doc and page is not None:
            pages.append((doc, int(page)))
    # deduplicate preserving order
    seen, deduped = set(), []
    for p in pages:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    return deduped[:TOP_K]


def compute_retrieval_metrics(result: dict, relevant: set, top_k: int = 4) -> dict:
    retrieved = _retrieved_pages(result)
    ctx_size  = len(retrieved)

    # hit / recall over entire context window
    hits_in_ctx = [p for p in retrieved if p in relevant]
    hit_ctx     = int(bool(hits_in_ctx))
    recall_ctx  = len(set(hits_in_ctx)) / len(relevant) if relevant else 0.0

    # MRR
    mrr = 0.0
    for rank, p in enumerate(retrieved, 1):
        if p in relevant:
            mrr = 1.0 / rank
            break

    # NDCG over context
    def dcg(ranked):
        return sum(1.0 / (i + 1) for i, p in enumerate(ranked) if p in relevant)

    ideal  = min(len(relevant), ctx_size)
    ndcg_ctx = dcg(retrieved) / dcg(list(relevant)[:ideal]) if ideal else 0.0

    # @K metrics
    at_k     = retrieved[:top_k]
    hit_k    = int(any(p in relevant for p in at_k))
    p_k      = sum(1 for p in at_k if p in relevant) / top_k if top_k else 0.0
    r_k      = sum(1 for p in at_k if p in relevant) / len(relevant) if relevant else 0.0
    ndcg_k   = dcg(at_k) / dcg(list(relevant)[:min(len(relevant), top_k)]) \
               if min(len(relevant), top_k) > 0 else 0.0

    return dict(ctx_size=ctx_size, hit=hit_ctx, recall=recall_ctx, mrr=mrr,
                ndcg_ctx=ndcg_ctx, hit_k=hit_k, p_k=p_k, r_k=r_k, ndcg_k=ndcg_k)


async def main():
    # ── 1. Create service ────────────────────────────────────────────────────
    print(f"Loading service '{app_name}' (LLM → GPU {LLM_GPU_IDS})…")
    try:
        await colette_api.service_delete(app_name)
    except Exception:
        pass

    saved_cfg = json.loads((app_dir / "config.json").read_text())
    saved_cfg["parameters"]["llm"]["gpu_ids"] = LLM_GPU_IDS
    colette_api.service_create(app_name, APIData(**saved_cfg))
    print("Service ready.\n")

    # ── 2. Load QA items ─────────────────────────────────────────────────────
    QA_ITEMS = []
    with open(BENCHMARK_DIR / "qa_dataset.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                QA_ITEMS.append(json.loads(line))
    print(f"Loaded {len(QA_ITEMS)} QA items\n")

    # ── 3. Load existing cache, keep non-hindex results ──────────────────────
    RESULTS_FILE = app_dir / "eval_results.json"
    ALL_RESULTS: dict = {}
    if RESULTS_FILE.exists():
        raw = json.loads(RESULTS_FILE.read_text())
        ALL_RESULTS = {
            tuple(k.split("|", 1)): v
            for k, v in raw.items()
            if k.split("|")[0] not in RERUN_NAMES   # drop stale hindex entries
        }
        dropped = len(raw) - len(ALL_RESULTS)
        print(f"Cache: kept {len(ALL_RESULTS)} non-hindex results, "
              f"dropped {dropped} stale hindex entries\n")

    # ── 4. Run hindex methods ─────────────────────────────────────────────────
    eval_items = [q for q in QA_ITEMS if q.get("relevant_docs")]
    total = len(eval_items) * len(RERUN_METHODS)
    done  = 0
    for item in eval_items:
        qid = item["id"]
        q   = item["question"]
        for m in RERUN_METHODS:
            done += 1
            print(f"[{done}/{total}] {m['label']:<16} {qid}  {q[:50]}…", end=" ", flush=True)
            try:
                r = run_query(m, q)
                ALL_RESULTS[(m["name"], qid)] = r
                img = len(r["image_hits"])
                print(f"✓ {r['latency_s']:.1f}s  img={img}")
            except Exception as exc:
                print(f"ERROR: {exc}")

    # ── 5. Save updated cache ─────────────────────────────────────────────────
    serializable = {f"{mn}|{qid}": v for (mn, qid), v in ALL_RESULTS.items()}
    RESULTS_FILE.write_text(json.dumps(serializable, indent=2, default=str))
    print(f"\nResults saved → {RESULTS_FILE}\n")

    # ── 6. Compute retrieval metrics ──────────────────────────────────────────
    ALL_METHODS = [
        {"name": "embedding",    "label": "Embedding"},
        {"name": "text_search",  "label": "Text Search"},
        {"name": "hybrid",       "label": "Hybrid"},
        *RERUN_METHODS,
    ]

    rows = []
    for item in eval_items:
        qid     = item["id"]
        qtype   = item["question_type"]
        diff    = item["difficulty"]
        relevant = build_relevant_set(item)
        if not relevant:
            continue
        for m in ALL_METHODS:
            result = ALL_RESULTS.get((m["name"], qid))
            if result is None:
                continue
            metrics = compute_retrieval_metrics(result, relevant)
            rows.append({"method": m["label"], **metrics,
                         "question_type": qtype, "difficulty": diff})

    import pandas as pd
    df = pd.DataFrame(rows)
    summary = (df.groupby("method")[["hit", "mrr", "ndcg_k", "hit_k", "ctx_size"]]
                 .mean()
                 .rename(columns={"hit": "Hit@ctx", "mrr": "MRR",
                                  "ndcg_k": f"NDCG@{TOP_K}",
                                  "hit_k": f"Hit@{TOP_K}",
                                  "ctx_size": "avg_ctx"})
                 .round(3))

    # Baseline from last run
    BASELINE = {
        "hindex only":   dict(hit=0.791, mrr=0.639, ndcg=0.657),
        "Embed+hindex":  dict(hit=0.835, mrr=0.655, ndcg=0.679),
        "Text+hindex":   dict(hit=0.991, mrr=0.679, ndcg=0.687),
        "Hybrid+hindex": dict(hit=0.983, mrr=0.685, ndcg=0.696),
    }

    print("=" * 70)
    print("RETRIEVAL SUMMARY (all methods)")
    print("=" * 70)
    print(summary.to_string())

    print("\n" + "=" * 70)
    print("DELTA vs. FLAT-SELECTION BASELINE (hindex methods only)")
    print("=" * 70)
    for label, base in BASELINE.items():
        row = summary.loc[label] if label in summary.index else None
        if row is None:
            print(f"  {label:<18} — not in results")
            continue
        d_hit  = row[f"Hit@{TOP_K}"] - base["hit"]
        d_mrr  = row["MRR"]         - base["mrr"]
        d_ndcg = row[f"NDCG@{TOP_K}"] - base["ndcg"]
        print(f"  {label:<18}  Hit@{TOP_K}: {d_hit:+.3f}   MRR: {d_mrr:+.3f}   "
              f"NDCG@{TOP_K}: {d_ndcg:+.3f}")

    await colette_api.service_delete(app_name)
    print("\nDone.")


asyncio.run(main())
