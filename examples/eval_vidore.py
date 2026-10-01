"""Evaluate vision_hybrid_bm25 on ViDoRe v1 (page-level retrieval, NDCG@5).

Each ViDoRe corpus page is converted to a single-page PDF, indexed via colette
(hindex VLM generation + BM25 + text embed + visual embed), and then queried.
Results are saved as JSON and printed as a comparison table against published baselines.

Usage:
  CUDA_VISIBLE_DEVICES=0,3 python examples/eval_vidore.py [dataset_nickname ...]

  dataset_nickname can be any of the keys in VIDORE_DATASETS below, e.g.:
    tabfquad   arxivqa   docvqa   infovqa   tatdqa
    ai   energy   government   healthcare   shift

  No args → runs all 10 sub-datasets sequentially.

GPU mapping (CUDA_VISIBLE_DEVICES=0,3):
  cuda:0 → physical GPU 0  (embedding model)
  cuda:1 → physical GPU 3  (LLM / VLM for hindex)
"""

import asyncio
import json
import math
import shutil
import sys
import time
from pathlib import Path

from datasets import load_dataset

colette_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(colette_root / "src"))

from colette.jsonapi import JSONApi
from colette.apidata import APIData

# ── GPU allocation (CUDA_VISIBLE_DEVICES=0,3) ───────────────────────────────
EMBED_GPU_ID = 0   # cuda:0 → physical GPU 0
LLM_GPU_IDS  = [1] # cuda:1 → physical GPU 3

# ── Config ───────────────────────────────────────────────────────────────────
TOP_K          = 5                             # retrieve top-5 for NDCG@5
EXISTING_APP   = colette_root / "app_eval_benchmark"
RESULTS_DIR    = colette_root / "examples"
KEEP_APP_DIRS  = True   # keep indexed app dirs for debugging

VIDORE_DATASETS = {
    "docvqa":      "vidore/docvqa_test_subsampled",
    "infovqa":     "vidore/infovqa_test_subsampled",
    "tatdqa":      "vidore/tatdqa_test",
    "arxivqa":     "vidore/arxivqa_test_subsampled",
    "tabfquad":    "vidore/tabfquad_test_subsampled",
    "ai":          "vidore/syntheticDocQA_artificial_intelligence_test",
    "energy":      "vidore/syntheticDocQA_energy_test",
    "government":  "vidore/syntheticDocQA_government_reports_test",
    "healthcare":  "vidore/syntheticDocQA_healthcare_industry_test",
    "shift":       "vidore/shiftproject_test",
}

# Published ColPali NDCG@5 baselines for comparison printing
COLPALI_BASELINES = {
    "arxivqa": 79.1, "docvqa": 54.4, "infovqa": 81.8,
    "tabfquad": 83.9, "tatdqa": 65.8, "shift": 73.2,
    "ai": 96.2, "energy": 91.0, "government": 92.7, "healthcare": 94.4,
}
BGE_BASELINES = {
    "arxivqa": 35.7, "docvqa": 32.9, "infovqa": 71.9,
    "tabfquad": 69.1, "tatdqa": 43.8, "shift": 73.1,
    "ai": 88.8, "energy": 83.3, "government": 80.4, "healthcare": 91.3,
}

colette_api = JSONApi()


# ── NDCG@5 ───────────────────────────────────────────────────────────────────

def ndcg_at_5(retrieved_ids: list[str], relevant_id: str) -> float:
    """Binary NDCG@5 — exactly 1 relevant document per query (ViDoRe standard)."""
    for rank, doc_id in enumerate(retrieved_ids[:5], 1):
        if doc_id == relevant_id:
            # IDCG = 1/log2(2) = 1.0  (ideal: relevant doc at rank 1)
            return 1.0 / math.log2(rank + 1)
    return 0.0


# ── Data helpers ──────────────────────────────────────────────────────────────

def parse_dataset(ds) -> tuple[list[tuple[str, str]], dict]:
    """Return (queries, corpus).

    queries: list of (query_text, relevant_image_filename)
    corpus:  {image_filename: PIL.Image}
    """
    corpus: dict = {}
    seen_queries: set = set()
    queries: list = []

    for row in ds:
        fn = row["image_filename"]
        corpus[fn] = row["image"]
        q = (row.get("query") or "").strip()
        if q and q not in seen_queries:
            seen_queries.add(q)
            queries.append((q, fn))

    return queries, corpus


def _safe_pdf_name(image_filename: str) -> str:
    """Convert image_filename to a safe PDF base name."""
    safe = image_filename.replace("/", "_").replace("\\", "_")
    if not safe.lower().endswith(".pdf"):
        safe += ".pdf"
    return safe


def save_corpus_as_pdfs(corpus: dict, pdf_dir: Path) -> dict[str, str]:
    """Save each PIL.Image as a single-page PDF.

    Returns {image_filename: pdf_stem} for reverse-mapping results.
    """
    pdf_dir.mkdir(parents=True, exist_ok=True)
    id_to_stem: dict[str, str] = {}

    for image_filename, image in corpus.items():
        pdf_name = _safe_pdf_name(image_filename)
        pdf_path = pdf_dir / pdf_name
        img = image.convert("RGB")
        img.save(str(pdf_path), "PDF", resolution=150)
        id_to_stem[image_filename] = Path(pdf_name).stem

    return id_to_stem


# ── Colette helpers ───────────────────────────────────────────────────────────

def run_query_vidore(app_name: str, question: str) -> dict:
    payload = {"parameters": {"input": {
        "message": question,
        "rag": {"retrieval_mode": "vision_hybrid_bm25", "top_k": TOP_K},
    }}}
    t0 = time.time()
    response = colette_api.service_predict(app_name, APIData(**payload))
    elapsed = round(time.time() - t0, 2)

    sources = response.sources or {}
    image_hits = []
    for item in sources.get("context", []):
        doc = Path(item.get("source", "")).name
        image_hits.append({"doc": doc, "page": item.get("page_number")})

    text_hits = []
    for item in sources.get("text_context", []):
        doc = Path(item.get("source", "")).name
        text_hits.append({"doc": doc, "page": item.get("page_number")})

    return {"image_hits": image_hits, "text_hits": text_hits, "latency_s": elapsed}


def _hits_to_doc_ids(result: dict, stem_to_id: dict[str, str]) -> list[str]:
    """Convert colette hits (doc=pdf_name, page=1) to original image_filename IDs."""
    ids: list[str] = []
    for h in result.get("image_hits", []) + result.get("text_hits", []):
        doc = h.get("doc", "")
        stem = Path(doc).stem
        original_id = stem_to_id.get(stem, stem)
        if original_id not in ids:
            ids.append(original_id)
    return ids[:TOP_K]


# ── Main eval loop ────────────────────────────────────────────────────────────

async def eval_one(nickname: str, hf_dataset: str) -> dict:
    print(f"\n{'='*65}")
    print(f"ViDoRe: {nickname}  ({hf_dataset})")
    print(f"{'='*65}")

    app_name = f"vidore_{nickname}"
    app_dir  = colette_root / f"app_vidore_{nickname}"
    results_path = RESULTS_DIR / f"vidore_{nickname}_results.json"

    # ── load dataset ──────────────────────────────────────────────────────────
    print("Loading from HuggingFace...")
    ds = load_dataset(hf_dataset, split="test")
    queries, corpus = parse_dataset(ds)
    print(f"  corpus={len(corpus)} pages  queries={len(queries)}")

    # ── resume check ─────────────────────────────────────────────────────────
    saved: dict = {}
    if results_path.exists():
        saved = json.loads(results_path.read_text())
        done = sum(1 for v in saved.values() if "ndcg5" in v)
        print(f"  Resuming: {done}/{len(queries)} already scored")

    # ── prepare PDF corpus ────────────────────────────────────────────────────
    pdf_dir = app_dir / "pdfs"
    if not pdf_dir.exists() or not any(pdf_dir.glob("*.pdf")):
        print(f"Converting {len(corpus)} images → PDFs...")
        t0 = time.time()
        id_to_stem = save_corpus_as_pdfs(corpus, pdf_dir)
        print(f"  Done in {time.time()-t0:.1f}s")
    else:
        # rebuild mapping from existing filenames
        id_to_stem = {}
        for image_filename in corpus:
            pdf_name = _safe_pdf_name(image_filename)
            id_to_stem[image_filename] = Path(pdf_name).stem
        print(f"  Reusing {len(list(pdf_dir.glob('*.pdf')))} existing PDFs")

    stem_to_id = {v: k for k, v in id_to_stem.items()}

    # ── create colette service ────────────────────────────────────────────────
    cfg = json.loads((EXISTING_APP / "config.json").read_text())
    cfg["app"]["repository"]                            = str(app_dir)
    cfg["parameters"]["llm"]["gpu_ids"]                 = LLM_GPU_IDS
    cfg["parameters"]["llm"].pop("external_vllm_server", None)  # use built-in vLLM
    cfg["parameters"]["input"]["rag"]["gpu_id"]         = EMBED_GPU_ID
    cfg["parameters"]["input"]["rag"]["retrieval_mode"] = "vision_hybrid_bm25"
    # num_tokens=1: skip LLM answer generation at predict time (outputc.num_tokens is set
    # at service creation and cannot be overridden per-request). We only need retrieved sources.
    cfg["parameters"]["output"] = {"num_tokens": 1}
    # Cap image size before visual embedding to avoid OOM in the Qwen2-VL ViT encoder.
    # max_pixels=2M causes ~10K patches → huge activation tensors that OOM on 24 GiB GPU
    # (model ~10 GiB + activations ~19 GiB > 24 GiB). 1024×768 ≈ 786K pixels fits safely.
    cfg["parameters"]["input"]["rag"].setdefault("ragm", {})
    cfg["parameters"]["input"]["rag"]["ragm"]["image_width"]  = 1024
    cfg["parameters"]["input"]["rag"]["ragm"]["image_height"] = 768

    try:
        await colette_api.service_delete(app_name)
    except Exception:
        pass

    colette_api.service_create(app_name, APIData(**cfg))

    # ── index (builds hindex via VLM + BM25 + embeddings) ────────────────────
    # Check for actual hindex content (not just dir existence — dir can be left by a failed run)
    index_path = app_dir / "mm_index"
    hindex_path = index_path / "hindex"
    need_index = not hindex_path.exists() or not any(hindex_path.glob("*.json"))
    # hindex JSONs exist but vector store may be incomplete (e.g. OOM crash during embed).
    # With preserve_existing_hindex=True (set when vlm_desc_requested), reindex=True skips
    # VLM re-generation and only rebuilds BM25 + ChromaDB from the saved JSONs.
    hindex_complete = hindex_path.exists() and len(list(hindex_path.glob("*.json"))) == len(corpus)
    if need_index or hindex_complete:
        if hindex_complete and not need_index:
            print(f"  hindex complete ({len(corpus)} JSONs), rebuilding vector stores only (no VLM)...")
        else:
            print(f"Indexing {len(corpus)} pages (VLM hindex + BM25 + embed)...")
            print("  This is the slow step. Grab a coffee.")
        # DPI=150 matches the PDF storage resolution (PDFs saved at 150 DPI) — avoids
        # upscaling to 200 DPI which caused 19 GiB allocation OOM in Qwen2-VL visual encoder.
        # index_protection=False: service config has protection=True; override to allow reindex.
        index_cfg = {"parameters": {"input": {
            "data": [str(pdf_dir)],
            "preprocessing": {"files": ["pdf"], "dpi": 150},
            "rag": {
                "retrieval_mode": "vision_hybrid_bm25",
                "reindex": True,
                "update_index": False,
                "index_protection": False,
                "hindex": {
                    "enable_hindex": True,
                    "add_node_summaries": False,
                    "toc_scan_pages": 1,
                },
            },
        }}}
        t0 = time.time()
        colette_api.service_index(app_name, APIData(**index_cfg))
        elapsed = time.time() - t0
        print(f"  Indexing done in {elapsed/60:.1f}min  ({elapsed/len(corpus):.1f}s/page)")
    else:
        print("  Reusing existing index")
        # still need to trigger index load (but skip rebuild)
        index_cfg = {"parameters": {"input": {
            "data": [str(pdf_dir)],
            "preprocessing": {"files": ["pdf"]},
            "rag": {"retrieval_mode": "vision_hybrid_bm25", "reindex": False},
        }}}
        colette_api.service_index(app_name, APIData(**index_cfg))

    # ── run queries ───────────────────────────────────────────────────────────
    ndcg_scores: list[float] = []
    for i, (query, relevant_fn) in enumerate(queries, 1):
        key = f"{i:04d}_{query[:40]}"

        if key in saved and "ndcg5" in saved[key]:
            ndcg_scores.append(saved[key]["ndcg5"])
            print(f"[{i:4d}/{len(queries)}] (cached) NDCG@5={saved[key]['ndcg5']:.3f}")
            continue

        print(f"[{i:4d}/{len(queries)}] {query[:60]}...", end=" ", flush=True)
        try:
            r = run_query_vidore(app_name, query)
            retrieved_ids = _hits_to_doc_ids(r, stem_to_id)
            score = ndcg_at_5(retrieved_ids, relevant_fn)
            ndcg_scores.append(score)
            saved[key] = {
                "query": query,
                "relevant": relevant_fn,
                "retrieved": retrieved_ids,
                "ndcg5": score,
                "latency_s": r["latency_s"],
            }
            results_path.write_text(json.dumps(saved, indent=2))
            print(f"NDCG@5={score:.3f}  ({r['latency_s']:.1f}s)")
        except Exception as e:
            print(f"ERROR: {e}")
            ndcg_scores.append(0.0)

    try:
        await colette_api.service_delete(app_name)
    except Exception:
        pass

    if not KEEP_APP_DIRS:
        shutil.rmtree(app_dir, ignore_errors=True)

    avg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0.0
    print(f"\n  ► {nickname}: NDCG@5 = {avg*100:.1f}")
    print(f"  ► ColPali baseline: {COLPALI_BASELINES.get(nickname, '—')}")
    print(f"  ► BGE-M3+Caption:   {BGE_BASELINES.get(nickname, '—')}")

    return {
        "nickname":    nickname,
        "hf_dataset":  hf_dataset,
        "ndcg5":       avg,
        "ndcg5_pct":   round(avg * 100, 2),
        "n_queries":   len(queries),
        "n_corpus":    len(corpus),
        "colpali_ref": COLPALI_BASELINES.get(nickname),
        "bge_ref":     BGE_BASELINES.get(nickname),
    }


async def main():
    target_keys = sys.argv[1:] if len(sys.argv) > 1 else list(VIDORE_DATASETS.keys())
    unknown = [k for k in target_keys if k not in VIDORE_DATASETS]
    if unknown:
        print(f"Unknown dataset nicknames: {unknown}")
        print(f"Valid: {list(VIDORE_DATASETS.keys())}")
        sys.exit(1)

    print(f"Running ViDoRe evaluation: {target_keys}")
    print(f"GPU: embed=cuda:{EMBED_GPU_ID}, LLM=cuda:{LLM_GPU_IDS[0]}")

    all_results = []
    summary_path = RESULTS_DIR / "vidore_results_summary.json"

    for nick in target_keys:
        result = await eval_one(nick, VIDORE_DATASETS[nick])
        all_results.append(result)
        summary_path.write_text(json.dumps(all_results, indent=2))

    # ── print comparison table ────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("ViDoRe v1 NDCG@5 — Comparison Table")
    print(f"{'='*70}")
    print(f"{'Dataset':<16} {'Ours':>8} {'ColPali':>9} {'BGE+Cap':>9} {'Δ ColPali':>10}")
    print("-" * 55)
    our_vals, col_vals = [], []
    for r in all_results:
        nick = r["nickname"]
        ours = r["ndcg5_pct"]
        cpali = r["colpali_ref"] or 0
        bge = r["bge_ref"] or 0
        delta = ours - cpali
        sign = "+" if delta >= 0 else ""
        print(f"{nick:<16} {ours:>8.1f} {cpali:>9.1f} {bge:>9.1f} {sign+f'{delta:.1f}':>10}")
        our_vals.append(ours)
        col_vals.append(cpali)
    if our_vals:
        our_avg = sum(our_vals) / len(our_vals)
        col_avg = sum(col_vals) / len(col_vals) if col_vals else 0
        delta_avg = our_avg - col_avg
        sign = "+" if delta_avg >= 0 else ""
        print("-" * 55)
        print(f"{'AVERAGE':<16} {our_avg:>8.1f} {col_avg:>9.1f} {'—':>9} {sign+f'{delta_avg:.1f}':>10}")
    print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
