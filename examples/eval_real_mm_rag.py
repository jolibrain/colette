"""Evaluate vision_hybrid_bm25 on REAL-MM-RAG (page-level retrieval, NDCG@5).

REAL-MM-RAG is an IBM Research benchmark for real-world multi-modal RAG with
industry documents (financial reports, financial slides, tech reports, tech slides).
Each corpus page is a PNG image; queries are natural-language questions generated
by a VLM and optionally rephrased to 3 difficulty levels.

Each image is converted to a single-page PDF, indexed via colette
(hindex VLM generation + BM25 + text embed + visual embed), and then queried.
Results are saved as JSON and a comparison table is printed.

Paper: https://arxiv.org/abs/2502.12342  (ACL 2025)
HF collection: https://huggingface.co/collections/ibm-research/real-mm-rag-bench-67d2dc0ddf2dfafe66f09d34

Usage:
  CUDA_VISIBLE_DEVICES=0,3 python examples/eval_real_mm_rag.py [domain ...] [--rephrase-level N]

  domain can be any of: fin_report  fin_slides  tech_report  tech_slides
  No args → runs all 4 domains sequentially.

  --rephrase-level 0   base queries (default) — original NDCG@5 from VLM-gen queries
  --rephrase-level 1   moderate rephrasing    — synonyms, sentence restructuring
  --rephrase-level 2   harder rephrasing      — abstractive paraphrase
  --rephrase-level 3   hardest rephrasing     — tests semantic understanding, not keyword match

GPU mapping (CUDA_VISIBLE_DEVICES=0,3):
  cuda:0 → physical GPU 0  (embedding model)
  cuda:1 → physical GPU 3  (LLM / VLM for hindex)
"""

import argparse
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
EMBED_GPU_ID = 0    # cuda:0 → physical GPU 0
LLM_GPU_IDS  = [1]  # cuda:1 → physical GPU 3

# ── Config ───────────────────────────────────────────────────────────────────
TOP_K         = 5                            # retrieve top-5 for NDCG@5
EXISTING_APP  = colette_root / "app_eval_benchmark"
RESULTS_DIR   = colette_root / "examples"
KEEP_APP_DIRS = True   # keep indexed app dirs for debugging / fast re-runs

REAL_MM_RAG_DATASETS = {
    "fin_report":  "ibm-research/REAL-MM-RAG_FinReport",
    "fin_slides":  "ibm-research/REAL-MM-RAG_FinSlides",
    "tech_report": "ibm-research/REAL-MM-RAG_TechReport",
    "tech_slides": "ibm-research/REAL-MM-RAG_TechSlides",
}

# Corpus/query counts from paper Table S1 (for reference)
DOMAIN_STATS = {
    "fin_report":  {"pages": 2687, "queries": 853},
    "fin_slides":  {"pages": 2280, "queries": 1052},
    "tech_report": {"pages": 1674, "queries": 1294},
    "tech_slides": {"pages": 1963, "queries": 1354},
}

# Published NDCG@5 baselines from REAL-MM-RAG paper (Table S2 / Table 2)
# ColPali = standard ColPali-v1.2; ColQwen = ColQwen2
COLPALI_L0 = {   # rephrase level 0 (base queries)
    "fin_report": 52.7, "fin_slides": 62.2, "tech_report": 80.6, "tech_slides": 89.7,
}
COLQWEN_L0 = {
    "fin_report": 60.8, "fin_slides": 59.3, "tech_report": 84.2, "tech_slides": 91.3,
}
COLPALI_L3 = {   # rephrase level 3 (hardest)
    "fin_report": 34.5, "fin_slides": 27.6, "tech_report": 62.0, "tech_slides": 75.8,
}
COLQWEN_L3 = {
    "fin_report": 41.8, "fin_slides": 31.1, "tech_report": 66.9, "tech_slides": 78.1,
}

colette_api = JSONApi()


# ── NDCG@5 ───────────────────────────────────────────────────────────────────

def ndcg_at_5(retrieved_ids: list[str], relevant_id: str) -> float:
    """Binary NDCG@5 — exactly 1 relevant document per query."""
    for rank, doc_id in enumerate(retrieved_ids[:5], 1):
        if doc_id == relevant_id:
            return 1.0 / math.log2(rank + 1)
    return 0.0


# ── Data helpers ──────────────────────────────────────────────────────────────

def parse_dataset(ds, rephrase_level: int = 0) -> tuple[list[tuple[str, str]], dict]:
    """Return (queries, corpus).

    queries: list of (query_text, relevant_image_filename)
    corpus:  {image_filename: PIL.Image}

    rephrase_level=0 → use the ``query`` field (original queries).
    rephrase_level=1/2/3 → use the ``rephrase_level_N`` field.
    Rows where the selected query field is null/empty are added to the
    corpus but skipped for the query list.
    """
    rephrase_key = None if rephrase_level == 0 else f"rephrase_level_{rephrase_level}"

    corpus: dict = {}
    seen_queries: set = set()
    queries: list = []

    for row in ds:
        fn = row["image_filename"]
        corpus[fn] = row["image"]

        q_raw = row.get(rephrase_key if rephrase_key else "query") or ""
        q = q_raw.strip()
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

    Returns {image_filename: pdf_stem} for reverse-mapping retrieval results.
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

def run_query(app_name: str, question: str) -> dict:
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

async def eval_one(nickname: str, hf_dataset: str, rephrase_level: int) -> dict:
    level_tag = f"_L{rephrase_level}" if rephrase_level > 0 else ""
    print(f"\n{'='*65}")
    print(f"REAL-MM-RAG: {nickname}{level_tag}  ({hf_dataset})")
    print(f"{'='*65}")

    app_name    = f"real_mm_rag_{nickname}"
    app_dir     = colette_root / f"app_real_mm_rag_{nickname}"
    results_path = RESULTS_DIR / f"real_mm_rag_{nickname}{level_tag}_results.json"

    # ── load dataset ──────────────────────────────────────────────────────────
    print("Loading from HuggingFace...")
    ds = load_dataset(hf_dataset, split="test")
    queries, corpus = parse_dataset(ds, rephrase_level=rephrase_level)
    print(f"  corpus={len(corpus)} pages  queries={len(queries)}  (rephrase_level={rephrase_level})")

    if not queries:
        print("  WARNING: no queries found — skipping this domain/level")
        return {}

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
    cfg["parameters"]["llm"].pop("external_vllm_server", None)
    cfg["parameters"]["input"]["rag"]["gpu_id"]         = EMBED_GPU_ID
    cfg["parameters"]["input"]["rag"]["retrieval_mode"] = "vision_hybrid_bm25"
    # num_tokens=1: skip LLM answer generation — retrieval-only eval
    cfg["parameters"]["output"] = {"num_tokens": 1}
    # Cap image size to avoid OOM in Qwen2-VL ViT encoder (same as ViDoRe eval)
    cfg["parameters"]["input"]["rag"].setdefault("ragm", {})
    cfg["parameters"]["input"]["rag"]["ragm"]["image_width"]  = 1024
    cfg["parameters"]["input"]["rag"]["ragm"]["image_height"] = 768

    try:
        await colette_api.service_delete(app_name)
    except Exception:
        pass

    colette_api.service_create(app_name, APIData(**cfg))

    # ── index ─────────────────────────────────────────────────────────────────
    index_path  = app_dir / "mm_index"
    hindex_path = index_path / "hindex"
    need_index  = not hindex_path.exists() or not any(hindex_path.glob("*.json"))
    hindex_complete = (
        hindex_path.exists()
        and len(list(hindex_path.glob("*.json"))) == len(corpus)
    )

    if need_index or hindex_complete:
        if hindex_complete and not need_index:
            print(f"  hindex complete ({len(corpus)} JSONs), rebuilding vector stores only (no VLM)...")
        else:
            print(f"Indexing {len(corpus)} pages (VLM hindex + BM25 + embed)...")
            print("  This is the slow step — grab a coffee.")
        # DPI=150 matches PDF storage resolution; avoids upscaling OOM
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
            r = run_query(app_name, query)
            retrieved_ids = _hits_to_doc_ids(r, stem_to_id)
            score = ndcg_at_5(retrieved_ids, relevant_fn)
            ndcg_scores.append(score)
            saved[key] = {
                "query":      query,
                "relevant":   relevant_fn,
                "retrieved":  retrieved_ids,
                "ndcg5":      score,
                "latency_s":  r["latency_s"],
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
    colpali_ref = (COLPALI_L3 if rephrase_level == 3 else COLPALI_L0).get(nickname)
    colqwen_ref = (COLQWEN_L3 if rephrase_level == 3 else COLQWEN_L0).get(nickname)

    print(f"\n  ► {nickname} (L{rephrase_level}): NDCG@5 = {avg*100:.1f}")
    if colpali_ref is not None:
        print(f"  ► ColPali baseline (L{rephrase_level}): {colpali_ref}")
    if colqwen_ref is not None:
        print(f"  ► ColQwen2 baseline (L{rephrase_level}): {colqwen_ref}")

    return {
        "nickname":        nickname,
        "hf_dataset":      hf_dataset,
        "rephrase_level":  rephrase_level,
        "ndcg5":           avg,
        "ndcg5_pct":       round(avg * 100, 2),
        "n_queries":       len(queries),
        "n_corpus":        len(corpus),
        "colpali_ref":     colpali_ref,
        "colqwen_ref":     colqwen_ref,
    }


async def main():
    parser = argparse.ArgumentParser(
        description="Evaluate vision_hybrid_bm25 on REAL-MM-RAG benchmark"
    )
    parser.add_argument(
        "domains",
        nargs="*",
        help="Domains to evaluate (default: all). Choices: " + " ".join(REAL_MM_RAG_DATASETS),
    )
    parser.add_argument(
        "--rephrase-level", "-r",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Query difficulty level (0=base, 3=hardest). Default: 0",
    )
    args = parser.parse_args()

    target_keys = args.domains if args.domains else list(REAL_MM_RAG_DATASETS.keys())
    unknown = [k for k in target_keys if k not in REAL_MM_RAG_DATASETS]
    if unknown:
        print(f"Unknown domain names: {unknown}")
        print(f"Valid: {list(REAL_MM_RAG_DATASETS.keys())}")
        sys.exit(1)

    rephrase_level = args.rephrase_level
    level_tag = f"_L{rephrase_level}" if rephrase_level > 0 else ""

    print(f"Running REAL-MM-RAG evaluation: {target_keys}  (rephrase_level={rephrase_level})")
    print(f"GPU: embed=cuda:{EMBED_GPU_ID}, LLM=cuda:{LLM_GPU_IDS[0]}")

    all_results = []
    summary_path = RESULTS_DIR / f"real_mm_rag{level_tag}_results_summary.json"

    for nick in target_keys:
        result = await eval_one(nick, REAL_MM_RAG_DATASETS[nick], rephrase_level)
        if result:
            all_results.append(result)
            summary_path.write_text(json.dumps(all_results, indent=2))

    if not all_results:
        print("No results collected.")
        return

    # ── print comparison table ────────────────────────────────────────────────
    colpali_refs = COLPALI_L3 if rephrase_level == 3 else COLPALI_L0
    colqwen_refs = COLQWEN_L3 if rephrase_level == 3 else COLQWEN_L0

    print(f"\n{'='*75}")
    print(f"REAL-MM-RAG NDCG@5 — rephrase_level={rephrase_level}")
    print(f"{'='*75}")
    print(f"{'Domain':<14} {'Ours':>8} {'ColPali':>9} {'ColQwen2':>10} {'Δ ColPali':>10}")
    print("-" * 55)
    our_vals, col_vals = [], []
    for r in all_results:
        nick   = r["nickname"]
        ours   = r["ndcg5_pct"]
        cpali  = colpali_refs.get(nick, 0)
        cqwen  = colqwen_refs.get(nick, 0)
        delta  = ours - cpali
        sign   = "+" if delta >= 0 else ""
        print(
            f"{nick:<14} {ours:>8.1f} {cpali:>9.1f} {cqwen:>10.1f} {sign+f'{delta:.1f}':>10}"
        )
        our_vals.append(ours)
        col_vals.append(cpali)
    if our_vals:
        our_avg   = sum(our_vals) / len(our_vals)
        col_avg   = sum(col_vals) / len(col_vals) if col_vals else 0
        delta_avg = our_avg - col_avg
        sign      = "+" if delta_avg >= 0 else ""
        print("-" * 55)
        print(
            f"{'AVERAGE':<14} {our_avg:>8.1f} {col_avg:>9.1f} {'—':>10} {sign+f'{delta_avg:.1f}':>10}"
        )
    print(f"\nFull results saved to: {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
