"""Rebuild the master hindex for all benchmark PDFs at the new 896×1120 resolution.

Triggers a full clean re-index (reindex=True) for vision_hybrid_bm25 mode so that:
  - hindex JSON files are rebuilt with 896×1120 image resolution (no more 320×480 hallucinations)
  - Tantivy BM25 index is rebuilt from the new hindex content
  - ChromaDB image embeddings are rebuilt (fast — no VLM, just the embed model)

GPU allocation:
  GPU 0 = embedding model (ChromaDB)
  GPU 1 = LLM / VLM for hindex building
  GPU 3 = DO NOT USE

After this script completes, re-run eval_new_vision_modes.py to benchmark all modes
with the new high-quality hindex.
"""

import asyncio
import json
import sys
import time
from pathlib import Path

colette_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(colette_root / "src"))

from colette.jsonapi import JSONApi
from colette.apidata import APIData

EXISTING_APP_DIR = colette_root / "app_eval_benchmark"
BENCHMARK_DIR    = colette_root / "examples" / "benchmark"
APP_NAME         = "rebuild_hindex"
LLM_GPU_IDS      = [1]   # GPU 1 = LLM for indexing (project allocation)
EMBED_GPU_ID     = 0

colette_api = JSONApi()


async def main():
    pdfs = sorted(BENCHMARK_DIR.glob("*.pdf"))
    print(f"Rebuilding hindex for {len(pdfs)} PDFs at 896×1120 resolution")
    for p in pdfs:
        print(f"  {p.name}")

    # ── build service config ──────────────────────────────────────────────────
    cfg = json.loads((EXISTING_APP_DIR / "config.json").read_text())
    cfg["app"]["repository"] = str(EXISTING_APP_DIR)
    cfg["parameters"]["llm"]["gpu_ids"] = LLM_GPU_IDS
    cfg["parameters"]["input"]["rag"]["gpu_id"] = EMBED_GPU_ID
    cfg["parameters"]["input"]["rag"]["index_protection"] = False  # allow reindex

    # Sanity-check resolution is updated
    w = cfg["parameters"]["llm"]["image_width"]
    h = cfg["parameters"]["llm"]["image_height"]
    print(f"\nImage resolution: {w}×{h}")
    assert w == 896 and h == 1120, f"Expected 896×1120, got {w}×{h} — update config.json first"

    # ── delete existing hindex JSON files so VLM is forced to rebuild them ──────
    # reindex=True clears Tantivy/ChromaDB but never rebuilds hindex JSON files for
    # description modes (vision_hybrid_bm25, vision_semantic, etc.) — those are
    # preserved by design to avoid expensive VLM re-runs. Deleting them here forces
    # a full VLM rebuild, which is the whole point of this script.
    hindex_dir = EXISTING_APP_DIR / "mm_index" / "hindex"
    deleted = 0
    for f in hindex_dir.glob("*.json"):
        if not f.name.startswith("_collection"):
            f.unlink()
            deleted += 1
    print(f"\nDeleted {deleted} stale hindex JSON files (forcing VLM rebuild)")

    # ── tear down any stale service ───────────────────────────────────────────
    try:
        await colette_api.service_delete(APP_NAME)
    except Exception:
        pass

    # ── create service (loads models) ────────────────────────────────────────
    print("\nLoading models…")
    t0 = time.time()
    colette_api.service_create(APP_NAME, APIData(**cfg))
    print(f"Models loaded in {time.time() - t0:.1f}s")

    # ── full clean re-index ───────────────────────────────────────────────────
    index_cfg = {
        "parameters": {
            "input": {
                "data": [str(BENCHMARK_DIR)],
                "preprocessing": {"files": ["pdf"], "dpi": 300},
                "rag": {
                    "retrieval_mode": "vision_hybrid_bm25",
                    "reindex": True,        # clean rebuild — clear ChromaDB + Tantivy
                    "update_index": False,
                    "hindex": {
                        "toc_scan_pages": 20,
                        "add_node_summaries": True,
                        "enable_hindex": True,
                    },
                },
            }
        }
    }

    print(f"\nStarting re-index of {len(pdfs)} PDFs (reindex=True)…")
    print("This rebuilds: hindex JSON files, Tantivy BM25, ChromaDB embeddings")
    t_idx = time.time()

    resp = colette_api.service_index(APP_NAME, APIData(**index_cfg))
    elapsed = time.time() - t_idx

    print(f"\nRe-index complete in {elapsed/60:.1f} min")
    print(f"Response: {resp}")

    # ── verify hindex files were rebuilt ──────────────────────────────────────
    hindex_dir = EXISTING_APP_DIR / "mm_index" / "hindex"
    hindex_files = [f for f in hindex_dir.glob("*.json") if not f.name.startswith("_collection")]
    print(f"\nHindex files in {hindex_dir.name}/: {len(hindex_files)}")
    for f in sorted(hindex_files):
        data = json.loads(f.read_text())
        pages = data.get("page_records", [])
        sample_text = pages[1].get("text_content", "")[:120] if len(pages) > 1 else ""
        print(f"  {f.name}")
        print(f"    page 2 text_content: {sample_text}")

    await colette_api.service_delete(APP_NAME)
    print("\nDone. Run eval_new_vision_modes.py to benchmark all modes.")


if __name__ == "__main__":
    asyncio.run(main())
