"""Test two-stage collection navigation on the 7-doc synthetic benchmark.

COLLECTION_THRESHOLD is temporarily lowered to 5 in hierarchy_builder.py so this
7-doc corpus triggers the two-stage path (cluster → doc → page).

Usage:
    cd /data1/ldejesus/repos/public/colette
    python examples/test_collection_index.py

GPU allocation (per project rule):
    GPU 0 = embedding model
    GPU 1 = LLM (VLM for hindex navigation + generation)
    GPU 2 = spare / evaluate notebook
    GPU 3 = DO NOT USE (external process)
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

colette_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(colette_root / "src"))

app_name        = "eval_benchmark"
app_dir         = str(colette_root / "app_eval_benchmark")
models_repo     = str(colette_root / "models")
embedding_model = "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct"
llm_model       = "Qwen/Qwen3.5-9B"
llm_gpu_ids     = [1]
embedding_gpu   = 0
PDFS_DIR        = str(colette_root / "examples" / "benchmark")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

from colette.jsonapi import JSONApi
from colette.apidata import APIData
from colette.backends.hf.hindex import COLLECTION_THRESHOLD

log.info("COLLECTION_THRESHOLD = %d (should be 5 for this test)", COLLECTION_THRESHOLD)
assert COLLECTION_THRESHOLD <= 7, (
    f"COLLECTION_THRESHOLD={COLLECTION_THRESHOLD} — must be ≤7 to trigger two-stage on the "
    "7-doc synthetic corpus. Lower it in hierarchy_builder.py."
)

colette_api = JSONApi()


async def main():
    # ── 1. Create service ───────────────────────────────────────────────────
    log.info("Creating service '%s'…", app_name)
    try:
        await colette_api.service_delete(app_name)
    except Exception:
        pass

    # Load the saved config (has the correct llm.inference structure) and patch GPU for LLM.
    import json as _json
    saved_cfg = _json.loads((Path(app_dir) / "config.json").read_text())
    saved_cfg["parameters"]["llm"]["gpu_ids"] = llm_gpu_ids
    service_config = saved_cfg
    result = colette_api.service_create(app_name, APIData(**service_config))
    log.info("Service created: %s", result)

    # ── 2. Run service_index with update_index=True ─────────────────────────
    # All 7 hindex files already exist → per-doc work is skipped.
    # The post-loop hook sees collection_index_exists()=False and builds it.
    log.info("Running service_index (update_index=True) to trigger collection index build…")
    index_config = {
        "parameters": {
            "input": {
                "data": [PDFS_DIR],
                "preprocessing": {"files": ["pdf"]},  # ignore qa_dataset.jsonl
                "rag": {
                    "retrieval_mode": "vision_hierarchical",
                    "update_index": True,
                    "reindex": False,
                },
            }
        }
    }
    colette_api.service_index(app_name, APIData(**index_config))
    log.info("service_index complete.")

    # ── 3. Verify collection index was saved ────────────────────────────────
    hindex_dir = Path(app_dir) / "mm_index" / "hindex"
    collection_file = hindex_dir / "_collection_index.json"
    if collection_file.exists():
        idx_data = json.loads(collection_file.read_text())
        clusters = idx_data.get("clusters", [])
        log.info("Collection index saved: %d clusters", len(clusters))
        for c in clusters:
            log.info("  [%s] %s: %d docs — %s",
                     c["cluster_id"], c["label"], len(c["doc_names"]), c["description"])
    else:
        log.error("Collection index NOT found at %s", collection_file)
        return

    # ── 4. Run a few predictions to exercise two-stage navigation ───────────
    questions = [
        "What was the Q2 2025 revenue growth compared to Q1?",
        "Which algorithm performed best in the comparison benchmark?",
        "What are the key risk factors identified in the annual summary?",
    ]

    for q in questions:
        log.info("\n--- Query: %s", q)
        response = colette_api.service_predict(
            app_name,
            APIData(**{
                "parameters": {
                    "input": {
                        "message": q,
                        "rag": {"retrieval_mode": "vision_hierarchical"},
                    }
                }
            }),
        )
        context = response.sources.get("context", [])
        log.info("Retrieved %d page(s):", len(context))
        for item in context:
            log.info("  page=%s  doc=%s  dist=%.3f",
                     item.get("page_number", "?"),
                     Path(item.get("source", "?")).name,
                     item.get("distance", -1))
        log.info("Answer: %s", (response.output or "")[:200])

    # ── 5. Cleanup ─────────────────────────────────────────────────────────
    await colette_api.service_delete(app_name)
    log.info("Done.")


asyncio.run(main())
