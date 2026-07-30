"""Probe the existing VLM's OCR capabilities with different prompts.

Compares four approaches on a benchmark page that has a complex table:
  1. PyMuPDF direct text extraction (existing fast path for digital PDFs)
  2. Current hindex text_content (stored — 300-char limit)
  3. Enhanced OCR prompts (full structured markdown — what the model can do)

GPU allocation: GPU 2 (LLM for eval/testing). DO NOT use GPU 3.

Usage:
  python examples/test_vlm_ocr.py
"""

import asyncio
import json
import sys
import time
from pathlib import Path

import fitz  # PyMuPDF
from PIL import Image

colette_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(colette_root / "src"))

from colette.jsonapi import JSONApi
from colette.apidata import APIData

EXISTING_APP_DIR = colette_root / "app_eval_benchmark"
BENCHMARK_DIR    = colette_root / "examples" / "benchmark"
APP_NAME         = "vlm_ocr_test"
LLM_GPU_IDS      = [2]

# Page 2 of this doc has the full benchmark comparison table
TARGET_PDF  = BENCHMARK_DIR / "VTQ-Q2-2025-Benchmarks.pdf"
TARGET_PAGE = 2  # 1-indexed

# Resolution configs to compare: (label, image_width, image_height)
RESOLUTIONS = [
    ("low_res  320×480  (current)", 320,  480),
    ("mid_res  896×1120 (half-budget)", 896, 1120),
    ("high_res 1120×1400 (near-max)", 1120, 1400),
]

# Single representative prompt — table extraction
TABLE_PROMPT = (
    "Extract every table from this page as markdown. "
    "Include all column headers and every data row with exact values. "
    "If there are multiple tables, label each one. "
    "Return only the markdown tables, nothing else."
)


def render_page_image(pdf_path: Path, page_number: int) -> Image.Image:
    with fitz.open(pdf_path) as doc:
        page = doc.load_page(page_number - 1)
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0))  # ~144 dpi
        return Image.frombytes("RGB", [pix.width, pix.height], pix.samples)


def pymupdf_text(pdf_path: Path, page_number: int) -> str:
    with fitz.open(pdf_path) as doc:
        return doc.load_page(page_number - 1).get_text("text").strip()


colette_api = JSONApi()


async def main():
    print(f"Target: {TARGET_PDF.name} — page {TARGET_PAGE}")
    page_image = render_page_image(TARGET_PDF, TARGET_PAGE)
    print(f"Rendered image: {page_image.size}\n")

    # ── 1. PyMuPDF (gold standard for digital PDFs) ──────────────────────────
    raw_text = pymupdf_text(TARGET_PDF, TARGET_PAGE)
    print("=" * 70)
    print("1. PyMuPDF direct extraction (no AI, digital PDF only)")
    print("=" * 70)
    print(raw_text)
    print(f"\n[{len(raw_text)} chars]")

    # ── 2. Current hindex output (already stored) ────────────────────────────
    hindex_file = (
        EXISTING_APP_DIR / "mm_index" / "hindex" /
        f"_data1_ldejesus_repos_public_colette_examples_benchmark_{TARGET_PDF.stem}_pdf.json"
    )
    if hindex_file.exists():
        h = json.loads(hindex_file.read_text())
        prec = h.get("page_records", [])
        p = prec[TARGET_PAGE - 1] if len(prec) >= TARGET_PAGE else {}
        print("\n" + "=" * 70)
        print("2. Current hindex output (stored, 300-char VLM limit)")
        print("=" * 70)
        print(f"text_content : {p.get('text_content', '')}")
        print(f"summary      : {p.get('summary', '')}")
        print(f"visual_desc  : {p.get('visual_description', '')}")

    # ── load model once, swap resolution between runs ────────────────────────
    cfg = json.loads((EXISTING_APP_DIR / "config.json").read_text())
    cfg["parameters"]["llm"]["gpu_ids"] = LLM_GPU_IDS
    cfg["app"]["repository"] = str(EXISTING_APP_DIR / "vlm_ocr_probe")

    try:
        await colette_api.service_delete(APP_NAME)
    except Exception:
        pass

    print(f"\nLoading model on GPU {LLM_GPU_IDS}…")
    t_load = time.time()
    colette_api.service_create(APP_NAME, APIData(**cfg))
    print(f"Model loaded in {time.time() - t_load:.1f}s")

    service   = colette_api.services[APP_NAME]
    llmmodel  = service.llmmodel
    vlm_fn    = service.inputc.ragobj.vlm_fn

    # ── 3. Resolution sweep — same prompt, three resolutions ─────────────────
    for i, (label, w, h) in enumerate(RESOLUTIONS, start=3):
        llmmodel.image_width  = w
        llmmodel.image_height = h
        print("\n" + "=" * 70)
        print(f"{i}. {label}")
        print("=" * 70)
        t0 = time.time()
        try:
            result = vlm_fn([page_image], TABLE_PROMPT, max_new_tokens=512)
            print(result)
            print(f"\n[{len(result)} chars | {time.time() - t0:.1f}s]")
        except Exception as exc:
            print(f"ERROR: {exc}")

    await colette_api.service_delete(APP_NAME)
    print("\nDone.")


if __name__ == "__main__":
    asyncio.run(main())
