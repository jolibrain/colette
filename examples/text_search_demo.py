"""
Text search engine demo (BM25)
===============================
This script demonstrates Colette's lexical text search engine
in isolation - no GPU, no embedding model, no running service required.

It shows:
    1. Indexing text extracted from the same PDF corpus style used by examples
    2. BM25 search - how keyword/phrase queries work
    3. Field filtering - restricting results to a specific document type
    4. The text budget cap - how injected context is trimmed for the LLM prompt
    5. Per-request retrieval_mode override and "hybrid" retrieval mode naming

Implementation note: this demo focuses on text-search behavior and stays engine-agnostic.

Run with:
        python examples/text_search_demo.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import fitz

# ── Colette imports ──────────────────────────────────────────────────────────
from colette.backends.tantivy import TantivyIndex, build_text_context
from colette.apidata import RAGObj, merge_rag_config


# ── Simple logger that prints to stdout ──────────────────────────────────────
class PrintLogger:
    def info(self, msg, *_): print(f"[INFO]  {msg}")
    def debug(self, msg, *_): print(f"[DEBUG] {msg}")
    def warning(self, msg, *_): print(f"[WARN]  {msg}")


# Fallback sample corpus (used only when no PDF is found).
FALLBACK_DOCUMENTS = [
    {
        "doc_id": "user_manual.pdf:page:1",
        "source": "user_manual.pdf",
        "page_number": 1,
        "crop_label": None,
        "label": None,
        "content": (
            "Chapter 1: Introduction. This manual describes the installation, "
            "configuration, and troubleshooting of the fuel pump relay. "
            "The relay controls power to the electric fuel pump."
        ),
    },
    {
        "doc_id": "user_manual.pdf:page:2",
        "source": "user_manual.pdf",
        "page_number": 2,
        "crop_label": None,
        "label": None,
        "content": (
            "Chapter 2: Installation. Mount the relay in a dry location. "
            "Connect the 30A terminal to the battery positive. "
            "Connect terminal 87 to the fuel pump power wire."
        ),
    },
    {
        "doc_id": "user_manual.pdf:page:3",
        "source": "user_manual.pdf",
        "page_number": 3,
        "crop_label": None,
        "label": None,
        "content": (
            "Chapter 3: Troubleshooting. If the fuel pump does not start, "
            "check the relay coil resistance (70–80 Ω). "
            "Replace the relay if the coil reads open circuit."
        ),
    },
    {
        "doc_id": "parts_catalog.pdf:page:1",
        "source": "parts_catalog.pdf",
        "page_number": 1,
        "crop_label": "table",
        "label": "parts",
        "content": (
            "Part number: 12V-RELAY-30A. Description: fuel pump relay 30 amp. "
            "Compatibility: all models 2018–2024. Price: $12.50."
        ),
    },
    {
        "doc_id": "parts_catalog.pdf:page:2",
        "source": "parts_catalog.pdf",
        "page_number": 2,
        "crop_label": "table",
        "label": "parts",
        "content": (
            "Part number: PUMP-EFI-001. Description: electric fuel pump assembly. "
            "Flow rate: 80 L/h. Pressure: 3.0 bar. Compatible with relay 12V-RELAY-30A."
        ),
    },
    {
        "doc_id": "service_note.pdf:page:1",
        "source": "service_note.pdf",
        "page_number": 1,
        "crop_label": "text",
        "label": "service",
        "content": (
            "Service Note SN-2024-07: Updated torque spec for engine cover bolts. "
            "Tighten to 22 Nm in a star pattern. Do not reuse old gasket."
        ),
    },
]


def separator(title: str):
    print(f"\n{'-'*60}")
    print(f"  {title}")
    print('-'*60)


def _collect_example_pdfs(repo_root: Path) -> list[Path]:
    candidates = [
        repo_root / "docs" / "pdf",
    ]
    pdfs: list[Path] = []
    for directory in candidates:
        if directory.exists():
            pdfs.extend(sorted(directory.glob("*.pdf")))
    return pdfs


def _build_documents_from_pdfs(pdf_files: list[Path]) -> list[dict]:
    documents: list[dict] = []
    for pdf_path in pdf_files:
        with fitz.open(pdf_path) as pdf_doc:
            for page_idx in range(pdf_doc.page_count):
                page_no = page_idx + 1
                page_text = pdf_doc.load_page(page_idx).get_text("text").strip()
                if not page_text:
                    continue
                documents.append(
                    {
                        "doc_id": f"{pdf_path.name}:page:{page_no}",
                        "source": pdf_path.name,
                        "page_number": page_no,
                        "crop_label": "text",
                        "label": None,
                        "content": page_text,
                    }
                )
    return documents


def _load_rag_defaults(repo_root: Path) -> RAGObj:
    config_path = repo_root / "src" / "colette" / "config" / "vrag_default_index.json"
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        rag_cfg = cfg.get("parameters", {}).get("input", {}).get("rag", {})
        return RAGObj(**rag_cfg)
    except Exception:
        # Fallback to model defaults if config is unavailable.
        return RAGObj()


def main():
    logger = PrintLogger()
    repo_root = Path(__file__).resolve().parents[1]
    pdf_files = _collect_example_pdfs(repo_root)
    rag_defaults = _load_rag_defaults(repo_root)

    if pdf_files:
        documents = _build_documents_from_pdfs(pdf_files)
        corpus_source = f"project PDFs ({', '.join(p.name for p in pdf_files)})"
    else:
        documents = FALLBACK_DOCUMENTS
        corpus_source = "fallback in-memory sample corpus"

    with tempfile.TemporaryDirectory() as tmpdir:
        index_path = Path(tmpdir) / "text_search_engine_demo"
        index = TantivyIndex(index_path, logger)

        # ─────────────────────────────────────────────────────────────────────
        # 1. INDEX
        # ─────────────────────────────────────────────────────────────────────
        separator("1. Indexing documents")
        index.add_documents(documents, recreate=True)
        print(f"Corpus source: {corpus_source}")
        print(f"Indexed {len(documents)} documents into {index_path}")
        print(
            "RAG defaults from JSON: "
            f"retrieval_mode={rag_defaults.retrieval_mode!r}, "
            f"text_search_engine_top_k={rag_defaults.text_search_engine_top_k}, "
            f"text_search_engine_max_chars_per_doc={rag_defaults.text_search_engine_max_chars_per_doc}, "
            f"text_search_engine_max_total_chars={rag_defaults.text_search_engine_max_total_chars}"
        )

        # ─────────────────────────────────────────────────────────────────────
        # 2. PLAIN BM25 SEARCH
        #    The user's question is sent verbatim; the text-search backend tokenizes it and
        #    scores each document using BM25 (tf-idf variant, same as Lucene).
        # ─────────────────────────────────────────────────────────────────────
        separator("2. BM25 search - plain question")
        query = "document title sources errors"
        hits = index.search(query, limit=rag_defaults.text_search_engine_top_k, fields=rag_defaults.text_search_engine_fields)
        print(f"Query : {query!r}")
        for i, h in enumerate(hits, 1):
            print(f"  Hit {i}: [{h['source']} p{h['page_number']}] score={h['score']:.3f}")
            print(f"          {h['content'][:80]}...")

        # ─────────────────────────────────────────────────────────────────────
        # 3. PHRASE SEARCH
        #    Wrap a phrase in quotes for exact-phrase matching.
        # ─────────────────────────────────────────────────────────────────────
        separator("3. Phrase search")
        phrase_query = '"source"'
        hits_phrase = index.search(
            phrase_query,
            limit=rag_defaults.text_search_engine_top_k,
            fields=rag_defaults.text_search_engine_fields,
        )
        print(f"Query : {phrase_query!r}")
        for i, h in enumerate(hits_phrase, 1):
            print(f"  Hit {i}: [{h['source']} p{h['page_number']}] score={h['score']:.3f}")

        # ─────────────────────────────────────────────────────────────────────
        # 4. FIELD FILTER
        #    Only return documents whose crop_label equals "text".
        #    In the HF image-RAG pipeline this can restrict by layout class
        #    (text / figure / table) when those labels are available.
        # ─────────────────────────────────────────────────────────────────────
        separator("4. Field filter - only 'text' documents")
        hits_table = index.search(
            "document",
            limit=rag_defaults.text_search_engine_top_k,
            fields=rag_defaults.text_search_engine_fields,
            crop_label="text",
        )
        print("Query : 'document'  filter: crop_label='text'")
        if hits_table:
            for i, h in enumerate(hits_table, 1):
                print(f"  Hit {i}: [{h['source']} p{h['page_number']}] crop={h['crop_label']!r}  score={h['score']:.3f}")
        else:
            print("  (no hits - crop_label filter excluded all matches)")

        # ─────────────────────────────────────────────────────────────────────
        # 5. TEXT BUDGET CAP
        #    build_text_context() trims results to fit in the LLM context window.
        #    text_search_engine_max_chars_per_doc limits each document; text_search_engine_max_total_chars
        #    caps the total injected text.
        # ─────────────────────────────────────────────────────────────────────
        separator("5. Text budget cap")
        hits_all = index.search("document", limit=rag_defaults.text_search_engine_top_k, fields=rag_defaults.text_search_engine_fields)
        bounded_hits, context_str = build_text_context(
            hits_all,
            max_chars_per_doc=rag_defaults.text_search_engine_max_chars_per_doc,
            max_total_chars=rag_defaults.text_search_engine_max_total_chars,
        )
        print(f"Raw hits   : {len(hits_all)}")
        print(f"After budget: {len(bounded_hits)} hits, {len(context_str)} chars total")
        print("\n--- Injected prompt block (what the LLM sees) ---")
        print(context_str)

        # ─────────────────────────────────────────────────────────────────────
        # 6. PER-REQUEST RETRIEVAL_MODE OVERRIDE via merge_rag_config
        #    The service is configured with mode="embedding_retrieval" (default).
        #    A single request can override it to "text_search_retrieval" or "hybrid".
        # ─────────────────────────────────────────────────────────────────────
        separator("6. Per-request retrieval_mode override (including hybrid)")

        # Service-level config (loaded when the service is created)
        service_rag = rag_defaults

        # Request-level override (sent by the API caller at query time)
        request_rag = RAGObj(retrieval_mode="hybrid", text_search_engine_top_k=service_rag.text_search_engine_top_k)

        merged = merge_rag_config(service_rag, request_rag)
        print(f"Service config : retrieval_mode={service_rag.retrieval_mode!r}  text_search_engine_top_k={service_rag.text_search_engine_top_k}")
        print(f"Request override: retrieval_mode={request_rag.retrieval_mode!r}  text_search_engine_top_k={request_rag.text_search_engine_top_k}")
        print("Note: retrieval_mode='hybrid' is the combined mode (embedding + text search).")
        print(f"Effective config: retrieval_mode={merged.retrieval_mode!r}  text_search_engine_top_k={merged.text_search_engine_top_k}")

        # Only the explicitly set fields in request_rag override; everything
        # else comes from the service config.
        request_rag_partial = RAGObj(text_search_engine_top_k=10)  # only override top_k
        merged_partial = merge_rag_config(service_rag, request_rag_partial)
        print(f"\nPartial override (top_k only):")
        print(f"  retrieval_mode={merged_partial.retrieval_mode!r}  text_search_engine_top_k={merged_partial.text_search_engine_top_k}")

        print("\nDone.")


if __name__ == "__main__":
    main()
