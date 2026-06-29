# Hierarchical Document Indexing (hindex)

Hierarchical indexing (`hindex`) is a retrieval strategy that understands document *structure* instead of relying solely on semantic similarity or keyword matching.

Rather than splitting documents into fixed-size chunks, hindex:

1. Uses the VLM (already loaded for V-RAG) to **OCR page images** and extract text.
2. Detects the document's **table of contents** (or infers a structure when none exists).
3. Builds a **hierarchical JSON tree** — a machine-readable version of the document's own organization.
4. At query time, the VLM **navigates the tree** to identify which sections are relevant, then retrieves the corresponding **page images** from the existing store.

This approach is inspired by the PageIndex concept (VectifyAI) but implemented from scratch as a vision-first strategy that reuses Colette's existing VLM infrastructure, preprocessing pipeline, and HDF5 image store.

---

## When to use hindex

| Use case | Recommended mode |
|---|---|
| Structured documents (technical manuals, financial reports, legal contracts) | `vision_hierarchical` or `hybrid` |
| Visual-heavy documents (schematics, photos, mixed layouts) | `embedding_retrieval` (standard V-RAG) |
| Plain text documents | `text_search_retrieval` |
| Maximum coverage (all signals) | `hybrid` — includes hindex when `enable_hindex: true` |

hindex shines when:
- The document has a clear hierarchical structure (chapters → sections → subsections).
- Answers are concentrated in a specific, named section rather than scattered throughout.
- You need **traceable retrieval** — every result is grounded in a specific named section and page range.

---

## Configuration

### Enable hindex in retrieval mode

```json
{
  "rag": {
    "retrieval_mode": "vision_hierarchical",
    "hindex": {
      "toc_scan_pages": 20,
      "add_node_summaries": true,
      "top_k": 4
    }
  }
}
```

To use hindex alongside embedding and text search (recommended for important documents):

```json
{
  "rag": {
    "retrieval_mode": "hybrid",
    "hindex": {
      "enable_hindex": true,
      "toc_scan_pages": 20,
      "add_node_summaries": true
    }
  }
}
```

### Parameters

| Parameter | Default | Description |
|---|---|---|
| `toc_scan_pages` | `20` | Number of pages to scan when looking for a table of contents |
| `add_node_summaries` | `true` | Whether the VLM generates a short summary for each section at index time |
| `top_k` | `4` | Maximum number of page images returned per query |
| `enable_hindex` | `false` | In `hybrid` mode, set to `true` to include hierarchy-based results |

---

## How indexing works

When `service_index` is called with hindex enabled, the following steps run after the standard preprocessing (PDF → page images):

```
PDF
  ↓  (existing preprocessing pipeline)
Page images (300 DPI, stored in HDF5 kvstore)
  ↓
VLM OCR on first N pages
  ↓
TOC detected?  ──YES──→ Parse entries → map to physical page indices (from PDF metadata)
      │                                        ↓
     NO                             Build hierarchical JSON tree
      ↓
 Infer structure from full document content
      ↓
 Generate section summaries (optional)
      ↓
 Persist JSON tree to mm_index/hindex/<doc>.json
```

The page-label-to-physical-index mapping is resolved from PDF metadata (via PyMuPDF) during preprocessing, so the VLM **never needs to read printed page numbers** from images.

---

## How retrieval works

```
User question
    ↓
[If multiple docs] VLM selects relevant document(s)
    ↓
VLM receives compact text tree, selects node IDs
    ↓
Page images for selected page ranges fetched from HDF5 kvstore
    ↓
Returned as retrieval results (same format as V-RAG embedding retrieval)
    ↓
[In hybrid mode] Merged with embedding and/or BM25 results
    ↓
LLM inference with full image context
```

Every retrieved result includes metadata:
- `source` — document name
- `page_number` — physical page index
- `crop_label` — always `full_page` for hindex results
- `retrieval_source` — `hindex` (allows downstream filtering)

---

## Index files

hindex creates one JSON file per document under:

```
<app_dir>/mm_index/hindex/<document_slug>.json
```

These files are human-readable and can be inspected or edited manually if needed.

Example tree structure:

```json
{
  "doc_name": "technical_manual.pdf",
  "doc_description": "Maintenance procedures for the XR-7 engine.",
  "page_count": 120,
  "nodes": [
    {
      "title": "Chapter 1 — Introduction",
      "node_id": "0001",
      "start_page": 3,
      "end_page": 10,
      "summary": "Overview of the XR-7 engine and safety precautions.",
      "children": [
        {
          "title": "1.1 Safety Warnings",
          "node_id": "0002",
          "start_page": 4,
          "end_page": 5,
          "summary": "Critical safety warnings before starting any procedure.",
          "children": []
        }
      ]
    }
  ]
}
```

---

## Limitations

- **Indexing cost**: hindex makes several VLM calls per document at index time (OCR + TOC detection + optional summaries). For a 100-page document with 10 sections, expect ~15–30 VLM calls. Since vLLM supports concurrent batching, multiple documents can be indexed in parallel.
- **No sub-page granularity**: hindex retrieves full pages, not cropped regions. For fine-grained visual retrieval (tables, figures), combine with `embedding_retrieval` in hybrid mode.
- **Requires structured content**: Documents without any discernible structure (collections of unrelated images, slide decks with no logical order) will produce a shallow one-level tree.
