# Hierarchical Indexing — Developer Reference

This document describes the internal architecture of the `hindex` backend and how to extend or integrate it.

---

## Package layout

```
src/colette/backends/hf/hindex/
    __init__.py            # Public exports: HierarchyBuilder, HierarchyStore, HierarchyRetriever
    hierarchy_builder.py   # VLM-driven TOC extraction and tree building
    hierarchy_store.py     # JSON persistence (one file per document)
    hierarchy_retriever.py # Agentic tree navigation + page image retrieval
```

---

## Component responsibilities

### `HierarchyBuilder`

Converts a list of rendered page images into a `DocumentHierarchy` tree.

**Inputs**
- `vlm_fn: Callable(images: list[PIL.Image], prompt: str) -> str` — the active VLM wrapped in a simple callable. The builder is backend-agnostic; callers are responsible for wiring the correct model.
- `page_label_map: dict[int, str]` — physical page index → printed page label, sourced from PDF metadata (PyMuPDF `page.get_label()`). Built at preprocessing time so the VLM never reads page numbers from images.
- `toc_scan_pages: int` — how many pages to OCR when searching for a TOC.
- `add_node_summaries: bool` — whether to call the VLM once per node to generate a short summary.

**Algorithm**

```
1. OCR first N pages  → raw text
2. Ask VLM: "is there a TOC? extract it as JSON"  → toc_data
3a. (TOC found)    parse entries → map labels to physical indices → build tree
3b. (no TOC)       ask VLM per page to infer structure → build flat tree
4. Generate doc-level description (first 3 pages)
5. (optional) Generate per-node summaries (first page of each section)
```

**VLM prompt contract**

The builder uses three distinct prompt patterns, each expecting a specific JSON shape:

| Step | Expected output shape |
|---|---|
| TOC detection | `{"has_toc": bool, "entries": [{"title", "level", "page_label"}]}` |
| Structure inference (no TOC) | `[{"title", "start_page", "end_page"}]` |
| Document description | Plain text sentence |
| Node summary | Plain text (≤50 words) |

JSON is extracted from VLM responses with `_parse_json_response()`, which strips markdown fences and falls back to regex extraction for models that add prose around JSON.

---

### `HierarchyStore`

Thin persistence layer. Stores one JSON file per document under `<index_dir>/hindex/`.

Document names are slugified (special characters → `_`, truncated at 200 chars) to create safe filenames.

Key methods:

```python
store.save(hierarchy)       # serialize to JSON
store.load(doc_name)        # deserialize; returns None if not found
store.load_all()            # load every document in the store
store.exists(doc_name)      # presence check
store.delete(doc_name)      # remove a document tree
store.list_doc_names()      # list slugified names of all stored documents
```

---

### `HierarchyRetriever`

Implements agentic navigation: the VLM reasons over the tree text to select relevant nodes, then page images are fetched from the HDF5 kvstore.

**Return format**

The return value is a dict compatible with `RAGImgRetriever.invoke()`:

```python
{
    "ids": [["manual_pdf_page_5", "manual_pdf_page_6"]],
    "distances": [[0.0, 0.0]],
    "metadatas": [[
        {
            "source": "manual.pdf",
            "page_number": 5,
            "crop_label": "full_page",
            "retrieval_source": "hindex",
        },
        ...
    ]],
}
```

`retrieval_source: "hindex"` allows downstream merge logic to distinguish hindex results from embedding or BM25 results.

**Kvstore key convention**

```python
_page_key(doc_name, page_idx) → "<doc_slug>_page_<N>"
```

This must match the keys written by the preprocessing pipeline when storing page images. The integration point is in `DocumentProcessor` — see the `Integration` section below.

**Navigation prompts**

1. *Document selection* (only when >1 document in store): asks VLM to return a JSON array of document names.
2. *Node selection*: asks VLM to return a JSON array of node IDs (the `[XXXX]` codes from the text tree).

Both use `_parse_json_list()` for robust JSON extraction.

Fallback behavior:
- If document selection returns nothing → use first document.
- If node selection returns nothing or no IDs match → fall back to top 2 root nodes, log a warning.

---

## Data model

```python
@dataclass
class HierarchyNode:
    title: str
    node_id: str        # zero-padded 4-digit string, e.g. "0003"
    start_page: int     # 0-based physical page index
    end_page: int       # 0-based physical page index (inclusive)
    summary: str
    children: list[HierarchyNode]

@dataclass
class DocumentHierarchy:
    doc_name: str
    doc_description: str
    page_count: int
    nodes: list[HierarchyNode]
```

Both support `to_dict()` / `from_dict()` for JSON serialization.

`DocumentHierarchy.as_text_tree()` produces the compact text representation fed to the VLM at retrieval time:

```
Document: manual.pdf (120 pages)
[0001] Chapter 1 — Introduction (pages 3–10)
  [0002] 1.1 Safety Warnings (pages 4–5)
  [0003] 1.2 Tools Required (pages 6–7)
[0004] Chapter 2 — Procedures (pages 11–45)
  ...
```

---

## Integration points in the existing pipeline

### 1. `apidata.py` — `RAGObj`

The `retrieval_mode` Literal must include `"vision_hierarchical"`. A new nested config object `HIndexObj` holds hindex-specific parameters:

```python
class HIndexObj(BaseModel):
    toc_scan_pages: int = 20
    add_node_summaries: bool = True
    enable_hindex: bool = False   # used when retrieval_mode == "hybrid"

class RAGObj(BaseModel):
    retrieval_mode: Literal[
        "embedding_retrieval",
        "text_search_retrieval",
        "hybrid",
        "vision_hierarchical",      # ← new
    ] = "embedding_retrieval"
    hindex: HIndexObj = Field(default_factory=HIndexObj)
```

### 2. `rag_img.py` — `RAGImg.index()`

After the standard image preprocessing loop (layout detection, embedding, Tantivy indexing), add a branch:

```python
if retrieval_mode_uses_hindex(rag_config.retrieval_mode, rag_config.hindex):
    builder = HierarchyBuilder(
        vlm_fn=self._vlm_call,          # wrap the existing VLM backend
        page_label_map=doc_page_labels,  # from PyMuPDF during PDF open
        toc_scan_pages=rag_config.hindex.toc_scan_pages,
        add_node_summaries=rag_config.hindex.add_node_summaries,
        logger=self.logger,
    )
    hierarchy = builder.build(doc_name, page_images)
    self.hindex_store.save(hierarchy)
```

`doc_page_labels` is built during the existing PDF rendering loop using `fitz.Document` (already imported in `rag_img.py`):

```python
fitz_doc = fitz.open(pdf_path)
page_label_map = {i: fitz_doc[i].get_label() or str(i + 1) for i in range(len(fitz_doc))}
```

### 3. `rag_img.py` — `RAGImgRetriever.invoke()`

Add a branch for `"vision_hierarchical"` mode and for `"hybrid"` when `enable_hindex` is true:

```python
if retrieval_mode_uses_hindex(retrieval_mode, hindex_config):
    hindex_docs = self.hindex_retriever.invoke(question)
    # Merge into docs the same way text_context is merged
    docs = _merge_hindex_results(docs, hindex_docs)
```

### 4. `tantivy/__init__.py` pattern

Add a helper following the Tantivy convention:

```python
# in a new colette/backends/hf/hindex/utils.py or directly in __init__.py
def retrieval_mode_uses_hindex(retrieval_mode: str, hindex_cfg) -> bool:
    if retrieval_mode == "vision_hierarchical":
        return True
    if retrieval_mode == "hybrid" and hindex_cfg.enable_hindex:
        return True
    return False
```

---

## Testing

Tests live in `tests/`:

| File | Coverage |
|---|---|
| `test_hindex_hierarchy_builder.py` | Builder unit tests — JSON parsing, tree nesting, build() with TOC and without |
| `test_hindex_hierarchy_store.py` | Store unit tests — save/load/delete/list, edge cases |
| `test_hindex_hierarchy_retriever.py` | Retriever unit tests — navigation, top_k, dedup, multi-doc, metadata |
| `test_hindex_integration.py` | End-to-end: builder → store → retriever → format compatibility with RAGImgRetriever |

All tests use fake VLMs and fake kvstores — no GPU or real models required.

Run with:

```bash
pytest tests/test_hindex_*.py -v
```

---

## Extending hindex

### Custom VLM backend

`HierarchyBuilder` and `HierarchyRetriever` accept any `vlm_fn: Callable(images, prompt) -> str`. To use a different model (e.g., a smaller distilled VLM for indexing):

```python
def my_vlm(images, prompt):
    # call any model
    return response_text

builder = HierarchyBuilder(vlm_fn=my_vlm, ...)
```

### Custom TOC parser

Override `HierarchyBuilder._detect_and_parse_toc()` to handle domain-specific TOC formats (e.g., multi-column financial reports, legal document indices).

### Custom node selection strategy

Override `HierarchyRetriever._navigate_tree()` to use a different selection strategy — e.g., keyword matching against node titles instead of VLM reasoning, or a hybrid of both.
