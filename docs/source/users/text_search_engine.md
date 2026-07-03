# Text Search Engine

Colette supports lexical retrieval through a text search engine using BM25-style ranking (powered by [Tantivy](https://github.com/quickwit-oss/tantivy)).

## Retrieval Modes

`parameters.input.rag.retrieval_mode` accepts:

- `embedding_retrieval`: vector embedding retrieval only (default)
- `text_search_retrieval`: BM25 text search only
- `hybrid`: runs embedding and text search in parallel; both result sets are sent to the LLM

## Query Processing

Before a question is sent to the BM25 engine, it goes through **keyword extraction**:

1. Hyphens are replaced by spaces — e.g. `PRODUCT-01-A` becomes `PRODUCT 01 A`, matching the tokens stored during indexing.
2. English and French stop words are removed (`what`, `is`, `the`, `of`, `quel`, `est`, …).
3. Single-character tokens are dropped.

This prevents natural-language questions from being sent verbatim to Tantivy, which would escape special characters (turning `PRODUCT-01-A` into `PRODUCT\-01\-A`, an unmatchable token) and degrade BM25 ranking.

**Example:**

| Input question | BM25 query sent |
|---|---|
| `What is the output voltage of PRODUCT-01-A?` | `output voltage PRODUCT 01 A` |
| `Quelles sont les sources d'erreurs identifiées ?` | `sources erreurs identifiées` |

## Response Sources

Depending on mode, `service_predict(...).sources` contains:

- `sources['context']`: embedding/image crops
- `sources['text_context']`: text-search page hits

| Mode | Keys returned |
|---|---|
| `embedding_retrieval` | `context` |
| `text_search_retrieval` | `text_context` |
| `hybrid` | both |

## Configuration Keys

All text-search settings live under `parameters.input.rag`:

| Key | Default | Description |
|---|---|---|
| `text_search_engine_top_k` | `4` | Number of page hits retrieved from the Tantivy index |
| `text_search_engine_fields` | `["content", "source"]` | Fields queried during full-text search |
| `text_search_engine_max_chars_per_doc` | `4000` | Max characters kept per retrieved page hit |
| `text_search_engine_max_total_chars` | `16000` | Total character budget contributed to the LLM context |

> **Tuning tip:** The default 4000 chars/doc covers 100 % of typical datasheet pages. Raising `text_search_engine_max_total_chars` beyond 16000 may improve recall on multi-page answers but increases LLM context usage.

## Index Paths

Default on-disk paths after indexing:

- HF backend (V-RAG): `<app_repository>/mm_index/text_search_engine/`
- LangChain backend (text RAG): `<app_repository>/index/text_search_engine/`

The text search index is only created when `retrieval_mode` is `text_search_retrieval` or `hybrid` at index time.

## Notes

- User-facing naming is engine-agnostic to allow backend replacement without API renaming.
- The BM25 index stores page-level text extracted during the PDF-to-image pipeline. Each Tantivy document corresponds to one PDF page.
- To inspect BM25 results interactively, use `examples/retrieval_debugger.ipynb`.
