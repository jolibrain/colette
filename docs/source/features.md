# Features

## Vision RAG (V-RAG)

Colette's primary retrieval mode treats every document page as an image. PDFs are rendered at high DPI, optionally split into layout-detected crops (text blocks, figures, tables, full pages), embedded with a vision-language embedding model (e.g. `gme-Qwen2-VL-2B-Instruct`), and stored in ChromaDB.

At query time the question is embedded and the most similar image crops are retrieved and passed directly to a VLM (e.g. `Qwen3-VL`). This preserves all visual structure — tables, diagrams, formulas, spatial layout — that text extraction would lose.

## Hybrid Retrieval (Embedding + BM25)

Set `retrieval_mode: hybrid` to run vector search and BM25 lexical search in parallel. The embedding channel finds semantically similar crops; the BM25 channel (powered by Tantivy) finds exact keyword matches. Both result sets are injected into the LLM context.

Hybrid mode is especially effective when queries contain product codes, part numbers, or rare identifiers that embedding models cannot reliably distinguish. See [Text Search Engine](text_search_engine.md) for configuration details.

## HyDE — Hypothetical Document Embeddings

When `llm.use_hyde: true`, Colette generates a short *hypothetical answer passage* with the LLM before retrieval, then embeds that passage instead of the raw question. Because the embedding model was trained on document-to-document similarity, a hypothetical passage lands much closer in vector space to real document content than a bare question does — even if the passage is factually imprecise.

The original question is still sent to the LLM unchanged for final answer generation. HyDE adds one LLM call per query; keep `hyde_num_tokens` low (128–256) to minimise latency. See [Configuration](users/configuration.md#hyde--hypothetical-document-embeddings) for usage.

## Layout Detection

When `parameters.input.rag.ragm.layout_detection: true` (default), Colette uses a layout detector to segment each page into typed crops before embedding:

| Crop label | Content |
|---|---|
| `text` | Paragraph / body text blocks |
| `figure` | Charts, photos, diagrams |
| `table` | Structured tabular data |
| `full_page` | Whole page fallback |

Querying can be restricted to one or more crop types via the `crop_label` parameter (e.g. `crop_label: "table"` to search only tables).

## Text RAG

In addition to V-RAG, Colette supports a classical text-extraction pipeline using `langchain` + `unstructured`. Documents are chunked, embedded with a text embedding model, and stored in ChromaDB. Useful when documents are text-only and visual fidelity is not required.

## Conversational Mode

Setting `llm.conversational: true` enables session-based multi-turn conversations. Each turn references the same `session_id`; prior turns are summarised and injected as context. See [Conversational RAG](conversational.md) for details.

## Query Rephrasing

When `llm.query_rephrasing: true`, the LLM rewrites the user question into a form more suited to embedding retrieval before the vector search step. This is distinct from HyDE: rephrasing rewrites the question; HyDE generates a hypothetical answer. Both can be combined.

## System Prompt

`llm.system_prompt` injects a custom system message before every user turn. Use it to enforce strict RAG behaviour, set language, or give the model a persona:

```json
"system_prompt": "Answer only using the provided context. If the answer is not in the context, say so explicitly."
```

## Multiple Inference Backends

Colette supports several LLM and embedding backends:

| Backend | Use case |
|---|---|
| `huggingface` | Local models via Transformers (default for V-RAG) |
| `vllm` | High-throughput local serving |
| `ollama` | Lightweight local models |
| `vllm_client` | Remote vLLM server |

## Example Notebooks

The `examples/` directory contains ready-to-run Jupyter notebooks:

| Notebook | Description |
|---|---|
| `get_start_python_api.ipynb` | End-to-end quickstart: index PDFs, run a RAG query |
| `retrieval_debugger.ipynb` | Step-by-step retrieval debugger: embedding search, HyDE comparison, BM25 inspection, PCA visualisation |
| `crop_viewer.ipynb` | Browse every indexed image crop by page and label |
| `visualize_embeddings.ipynb` | PCA scatter of the full ChromaDB index |
