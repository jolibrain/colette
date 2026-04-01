# Text Search Engine

Colette supports lexical retrieval through a text search engine using BM25-style ranking.

## Retrieval Modes

`parameters.input.rag.retrieval_mode` accepts only:

- `embedding_retrieval`: embedding retrieval only
- `text_search_retrieval`: text search retrieval only
- `hybrid`: runs embedding and text search retrieval together

## Response Sources

Depending on mode, `service_predict(...).sources` contains:

- `sources['context']`: embedding/image context
- `sources['text_context']`: text-search hits

Mode behavior:

- `embedding_retrieval` -> `context`
- `text_search_retrieval` -> `text_context`
- `hybrid` -> both keys

## Configuration Keys

Text-search configuration lives under `parameters.input.rag`:

- `text_search_engine_top_k`
- `text_search_engine_fields`
- `text_search_engine_max_chars_per_doc`
- `text_search_engine_max_total_chars`

## Index Paths

Default on-disk paths:

- HF backend: `<app_repository>/mm_index/text_search_engine`
- LangChain backend: `<app_repository>/index/text_search_engine`

## Notes

- User-facing naming is engine-agnostic to allow backend replacement without API renaming.
