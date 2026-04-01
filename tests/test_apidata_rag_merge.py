from colette.apidata import RAGObj, merge_rag_config


def test_merge_rag_config_keeps_base_defaults():
    base = RAGObj()
    merged = merge_rag_config(base, None)

    assert merged.retrieval_mode == "embedding_retrieval"
    assert merged.text_search_engine_top_k == 4
    assert merged.text_search_engine_max_chars_per_doc == 1500
    assert merged.text_search_engine_max_total_chars == 6000


def test_merge_rag_config_applies_partial_override_only():
    base = RAGObj(retrieval_mode="embedding_retrieval", top_k=8, text_search_engine_top_k=2)
    override = RAGObj(retrieval_mode="hybrid")

    merged = merge_rag_config(base, override)

    assert merged.retrieval_mode == "hybrid"
    assert merged.top_k == 8
    assert merged.text_search_engine_top_k == 2


def test_merge_rag_config_builds_from_override_when_base_missing():
    override = RAGObj(retrieval_mode="text_search_retrieval", text_search_engine_top_k=6)
    merged = merge_rag_config(None, override)

    assert merged.retrieval_mode == "text_search_retrieval"
    assert merged.text_search_engine_top_k == 6