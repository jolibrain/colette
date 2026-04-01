from types import SimpleNamespace

from langchain_core.documents import Document

from colette.apidata import RAGObj
from colette.backends.hf.rag.rag_img import RAGImgRetriever
from colette.backends.langchain.rag.rag_txt import RAGTxt


class _DummyIndexDB:
    def __init__(self):
        self.calls = 0

    def query(self, query_texts, n_results, where=None):
        self.calls += 1
        return {
            "ids": [["img-1", "img-2"]],
            "distances": [[0.1, 0.3]],
            "metadatas": [[{"source": "a.pdf"}, {"source": "b.pdf"}]],
        }


class _DummyTantivy:
    def __init__(self):
        self.calls = 0

    def search(self, query, *, limit, fields, crop_label=None):
        self.calls += 1
        return [
            {
                "doc_id": "txt-1",
                "source": "manual.pdf",
                "page_number": 1,
                "crop_label": "full_page",
                "content": "The answer is in this paragraph.",
                "score": 1.0,
            }
        ]


class _DummyLogger:
    def debug(self, *_args, **_kwargs):
        return None

    def warning(self, *_args, **_kwargs):
        return None


def test_hf_retriever_vector_mode_uses_only_vector_search():
    indexdb = _DummyIndexDB()
    tantivy = _DummyTantivy()
    retriever = RAGImgRetriever(
        indexlib="chromadb",
        indexdb=indexdb,
        top_k=1,
        remove_duplicates=False,
        filter_width=-1,
        filter_height=-1,
        app_repository=None,
        kvstore=None,
        text_search_engine_index=tantivy,
        logger=_DummyLogger(),
    )

    docs = retriever.invoke("where is it", 5, retrieval_mode="embedding_retrieval")

    assert indexdb.calls == 1
    assert tantivy.calls == 0
    assert docs["ids"][0] == ["img-1"]
    assert "text_context" not in docs


def test_hf_retriever_text_search_mode_uses_only_text_search():
    indexdb = _DummyIndexDB()
    tantivy = _DummyTantivy()
    retriever = RAGImgRetriever(
        indexlib="chromadb",
        indexdb=indexdb,
        top_k=1,
        remove_duplicates=False,
        filter_width=-1,
        filter_height=-1,
        app_repository=None,
        kvstore=None,
        text_search_engine_index=tantivy,
        logger=_DummyLogger(),
    )

    docs = retriever.invoke("where is it", 5, retrieval_mode="text_search_retrieval", text_search_engine_top_k=2)

    assert indexdb.calls == 0
    assert tantivy.calls == 1
    assert docs["ids"][0] == []
    assert len(docs["text_context"]) == 1


def test_hf_retriever_both_mode_uses_both_searches():
    indexdb = _DummyIndexDB()
    tantivy = _DummyTantivy()
    retriever = RAGImgRetriever(
        indexlib="chromadb",
        indexdb=indexdb,
        top_k=2,
        remove_duplicates=False,
        filter_width=-1,
        filter_height=-1,
        app_repository=None,
        kvstore=None,
        text_search_engine_index=tantivy,
        logger=_DummyLogger(),
    )

    docs = retriever.invoke("where is it", 5, retrieval_mode="hybrid", text_search_engine_top_k=2)

    assert indexdb.calls == 1
    assert tantivy.calls == 1
    assert docs["ids"][0] == ["img-2", "img-1"]
    assert len(docs["text_context"]) == 1


def test_rag_txt_retrieve_merges_embedding_and_text_search_documents():
    rag_txt = RAGTxt()
    rag_txt.ad = SimpleNamespace(rag=RAGObj(retrieval_mode="hybrid", text_search_engine_top_k=2))
    rag_txt.rag_retriever = SimpleNamespace(
        invoke=lambda _q: [
            Document(page_content="vector content", metadata={"source": "a.pdf", "page": 1}),
        ]
    )
    rag_txt.text_search_engine_index = SimpleNamespace(
        search=lambda *_args, **_kwargs: [
            {
                "source": "b.pdf",
                "page_number": 2,
                "crop_label": "text",
                "content": "tantivy content",
                "score": 0.9,
            }
        ]
    )

    docs = rag_txt.retrieve("question", request_rag=RAGObj(retrieval_mode="hybrid", text_search_engine_top_k=2))

    assert len(docs) == 2
    sources = [doc.metadata.get("source") for doc in docs]
    assert "a.pdf" in sources
    assert "b.pdf" in sources