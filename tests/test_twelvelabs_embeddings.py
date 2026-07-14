import os

import pytest

from colette.backends.langchain.rag.twelvelabs_embeddings import (
    DEFAULT_MARENGO_MODEL,
    MarengoEmbeddings,
    analyze_video,
)


@pytest.mark.smoke
def test_marengo_embeddings_monkeypatch(monkeypatch):
    """No-network unit test: the LangChain Embeddings protocol is wired correctly."""

    class DummySegment:
        float_ = [0.1] * 512

    class DummyTextEmbedding:
        segments = [DummySegment()]

    class DummyResp:
        text_embedding = DummyTextEmbedding()

    class DummyEmbed:
        def create(self, *, model_name, text):
            assert model_name == DEFAULT_MARENGO_MODEL
            assert isinstance(text, str)
            return DummyResp()

    class DummyClient:
        embed = DummyEmbed()

    monkeypatch.setattr(
        "colette.backends.langchain.rag.twelvelabs_embeddings._make_client",
        lambda api_key=None: DummyClient(),
    )

    emb = MarengoEmbeddings()
    assert emb.embed_query("a cat playing piano") == [0.1] * 512
    docs = emb.embed_documents(["one", "two"])
    assert len(docs) == 2 and len(docs[0]) == 512


def test_analyze_video_requires_single_source():
    with pytest.raises(ValueError):
        analyze_video(prompt="summarize", url="http://x", video_id="abc")
    with pytest.raises(ValueError):
        analyze_video(prompt="summarize")


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("TWELVELABS_API_KEY"),
    reason="requires TWELVELABS_API_KEY (free key at https://twelvelabs.io)",
)
def test_marengo_live_embedding_dim():
    """Live smoke: Marengo returns a 512-dim text embedding."""
    emb = MarengoEmbeddings()
    vec = emb.embed_query("a red sports car on a coastal highway")
    assert len(vec) == 512
    assert all(isinstance(x, float) for x in vec[:8])
