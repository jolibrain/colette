"""TwelveLabs embedding and video-analysis helpers for the text RAG backend.

This is an opt-in integration that lets Colette use TwelveLabs' multimodal
models inside the existing LangChain/Chroma text-RAG pipeline:

- ``MarengoEmbeddings`` exposes the Marengo model (512-dim multimodal
  embeddings) through the LangChain ``Embeddings`` interface, so it can be
  passed wherever ``HuggingFaceEmbeddings`` is used today.
- ``analyze_video`` calls the Pegasus video-understanding model to turn a
  video (URL, uploaded asset id, or already-indexed video id) into text that
  can then be chunked and indexed like any other document.

The API key is read from the ``TWELVELABS_API_KEY`` environment variable and
is never logged. Grab a free key at https://twelvelabs.io.
"""

from __future__ import annotations

import os

# Default model names; can be overridden via RAGObj.embedding_model / function args.
DEFAULT_MARENGO_MODEL = "marengo3.0"
DEFAULT_PEGASUS_MODEL = "pegasus1.5"


def _make_client(api_key: str | None = None):
    """Build a TwelveLabs client, importing the SDK lazily so it stays optional."""
    try:
        from twelvelabs import TwelveLabs
    except ImportError as exc:  # pragma: no cover - exercised only without the dep
        msg = "The 'twelvelabs' package is required for the TwelveLabs embedding_lib (pip install twelvelabs)"
        raise ImportError(msg) from exc

    key = api_key or os.environ.get("TWELVELABS_API_KEY")
    if not key:
        msg = "TWELVELABS_API_KEY is not set. Get a free key at https://twelvelabs.io"
        raise ValueError(msg)
    return TwelveLabs(api_key=key)


class MarengoEmbeddings:
    """LangChain-compatible embeddings backed by TwelveLabs Marengo.

    Implements the minimal ``Embeddings`` protocol (``embed_documents`` and
    ``embed_query``) so it is a drop-in replacement for ``HuggingFaceEmbeddings``
    in the text-RAG Chroma vector store. Marengo returns 512-dim vectors that
    live in a shared multimodal space (text/image/audio/video), which is what
    makes it useful for video-aware retrieval.
    """

    def __init__(self, model_name: str | None = None, api_key: str | None = None):
        self.model_name = model_name or DEFAULT_MARENGO_MODEL
        self.client = _make_client(api_key)

    def _embed(self, text: str) -> list[float]:
        resp = self.client.embed.create(model_name=self.model_name, text=text)
        segments = resp.text_embedding.segments
        if not segments or segments[0].float_ is None:
            msg = "TwelveLabs Marengo returned no text embedding"
            raise RuntimeError(msg)
        return list(segments[0].float_)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


def analyze_video(
    *,
    prompt: str,
    url: str | None = None,
    asset_id: str | None = None,
    video_id: str | None = None,
    model_name: str | None = None,
    max_tokens: int = 2048,
    api_key: str | None = None,
) -> str:
    """Run TwelveLabs Pegasus video understanding and return the generated text.

    Provide exactly one source: a public ``url``, an uploaded ``asset_id``, or
    an already-indexed ``video_id``. The returned text can be fed into the
    normal text-RAG chunking/indexing path so video content becomes searchable
    alongside documents.
    """
    sources = [s for s in (url, asset_id, video_id) if s]
    if len(sources) != 1:
        msg = "analyze_video requires exactly one of: url, asset_id, video_id"
        raise ValueError(msg)

    client = _make_client(api_key)
    kwargs = {"model_name": model_name or DEFAULT_PEGASUS_MODEL, "prompt": prompt, "max_tokens": max_tokens}
    if video_id:
        kwargs["video_id"] = video_id
    else:
        from twelvelabs.types.video_context import VideoContext_AssetId, VideoContext_Url

        kwargs["video"] = VideoContext_Url(url=url) if url else VideoContext_AssetId(asset_id=asset_id)

    return client.analyze(**kwargs).data
