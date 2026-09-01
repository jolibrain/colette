import logging
import types
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from colette.apidata import InputConnectorObj, RAGObj
from colette.backends.hf.rag.rag_img import ImageEmbeddingFunction


@pytest.mark.smoke
def test_image_embedding_loader_monkeypatch(monkeypatch):
    class DummyProcessor:
        def __init__(self):
            self.tokenizer = types.SimpleNamespace(padding_side=None)

        def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True):
            return "dummy"

    class DummyModel:
        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

    captured_kwargs = {}

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            captured_kwargs.update(kwargs)
            self.model = DummyModel()
            self.processor = DummyProcessor()

    monkeypatch.setattr(
        "colette.backends.hf.rag.rag_img.Qwen3VLEmbedder",
        DummyEmbedder,
    )

    ad = InputConnectorObj(
        rag=RAGObj(embedding_model="Qwen/Qwen3-VL-Embedding-2B", gpu_id=0, shared_model=False, embedding_lib="huggingface")
    )
    embf = ImageEmbeddingFunction(ad, Path("."), logging.getLogger())
    assert isinstance(embf.model, DummyModel)
    assert isinstance(embf.processor, DummyProcessor)

    # The embedder forwards these to from_pretrained. Without an explicit dtype it
    # falls back to torch.get_default_dtype() (float32) although the checkpoints are
    # bfloat16, which doubles resident memory (~32GB instead of ~16GB for the 8B).
    assert captured_kwargs.get("torch_dtype") is torch.bfloat16

    # Qwen3-VL uses 3D mrope and must run on sdpa. Resolving the attention
    # implementation without the model name returns flash_attention_2 wherever
    # flash-attn is installed; in an embedder that does not crash, it silently
    # writes garbage vectors into the index.
    assert captured_kwargs.get("attn_implementation") == "sdpa"
