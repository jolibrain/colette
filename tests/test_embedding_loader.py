import logging
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

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

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
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
