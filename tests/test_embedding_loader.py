import logging
import types
from pathlib import Path

import pytest

from colette.apidata import InputConnectorObj, RAGObj
from colette.backends.hf.rag.rag_img import ImageEmbeddingFunction


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

    dummy_processor = DummyProcessor()
    dummy_model = DummyModel()

    monkeypatch.setattr(
        "colette.backends.hf.rag.rag_img.AutoProcessor.from_pretrained",
        lambda *a, **k: dummy_processor,
    )
    monkeypatch.setattr(
        "colette.backends.hf.rag.rag_img.AutoModel.from_pretrained", lambda *a, **k: dummy_model
    )

    ad = InputConnectorObj(
        rag=RAGObj(embedding_model="Qwen/Qwen3-VL-Embedding-2B", gpu_id=0, shared_model=False, embedding_lib="huggingface")
    )
    embf = ImageEmbeddingFunction(ad, Path("."), logging.getLogger())
    assert embf.model is dummy_model
    assert embf.processor is dummy_processor
