import logging
from pathlib import Path
from types import SimpleNamespace

import torch
import pytest

from colette.apidata import InputConnectorObj, RAGObj
from colette.backends.hf.rag.rag_img import ImageEmbeddingFunction


@pytest.mark.smoke
def test_embedding_integration_flow(monkeypatch):
    # Create a dummy processor that returns tensors compatible with a model call
    class DummyProcessor:
        def __init__(self):
            self.tokenizer = type("T", (), {"padding_side": None})()

        def apply_chat_template(self, msg, tokenize=False, add_generation_prompt=True):
            return "dummy"

        def __call__(self, text, images, videos, padding, truncation, return_tensors):
                # Return a dict-like object that implements .to(device)
                class DummyInputs(dict):
                    def to(self, device):
                        return self

                return DummyInputs(
                    {
                        "input_ids": torch.tensor([[1, 2, 3]]),
                        "attention_mask": torch.tensor([[1, 1, 1]]),
                        "pixel_values": torch.randn(1, 3, 64, 64),
                    }
                )

    class DummyModel:
        def to(self, *args, **kwargs):
            return self

        def eval(self):
            return self

        def prepare_inputs_for_generation(self, **kwargs):
            return kwargs

        def __call__(self, **kwargs):
            # Emulate HF output with last_hidden_state and hidden_states
            return type("O", (), {"last_hidden_state": torch.randn(1, 1, 128), "hidden_states": [torch.randn(1, 1, 128)]})()

    class DummyEmbedder:
        def __init__(self, *args, **kwargs):
            self.model = DummyModel()
            self.processor = DummyProcessor()

    monkeypatch.setattr(
        "colette.backends.hf.rag.rag_img.Qwen3VLEmbedder", DummyEmbedder
    )

    ad = InputConnectorObj(
        rag=RAGObj(embedding_model="Qwen/Qwen3-VL-Embedding-2B", gpu_id=0, shared_model=False, embedding_lib="huggingface")
    )
    embf = ImageEmbeddingFunction(ad, Path("."), logging.getLogger())

    # Call the embedder with a simple text input and ensure we get a list-like embedding back
    embs = embf(["hello world"])
    assert embs is not None
