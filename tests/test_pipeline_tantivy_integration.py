"""
Integration tests for Tantivy retrieval modes.

These tests exercise the full create → index → predict loop with a real
embedding model + LLM.  They require:
  - A GPU with sufficient VRAM
  - Models already present under the workspace models/ directory
  - env var COLETTE_PIPELINE_HF_MODEL pointing to an available VL model

Run explicitly with:
    pytest tests/test_pipeline_tantivy_integration.py -m integration -v
"""

import asyncio
import json
import os
import time
import uuid
from pathlib import Path

import pytest

from colette.apidata import APIData
from colette.jsonapi import JSONApi

pytestmark = pytest.mark.integration

_ROOT = Path(__file__).resolve().parents[1]
_EMBEDDING_MODEL = "Qwen/Qwen3-VL-Embedding-2B"


def _poll_index_finished(api: JSONApi, app_name: str, timeout_s: int = 180) -> str:
    deadline = time.monotonic() + timeout_s
    last = ""
    while time.monotonic() < deadline:
        status = asyncio.run(api.service_index_status(app_name))
        last = (status.message or "").lower()
        if "finished" in last or "error" in last:
            return last
        time.sleep(0.5)
    pytest.fail(f"Indexing timed out. Last status: {last}")


@pytest.fixture
def text_search_engine_context(tmp_path):
    """Shared fixture: service configs ready for retrieval-mode experiments."""
    cfg_create = _ROOT / "src" / "colette" / "config" / "vrag_default.json"
    cfg_index = _ROOT / "src" / "colette" / "config" / "vrag_default_index.json"

    create_config = json.loads(cfg_create.read_text())
    index_config = json.loads(cfg_index.read_text())

    app_dir = tmp_path / "text_search_engine_it"
    create_config["app"]["repository"] = str(app_dir)
    create_config["app"]["models_repository"] = str((_ROOT / "models").resolve())
    create_config["parameters"].setdefault("input", {}).setdefault("rag", {})
    create_config["parameters"]["input"]["rag"]["embedding_model"] = _EMBEDDING_MODEL
    create_config["parameters"]["llm"] = {
        "source": os.getenv("COLETTE_PIPELINE_HF_MODEL", "Qwen/Qwen2-VL-2B-Instruct"),
        "gpu_ids": [0],
        "image_width": 320,
        "image_height": 480,
        "inference": {"lib": "huggingface"},
    }

    index_config["parameters"]["input"]["data"] = [str(_ROOT / "tests" / "data_pdf1")]
    index_config["parameters"]["input"]["rag"]["reindex"] = True
    index_config["parameters"]["input"]["rag"]["index_protection"] = False

    return {
        "create": create_config,
        "index": index_config,
    }


def _create_and_index(api: JSONApi, app_name: str, create_config: dict, index_config: dict):
    create_resp = api.service_create(app_name, APIData(**create_config))
    assert create_resp.service_name == app_name

    api.service_index(app_name, APIData(**index_config))
    state = _poll_index_finished(api, app_name)
    assert "finished" in state, state


@pytest.mark.parametrize("mode", ["text_search_retrieval", "hybrid"])
def test_text_search_engine_retrieval_mode_at_index_time(text_search_engine_context, mode):
    """
    When the service is indexed with retrieval_mode='text_search_retrieval' or 'hybrid', a
    subsequent predict call must return a non-empty 'text_context' list in
    response.sources.
    """
    api = JSONApi()
    app_name = f"it_text_search_engine_{mode}_{uuid.uuid4().hex[:8]}"

    create = json.loads(json.dumps(text_search_engine_context["create"]))
    index = json.loads(json.dumps(text_search_engine_context["index"]))
    index["parameters"]["input"]["rag"]["retrieval_mode"] = mode

    try:
        _create_and_index(api, app_name, create, index)

        query = APIData(
            **{
                "parameters": {
                    "input": {"message": "What is the title of the document?"}
                }
            }
        )
        resp = api.service_predict(app_name, query)

        assert resp is not None
        assert str(resp.output).strip() != ""
        assert resp.sources is not None

        # text_context must be present and have at least one entry
        assert "text_context" in resp.sources, (
            f"Expected 'text_context' key in sources for mode={mode!r}; "
            f"got keys: {list(resp.sources.keys())}"
        )
        assert len(resp.sources["text_context"]) > 0

        first = resp.sources["text_context"][0]
        assert "source" in first
        assert "content" in first

        if mode == "hybrid":
            # vector path should also return image crops
            assert "context" in resp.sources
            assert len(resp.sources["context"]) > 0

    finally:
        asyncio.run(api.service_delete(app_name))


def test_text_search_engine_per_request_override(text_search_engine_context):
    """
    Service is indexed with mode='hybrid'.  A predict request that overrides to
    mode='text_search_retrieval' must return text_context but NOT image crop context.
    """
    api = JSONApi()
    app_name = f"it_text_search_engine_override_{uuid.uuid4().hex[:8]}"

    create = json.loads(json.dumps(text_search_engine_context["create"]))
    index = json.loads(json.dumps(text_search_engine_context["index"]))
    index["parameters"]["input"]["rag"]["retrieval_mode"] = "hybrid"

    try:
        _create_and_index(api, app_name, create, index)

        # Override to tantivy-only at request time
        query = APIData(
            **{
                "parameters": {
                    "input": {
                        "message": "What is the title of the document?",
                        "rag": {"retrieval_mode": "text_search_retrieval"},
                    }
                }
            }
        )
        resp = api.service_predict(app_name, query)

        assert resp is not None
        assert str(resp.output).strip() != ""
        assert resp.sources is not None

        assert "text_context" in resp.sources
        assert len(resp.sources["text_context"]) > 0

        # With tantivy-only mode, image sources should be empty
        image_context = resp.sources.get("context", [])
        assert image_context == [], (
            f"Expected empty image context for tantivy-only override; got {image_context}"
        )

    finally:
        asyncio.run(api.service_delete(app_name))


def test_embedding_mode_has_no_text_context(text_search_engine_context):
    """
    Default mode='embedding_retrieval' must NOT produce a text_context list in sources
    (backward compatibility guarantee).
    """
    api = JSONApi()
    app_name = f"it_vector_notext_{uuid.uuid4().hex[:8]}"

    create = json.loads(json.dumps(text_search_engine_context["create"]))
    index = json.loads(json.dumps(text_search_engine_context["index"]))
    # vector is the default; set explicitly for clarity
    index["parameters"]["input"]["rag"]["retrieval_mode"] = "embedding_retrieval"

    try:
        _create_and_index(api, app_name, create, index)

        query = APIData(
            **{
                "parameters": {
                    "input": {"message": "What is the title of the document?"}
                }
            }
        )
        resp = api.service_predict(app_name, query)

        assert resp.sources is not None
        assert "context" in resp.sources
        assert len(resp.sources["context"]) > 0

        # text_context must be absent or empty for pure vector mode
        text_ctx = resp.sources.get("text_context", [])
        assert text_ctx == [], (
            f"Expected no text_context for vector-only mode; got {text_ctx}"
        )

    finally:
        asyncio.run(api.service_delete(app_name))
