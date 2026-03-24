import asyncio
import json
import os
import socket
import time
import uuid
from pathlib import Path

import pytest
from pydantic import ValidationError

from colette.apidata import APIData
from colette.jsonapi import JSONApi

pytestmark = pytest.mark.integration


def _is_service_available(host: str, port: int, timeout: int = 2) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except (socket.timeout, OSError):
        return False


def _poll_index_finished(
    api: JSONApi,
    app_name: str,
    timeout_s: int = 180,
    poll_interval_s: float = 0.5,
) -> str:
    """Poll async index status endpoint until finished/error/timeout."""
    deadline = time.monotonic() + timeout_s
    last_message = ""
    while time.monotonic() < deadline:
        status_resp = asyncio.run(api.service_index_status(app_name))
        message = (status_resp.message or "").lower()
        last_message = message
        if "finished" in message or "error" in message:
            return message
        time.sleep(poll_interval_s)
    pytest.fail(f"Indexing did not reach terminal state within {timeout_s}s. Last status: {last_message}")


@pytest.fixture
def pipeline_context(tmp_path):
    root = Path(__file__).resolve().parents[1]
    cfg_create = root / "src" / "colette" / "config" / "vrag_default.json"
    cfg_index = root / "src" / "colette" / "config" / "vrag_default_index.json"

    create_config = json.loads(cfg_create.read_text())
    index_config = json.loads(cfg_index.read_text())

    app_name = f"it_pipeline_{uuid.uuid4().hex[:8]}"
    app_dir = tmp_path / app_name

    create_config["app"]["repository"] = str(app_dir)
    create_config["app"]["models_repository"] = str((root / "models").resolve())

    # Keep the same Python API pipeline semantics as examples, but use a
    # model that is already present in this workspace.
    create_config["parameters"]["llm"] = {
        "source": os.getenv("COLETTE_PIPELINE_HF_MODEL", "Qwen/Qwen2-VL-2B-Instruct"),
        "gpu_ids": [0],
        "image_width": 320,
        "image_height": 480,
        "inference": {"lib": "huggingface"},
    }

    index_config["parameters"]["input"]["data"] = [str(root / "tests" / "data_pdf1")]
    index_config["parameters"]["input"]["rag"]["reindex"] = True
    index_config["parameters"]["input"]["rag"]["index_protection"] = False

    return {
        "app_name": app_name,
        "create": create_config,
        "index": index_config,
    }


@pytest.mark.parametrize(
    "embedding_model",
    [
        "Qwen/Qwen3-VL-Embedding-2B",
        "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
    ],
)
def test_pipeline_python_api_create_index_predict(pipeline_context, embedding_model):
    api = JSONApi()
    create_config = pipeline_context["create"]
    rag = create_config["parameters"]["input"].setdefault("rag", {})
    rag["embedding_model"] = embedding_model

    embedding_suffix = embedding_model.split("/")[-1].lower().replace("-", "_")
    app_name = f"{pipeline_context['app_name']}_{embedding_suffix[:20]}"

    # Make data-path deterministic for each app instance.
    index_config = json.loads(json.dumps(pipeline_context["index"]))
    index_config["parameters"]["input"]["data"] = [str(Path(__file__).resolve().parents[1] / "tests" / "data_pdf1")]

    try:
        create_resp = api.service_create(app_name, APIData(**create_config))
        assert create_resp.service_name == app_name

        info_resp = api.service_info()
        assert app_name in info_resp.info.services

        index_resp = api.service_index(app_name, APIData(**index_config))
        # service_index can return either explicit success payload or default APIResponse.
        assert index_resp is not None

        index_state = _poll_index_finished(api, app_name)
        assert "finished" in index_state, index_state

        query = APIData(
            **{
                "parameters": {
                    "input": {
                        "message": "What is the title of the document?",
                    }
                }
            }
        )
        predict_resp = api.service_predict(app_name, query)

        # Validate the end-to-end pipeline contract from the Python API perspective.
        assert predict_resp is not None
        assert predict_resp.output is not None
        assert str(predict_resp.output).strip() != ""

        # Validate RAG sources contract used by API consumers.
        assert predict_resp.sources is not None
        assert isinstance(predict_resp.sources, dict)
        assert "context" in predict_resp.sources
        assert isinstance(predict_resp.sources["context"], list)
        assert len(predict_resp.sources["context"]) > 0
        first = predict_resp.sources["context"][0]
        assert "key" in first
        assert "content" in first
        assert "distance" in first

    finally:
        # First delete should remove the service.
        delete_resp = asyncio.run(api.service_delete(app_name))
        assert delete_resp.status.code in (200, 404)
        # Second delete validates idempotent cleanup behavior.
        delete_resp_2 = asyncio.run(api.service_delete(app_name))
        assert delete_resp_2.status.code == 404


def test_pipeline_python_api_index_invalid_data_path(pipeline_context):
    api = JSONApi()
    app_name = f"{pipeline_context['app_name']}_invalid"
    create_config = json.loads(json.dumps(pipeline_context["create"]))
    create_config["app"]["repository"] = str(Path(create_config["app"]["repository"]).parent / app_name)

    bad_index = json.loads(json.dumps(pipeline_context["index"]))
    bad_index["parameters"]["input"]["data"] = ["tests/path_that_does_not_exist_for_pipeline_test"]

    try:
        create_resp = api.service_create(app_name, APIData(**create_config))
        assert create_resp.service_name == app_name

        with pytest.raises(ValidationError):
            api.service_index(app_name, APIData(**bad_index))
    finally:
        asyncio.run(api.service_delete(app_name))
