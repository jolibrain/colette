import asyncio
import json
import os
import socket
import uuid
from pathlib import Path

import pytest

from colette.apidata import APIData
from colette.jsonapi import JSONApi

pytestmark = pytest.mark.integration


def _is_service_available(host: str, port: int, timeout: int = 2) -> bool:
    try:
        socket.create_connection((host, port), timeout=timeout)
        return True
    except (socket.timeout, OSError):
        return False


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

    rag = create_config["parameters"]["input"].setdefault("rag", {})
    rag["embedding_model"] = os.getenv(
        "COLETTE_PIPELINE_EMBEDDING_MODEL",
        "Qwen/Qwen3-VL-Embedding-2B",
    )

    index_config["parameters"]["input"]["data"] = [str(root / "tests" / "data_pdf1")]
    index_config["parameters"]["input"]["rag"]["reindex"] = True
    index_config["parameters"]["input"]["rag"]["index_protection"] = False

    return {
        "app_name": app_name,
        "create": create_config,
        "index": index_config,
    }


def test_pipeline_python_api_create_index_predict(pipeline_context):
    api = JSONApi()
    app_name = pipeline_context["app_name"]

    try:
        create_resp = api.service_create(app_name, APIData(**pipeline_context["create"]))
        assert create_resp.service_name == app_name

        info_resp = api.service_info()
        assert app_name in info_resp.info.services

        index_resp = api.service_index(app_name, APIData(**pipeline_context["index"]))
        # service_index can return either explicit success payload or default APIResponse.
        assert index_resp is not None

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

    finally:
        asyncio.run(api.service_delete(app_name))
