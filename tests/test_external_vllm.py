import os
import shutil
import signal
import subprocess
import time
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from utils import pretty_print_response

from colette.httpjsonapi import app


@pytest_asyncio.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


models_repo = os.getenv("MODELS_REPO", "models")


def generic_index(client, sname, index_json):
    response = client.put(f"/v1/index/{sname}", json=index_json)
    assert response.status_code == 200
    response = client.get(f"/v1/index/{sname}/status")
    pretty_print_response(response.json())
    assert response.status_code == 200

    while "running" in response.json()["message"]:
        time.sleep(2)
        response = client.get(f"/v1/index/{sname}/status")
        pretty_print_response(response.json())
    return response


@pytest.fixture
def temp_dir(request):
    # Get the repository path from the test function's parameters
    temp_dir = Path(request.node.get_closest_marker("repository_path").args[0])
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.repository_path("test_vllm_external")
@pytest.mark.asyncio
def test_external_vllm_single_image(temp_dir, client):
    json_create_img_hf = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
            "verbose": "debug",
        },
        "parameters": {
            "input": {
                "lib": "hf",
                "data_output_type": "img",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "vllm",
                    "embedding_model": "Qwen/Qwen2-VL-2B-Instruct",
                    "top_k": 1,
                    "shared_model": True,
                    "ragm": {
                        "layout_detection": True,
                        "index_overview": False,
                        "image_width": 512,
                        "image_height": 512,
                    },
                },
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Réponse: ",
                    "template_prompt_variables": ["question"],
                },
            },
            "llm": {
                "source": "Qwen/Qwen2-VL-2B-Instruct",
                "gpu_ids": [0],
                "image_width": 640,
                "image_height": 320,
                "inference": {"lib": "vllm_client"},
                "shared": False,
                "vllm_memory_utilization": 0.45,
                "external_vllm_server": {
                    "url": "http://localhost:8000/v1",
                    "api_key": "token-abc123",
                },
            },
        },
    }
    json_index_img_hf = {
        "parameters": {
            "input": {
                "preprocessing": {
                    "files": ["all"],
                    "save_output": True,
                },
                "data": ["tests/data_img1"],
                "rag": {
                    "reindex": True,
                    "index_protection": False,
                    "gpu_id": 0,
                },
            },
        }
    }

    try:
        # spawn vllm server for test
        pro = subprocess.Popen(
            [
                "vllm",
                "serve",
                "Qwen/Qwen2-VL-2B-Instruct",
                "--dtype",
                "auto",
                "--api-key",
                "token-abc123",
                "--enforce-eager",
                "--gpu_memory_utilization",
                "0.5",
                "--max-model-len",
                "4096",
            ],
            preexec_fn=os.setsid,
        )

        time.sleep(180)
        # colette
        response = client.put("/v1/app/test_external_vllm_single_image", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_external_vllm_single_image"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_external_vllm_single_image" in response.json()["info"]["services"]

        generic_index(client, "test_external_vllm_single_image", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}

        response = client.post("/v1/predict/test_external_vllm_single_image", json=json_predict)
        pretty_print_response(response.json())
        assert response.json()["sources"]["context"][0]["distance"] > 0.0
        # assert "nationale" in response.json()["output"]

    finally:
        # delete the service
        response = client.delete("/v1/app/test_external_vllm_single_image")
        assert response.status_code == 200
        os.killpg(os.getpgid(pro.pid), signal.SIGKILL)
