import json
import os
import shutil
import time

import pytest_asyncio
from fastapi.testclient import TestClient
from utils import compare_dicts, pretty_print_response

from colette.httpjsonapi import app


@pytest_asyncio.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


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


def test_logging_payload(client):
    try:
        # Step 1: Validate base endpoint
        response = client.get("/v1/info")
        assert "commit" in response.json()["version"]

        # Step 2: Create a service
        json_create_img_hf = {
            "app": {"repository": "test_logging_payload_2", "verbose": "debug"},
            "parameters": {
                "input": {
                    "lib": "hf",
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "huggingface",
                        "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                        "chunk_num": 3,
                        "chunk_overlap": 20,
                        "top_k": 4,
                        "ragm": {"layout_detection": False},
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions."
                        " Question: {question} Réponse: ",
                        "template_prompt_variables": ["question"],
                    },
                    "data": ["tests/data_img2"],
                },
                "llm": {
                    "source": "Qwen/Qwen2-VL-2B-Instruct",
                    "gpu_ids": [0],
                    "image_width": 640,
                    "image_height": 320,
                    "inference": {"lib": "huggingface"},
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
                    "data": ["tests/data_img2"],
                    "rag": {
                        "reindex": True,
                        "index_protection": False,
                        "gpu_id": 0,
                    },
                },
            }
        }

        response = client.put("/v1/app/test_logging_payload", json=json_create_img_hf)
        assert response.status_code == 200, response.json()
        assert response.json()["service_name"] == "test_logging_payload"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "commit" in response.json()["version"]
        assert "test_logging_payload" in response.json()["info"]["services"]

        generic_index(client, "test_logging_payload", json_index_img_hf)

        predictions = [
            {"message": "Quel est le titre du document ?", "expected_output": None},
            {"message": "Quels sont les députés ?", "expected_output": None},
            {
                "message": "Quel est le pourcentage d'investissement en Space Transportation ?",
                "expected_output": "%",
            },
        ]
        for prediction in predictions:
            json_predict = {"parameters": {"input": {"message": prediction["message"]}}}
            response = client.post("/v1/predict/test_logging_payload", json=json_predict)
            pretty_print_response(response.json())
            if prediction["expected_output"]:
                assert prediction["expected_output"] in response.json()["output"]

        config_path = "test_logging_payload_2/config.json"
        assert os.path.exists(config_path), f"Config file not found: {config_path}"
        with open(config_path) as f:
            config = json.load(f)

        differences = compare_dicts(json_create_img_hf, config)
        assert len(differences) == 0, differences

    finally:
        response = client.delete("/v1/app/test_logging_payload")
        assert response.status_code == 200, "Failed to delete the service"
        shutil.rmtree("test_logging_payload_2", ignore_errors=True)
