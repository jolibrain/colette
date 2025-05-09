import os
import shutil
import time
from pathlib import Path

import pytest
import pytest_asyncio
from chromadb import PersistentClient
from chromadb.config import Settings
from fastapi.testclient import TestClient
from utils import pretty_print_response

from colette.httpjsonapi import app

models_repo = os.getenv("MODELS_REPO", "models")


@pytest_asyncio.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


# client = TestClient(app)


@pytest.fixture
def temp_dir(request):
    # Get the repository path from the test function's parameters
    temp_dir = Path(request.node.get_closest_marker("repository_path").args[0])
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.repository_path(Path("test_create_app_and_index"))
@pytest.mark.asyncio
def test_create_app_and_index(temp_dir, client):
    response = client.get("/v1/info")
    assert "commit" in response.json()["version"]

    ##############################################
    # build the service with hf backend with multiple indexing rounds
    app_definition = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
            "verbose": "debug",
        },
        "parameters": {
            "input": {
                "lib": "hf",
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Réponse: ",
                    "template_prompt_variables": ["question"],
                },
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    "top_k": 2,
                    "ragm": {
                        "layout_detection": False,
                    },
                },
            },
            "llm": {
                "source": "Qwen/Qwen2-VL-2B-Instruct",
                "gpu_ids": [0],
                "image_width": 640,
                "image_height": 960,
                "inference": {"lib": "huggingface"},
            },
        },
    }

    index_definition = {
        "parameters": {
            "input": {
                "preprocessing": {
                    "files": ["all"],
                    "save_output": False,
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
        response = client.put(
            "/v1/app/test_create_app_and_index",
            json=app_definition,
        )
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_create_app_and_index"

        response = client.get("/v1/index/test_create_app_and_index/status")
        pretty_print_response(response.json())
        assert response.status_code == 200

        response = client.put("/v1/index/test_create_app_and_index", json=index_definition)
        pretty_print_response(response.json())
        assert response.status_code == 200

        response = client.get("/v1/index/test_create_app_and_index/status")
        pretty_print_response(response.json())
        assert response.status_code == 200

        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get("/v1/index/test_create_app_and_index/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_create_app_and_index" in response.json()["info"]["services"]

        persist_dir = app_definition["app"]["repository"] + os.sep + "mm_index"
        cc = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        assert cc.count_collections() == 1
        assert cc.get_collection("mm_db") is not None
        assert cc.get_collection("mm_db").count() == 1

        index_definition["parameters"]["input"]["rag"]["reindex"] = False
        index_definition["parameters"]["input"]["rag"]["update_index"] = True
        index_definition["parameters"]["input"]["data"] = ["tests/data_img3"]

        response = client.put("/v1/index/test_create_app_and_index", json=index_definition)
        pretty_print_response(response.json())
        assert response.status_code == 200, "App could not be indexed"

        time.sleep(2)
        response = client.get("/v1/index/test_create_app_and_index/status")
        pretty_print_response(response.json())
        assert response.status_code == 200
        while "finished" not in response.json()["message"]:
            time.sleep(2)
            response = client.get("/v1/index/test_create_app_and_index/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        persist_dir = app_definition["app"]["repository"] + os.sep + "mm_index"
        cc = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        assert cc.count_collections() == 1
        assert cc.get_collection("mm_db") is not None
        assert cc.get_collection("mm_db").count() == 2
    finally:
        # delete the service
        response = client.delete("/v1/app/test_create_app_and_index")
        assert response.status_code == 200


@pytest.mark.repository_path(Path("test_create_app_twice_true_true"))
@pytest.mark.asyncio
def test_create_app_twice(temp_dir, client):
    response = client.get("/v1/info")
    assert "commit" in response.json()["version"]

    ##############################################
    # build the service with hf backend with multiple index
    app_definition = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
            "verbose": "debug",
        },
        "parameters": {
            "input": {
                "lib": "hf",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    "gpu_id": 0,
                    "top_k": 2,
                    "ragm": {
                        "layout_detection": False,
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
                "image_height": 960,
                "inference": {"lib": "huggingface"},
            },
        },
    }
    index_definition = {
        "parameters": {
            "input": {
                "preprocessing": {
                    "files": ["all"],
                    "save_output": False,
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
        response = client.put(
            "/v1/app/test_create_app_twice",
            json=app_definition,
        )
        pretty_print_response(response.json())
        assert response.status_code == 200, "App could not be created"
        assert response.json()["service_name"] == "test_create_app_twice"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_create_app_twice" in response.json()["info"]["services"]

        response = client.put("/v1/index/test_create_app_twice", json=index_definition)
        pretty_print_response(response.json())
        assert response.status_code == 200

        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get("/v1/index/test_create_app_twice/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        persist_dir = app_definition["app"]["repository"] + os.sep + "mm_index"
        cc = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        assert cc.count_collections() == 1
        assert cc.get_collection("mm_db") is not None
        assert cc.get_collection("mm_db").count() == 1

        index_definition["parameters"]["input"]["rag"]["reindex"] = True
        index_definition["parameters"]["input"]["data"] = ["tests/data_img3"]

        response = client.put(
            "/v1/app/test_create_app_twice_true_true",
            json=app_definition,
        )
        pretty_print_response(response.json()), "App could not be created"
        assert response.status_code == 200, "App could not be created"

        response = client.put("/v1/index/test_create_app_twice_true_true", json=index_definition)
        pretty_print_response(response.json())
        assert response.status_code == 200

        response = client.get("/v1/index/test_create_app_twice_true_true/status")
        pretty_print_response(response.json())
        assert response.status_code == 200

        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get("/v1/index/test_create_app_twice_true_true/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        persist_dir = app_definition["app"]["repository"] + os.sep + "mm_index"
        cc = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        assert cc.count_collections() == 1
        assert cc.get_collection("mm_db") is not None
        assert cc.get_collection("mm_db").count() == 1
    finally:
        # delete the service
        response = client.delete("/v1/app/test_create_app_twice")
        assert response.status_code == 200
        response = client.delete("/v1/app/test_create_app_twice_true_true")
        assert response.status_code == 200


@pytest.mark.repository_path(Path("test_index_first"))
@pytest.mark.asyncio
def test_index_first(temp_dir, client):
    response = client.get("/v1/info")
    assert "commit" in response.json()["version"]

    ##############################################
    # build the service with hf backend with multiple index
    app_definition = {
        "app": {"repository": str(temp_dir), "models_repository": models_repo, "verbose": "debug"},
        "parameters": {
            "input": {
                "lib": "hf",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    "top_k": 2,
                    "ragm": {"layout_detection": False},
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
                "image_height": 960,
                "inference": {"lib": "huggingface"},
            },
        },
    }
    # index_definition = {
    #     "parameters": {
    #         "input": {
    #             "preprocessing": {
    #                 "files": ["all"],
    #                 "save_output": False,
    #             },
    #             "data": ["tests/data_img1"],
    #             "rag": {
    #                 "reindex": True,
    #                 "index_protection": False,
    #                 "gpu_id": 0,
    #             },
    #         },
    #     }
    # }

    try:
        response = client.put(
            "/v1/app/test_index_first",
            json=app_definition,
        )
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_index_first"
    finally:
        # delete the service
        response = client.delete("/v1/app/test_index_first")
        assert response.status_code == 200


@pytest.mark.repository_path(Path("test_create_app_and_multiple_index"))
@pytest.mark.asyncio
def test_create_app_and_multiple_index(temp_dir, client):
    response = client.get("/v1/info")
    assert "commit" in response.json()["version"]

    ##############################################
    # build the service with hf backend with multiple index
    app_definition = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
            "verbose": "debug",
        },
        "parameters": {
            "input": {
                "lib": "hf",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    "top_k": 2,
                    "ragm": {
                        "layout_detection": False,
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
                "image_height": 800,
                "inference": {"lib": "huggingface"},
            },
        },
    }

    index_definition = {
        "parameters": {
            "input": {
                "preprocessing": {
                    "files": ["all"],
                    "save_output": False,
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
        response = client.put(
            "/v1/app/test_create_app_and_multiple_index",
            json=app_definition,
        )
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_create_app_and_multiple_index"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_create_app_and_multiple_index" in response.json()["info"]["services"]

        response = client.put("/v1/index/test_create_app_and_multiple_index", json=index_definition)
        pretty_print_response(response.json())
        assert response.status_code == 200
        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get("/v1/index/test_create_app_and_multiple_index/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        persist_dir = app_definition["app"]["repository"] + os.sep + "mm_index"
        cc = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        assert cc.count_collections() == 1
        assert cc.get_collection("mm_db") is not None
        assert cc.get_collection("mm_db").count() == 1

        index_definition["parameters"]["input"]["rag"]["reindex"] = False
        index_definition["parameters"]["input"]["rag"]["update_index"] = True
        index_definition["parameters"]["input"]["data"].append("tests/data_img3")

        response = client.put("/v1/index/test_create_app_and_multiple_index", json=index_definition)
        pretty_print_response(response.json())
        assert response.status_code == 200, "App could not be indexed"

        pretty_print_response(response.json())
        assert response.status_code == 200
        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get("/v1/index/test_create_app_and_multiple_index/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        persist_dir = app_definition["app"]["repository"] + os.sep + "mm_index"
        cc = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        assert cc.count_collections() == 1
        assert cc.get_collection("mm_db") is not None
        assert cc.get_collection("mm_db").count() == 2

        index_definition["parameters"]["input"]["data"].append("tests/data_img2")

        response = client.put("/v1/index/test_create_app_and_multiple_index", json=index_definition)
        pretty_print_response(response.json())
        assert response.status_code == 200, "App could not be indexed"

        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get("/v1/index/test_create_app_and_multiple_index/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        persist_dir = app_definition["app"]["repository"] + os.sep + "mm_index"
        cc = PersistentClient(path=persist_dir, settings=Settings(anonymized_telemetry=False))
        assert cc.count_collections() == 1
        assert cc.get_collection("mm_db") is not None
        assert cc.get_collection("mm_db").count() == 4

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}
        response = client.post("/v1/predict/test_create_app_and_multiple_index", json=json_predict)
        pretty_print_response(response.json())
        assert "Rapport" in response.json()["output"] or "RAPPORT" in response.json()["output"]

        json_predict2 = {"parameters": {"input": {"message": "Quels sont les députés ?"}}}
        response = client.post("/v1/predict/test_create_app_and_multiple_index", json=json_predict2)
        pretty_print_response(response.json())
        assert "LOPEZ" in response.json()["output"]

        json_predict3 = {
            "parameters": {"input": {"message": "Quel est le pourcentage d'investissement en Space Transportation ?"}}
        }
        response = client.post("/v1/predict/test_create_app_and_multiple_index", json=json_predict3)
        pretty_print_response(response.json())

        json_predict4 = {"parameters": {"input": {"message": "Quel domaine représente 17% du budget?"}}}
        response = client.post("/v1/predict/test_create_app_and_multiple_index", json=json_predict4)
        pretty_print_response(response.json())
    finally:
        # delete the service
        response = client.delete("/v1/app/test_create_app_and_multiple_index")
        assert response.status_code == 200
