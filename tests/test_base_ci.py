import os
import multiprocessing as mp
import shutil
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from utils import pretty_print_response

from colette.httpjsonapi import app

models_repo = os.getenv("MODELS_REPO", "models")
VLLM_E2E_ENABLED = os.getenv("COLETTE_ENABLE_VLLM_E2E") == "1"

# messages

json_create = {
    "app": {
        "repository": "colette_test",
        "models_repository": models_repo,
    },
    "parameters": {
        "input": {
            "lib": "langchain",
            "rag": {
                "indexdb_lib": "chromadb",
                "embedding_lib": "huggingface",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            "template": {
                "template_prompt": "Tu es un assistant de réponse à des questions."
                "Question: {question} Contexte: {context} Réponse: ",
                "template_prompt_variables": ["context", "question"],
            },
        },
        "llm": {
            "source": "qwen2.5:0.5b",
            "inference": {"lib": "ollama"},
        },
    },
}

json_index = {
    "parameters": {
        "input": {
            "preprocessing": {"files": ["all"], "lib": "unstructured"},
            "rag": {
                "reindex": True,
                "index_protection": False,
            },
            "data": ["tests/data"],
        },
    }
}


json_predict = {
    "app": {"repository": "colette_test/"},
    "parameters": {"input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}},
}

json_predict_prompt = {
    "app": {"repository": "colette_test/"},
    "parameters": {
        "input": {
            "template": {
                "template_prompt": "Répond en une seule phrase. Question: {question} Contexte: {context} Réponse: "
            },
            "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?",
        }
    },
}


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


# testing
# client = TestClient(app)


@pytest.fixture
def temp_dir(request):
    # Get the repository path from the test function's parameters
    temp_dir = Path(request.node.get_closest_marker("repository_path").args[0])
    shutil.rmtree(temp_dir, ignore_errors=True)
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


def ensure_service_deleted(client, sname):
    try:
        client.delete(f"/v1/app/{sname}")
    except Exception:
        pass


def require_vllm_runtime():
    """vLLM e2e is opt-in and can fail under fork-based multiprocessing with CUDA."""
    if os.getenv("COLETTE_ENABLE_VLLM_E2E") != "1":
        pytest.skip("set COLETTE_ENABLE_VLLM_E2E=1 to run vLLM e2e tests")
    start_method = mp.get_start_method(allow_none=True) or "fork"
    if start_method != "spawn":
        pytest.skip(f"vLLM e2e requires multiprocessing start method 'spawn' (got '{start_method}')")


@pytest.mark.smoke
def test_info(client):
    response = client.get("/v1/info")
    assert response.status_code == 200
    assert "commit" in response.json()["version"]


# def test_service_create_ollama():
#     response = client.put("/v1/app/test", json=json_create)
#     assert response.status_code == 200
#     assert response.json()["service_name"] == "test"


# def test_service_create_llamacpp():
#     json_create_llamacpp = copy.deepcopy(json_create)
#     json_create_llamacpp["parameters"]["llm"]["source"] = (
#         "Qwen/Qwen2.5-0.5B-Instruct-GGUF"
#     )
#     json_create_llamacpp["parameters"]["llm"]["filename"] = (
#         "qwen2.5-0.5b-instruct-q8_0.gguf"
#     )
#     json_create_llamacpp["parameters"]["llm"]["inference"]["lib"] = "llamacpp"

#     response = client.put("/v1/app/test", json=json_create_llamacpp)
#     assert response.status_code == 200

#     response = client.delete("/v1/app/test")
#     print(response.json())
#     assert response.status_code == 200


# def test_service_create_norag():
#     json_create_norag = copy.deepcopy(json_create)
#     del json_create_norag["parameters"]["input"]["rag"]
#     del json_create_norag["parameters"]["input"]["data"]
#     template = json_create_norag["parameters"]["input"]["template"]
#     template["template_prompt"] = "Question: {question}"
#     template["template_prompt_variables"] = ["question"]
#     response = client.put("/v1/app/test", json=json_create_norag)
#     assert response.status_code == 200
#     print(response.json())

#     response = client.post("/v1/predict/test", json=json_predict)
#     assert response.status_code == 200
#     print(response.json())

#     response = client.delete("/v1/app/test")
#     print(response.json())
#     assert response.status_code == 200


# def test_service_session():
#     # create a service with history in the prompt
#     json_create_session = copy.deepcopy(json_create)
#     del json_create_session["parameters"]["input"]["rag"]
#     del json_create_session["parameters"]["input"]["data"]
#     json_create_session["parameters"]["llm"]["conversational"] = True
#     template = json_create_session["parameters"]["input"]["template"]
#     template["template_prompt"] = "Tu es un assistant de réponse à des questions."
#     template["template_prompt_variables"] = ["question"]
#     response = client.put("/v1/app/test", json=json_create_session)
#     assert response.status_code == 200
#     # simulate 2 sessions
#     sessions = {"a": "Alice", "b": "Bob"}
#     # both state something
#     for session, name in sessions.items():
#         json_predict_session = copy.deepcopy(json_predict)
#         json_predict_session["parameters"]["input"]["session_id"] = session
#         json_predict_session["parameters"]["input"]["message"] = f"Je suis {name}."
#         response = client.post("/v1/predict/test", json=json_predict_session)
#         assert response.status_code == 200
#     # both ask for statement
#     for session in sessions.keys():
#         json_predict_session = copy.deepcopy(json_predict)
#         json_predict_session["parameters"]["input"]["session_id"] = session
#         json_predict_session["parameters"]["input"]["message"] = "Quel est mon nom ?"
#         response = client.post("/v1/predict/test", json=json_predict_session)
#         assert response.status_code == 200
#         print(response.json())

#     response = client.delete("/v1/app/test")
#     print(response.json())
#     assert response.status_code == 200


# @pytest.mark.repository_path("test_llamacpp_gpt4all_all-MiniLM-L6-v2")
# def test_llamacpp_gpt4all(temp_dir):
#     response = client.get("/v1/info")
#     assert "commit" in response.json()["version"]

#     # build the service with llamacpp and gpt4all embeddings
#     json_create_llamacpp_gpt4all_all = {
#         "app": {
#             "repository": str(temp_dir)
#         },
#         "parameters": {
#             "input": {
#                 "lib": "langchain",
#                 "preprocessing": {
#                     "files": ["all"],
#                     "lib": "unstructured",
#                     "strategy": "fast",
#                     "cleaning": False
#                 },
#                 "rag": {
#                     "indexdb_lib": "chromadb",
#                     "embedding_lib": "gpt4all",
#                     "embedding_model": "all-MiniLM-L6-v2.gguf2.f16.gguf",
#                     "search": False,
#                     "reindex": True,
#                     "index_protection": False,
#                     "gpu_id": -1
#                 },
#                 "template": {
#                     "template_prompt": "Tu es un assistant de réponse à des questions."
#                     "Question: {question} Contexte: {context} Réponse: ",
#                     "template_prompt_variables": ["context", "question"],
#                 },
#                 "data": ["tests/data"]
#             },
#             "llm": {
#                 "source": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
#                 "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
#                 "context_size": 3000,
#                 "inference": {
#                     "lib": "llamacpp"
#                     }
#             }
#         }
#     }

#     ad_json = json.dumps(json_create_llamacpp_gpt4all_all)

#     response = client.put(
#         "/v1/app/test_llamacpp_gpt4all_all-MiniLM-L6-v2",
#         data={"ad": ad_json}
#     )
#     assert response.status_code == 200
#     assert (
#         response.json()["service_name"] == "test_llamacpp_gpt4all_all-MiniLM-L6-v2"
#     )

#     # predict with the service
#     json_predict = {
#         "app": {"repository": "test_llamacpp_gpt4all_all-MiniLM-L6-v2"},
#         "parameters": {
#             "input": {
#                 "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"
#             }
#         },
#     }
#     response = client.post(
#         "/v1/predict/test_llamacpp_gpt4all_all-MiniLM-L6-v2", json=json_predict
#     )
#     assert response.status_code == 200
#     # assert "36500" in response.json()["output"]
#     print(response.json()["output"])

#     # delete the service
#     response = client.delete("/v1/app/test_llamacpp_gpt4all_all-MiniLM-L6-v2")
#     assert response.status_code == 200


####################################################################
# build the service with llamacpp and huggingface embeddings
@pytest.mark.repository_path("test_llamacpp_hf_all-MiniLM-L6-v2")
@pytest.mark.integration
@pytest.mark.e2e
def test_llamacpp_hf(temp_dir, client):
    json_create_llamacpp_hf_all = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
        },
        "parameters": {
            "input": {
                "lib": "langchain",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "search": False,
                },
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions."
                    " Question: {question} Contexte: {context} Réponse: ",
                    "template_prompt_variables": ["context", "question"],
                },
            },
            "llm": {
                "source": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
                "context_size": 3500,
                "inference": {"lib": "llamacpp"},
            },
        },
    }
    json_index_llamacpp_hf_all = {
        "parameters": {
            "input": {
                "preprocessing": {"files": ["all"], "lib": "unstructured", "strategy": "fast"},
                "rag": {"reindex": True, "index_protection": False, "gpu_id": -1},
                "data": ["tests/data"],
            },
        }
    }

    sname = "test_llamacpp_hf_all-MiniLM-L6-v2"
    ensure_service_deleted(client, sname)
    try:
        response = client.put(f"/v1/app/{sname}", json=json_create_llamacpp_hf_all)
        assert response.status_code == 200
        assert response.json()["service_name"] == sname

        response = client.put(f"/v1/index/{sname}", json=json_index_llamacpp_hf_all)
        pretty_print_response(response.json())
        assert response.status_code == 200
        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get(f"/v1/index/{sname}/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        json_predict = {
            "app": {"repository": sname},
            "parameters": {"input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}},
        }
        response = client.post(f"/v1/predict/{sname}", json=json_predict)
        assert response.status_code == 200
        print(response.json()["output"])
    finally:
        ensure_service_deleted(client, sname)


####################################################################
# build a new service with same embeddings but different lib i.e. huggingface
@pytest.mark.repository_path("test_llamacpp_hf_e5")
@pytest.mark.integration
@pytest.mark.e2e
def test_llamacpp_hf_e5(temp_dir, client):
    json_create_llamacpp_e5 = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
        },
        "parameters": {
            "input": {
                "lib": "langchain",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "intfloat/multilingual-e5-small",
                },
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions."
                    " Question: {question} Contexte: {context} Réponse: ",
                    "template_prompt_variables": ["context", "question"],
                },
            },
            "llm": {
                "source": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
                "filename": "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
                "context_size": 3500,
                "inference": {"lib": "llamacpp"},
            },
        },
    }

    json_index_llamacpp_e5 = {
        "parameters": {
            "input": {
                "preprocessing": {"files": ["all"], "lib": "unstructured", "strategy": "fast"},
                "rag": {"reindex": True, "index_protection": False, "gpu_id": -1},
                "data": ["tests/data"],
            },
        }
    }
    sname = "test_llamacpp_e5"
    ensure_service_deleted(client, sname)
    try:
        response = client.put(f"/v1/app/{sname}", json=json_create_llamacpp_e5)
        assert response.status_code == 200

        response = client.put(f"/v1/index/{sname}", json=json_index_llamacpp_e5)
        pretty_print_response(response.json())
        assert response.status_code == 200
        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get(f"/v1/index/{sname}/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        json_predict = {
            "app": {"repository": "test_llamacpp_hf_e5"},
            "parameters": {"input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}},
        }
        response = client.post(f"/v1/predict/{sname}", json=json_predict)
        assert response.status_code == 200
        print(response.json()["output"])
    finally:
        ensure_service_deleted(client, sname)


#################################################################################
# build the service with vllm
@pytest.mark.repository_path("test_vllm")
@pytest.mark.integration
@pytest.mark.e2e
@pytest.mark.skipif(
    not VLLM_E2E_ENABLED,
    reason="set COLETTE_ENABLE_VLLM_E2E=1 to run vLLM e2e tests",
)
def test_vllm(temp_dir, client):
    require_vllm_runtime()
    json_create_vllm = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
        },
        "parameters": {
            "input": {
                "lib": "langchain",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "search": False,
                },
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions."
                    " Question: {question} Contexte: {context} Réponse: ",
                    "template_prompt_variables": ["context", "question"],
                },
            },
            "llm": {
                "source": "Qwen/Qwen2.5-0.5B",
                "context_size": 2048,
                "vllm_memory_utilization": 0.3,
                "vllm_enforce_eager": True,
                "dtype": "float32",
                "inference": {"lib": "vllm"},
            },
        },
    }
    json_index_vllm = {
        "parameters": {
            "input": {
                "preprocessing": {"files": ["all"], "lib": "unstructured", "strategy": "fast"},
                "rag": {"reindex": True, "index_protection": True, "gpu_id": -1},
                "data": ["tests/data"],
            },
        }
    }

    sname = "test_vllm"
    ensure_service_deleted(client, sname)
    try:
        response = client.put(f"/v1/app/{sname}", json=json_create_vllm)
        if response.status_code != 200:
            detail = response.json().get("status", {}).get("colette_message", "")
            if "Engine core initialization failed" in str(detail):
                pytest.skip("vLLM engine cannot initialize in this runtime (CUDA fork/spawn limitation)")
        assert response.status_code == 200
        assert response.json()["service_name"] == sname

        response = client.put(f"/v1/index/{sname}", json=json_index_vllm)
        pretty_print_response(response.json())
        assert response.status_code == 200
        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get(f"/v1/index/{sname}/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        response = client.delete(f"/v1/app/{sname}")
        assert response.status_code == 200

        response = client.put(f"/v1/app/{sname}", json=json_create_vllm)
        assert response.status_code == 200
        response = client.put(f"/v1/index/{sname}", json=json_index_vllm)
        pretty_print_response(response.json())
        assert response.status_code == 200
        while "error" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get(f"/v1/index/{sname}/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        json_index_vllm["parameters"]["input"]["rag"]["reindex"] = False
        response = client.put(f"/v1/index/{sname}", json=json_index_vllm)
        pretty_print_response(response.json())
        assert response.status_code == 200
        while "finished" not in response.json()["message"]:
            time.sleep(0.5)
            response = client.get(f"/v1/index/{sname}/status")
            pretty_print_response(response.json())
            assert response.status_code == 200

        json_predict = {
            "app": {"repository": sname},
            "parameters": {"input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}},
        }
        response = client.post(f"/v1/predict/{sname}", json=json_predict)
        assert response.status_code == 200
        print(response.json()["output"])
    finally:
        ensure_service_deleted(client, sname)
