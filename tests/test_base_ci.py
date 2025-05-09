import os
import shutil
import time
from pathlib import Path

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from utils import pretty_print_response

from colette.httpjsonapi import app

models_repo = os.getenv("MODELS_REPO", "models")

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


@pytest_asyncio.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


# testing
# client = TestClient(app)


@pytest.fixture
def temp_dir(request):
    # Get the repository path from the test function's parameters
    temp_dir = Path(request.node.get_closest_marker("repository_path").args[0])
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.asyncio
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
@pytest.mark.asyncio
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

    response = client.put("/v1/app/test_llamacpp_hf_all-MiniLM-L6-v2", json=json_create_llamacpp_hf_all)
    assert response.status_code == 200
    assert response.json()["service_name"] == "test_llamacpp_hf_all-MiniLM-L6-v2"

    response = client.put("/v1/index/test_llamacpp_hf_all-MiniLM-L6-v2", json=json_index_llamacpp_hf_all)
    pretty_print_response(response.json())
    assert response.status_code == 200
    # assert response.json()["service_name"] == "test_llamacpp_hf_all-MiniLM-L6-v2"
    while "finished" not in response.json()["message"]:
        time.sleep(0.5)
        response = client.get("/v1/index/test_llamacpp_hf_all-MiniLM-L6-v2/status")
        pretty_print_response(response.json())
        assert response.status_code == 200

    # predict with the service
    json_predict = {
        "app": {"repository": "test_llamacpp_hf_all-MiniLM-L6-v2"},
        "parameters": {"input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}},
    }
    response = client.post("/v1/predict/test_llamacpp_hf_all-MiniLM-L6-v2", json=json_predict)
    assert response.status_code == 200
    # assert "36500" in response.json()["output"]
    print(response.json()["output"])

    # delete the service
    response = client.delete("/v1/app/test_llamacpp_hf_all-MiniLM-L6-v2")
    assert response.status_code == 200


####################################################################
# build a new service with same embeddings but different lib i.e. huggingface
@pytest.mark.repository_path("test_llamacpp_hf_e5")
@pytest.mark.asyncio
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
    response = client.put("/v1/app/test_llamacpp_e5", json=json_create_llamacpp_e5)
    assert response.status_code == 200

    response = client.put("/v1/index/test_llamacpp_e5", json=json_index_llamacpp_e5)
    pretty_print_response(response.json())
    assert response.status_code == 200
    # assert response.json()["service_name"] == "test_llamacpp_hf_all-MiniLM-L6-v2"
    while "finished" not in response.json()["message"]:
        time.sleep(0.5)
        response = client.get("/v1/index/test_llamacpp_e5/status")
        pretty_print_response(response.json())
        assert response.status_code == 200

    # predict with the service
    json_predict = {
        "app": {"repository": "test_llamacpp_hf_e5"},
        "parameters": {"input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}},
    }
    response = client.post("/v1/predict/test_llamacpp_e5", json=json_predict)
    assert response.status_code == 200
    print(response.json()["output"])

    # delete the service
    response = client.delete("/v1/app/test_llamacpp_e5")
    assert response.status_code == 200


#################################################################################
# build the service with vllm
@pytest.mark.repository_path("test_vllm")
@pytest.mark.asyncio
def test_vllm(temp_dir, client):
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

    response = client.put("/v1/app/test_vllm", json=json_create_vllm)
    assert response.status_code == 200
    assert response.json()["service_name"] == "test_vllm"

    response = client.put("/v1/index/test_vllm", json=json_index_vllm)
    pretty_print_response(response.json())
    assert response.status_code == 200
    # assert response.json()["service_name"] == "test_llamacpp_hf_all-MiniLM-L6-v2"
    while "finished" not in response.json()["message"]:
        time.sleep(0.5)
        response = client.get("/v1/index/test_vllm/status")
        pretty_print_response(response.json())
        assert response.status_code == 200

    response = client.delete("/v1/app/test_vllm")
    assert response.status_code == 200

    response = client.put("/v1/app/test_vllm", json=json_create_vllm)
    response = client.put("/v1/index/test_vllm", json=json_index_vllm)
    pretty_print_response(response.json())
    assert response.status_code == 200
    # assert response.json()["service_name"] == "test_llamacpp_hf_all-MiniLM-L6-v2"
    while "error" not in response.json()["message"]:
        time.sleep(0.5)
        response = client.get("/v1/index/test_vllm/status")
        pretty_print_response(response.json())
        assert response.status_code == 200
    assert response.status_code == 200

    json_index_vllm["parameters"]["input"]["rag"]["reindex"] = False
    response = client.put("/v1/index/test_vllm", json=json_index_vllm)
    pretty_print_response(response.json())
    assert response.status_code == 200
    # assert response.json()["service_name"] == "test_llamacpp_hf_all-MiniLM-L6-v2"
    while "finished" not in response.json()["message"]:
        time.sleep(0.5)
        response = client.get("/v1/index/test_vllm/status")
        pretty_print_response(response.json())
        assert response.status_code == 200
    assert response.status_code == 200

    # predict with the service
    json_predict = {
        "app": {"repository": "test_vllm"},
        "parameters": {"input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}},
    }
    response = client.post("/v1/predict/test_vllm", json=json_predict)
    assert response.status_code == 200
    # assert "36500" in response.json()["output"]
    print(response.json()["output"])

    # delete the service
    response = client.delete("/v1/app/test_vllm")
    assert response.status_code == 200
