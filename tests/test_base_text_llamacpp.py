import copy

import pytest
from utils import ensure_repo_deleted, ensure_service_deleted, wait_for_index_status

from base_text_helpers import models_repo

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


class TestRagLangchainTxtLlamacpp:
    def test_llamacpp_gpt4all(self, client):
        ##############################################
        # build the service with llamacpp and gpt4all embeddings
        json_create_llamacpp_gpt4all_all = {
            "app": {
                "repository": "test_llamacpp_gpt4all_all-MiniLM-L6-v2",
                "models_repository": models_repo,
            },
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "strategy": "fast",
                        "cleaning": False,
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "huggingface",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "search": False,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                },
                "llm": {
                    "source": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                    "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
                    "inference": {"lib": "llamacpp"},
                },
            },
        }
        json_index_llamacpp_gpt4all_all = {
            "parameters": {
                "input": {
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "strategy": "fast",
                        "cleaning": False,
                    },
                    "rag": {"reindex": True, "index_protection": False},
                    "data": ["tests/data"],
                },
            }
        }

        sname = "test_llamacpp_gpt4all_all-MiniLM-L6-v2"
        ensure_service_deleted(client, sname)
        ensure_repo_deleted(sname)
        try:
            response = client.put(
                f"/v1/app/{sname}",
                json=json_create_llamacpp_gpt4all_all,
            )
            assert response.status_code == 200
            assert response.json()["service_name"] == sname

            response = client.put(f"/v1/index/{sname}", json=json_index_llamacpp_gpt4all_all)
            assert response.status_code == 200
            response = wait_for_index_status(client, sname)
            assert "finished" in response.json()["message"]

            json_predict = {
                "app": {"repository": sname},
                "parameters": {
                    "input": {
                        "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"
                    }
                },
            }
            response = client.post(f"/v1/predict/{sname}", json=json_predict)
            print("response status code=", response.status_code)
            print("response=", response.json())
            assert response.status_code == 200
        finally:
            ensure_service_deleted(client, sname)
            ensure_repo_deleted(sname)

    def test_llamacpp_hf(self, client):
        ##############################################
        # build the service with llamacpp and huggingface embeddings
        json_create_llamacpp_hf_all = {
            "app": {
                "repository": "test_llamacpp_hf_all-MiniLM-L6-v2",
                "models_repository": models_repo,
            },
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "strategy": "fast",
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "huggingface",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "search": False,
                        "reindex": True,
                        "index_protection": False,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                },
                "llm": {
                    "source": "Qwen/Qwen2.5-0.5B-Instruct-GGUF",
                    "filename": "qwen2.5-0.5b-instruct-q8_0.gguf",
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
        ensure_repo_deleted(sname)
        try:
            response = client.put(
                f"/v1/app/{sname}",
                json=json_create_llamacpp_hf_all,
            )
            assert response.status_code == 200
            assert response.json()["service_name"] == sname

            response = client.put(f"/v1/index/{sname}", json=json_index_llamacpp_hf_all)
            assert response.status_code == 200
            response = wait_for_index_status(client, sname)
            assert "finished" in response.json()["message"]

            json_predict = {
                "app": {"repository": sname},
                "parameters": {
                    "input": {
                        "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"
                    }
                },
            }
            response = client.post(f"/v1/predict/{sname}", json=json_predict)
            assert response.status_code == 200
            print(response.json())
        finally:
            ensure_service_deleted(client, sname)
            ensure_repo_deleted(sname)

    def test_llamacpp_e5(self, client):
        ##############################################
        # build a new service with same embeddings but different lib i.e. huggingface
        json_create_llamacpp_e5 = {
            "app": {
                "repository": "test_llamacpp_hf_e5",
                "models_repository": models_repo,
            },
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "strategy": "fast",
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "huggingface",
                        "embedding_model": "intfloat/multilingual-e5-small",
                        "reindex": True,
                        "index_protection": False,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                },
                "llm": {
                    "source": "bartowski/Qwen2.5-0.5B-Instruct-GGUF",
                    "filename": "Qwen2.5-0.5B-Instruct-Q8_0.gguf",
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
        ensure_repo_deleted("test_llamacpp_hf_e5")
        try:
            response = client.put(f"/v1/app/{sname}", json=json_create_llamacpp_e5)
            assert response.status_code == 200

            response = client.put(f"/v1/index/{sname}", json=json_index_llamacpp_e5)
            assert response.status_code == 200
            response = wait_for_index_status(client, sname)
            assert "finished" in response.json()["message"]

            json_predict = {
                "app": {"repository": "test_llamacpp_hf_e5"},
                "parameters": {
                    "input": {
                        "message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"
                    }
                },
            }
            response = client.post(f"/v1/predict/{sname}", json=json_predict)
            assert response.status_code == 200
            print(response.json()["output"])
        finally:
            ensure_service_deleted(client, sname)
            ensure_repo_deleted("test_llamacpp_hf_e5")

