import copy

import pytest
from utils import ensure_repo_deleted, ensure_service_deleted, wait_for_index_status

from base_text_helpers import models_repo

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

json_predict = {
    "app": {"repository": "colette_test/"},
    "parameters": {
        "input": {"message": "Quel est le nombre d'objets spatiaux de plus de 10cm ?"}
    },
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


class TestRagLangchainTxtOllama:
    def test_ollama_gpt4all(self, client):
        ##############################################
        # build the service with ollama and no rag
        json_create_ollama_gpt4all_all = {
            "app": {"repository": "test_ollama_norag"},
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "save_output": True,
                        "strategy": "fast",
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Réponse: ",
                        "template_prompt_variables": ["question"],
                    },
                },
                "llm": {"source": "qwen2.5:0.5b", "inference": {"lib": "ollama"}},
            },
        }
        sname = "test_ollama_norag"
        ensure_service_deleted(client, sname)
        ensure_repo_deleted("test_ollama_norag")
        try:
            response = client.put(
                f"/v1/app/{sname}",
                json=json_create_ollama_gpt4all_all,
            )
            assert response.status_code == 200
            assert response.json()["service_name"] == sname

            response = client.get("/v1/info")
            assert sname in response.json()["info"]["services"]

            json_predict_norag = {
                "parameters": {"input": {"message": "Quel est la capitale de la France ?"}},
            }
            response = client.post(f"/v1/predict/{sname}", json=json_predict_norag)
            print("response=", response.json())
            assert "Paris" in response.json()["output"]
        finally:
            ensure_service_deleted(client, sname)
            ensure_repo_deleted("test_ollama_norag")

    def test_ollama_gpt4all_rag(self, client):
        ##############################################
        # build the service with ollama
        json_create_ollama_gpt4all_all = {
            "app": {
                "repository": "test_ollama_gpt4all_all-MiniLM-L6-v2",
                "models_repository": models_repo,
            },
            "parameters": {
                "input": {
                    "lib": "langchain",
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "save_output": True,
                        "strategy": "fast",
                    },
                    "rag": {
                        "indexdb_lib": "chromadb",
                        "embedding_lib": "huggingface",
                        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                        "search": True,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                },
                # Keep this aligned with preflight/model bootstrap expectations.
                "llm": {"source": "qwen2.5:0.5b", "inference": {"lib": "ollama"}},
            },
        }
        json_index_ollama = {
            "parameters": {
                "input": {
                    "preprocessing": {
                        "files": ["all"],
                        "lib": "unstructured",
                        "save_output": True,
                        "strategy": "fast",
                    },
                    "rag": {"reindex": True, "index_protection": False},
                    "data": ["tests/data"],
                },
            }
        }
        sname = "test_ollama_gpt4all_all-MiniLM-L6-v2"
        ensure_service_deleted(client, sname)
        ensure_repo_deleted(sname)
        try:
            response = client.put(
                f"/v1/app/{sname}",
                json=json_create_ollama_gpt4all_all,
            )
            assert response.status_code == 200
            assert response.json()["service_name"] == sname

            response = client.put(f"/v1/index/{sname}", json=json_index_ollama)
            assert response.status_code == 200
            response = wait_for_index_status(client, sname)
            assert "finished" in response.json()["message"]

            response = client.post(f"/v1/predict/{sname}", json=json_predict)
            print("response=", response.json())
            assert response.status_code == 200

            response = client.post(f"/v1/predict/{sname}", json=json_predict_prompt)
            print("response=", response.json())
            assert response.status_code == 200
        finally:
            ensure_service_deleted(client, sname)
            ensure_repo_deleted(sname)

