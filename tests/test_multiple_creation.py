import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

from colette.httpjsonapi import app


@pytest_asyncio.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


@pytest.mark.asyncio
def test_multiple_creations(client):
    json_create_llamacpp_gpt4all_all = {
        "app": {"repository": "test_llamacpp_gpt4all_all-MiniLM-L6-v2"},
        "parameters": {
            "input": {
                "lib": "langchain",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                    "search": False,
                    "reindex": True,
                    "index_protection": False,
                    "gpu_id": -1,
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
                "inference": {"lib": "llamacpp"},
            },
        },
    }

    response = client.put("/v1/app/test_llamacpp_gpt4all_all-MiniLM-L6-v2", json=json_create_llamacpp_gpt4all_all)
    assert response.status_code == 200
    assert response.json()["service_name"] == "test_llamacpp_gpt4all_all-MiniLM-L6-v2"

    response = client.put("/v1/app/test_llamacpp_gpt4all_all-MiniLM-L6-v2", json=json_create_llamacpp_gpt4all_all)
    assert response.status_code == 400
    print("reponse.json", response.json())
