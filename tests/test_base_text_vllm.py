import copy
import sys

import pytest
from utils import ensure_repo_deleted, ensure_service_deleted, wait_for_index_status

from base_text_helpers import VLLM_E2E_ENABLED, models_repo, require_vllm_runtime

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


class TestRagLangchainTxtVllm:
    @pytest.mark.skipif(
        not VLLM_E2E_ENABLED,
        reason="set COLETTE_ENABLE_VLLM_E2E=1 to run vLLM e2e tests",
    )
    def test_vllm(self, client):
        require_vllm_runtime()
        ##############################################
        # build a service with vllm
        json_create_vllm = {
            "app": {
                "repository": "test_vllm",
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
                        "gpu_id": -1,
                    },
                    "template": {
                        "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Contexte: {context} Réponse: ",
                        "template_prompt_variables": ["context", "question"],
                    },
                    "data": ["tests/data"],
                },
                "llm": {
                    "source": "Qwen/Qwen2.5-0.5B",
                    "context_size": 2048,
                    "memory_utilization": 0.3,
                    "dtype": "float32",
                    "inference": {"lib": "vllm"},
                },
            },
        }

        json_index_vllm = {
            "parameters": {
                "input": {
                    "preprocessing": {"files": ["all"], "lib": "unstructured", "strategy": "fast"},
                    "rag": {"reindex": True, "index_protection": False, "gpu_id": -1},
                    "data": ["tests/data"],
                },
            }
        }
        sname = "test_vllm"
        ensure_service_deleted(client, sname)
        ensure_repo_deleted(sname)
        try:
            response = client.put(f"/v1/app/{sname}", json=json_create_vllm)
            if response.status_code != 200:
                detail = response.json().get("status", {}).get("colette_message", "")
                if "Engine core initialization failed" in str(detail):
                    pytest.skip("vLLM engine cannot initialize in this runtime (CUDA fork/spawn limitation)")
            assert response.status_code == 200
            assert response.json()["service_name"] == sname

            response = client.put(f"/v1/index/{sname}", json=json_index_vllm)
            assert response.status_code == 200
            response = wait_for_index_status(client, sname)
            assert "finished" in response.json()["message"] or "error" in response.json()["message"]

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
            print(response.json()["output"])
        finally:
            ensure_service_deleted(client, sname)
            ensure_repo_deleted(sname)
