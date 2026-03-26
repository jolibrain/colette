import shutil

import pytest

from base_img_helpers import generic_index, models_repo, pretty_print_response

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


@pytest.mark.repository_path("test_hf_shared_models")
def test_hf_shared_models(temp_dir, client):
    json_create_img_hf = {
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
                    "shared_model": True,
                    "top_k": 4,
                    "gpu_id": 0,
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
                "shared_model": True,
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
                    "dpi": 200,
                },
                "data": ["tests/data"],
                "rag": {
                    "reindex": False,
                    "index_protection": False,
                    "gpu_id": 0,
                },
            },
        }
    }
    try:
        response = client.put("/v1/app/test_hf_shared_models", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200

        json_create_img_hf["app"]["repository"] = "test_hf_shared_models2"
        response = client.put("/v1/app/test_hf_shared_models2", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_shared_models2" in response.json()["info"]["services"]

        generic_index(client, "test_hf_shared_models2", json_index_img_hf)
        generic_index(client, "test_hf_shared_models", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document RINFANR5L16B2040 ?"}}}
        response = client.post("/v1/predict/test_hf_shared_models", json=json_predict)
        pretty_print_response(response.json())
        # assert "Rapport" in response.json()["output"]

        response = client.post("/v1/predict/test_hf_shared_models2", json=json_predict)
        pretty_print_response(response.json())
    finally:
        # delete the services
        response = client.delete("/v1/app/test_hf_shared_models")
        assert response.status_code == 200
        response = client.delete("/v1/app/test_hf_shared_models2")
        assert response.status_code == 200

        # delete the second repository
        shutil.rmtree("test_hf_shared_models2", ignore_errors=True)


