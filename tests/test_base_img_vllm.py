import pytest

from base_img_helpers import generic_index, models_repo, pretty_print_response

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

def test_vllm_single_image(temp_dir, client):
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
                    "embedding_lib": "vllm",
                    "embedding_model": "Qwen/Qwen2-VL-2B-Instruct",
                    "top_k": 1,
                    "shared_model": True,
                    "vllm_rag_memory_utilization": 0.4,
                    # "vllm_rag_quantization": "bitsandbytes",
                    "vllm_rag_enforce_eager": True,
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
                # "source": "Qwen/Qwen2-VL-2B-Instruct",
                "source": "HuggingFaceTB/SmolVLM-256M-Instruct",
                "gpu_ids": [0],
                "image_width": 640,
                "image_height": 320,
                "inference": {"lib": "vllm"},
                "shared": True,
                "vllm_memory_utilization": 0.4,
                "vllm_quantization": "bitsandbytes",
                "context_size": 256,
            },
        },
    }
    json_index_img_hf = {
        "parameters": {
            "input": {
                "preprocessing": {"files": ["all"], "save_output": True},
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
        response = client.put("/v1/app/test_vllm_single_image", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_vllm_single_image"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_vllm_single_image" in response.json()["info"]["services"]

        generic_index(client, "test_vllm_single_image", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}

        response = client.post("/v1/predict/test_vllm_single_image", json=json_predict)
        pretty_print_response(response.json())
        assert response.json()["sources"]["context"][0]["distance"] > 0.0
        # assert "Rapport" in response.json()["output"]

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les auteurs ?"}}
        # }
        # response = client.post("/v1/predict/test_vllm_single_image", json=json_predict2)
        # pretty_print_response(response.json())
        # assert (
        #     "LOPEZ" in response.json()["output"] or "Lopez" in response.json()["output"]
        # )
    finally:
        # delete the service
        response = client.delete("/v1/app/test_vllm_single_image")
        assert response.status_code == 200


@pytest.mark.repository_path("test_vllm_q25")
@pytest.mark.asyncio
@pytest.mark.skip(reason="Blocked legacy scenario: Qwen2.5-VL vLLM path is not stable enough for maintained integration coverage.")
def test_vllm_single_image_25(temp_dir, client):
    pytest.skip("Blocked legacy scenario: Qwen2.5-VL vLLM path is not stable enough for maintained integration coverage.")


