import pytest

from base_img_helpers import generic_index, models_repo, pretty_print_response

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

def test_info_call(client):
    response = client.get("/v1/info")
    assert "commit" in response.json()["version"]


##############################################
# build the service with hf backend with a single image
@pytest.mark.repository_path("test_hf_single_image")
@pytest.mark.asyncio
def test_hf_single_image(temp_dir, client):
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
                    "top_k": 1,
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
                "image_width": 512,
                "image_height": 512,
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
        response = client.put("/v1/app/test_hf_single_image", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_hf_single_image"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_single_image" in response.json()["info"]["services"]

        generic_index(client, "test_hf_single_image", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}

        response = client.post("/v1/predict/test_hf_single_image", json=json_predict)
        pretty_print_response(response.json())
        assert response.json()["sources"]["context"][0]["distance"] > 0.0
        assert "rapport" in response.json()["output"].lower()

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les auteurs ?"}}
        # }
        # response = client.post("/v1/predict/test_hf_single_image", json=json_predict2)
        # pretty_print_response(response.json())
        # assert (
        #     "LOPEZ" in response.json()["output"]
        #     or "Lopez" in response.json()["output"]["output"]
        # )
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_single_image")
        assert response.status_code == 200


##############################################
# build the service with hf backend with a single image with query rephrasing
@pytest.mark.repository_path("test_hf_single_image_rephrasing")
@pytest.mark.asyncio
def test_hf_single_image_rephrasing(temp_dir, client):
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
                    "top_k": 1,
                    "ragm": {"layout_detection": False},
                },
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Réponse: ",
                    "template_prompt_variables": ["question"],
                },
            },
            "llm": {
                "source": "Qwen/Qwen2-VL-7B-Instruct",
                "gpu_ids": [0],
                "image_width": 640,
                "image_height": 960,
                "query_rephrasing": True,
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
        response = client.put("/v1/app/test_hf_single_image_rephrasing", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_hf_single_image_rephrasing"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_single_image_rephrasing" in response.json()["info"]["services"]

        generic_index(client, "test_hf_single_image_rephrasing", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}

        response = client.post("/v1/predict/test_hf_single_image_rephrasing", json=json_predict)
        pretty_print_response(response.json())
        assert response.json()["sources"]["context"][0]["distance"] > 0.0
        # assert "Rapport" in response.json()["output"]

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les auteurs ?"}}
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_single_image_rephrasing", json=json_predict2
        # )
        # pretty_print_response(response.json())
        # assert "LOPEZ" in response.json()["output"]
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_single_image_rephrasing")
        assert response.status_code == 200


@pytest.mark.repository_path("test_hf_single_image_autoscale")
@pytest.mark.asyncio
def test_hf_single_image_autoscale(temp_dir, client):
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
                    "top_k": 1,
                    "ragm": {
                        "layout_detection": False,
                        "auto_scale_for_font": True,
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
    json_index_img_hf = {
        "parameters": {
            "input": {
                "preprocessing": {
                    "files": ["all"],
                    "save_output": True,
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
        response = client.put("/v1/app/test_hf_single_image_autoscale", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_hf_single_image_autoscale"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_single_image_autoscale" in response.json()["info"]["services"]

        generic_index(client, "test_hf_single_image_autoscale", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}

        response = client.post("/v1/predict/test_hf_single_image_autoscale", json=json_predict)
        pretty_print_response(response.json())
        assert response.json()["sources"]["context"][0]["distance"] > 0.0
        assert "Rapport" in response.json()["output"]

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les auteurs ?"}}
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_single_image_autoscale", json=json_predict2
        # )
        # pretty_print_response(response.json())
        # assert "LOPEZ" in response.json()["output"]
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_single_image_autoscale")
        assert response.status_code == 200


