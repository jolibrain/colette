import pytest

from base_img_helpers import generic_index, models_repo, pretty_print_response

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


@pytest.mark.repository_path("test_hf_single_image_coldb")
def test_hf_single_image_coldb(temp_dir, client):
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
                    "indexdb_lib": "coldb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "vidore/colpali-v1.2-hf",
                    "top_k": 1,
                    "gpu_id": 0,
                    "num_partitions": 100,
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
        response = client.put("/v1/app/test_hf_single_image_coldb", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_hf_single_image_coldb"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_single_image_coldb" in response.json()["info"]["services"]

        generic_index(client, "test_hf_single_image_coldb", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}

        response = client.post("/v1/predict/test_hf_single_image_coldb", json=json_predict)
        pretty_print_response(response.json())
        assert "Rapport" in response.json()["output"]

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les auteurs ?"}}
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_single_image_coldb", json=json_predict2
        # )
        # pretty_print_response(response.json())
        # assert "LOPEZ" in response.json()["output"]
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_single_image_coldb")
        assert response.status_code == 200


##############################################
# build the service with hf backend with preprocessing and layout crops using coldb
@pytest.mark.repository_path("test_hf_layout_coldb")
@pytest.mark.asyncio
def test_hf_layout_coldb(temp_dir, client):
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
                    "indexdb_lib": "coldb",
                    "gpu_id": 0,
                    "embedding_lib": "huggingface",
                    "embedding_model": "vidore/colpali-v1.2-hf",
                    "top_k": 4,
                    "num_partitions": 100,
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
                "preprocessing": {"files": ["all"], "dpi": 200},
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
        # test index protection
        response = client.put("/v1/app/test_hf_layout_coldb", json=json_create_img_hf)
        pretty_print_response(response.json())

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_layout_coldb" in response.json()["info"]["services"]

        generic_index(client, "test_hf_layout_coldb", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document RINFANR5L16B2040 ?"}}}
        response = client.post("/v1/predict/test_hf_layout_coldb", json=json_predict)
        pretty_print_response(response.json())
        # assert "Rapport" in response.json()["output"]

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les députés ?"}}
        # }
        # response = client.post("/v1/predict/test_hf_layout_coldb", json=json_predict2)
        # pretty_print_response(response.json())
        # # assert "LOPEZ" in response.json()["output"]

        # json_predict3 = {
        #     "parameters": {
        #         "input": {
        #             "message": "Quel est le pourcentage d'investissement en Space Transportation ?"
        #         }
        #     }
        # }
        # response = client.post("/v1/predict/test_hf_layout_coldb", json=json_predict3)
        # pretty_print_response(response.json())
        # # assert "17%" in response.json()["output"]
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_layout_coldb")
        assert response.status_code == 200

        ##############################################
