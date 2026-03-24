import pytest

from base_img_helpers import generic_index, models_repo, pretty_print_response

pytestmark = [pytest.mark.integration, pytest.mark.e2e]

def test_hf_multiple_images(temp_dir, client):
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
                    "top_k": 2,
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
                "data": ["tests/data_img2"],
                "rag": {
                    "reindex": True,
                    "index_protection": False,
                    "gpu_id": 0,
                },
            },
        }
    }
    try:
        response = client.put("/v1/app/test_hf_multiple_images", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_hf_multiple_images"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_multiple_images" in response.json()["info"]["services"]

        generic_index(client, "test_hf_multiple_images", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}
        response = client.post("/v1/predict/test_hf_multiple_images", json=json_predict)
        pretty_print_response(response.json())
        assert "Rapport" in response.json()["output"] or "RAPPORT" in response.json()["output"]

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les députés ?"}}
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_multiple_images", json=json_predict2
        # )
        # pretty_print_response(response.json())
        # assert "LOPEZ" in response.json()["output"]

        # json_predict3 = {
        #     "parameters": {
        #         "input": {
        #             "message": "Quel est le pourcentage d'investissement en Space Transportation ?"
        #         }
        #     }
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_multiple_images", json=json_predict3
        # )
        # pretty_print_response(response.json())
        # # assert "17%" in response.json()["output"]
        # assert "%" in response.json()["output"]  # 17% should be good here
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_multiple_images")
        assert response.status_code == 200


##############################################
# build the service with hf backend with multiple_images with smolvlm
# @pytest.mark.repository_path("test_hf_multiple_images_smolvlm")
# def test_hf_multiple_images_smolvlm(temp_dir):
#     json_create_img_hf = {
#         "app": {
#             "repository": str(temp_dir),
#             "models_repository": models_repo,
#             "verbose": "debug",
#         },
#         "parameters": {
#             "input": {
#                 "lib": "hf",
#                 "preprocessing": {
#                     "files": ["all"],
#                     "save_output": True,
#                 },
#                 "rag": {
#                     "indexdb_lib": "chromadb",
#                     "embedding_lib": "huggingface",
#                     "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
#                     "reindex": True,
#                     "index_protection": False,
#                     "top_k": 2,
#                     "gpu_id": 0,
#                     "ragm": {"layout_detection": False},
#                 },
#                 "template": {
#                     "template_prompt": "Tu es un assistant de réponse à des questions."
#                     " Question: {question} Réponse: ",
#                     "template_prompt_variables": ["question"],
#                 },
#                 "data": ["tests/data_img2"],
#             },
#             "llm": {
#                 "source": "HuggingFaceTB/SmolVLM-Instruct",
#                 "gpu_ids": [0],
#                 "image_width": 640,
#                 "image_height": 960,
#                 "inference": {"lib": "huggingface"},
#             },
#         },
#     }
#     try:
#         ad_json = json.dumps(json_create_img_hf)
#         response = client.put(
#             "/v1/app/test_hf_multiple_images_smolvlm",
#             data={"ad": ad_json},
#         )
#         pretty_print_response(response.json())
#         assert response.status_code == 200
#         assert response.json()["service_name"] == "test_hf_multiple_images_smolvlm"

#         response = client.get("/v1/info")
#         pretty_print_response(response.json())
#         assert "test_hf_multiple_images_smolvlm" in response.json()["info"]["services"]

#         json_predict = {
#             "parameters": {"input": {"message": "Quel est le titre du document ?"}}
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_smolvlm", json=json_predict
#         )
#         pretty_print_response(response.json())
#         assert (
#             "Rapport" in response.json()["output"]
#             or "RAPPORT" in response.json()["output"]
#         )

#         json_predict2 = {
#             "parameters": {"input": {"message": "Quels sont les députés ?"}}
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_smolvlm", json=json_predict2
#         )
#         pretty_print_response(response.json())
#         assert "LOPEZ" in response.json()["output"]

#         json_predict3 = {
#             "parameters": {
#                 "input": {
#                     "message": "Quel est le pourcentage d'investissement en Space Transportation ?"
#                 }
#             }
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_smolvlm", json=json_predict3
#         )
#         pretty_print_response(response.json())
#         # assert "17%" in response.json()["output"]
#     finally:
#         # delete the service
#         response = client.delete("/v1/app/test_hf_multiple_images_smolvlm")
#         assert response.status_code == 200


##############################################
# build the service with hf backend with chunks
# @pytest.mark.repository_path("test_hf_multiple_images_chunks")
# def test_hf_multiple_images_chunks(temp_dir):
#     json_create_img_hf = {
#         "app": {
#             "repository": str(temp_dir),
#             "models_repository": models_repo,
#             "verbose": "debug",
#         },
#         "parameters": {
#             "input": {
#                 "lib": "hf",
#                 "preprocessing": {
#                     "files": ["all"],
#                     "save_output": True,
#                 },
#                 "rag": {
#                     "indexdb_lib": "chromadb",
#                     "embedding_lib": "huggingface",
#                     "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
#                     "chunk_num": 3,
#                     "chunk_overlap": 20,
#                     "reindex": True,
#                     "index_protection": False,
#                     "top_k": 4,
#                     "gpu_id": 0,
#                     "ragm": {
#                         "layout_detection": False,
#                         "index_overview": False
#                     },
#                 },
#                 "template": {
#                     "template_prompt": "Tu es un assistant de réponse à des questions."
#                     " Question: {question} Réponse: ",
#                     "template_prompt_variables": ["question"],
#                 },
#                 "data": ["tests/data_img2"],
#             },
#             "llm": {
#                 "source": "Qwen/Qwen2-VL-2B-Instruct",
#                 "gpu_ids": [0],
#                 "image_width": 640,
#                 "image_height": 320,
#                 "inference": {"lib": "huggingface"},
#             },
#         },
#     }
#     try:
#         ad_json = json.dumps(json_create_img_hf)
#         response = client.put(
#             "/v1/app/test_hf_multiple_images_chunks",
#             data={"ad": ad_json},
#         )
#         pretty_print_response(response.json())
#         assert response.status_code == 200
#         assert response.json()["service_name"] == "test_hf_multiple_images_chunks"
#         response = client.get("/v1/info")
#         pretty_print_response(response.json())
#         assert "test_hf_multiple_images_chunks" in response.json()["info"]["services"]

#         json_predict = {
#             "parameters": {"input": {"message": "Quel est le titre du document ?"}}
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_chunks", json=json_predict
#         )
#         pretty_print_response(response.json())
#         # assert "Rapport" in response.json()["output"]

#         json_predict2 = {
#             "parameters": {"input": {"message": "Quels sont les députés ?"}}
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_chunks", json=json_predict2
#         )
#         pretty_print_response(response.json())
#         assert "LOPEZ" in response.json()["output"]

#         json_predict3 = {
#             "parameters": {
#                 "input": {
#                     "message": "Quel est le pourcentage d'investissement en Space Transportation ?"
#                 }
#             }
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_chunks", json=json_predict3
#         )
#         pretty_print_response(response.json())
#         assert "17%" in response.json()["output"]  # 17% should be good here
#     finally:
#         # delete the service
#         response = client.delete("/v1/app/test_hf_multiple_images_chunks")
#         assert response.status_code == 200


##############################################
# build the service with hf backend with crops
@pytest.mark.repository_path("test_hf_multiple_images_crops")
@pytest.mark.asyncio
def test_hf_multiple_images_crops(temp_dir, client):
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
                    "top_k": 4,
                    "ragm": {
                        "layout_detection": True,
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
                "preprocessing": {
                    "files": ["all"],
                    "save_output": True,
                },
                "data": ["tests/data_img2"],
                "rag": {
                    "reindex": True,
                    "index_protection": False,
                    "gpu_id": 0,
                },
            },
        }
    }
    try:
        response = client.put("/v1/app/test_hf_multiple_images_crops", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_hf_multiple_images_crops"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_multiple_images_crops" in response.json()["info"]["services"]

        generic_index(client, "test_hf_multiple_images_crops", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}
        response = client.post("/v1/predict/test_hf_multiple_images_crops", json=json_predict)
        pretty_print_response(response.json())
        # assert "Rapport" in response.json()["output"]

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les députés ?"}}
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_multiple_images_crops", json=json_predict2
        # )
        # pretty_print_response(response.json())
        # assert "LOPEZ" in response.json()["output"]

        # json_predict3 = {
        #     "parameters": {
        #         "input": {
        #             "message": "Quel est le pourcentage d'investissement en Space Transportation ?"
        #         }
        #     }
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_multiple_images_crops", json=json_predict3
        # )
        # pretty_print_response(response.json())
        # assert "17%" in response.json()["output"]
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_multiple_images_crops")
        assert response.status_code == 200


##############################################
# build the service with hf backend with stitched crops
# @pytest.mark.repository_path("test_hf_multiple_images_stitched_crops")
# def test_hf_multiple_images_stitched_crops(temp_dir):
#     json_create_img_hf = {
#         "app": {
#             "repository": str(temp_dir),
#             "models_repository": models_repo,
#             "verbose": "debug",
#         },
#         "parameters": {
#             "input": {
#                 "lib": "hf",
#                 "preprocessing": {
#                     "files": ["all"],
#                     "save_output": True,
#                 },
#                 "rag": {
#                     "indexdb_lib": "chromadb",
#                     "embedding_lib": "huggingface",
#                     "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
#                     "reindex": True,
#                     "index_protection": False,
#                     "top_k": 4,
#                     "gpu_id": 0,
#                     "ragm": {
#                         "layout_detection": True,
#                         "index_overview": False,
#                     },
#                 },
#                 "template": {
#                     "template_prompt": "Tu es un assistant de réponse à des questions."
#                      " Question: {question} Réponse: ",
#                     "template_prompt_variables": ["question"],
#                 },
#                 "data": ["tests/data_img2"],
#             },
#             "llm": {
#                 "source": "google/gemma-3-4b-it",
#                 "gpu_ids": [0],
#                 "stitch_crops": True,
#                 "inference": {"lib": "huggingface"},
#             },
#         },
#     }
#     try:
#         ad_json = json.dumps(json_create_img_hf)
#         response = client.put(
#             "/v1/app/test_hf_multiple_images_stitched_crops",
#             data={"ad": ad_json},
#         )
#         pretty_print_response(response.json())
#         assert response.status_code == 200
#         assert (
#             response.json()["service_name"] == "test_hf_multiple_images_stitched_crops"
#         )

#         response = client.get("/v1/info")
#         pretty_print_response(response.json())
#         assert (
#             "test_hf_multiple_images_stitched_crops"
#             in response.json()["info"]["services"]
#         )

#         json_predict = {
#             "parameters": {"input": {"message": "Quel est le titre du document ?"}}
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_stitched_crops", json=json_predict
#         )
#         pretty_print_response(response.json())
#         # assert "Rapport" in response.json()["output"]

#         json_predict2 = {
#             "parameters": {"input": {"message": "Quels sont les députés ?"}}
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_stitched_crops", json=json_predict2
#         )
#         pretty_print_response(response.json())
#         # assert "LOPEZ" in response.json()["output"]

#         json_predict3 = {
#             "parameters": {
#                 "input": {
#                     "message": "Quel est le pourcentage d'investissement en Space Transportation ?"
#                 }
#             }
#         }
#         response = client.post(
#             "/v1/predict/test_hf_multiple_images_stitched_crops", json=json_predict3
#         )
#         pretty_print_response(response.json())
#         assert "%" in response.json()["output"]
#     finally:
#         # delete the service
#         response = client.delete("/v1/app/test_hf_multiple_images_stitched_crops")
#         assert response.status_code == 200


##############################################
# build the service with hf backend with pixtral
# @pytest.mark.repository_path("test_hf_single_image_pixtral")
# def test_hf_single_image_pixtral(temp_dir):
#     json_create_img_hf = {
#         "app": {
#             "repository": str(temp_dir),
#             "models_repository": models_repo,
#             "verbose": "debug",
#         },
#         "parameters": {
#             "input": {
#                 "lib": "hf",
#                 "preprocessing": {"files": ["all"], "save_output": True},
#                 "rag": {
#                     "indexdb_lib": "chromadb",
#                     "embedding_lib": "huggingface",
#                     "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
#                     "reindex": True,
#                     "index_protection": False,
#                     "top_k": 4,
#                     "gpu_id": 0,
#                 },
#                 "template": {
#                     "template_prompt": "Tu es un assistant de réponse à des questions."
#                     " Question: {question} Réponse: ",
#                     "template_prompt_variables": ["question"],
#                 },
#                 "data": ["tests/data_img2"],
#             },
#             "llm": {
#                 "source": "mistral-community/pixtral-12b",
#                 "load_in_8bit": True,
#                 "gpu_ids": [0],
#                 "image_width": 640,
#                 "image_height": 320,
#                 "inference": {"lib": "huggingface"},
#             },
#         },
#     }
#     try:
#         ad_json = json.dumps(json_create_img_hf)
#         response = client.put(
#             "/v1/app/test_hf_single_image_pixtral",
#             data={"ad": ad_json},
#         )
#         pretty_print_response(response.json())
#         assert response.status_code == 200
#         assert response.json()["service_name"] == "test_hf_single_image_pixtral"

#         response = client.get("/v1/info")
#         pretty_print_response(response.json())
#         assert "test_hf_single_image_pixtral" in response.json()["info"]["services"]

#         json_predict = {
#             "parameters": {"input": {"message": "Quel est le titre du document ?"}}
#         }
#         response = client.post(
#             "/v1/predict/test_hf_single_image_pixtral", json=json_predict
#         )
#         pretty_print_response(response.json())
#         # assert "Rapport" in response.json()["output"]

#         json_predict2 = {
#             "parameters": {"input": {"message": "Quels sont les députés ?"}}
#         }
#         response = client.post(
#             "/v1/predict/test_hf_single_image_pixtral", json=json_predict2
#         )
#         pretty_print_response(response.json())
#         # assert "LOPEZ" in response.json()["output"]

#         json_predict3 = {
#             "parameters": {
#                 "input": {
#                     "message": "Quel est le pourcentage d'investissement en Space Transportation ?"
#                 }
#             }
#         }
#         response = client.post(
#             "/v1/predict/test_hf_single_image_pixtral", json=json_predict3
#         )
#         pretty_print_response(response.json())
#         # assert "17 %" in response.json()["output"]
#     finally:
#         # delete the service
#         response = client.delete("/v1/app/test_hf_single_image_pixtral")
#         assert response.status_code == 200


# ##############################################
# build the service with hf backend with preprocessing and layout crops
@pytest.mark.repository_path("test_hf_multiple_images_preproc_layout")
@pytest.mark.asyncio
def test_hf_multiple_images_preproc_layout(temp_dir, client):
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
                    "gpu_id": 0,
                    "top_k": 4,
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
                # "source": "google/gemma-3-4b-it", # does not fit on 24GB with the embedder
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
                    "reindex": True,
                    "index_protection": True,
                    "gpu_id": 0,
                },
            },
        }
    }
    try:
        response = client.put("/v1/app/test_hf_multiple_images_preproc_layout", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        generic_index(client, "test_hf_multiple_images_preproc_layout", json_index_img_hf)

        # indexing a second time under a new name
        response = client.put("/v1/app/test_hf_multiple_images_preproc_layout_bis", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200, response.json()
        response = generic_index(client, "test_hf_multiple_images_preproc_layout_bis", json_index_img_hf)
        pretty_print_response(response.json())

        # the first indexing should be ok
        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_multiple_images_preproc_layout" in response.json()["info"]["services"]
    finally:
        # removing the service
        response = client.delete("/v1/app/test_hf_multiple_images_preproc_layout")
        assert response.status_code == 200

    try:
        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_multiple_images_preproc_layout" not in response.json()["info"]["services"]

        # change the index protection to False
        json_index_img_hf["parameters"]["input"]["rag"]["index_protection"] = False

        response = client.put("/v1/app/test_hf_multiple_images_preproc_layout", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert "test_hf_multiple_images_preproc_layout" in response.json()["service_name"]
        generic_index(client, "test_hf_multiple_images_preproc_layout", json_index_img_hf)

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_multiple_images_preproc_layout" in response.json()["info"]["services"]

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document RINFANR5L16B2040 ?"}}}
        response = client.post("/v1/predict/test_hf_multiple_images_preproc_layout", json=json_predict)
        pretty_print_response(response.json())
        # assert "Rapport" in response.json()["output"]

        json_predict2 = {"parameters": {"input": {"message": "Quels sont les députés ?"}}}
        response = client.post("/v1/predict/test_hf_multiple_images_preproc_layout", json=json_predict2)
        pretty_print_response(response.json())
        # assert "LOPEZ" in response.json()["output"]

        json_predict3 = {
            "parameters": {"input": {"message": "Quel est le pourcentage d'investissement en Space Transportation ?"}}
        }
        response = client.post("/v1/predict/test_hf_multiple_images_preproc_layout", json=json_predict3)
        pretty_print_response(response.json())
        assert response.json()["sources"]["context"][0]["distance"] <= 0.6
        assert "17%" in response.json()["output"]
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_multiple_images_preproc_layout")
        assert response.status_code == 200


##############################################
# test shared models
def test_hf_multiple_images_with_duplicates(temp_dir, client):
    json_create_img_hf = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
            "verbose": "debug",
        },
        "parameters": {
            "input": {
                "lib": "hf",
                "data_output_type": "img",
                "preprocessing": {
                    "files": ["all"],
                    "dpi": 200,
                },
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    "top_k": 2,
                    "gpu_id": 0,
                    "ragm": {
                        "layout_detection": False,
                    },
                },
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Réponse: ",
                    "template_prompt_variables": ["question"],
                },
                "data": ["tests/data_dup"],
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
                "data": ["tests/data_dup"],
                "rag": {
                    "reindex": True,
                    "index_protection": False,
                    "gpu_id": 0,
                },
            },
        }
    }
    try:
        response = client.put("/v1/app/test_hf_multiple_images_with_duplicates", json=json_create_img_hf)
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_hf_multiple_images_with_duplicates"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_multiple_images_with_duplicates" in response.json()["info"]["services"]

        generic_index(client, "test_hf_multiple_images_with_duplicates", json_index_img_hf)

        json_predict = {"parameters": {"input": {"message": "Quel est le titre du document ?"}}}
        response = client.post("/v1/predict/test_hf_multiple_images_with_duplicates", json=json_predict)
        pretty_print_response(response.json())
        assert "Rapport" in response.json()["output"] or "RAPPORT" in response.json()["output"]

        # json_predict2 = {
        #     "parameters": {"input": {"message": "Quels sont les députés ?"}}
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_multiple_images_with_duplicates", json=json_predict2
        # )
        # pretty_print_response(response.json())
        # assert "LOPEZ" in response.json()["output"]

        # json_predict3 = {
        #     "parameters": {
        #         "input": {
        #             "message": "Quel est le pourcentage d'investissement en Space Transportation ?"
        #         }
        #     }
        # }
        # response = client.post(
        #     "/v1/predict/test_hf_multiple_images_with_duplicates", json=json_predict3
        # )
        # pretty_print_response(response.json())
        # # assert "17%" in response.json()["output"]
        # assert "%" in response.json()["output"]  # 17% should be good here
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_multiple_images_with_duplicates")
        assert response.status_code == 200
