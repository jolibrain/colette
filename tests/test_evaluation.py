import json
import os
import shutil
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from fastapi.testclient import TestClient
from utils import pretty_print_response

col_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../src")
sys.path.append(col_dir)

tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../tools")
sys.path.append(tools_dir)

from httpjsonapi import app  # noqa
from evaluation import run_evaluation #noqa

models_repo = os.getenv("MODELS_REPO", 'models')

client = TestClient(app)


@pytest.fixture
def temp_dir(request):
    # Get the repository path from the test function's parameters
    temp_dir = Path(request.node.get_closest_marker("repository_path").args[0])
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)


def test_info_call():
    response = client.get("/v1/info")
    assert "commit" in response.json()["version"]


##############################################
# build the service with hf backend with a single image
@pytest.mark.repository_path("test_hf_single_image")
def test_hf_single_image(temp_dir):
    json_create_img_hf = {
        "app": {
            "repository": str(temp_dir),
            "models_repository": models_repo,
            "verbose": "info"
        },
        "parameters": {
            "input": {
                "lib": "hf",
                "preprocessing": {
                    "files": ["all"],
                    "save_output": True
                },
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    "reindex": True,
                    "index_protection": False,
                    "top_k": 1,
                    "gpu_id": 0,
                    "ragm": {
                        "layout_detection": False
                    }
                },
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Réponse: ",
                    "template_prompt_variables": ["question"]
                },
                "data": ["tests/data_img1"]
            },
            "llm": {
                "source": "Qwen/Qwen2-VL-2B-Instruct",
                "gpu_ids": [0],
                "image_width": 640,
                "image_height": 960,
                "inference": {
                    "lib": "huggingface"
                }
            }
        }
    }
    try:
        ad_json = json.dumps(json_create_img_hf)
        response = client.put("/v1/app/test_hf_single_image", data={"ad": ad_json})
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == "test_hf_single_image"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert "test_hf_single_image" in response.json()["info"]["services"]

        print(str(temp_dir))

        qa = [dict(
            id=1, question="Quels sont les auteurs?",
            answer="Les auteurs sont Mme Cécile RILHAC et M. Aurélien LOPEZ-LIGUORI.",
            short_answer=["RILHAC", "LOPEZ-LIGUORI"],
            references=[dict(file="RINFANR5L16B2040.jpg-001", pages=["0"])],
            lang="fr"
        )]

        with TemporaryDirectory() as temp_dir_2:
            # create the question/answer file
            file_name = Path(temp_dir_2) / "qa.json"
            print(qa)
            json.dump(qa, open(file_name, "w"))
            assert file_name.exists()

            class Args:
                app_dir = str(temp_dir)
                qa = str(file_name)
                debug = False


            print(Args())

            # run the evaluation
            _, results_df, retriever_df, _ = run_evaluation(Args())
            assert results_df.shape == (1, 10), results_df.shape
            row = results_df.row(0)
            assert "RILHAC" in row[2], row[2]
            assert "RINFANR5L16B2040.jpg-001" in row[8], row[8]
            assert retriever_df.shape == (1, 4), retriever_df.shape
    finally:
        # delete the service
        response = client.delete("/v1/app/test_hf_single_image")
        assert response.status_code == 200
