import os

from fastapi.testclient import TestClient
from utils import pretty_print_response

from colette.httpjsonapi import app  # noqa

models_repo = os.getenv("MODELS_REPO", "models")

# testing
client = TestClient(app)


def test_diffusr():
    response = client.get("/v1/info")
    assert "commit" in response.json()["version"]

    ##############################################
    # build the service with diffusr backend and flux model
    json_create_img_diffusr_flux = {
        "app": {
            "repository": "test_diffusr_flux",
            "models_repository": models_repo,
            "verbose": "debug",
        },
        "parameters": {
            "input": {
                "lib": "diffusr",
            },
            "llm": {
                # "source": "black-forest-labs/FLUX.1-dev",
                # "source": "black-forest-labs/FLUX.1-schnell",
                "source": "Tencent-Hunyuan/HunyuanDiT-v1.2-Diffusers-Distilled",
                "gpu_ids": [0],
                "image_width": 1024,
                "image_height": 1024,
                "inference": {"lib": "diffusers"},
                "load_in_8bit": True,
            },
        },
    }
    response = client.put("/v1/app/test_diffusr_flux", json=json_create_img_diffusr_flux)

    pretty_print_response(response.json())
    assert response.status_code == 200
    assert response.json()["service_name"] == "test_diffusr_flux"

    json_predict = {"parameters": {"input": {"message": "Dessines-moi un mouton"}}}
    response = client.post("/v1/predict/test_diffusr_flux", json=json_predict)
    pretty_print_response(response.json())

    # json_predict2 = {
    #     "parameters": {
    #         "input": {
    #             "message": "A blade-runner is having lunch at a chinese"
    #                        " soup restaurant below an ATARI advertising sign"
    #         }
    #     }
    # }
    # response = client.post("/v1/predict/test_diffusr_flux", json=json_predict2)
    pretty_print_response(response.json())
