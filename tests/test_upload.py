import json
import os
import shutil
import time
from pathlib import Path
from pprint import pprint

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from utils import pretty_print_response

from colette.httpjsonapi import app
from colette.kvstore import ImageStorageFactory

models_repo = os.getenv("MODELS_REPO", "models")


@pytest_asyncio.fixture(scope="module")
def client():
    with TestClient(app) as client:
        yield client


def count_files_recursively(directory):
    count = 0
    for _, _, files in os.walk(directory):
        count += len(files)
    return count


def generic_index(client, sname, index_json):
    response = client.put(f"/v1/index/{sname}", json=index_json)
    pretty_print_response(response.json())
    assert response.status_code == 200
    response = client.get(f"/v1/index/{sname}/status")
    assert response.status_code == 200

    while "running" in response.json()["message"]:
        time.sleep(2)
        response = client.get(f"/v1/index/{sname}/status")
    return response


def generic_upload(client, sname, ad_index, files):
    ad_json_str = json.dumps(ad_index)
    response = client.put(f"/v1/upload/{sname}", data={"ad": ad_json_str}, files=files)
    pretty_print_response(response.json())
    assert response.status_code == 200
    response = client.get(f"/v1/index/{sname}/status")
    assert response.status_code == 200

    while "running" in response.json()["message"]:
        time.sleep(2)
        response = client.get(f"/v1/index/{sname}/status")
    return response


@pytest.fixture
def temp_dir(request):
    # Get the repository path from the test function's parameters
    temp_dir = Path(request.node.get_closest_marker("repository_path").args[0])
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.mark.repository_path("test_create_without_upload")
@pytest.mark.asyncio
def test_create_without_upload(temp_dir, client):
    # Define the content of the JSON to send as part of the multipart
    ad_content = {
        "app": {"repository": str(temp_dir), "models_repository": models_repo, "verbose": "debug"},
        "parameters": {
            "input": {
                "lib": "hf",
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    "top_k": 3,
                    "ragm": {"layout_detection": True, "image_width": 512, "image_height": 512},
                },
            },
            "llm": {
                "source": "Qwen/Qwen2-VL-2B-Instruct",
                "gpu_ids": [0],
                "image_width": 320,
                "image_height": 480,
                "inference": {"lib": "huggingface"},
            },
        },
    }

    ad_index = {
        "parameters": {
            "input": {
                "preprocessing": {"files": ["all"], "filters": ["/~[^/]*$"], "dpi": 300},
                "data": ["tests/data_img1"],
                "rag": {
                    "reindex": True,
                    "index_protection": False,
                    "gpu_id": 0,
                },
            },
        }
    }

    upload_data_dir = Path("tests/upload_data")

    ad_index["parameters"]["input"]["data"].append(str(upload_data_dir))

    # Post to the correct endpoint
    response = client.put("/v1/app/test_create_without_upload", json=ad_content)

    print(response.status_code)
    pprint(response.json())

    generic_index(client, "test_create_without_upload", ad_index)
    # Assert the expected response
    assert response.status_code == 200

    # Check the number of crops
    kv = ImageStorageFactory.create_storage("hdf5", temp_dir / "kvstore.db")
    assert len(list(kv.iter_keys())) == 32

    response = client.delete("/v1/app/test_create_without_upload")
    assert response.status_code == 200


@pytest.mark.repository_path("test_create_with_upload")
@pytest.mark.asyncio
def test_create_with_upload(temp_dir, client):
    # Define the content of the JSON to send as part of the multipart
    ad_content = {
        "app": {"repository": str(temp_dir), "models_repository": models_repo, "verbose": "debug"},
        "parameters": {
            "input": {
                "lib": "hf",
                "preprocessing": {"files": ["all"], "filters": ["/~[^/]*$"], "dpi": 300},
                "rag": {
                    "indexdb_lib": "chromadb",
                    "embedding_lib": "huggingface",
                    "embedding_model": "Alibaba-NLP/gme-Qwen2-VL-2B-Instruct",
                    "gpu_id": 0,
                    "top_k": 3,
                    "reindex": True,
                    "index_protection": False,
                    "ragm": {"layout_detection": True, "image_width": 512, "image_height": 512},
                },
                "data": ["tests/data_img1"],
            },
            "llm": {
                "source": "Qwen/Qwen2-VL-2B-Instruct",
                "gpu_ids": [0],
                "image_width": 320,
                "image_height": 480,
                "inference": {"lib": "huggingface"},
            },
        },
    }

    ad_index = {
        "parameters": {
            "input": {
                "preprocessing": {"files": ["all"], "filters": ["/~[^/]*$"], "dpi": 300},
                "data": ["tests/data_img1"],
                "rag": {
                    "reindex": True,
                    "index_protection": False,
                    "gpu_id": 0,
                },
            },
        }
    }

    upload_data_dir = Path("tests/upload_data")

    pdf_file_path = upload_data_dir / "DeepSeek_R1.pdf"
    jpg_file_path = upload_data_dir / "RINFANR5L16B2040.jpg-016.jpg"

    # Ensure the files exist
    assert pdf_file_path.exists(), f"PDF file does not exist: {pdf_file_path}"
    assert jpg_file_path.exists(), f"JPEG file does not exist: {jpg_file_path}"

    response = client.put("/v1/app/test_create_with_upload", json=ad_content)
    print(response.status_code)
    pprint(response.json())

    # Assert the expected response
    assert response.status_code == 200

    generic_index(client, "test_create_with_upload", ad_index)

    # Open the files in binary mode using 'with' statement
    with open(pdf_file_path, "rb") as pdf_file, open(jpg_file_path, "rb") as jpg_file:
        # Prepare the files to upload
        files = [
            ("files", ("DeepSeek_R1.pdf", pdf_file, "application/pdf")),
            ("files", ("RINFANR5L16B2040.jpg-016.jpg", jpg_file, "image/jpeg")),
        ]
        del ad_index["parameters"]["input"]["data"]

        # Post to the correct endpoint
        response = generic_upload(client, "test_create_with_upload", ad_index, files)
        # response = client.put("/v1/upload/test_create_with_upload", json=ad_index, files=files)

    print(response.status_code)
    print(response.json())

    # Assert the expected response
    assert response.status_code == 200

    # check that tempdir / uploads / files exist
    assert (temp_dir / "uploads").exists(), "Upload directory does not exist"

    assert count_files_recursively(temp_dir / "uploads") == 2

    # Check the number of crops
    kv = ImageStorageFactory.create_storage("hdf5", temp_dir / "kvstore.db")
    assert len(list(kv.iter_keys())) == 32
