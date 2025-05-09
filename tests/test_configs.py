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
from evaluation import run_evaluation  # noqa

models_repo = os.getenv("MODELS_REPO", 'models')

client = TestClient(app)


@pytest.fixture
def temp_dir(request):
    """Fixture to create and clean up a temporary directory."""
    temp_dir = Path(request.node.get_closest_marker("repository_path").args[0])
    temp_dir.mkdir(parents=True, exist_ok=True)
    yield temp_dir
    shutil.rmtree(temp_dir)


# Define test configurations (JSON config file path + QA list)
test_configs = [
    # (
    #     "trag_default.json",
    #     [
    #         dict(
    #             id=1,
    #             question="De combien d'étages est composée la fusée Ariane 5?",
    #             answer="",
    #             short_answer=["2", "deux", "10"],
    #             references=[dict(file="fusee_ariane_5", pages=["0"])],
    #             lang="fr"
    #         )
    #     ]
    # ),
    (
        "vrag_default.json",
        [
            dict(
                id=1,
                question="Quel est le titre du document?",
                answer="Le titre du document est 'Moteur Vulcain', 1996, Inv. 40959",
                short_answer=["Vulcain", "Ariane"],
                references=[dict(file="fusee_ariane_5", pages=["1", "2"])],
                lang="fr"
            )
        ]
    ),
]


@pytest.mark.parametrize("config_file, qa_data", test_configs)
@pytest.mark.repository_path("test_config")
def test_trag(temp_dir, config_file, qa_data):
    config = Path(__file__).parent.parent / "tools" / "config" / config_file
    json_config = json.load(open(config))
    json_config["app"]["models_repository"] = str(Path(models_repo).absolute())
    json_config["app"]["repository"] = str(temp_dir.absolute())
    json_config["parameters"]["input"]["data"] = ["tests/data_pdf1"]

    print(json.dumps(json_config, indent=2))

    app_name = temp_dir.name

    try:
        response = client.put(f"/v1/app/{app_name}", data={"ad": json.dumps(json_config)})
        pretty_print_response(response.json())
        assert response.status_code == 200
        assert response.json()["service_name"] == f"{app_name}"

        response = client.get("/v1/info")
        pretty_print_response(response.json())
        assert f"{app_name}" in response.json()["info"]["services"]

        with TemporaryDirectory() as temp_dir_2:
            # Create the question/answer file
            file_name = Path(temp_dir_2) / "qa.json"
            json.dump(qa_data, open(file_name, "w"))
            assert file_name.exists()

            class Args:
                app_dir = str(temp_dir)
                qa = str(file_name)
                debug = False

            # Run the evaluation
            _, results_df, retriever_df, _ = run_evaluation(Args())

            # Check results
            assert results_df.shape == (1, 10), results_df.shape
            for i, qa_entry in enumerate(qa_data):
                row = results_df.row(i)
                assert any(keyword.lower() in row[2].lower() for keyword in qa_entry["short_answer"]), f"ROW: {row}"
                assert any(ref["file"] in row[8] for ref in qa_entry["references"]), row[8]

            assert retriever_df.shape[0] == len(qa_data), retriever_df.shape

    finally:
        # Delete the service
        response = client.delete(f"/v1/app/{app_name}")
        assert response.status_code == 200
