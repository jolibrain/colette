import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


pytestmark = [pytest.mark.integration, pytest.mark.e2e, pytest.mark.evaluation]

models_repo = os.getenv("MODELS_REPO", "models")
test_configs = [
    (
        "vrag_default.json",
        [
            dict(
                id=1,
                question="Quel est le titre du document?",
                answer="Le titre du document est 'Moteur Vulcain', 1996, Inv. 40959",
                short_answer=["Vulcain", "Ariane"],
                references=[dict(file="fusee_ariane_5", pages=["1", "2"])],
                lang="fr",
            )
        ],
    ),
]


@pytest.mark.parametrize("config_file,qa_data", test_configs)
def test_vrag_config_evaluation_contract(client, repo_dir, run_evaluation, config_file, qa_data):
    config_path = Path(__file__).resolve().parents[2] / "tools" / "config" / config_file
    with config_path.open(encoding="utf-8") as handle:
        json_config = json.load(handle)

    json_config["app"]["models_repository"] = str(Path(models_repo).resolve())
    json_config["app"]["repository"] = str(repo_dir.resolve())
    json_config["parameters"]["input"]["data"] = ["tests/data_pdf1"]

    app_name = f"evaluation-config-{repo_dir.parent.name}"
    created = False

    try:
        response = client.put(f"/v1/app/{app_name}", data={"ad": json.dumps(json_config)})
        assert response.status_code == 200
        assert response.json()["service_name"] == app_name
        created = True

        response = client.get("/v1/info")
        assert app_name in response.json()["info"]["services"]

        with TemporaryDirectory() as qa_dir:
            qa_file = Path(qa_dir) / "qa.json"
            with qa_file.open("w", encoding="utf-8") as handle:
                json.dump(qa_data, handle)

            class Args:
                app_dir = str(repo_dir)
                qa = str(qa_file)
                debug = False

            _, results_df, retriever_df, _ = run_evaluation(Args())

        assert results_df.shape == (1, 10), results_df.shape
        for index, qa_entry in enumerate(qa_data):
            row = results_df.row(index)
            assert any(keyword.lower() in row[2].lower() for keyword in qa_entry["short_answer"]), row
            assert any(ref["file"] in row[8] for ref in qa_entry["references"]), row[8]

        assert retriever_df.shape[0] == len(qa_data), retriever_df.shape
    finally:
        if created:
            response = client.delete(f"/v1/app/{app_name}")
            assert response.status_code == 200