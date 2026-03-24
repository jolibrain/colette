import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest


pytestmark = [
    pytest.mark.integration,
    pytest.mark.e2e,
    pytest.mark.evaluation,
    pytest.mark.gpu,
]

models_repo = os.getenv("MODELS_REPO", "models")


def test_hf_single_image_evaluation_contract(client, repo_dir, run_evaluation):
    app_name = f"evaluation-hf-single-{repo_dir.parent.name}"
    payload = {
        "app": {
            "repository": str(repo_dir),
            "models_repository": models_repo,
            "verbose": "info",
        },
        "parameters": {
            "input": {
                "lib": "hf",
                "preprocessing": {
                    "files": ["all"],
                    "save_output": True,
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
                        "layout_detection": False,
                    },
                },
                "template": {
                    "template_prompt": "Tu es un assistant de réponse à des questions. Question: {question} Réponse: ",
                    "template_prompt_variables": ["question"],
                },
                "data": ["tests/data_img1"],
            },
            "llm": {
                "source": "Qwen/Qwen2-VL-2B-Instruct",
                "gpu_ids": [0],
                "image_width": 640,
                "image_height": 960,
                "inference": {
                    "lib": "huggingface",
                },
            },
        },
    }
    qa_data = [
        dict(
            id=1,
            question="Quels sont les auteurs?",
            answer="Les auteurs sont Mme Cécile RILHAC et M. Aurélien LOPEZ-LIGUORI.",
            short_answer=["RILHAC", "LOPEZ-LIGUORI"],
            references=[dict(file="RINFANR5L16B2040.jpg-001", pages=["0"])],
            lang="fr",
        )
    ]
    created = False

    try:
        response = client.put(f"/v1/app/{app_name}", data={"ad": json.dumps(payload)})
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
        row = results_df.row(0)
        assert "RILHAC" in row[2], row[2]
        assert "RINFANR5L16B2040.jpg-001" in row[8], row[8]
        assert retriever_df.shape == (1, 4), retriever_df.shape
    finally:
        if created:
            response = client.delete(f"/v1/app/{app_name}")
            assert response.status_code == 200