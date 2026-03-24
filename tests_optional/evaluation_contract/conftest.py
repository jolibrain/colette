import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[2]
for extra_path in (ROOT / "src", ROOT / "tools"):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)


@pytest.fixture
def client(run_evaluation):
    from fastapi.testclient import TestClient
    from colette.httpjsonapi import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def repo_dir(tmp_path):
    repo_path = tmp_path / "repo"
    repo_path.mkdir()
    yield repo_path


@pytest.fixture(scope="session")
def run_evaluation():
    evaluation = pytest.importorskip(
        "evaluation",
        reason="install optional evaluation dependencies and run `make test-evaluation`",
    )
    return evaluation.run_evaluation