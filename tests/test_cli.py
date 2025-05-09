import shutil
from pathlib import Path

import pytest
from typer.testing import CliRunner

from colette.colette_cli import cli  # noqa

# Define paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_img2"
TOOLS_DIR = BASE_DIR.parent / "src/colette"
CONFIG_FILE = TOOLS_DIR / "config" / "vrag_default.json"
INDEX_FILE = TOOLS_DIR / "config" / "vrag_default_index.json"
APP_DIR = BASE_DIR / "app_colette"
MODELS_DIR = APP_DIR / "models"


@pytest.fixture(scope="module", autouse=True)
def setup_and_cleanup():
    """Ensure required files exist before tests, then clean up APP_DIR after tests."""
    assert DATA_DIR.exists(), f"Test data directory {DATA_DIR} is missing."
    assert CONFIG_FILE.exists(), f"Config file {CONFIG_FILE} is missing."

    # Ensure APP_DIR exists before running tests
    APP_DIR.mkdir(parents=True, exist_ok=True)

    yield  # Run tests

    # Cleanup: Remove APP_DIR after tests, regardless of success/failure
    try:
        if APP_DIR.exists():
            shutil.rmtree(APP_DIR)
            print(f"Deleted test directory: {APP_DIR}")
    except Exception as e:
        print(f"Failed to delete {APP_DIR}: {e}")


def test_index(setup_and_cleanup):
    """Test the indexing function via CLI."""
    runner = CliRunner()

    result = runner.invoke(
        cli,
        [
            "index",
            "--app-dir",
            str(APP_DIR),
            "--data-dir",
            str(DATA_DIR),
            "--config-file",
            str(CONFIG_FILE),
            "--index-file",
            str(INDEX_FILE),
        ],
    )

    assert result.exit_code == 0, f"Indexing failed: {result.output}"
    assert "Indexing completed" in result.output, "Indexing did not complete successfully"


def test_chat(setup_and_cleanup):
    """Test the chat function via CLI."""
    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "chat",
            "--app-dir",
            str(APP_DIR),
            "--msg",
            "What is the budget allocated to Space Transportation?",
        ],
    )

    assert result.exit_code == 0, f"Chat failed: {result.output}"
    assert "Chat completed" in result.output, "Chat did not complete successfully"
