PYTHON ?= $(if $(wildcard venv_colette/bin/python),venv_colette/bin/python,python3)
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
SMOKE_TESTS ?= tests/test_base_ci.py::test_info tests/test_embedding_loader.py tests/test_embedding_integration.py

.PHONY: style lint format-check lint-check test-smoke

style:
	$(RUFF) format .

lint:
	$(RUFF) check --fix .

format-check:
	$(RUFF) format --check .

lint-check:
	$(RUFF) check .

test-smoke:
	$(PYTEST) $(SMOKE_TESTS) -m smoke -q