PYTHON ?= $(if $(wildcard venv_colette/bin/python),venv_colette/bin/python,python3)
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
SMOKE_TESTS ?= tests/test_base_ci.py::test_info tests/test_embedding_loader.py tests/test_embedding_integration.py tests/test_services_smoke.py tests/test_http_openwebui_smoke.py tests/test_cli_smoke.py tests/test_jsonapi_helpers_smoke.py
COV_MIN ?= 30
COV_TARGET ?= src/colette

.PHONY: style lint format-check lint-check test-smoke test-coverage

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

test-coverage:
	$(PYTEST) $(SMOKE_TESTS) -m smoke --cov=$(COV_TARGET) --cov-report=term-missing --cov-report=xml:coverage.xml --cov-fail-under=$(COV_MIN) -q