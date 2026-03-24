PYTHON ?= $(if $(wildcard venv_colette/bin/python),venv_colette/bin/python,python3)
PYTEST ?= $(PYTHON) -m pytest
RUFF ?= $(PYTHON) -m ruff
SMOKE_TESTS ?= tests/test_base_ci.py::test_info tests/test_embedding_loader.py tests/test_embedding_integration.py tests/test_services_smoke.py tests/test_http_openwebui_smoke.py tests/test_cli_smoke.py tests/test_jsonapi_helpers_smoke.py tests/test_kvstore_smoke.py tests/test_logger_smoke.py tests/test_jsonapi_service_smoke.py tests/test_core_services_smoke.py
COV_MIN ?= 35
COV_TARGET ?= src/colette
CI_ARTIFACTS_DIR ?= .ci-artifacts
# GPU to use for e2e / integration tests (override with: make test-e2e GPU_ID=0)
GPU_ID ?= 1
GPU_ENV = CUDA_VISIBLE_DEVICES=$(GPU_ID) COLETTE_GPU_ID=$(GPU_ID)
JUNIT_SMOKE ?= $(CI_ARTIFACTS_DIR)/junit-smoke.xml
JUNIT_COVERAGE ?= $(CI_ARTIFACTS_DIR)/junit-coverage.xml
JUNIT_INTEGRATION ?= $(CI_ARTIFACTS_DIR)/junit-integration.xml
JUNIT_PIPELINE_INTEGRATION ?= $(CI_ARTIFACTS_DIR)/junit-pipeline-integration.xml
COVERAGE_XML ?= $(CI_ARTIFACTS_DIR)/coverage.xml

.PHONY: style lint format-check lint-check test-smoke test-coverage test-integration test-integration-pipeline test-e2e ci-smoke ci-coverage ci-integration ci-pipeline-integration

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

test-integration:
	COLETTE_RUN_INTEGRATION=1 $(PYTEST) tests/ -m "integration and not e2e" -v --tb=short

test-integration-pipeline:
	$(GPU_ENV) COLETTE_RUN_INTEGRATION=1 $(PYTEST) tests/test_pipeline_python_api_integration.py -m integration -v --tb=short

test-e2e:
	$(GPU_ENV) COLETTE_RUN_INTEGRATION=1 $(PYTEST) tests/ -m e2e -v --tb=short

ci-smoke:
	mkdir -p $(CI_ARTIFACTS_DIR)
	$(PYTEST) $(SMOKE_TESTS) -m smoke -q --junitxml=$(JUNIT_SMOKE)

ci-coverage:
	mkdir -p $(CI_ARTIFACTS_DIR)
	$(PYTEST) $(SMOKE_TESTS) -m smoke --cov=$(COV_TARGET) --cov-report=term-missing --cov-report=xml:$(COVERAGE_XML) --cov-fail-under=$(COV_MIN) -q --junitxml=$(JUNIT_COVERAGE)

ci-integration:
	mkdir -p $(CI_ARTIFACTS_DIR)
	$(GPU_ENV) COLETTE_RUN_INTEGRATION=1 $(PYTEST) tests/ -m "integration and not e2e" -v --tb=short --junitxml=$(JUNIT_INTEGRATION)

ci-pipeline-integration:
	mkdir -p $(CI_ARTIFACTS_DIR)
	$(GPU_ENV) COLETTE_RUN_INTEGRATION=1 $(PYTEST) tests/test_pipeline_python_api_integration.py -m integration -v --tb=short --junitxml=$(JUNIT_PIPELINE_INTEGRATION)

ci-e2e:
	mkdir -p $(CI_ARTIFACTS_DIR)
	$(GPU_ENV) COLETTE_RUN_INTEGRATION=1 $(PYTEST) tests/ -m e2e -v --tb=short --junitxml=$(CI_ARTIFACTS_DIR)/junit-e2e.xml