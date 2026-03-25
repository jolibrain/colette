# Jenkins Pipeline

This page documents the local Jenkins CI flow for Colette.

## Pipeline Overview

The pipeline is defined in `Jenkinsfile` and uses `agent none` with stage-level node selection.

Main stages:

1. `Setup`: creates or reuses the smoke virtualenv cache.
2. `Setup Full` (optional): creates or reuses the full virtualenv cache for deeper lanes.
3. `PR Smoke`: always runs the fast smoke gate.
4. `Integration Stable (Optional)`: runs when `RUN_INTEGRATION_STABLE=true`.
5. `Nightly GPU Matrix` (optional): runs `Integration`, `Pipeline Integration`, and `E2E` in parallel when `RUN_NIGHTLY_MATRIX=true`.

## Node And GPU Targeting

All stages run on:

```text
linux && gpu && n5
```

This keeps CI pinned to the intended machine class (for example `neptune05t` carrying the `n5` label).

GPU execution uses a Jenkins lock keyed by node name:

```text
${NODE_NAME}-gpu
```

The pipeline exports `CUDA_VISIBLE_DEVICES` from the lock metadata, with fallback to the `GPU_ID` parameter.

## Virtualenv Cache Model

Caches persist one level above the Jenkins workspace:

- `../venv_colette_smoke_cache`
- `../venv_colette_full_cache`

Inside the workspace, `venv_colette` is a symlink to the selected cache.

`cleanWs()` only cleans the workspace and does not remove these caches.

## Deterministic Smoke Environment

Smoke setup is handled by `ci/setup_venv.sh smoke` and `ci/requirements-smoke.txt`.

Behavior:

1. Builds cache when missing or broken.
2. Computes SHA256 of `ci/requirements-smoke.txt`.
3. Rebuilds the smoke cache when the requirements hash changes.
4. Runs `ci/verify_smoke_imports.py` to import all smoke test files before pytest starts.
5. Keeps runtime dependencies deterministic instead of ad-hoc missing-module installs.

The smoke requirements file includes key import-time providers used by smoke tests
(for example `python-doctr` for `doctr.models` imports).

## Deterministic Full Environment

Full setup is handled by `ci/setup_venv.sh full` and `ci/requirements-full.txt`.

Behavior:

1. Builds cache when missing or broken.
2. Computes SHA256 of `ci/requirements-full.txt`.
3. Rebuilds the full cache when the requirements hash changes.
4. Rebuilds the full cache if core tooling such as `pytest` is missing.
5. Installs the editable package with `--no-deps` after the pinned full requirements are installed.
6. Attempts `flash-attn==2.5.6` separately with `--no-build-isolation`, but does not fail the pipeline if that optional acceleration package cannot be built.

This avoids reusing stale integration caches and prevents `flash-attn` from failing early through pip build isolation during the main full-profile install.

## Jenkins Parameters

- `GPU_ID` (string): fallback GPU index for `CUDA_VISIBLE_DEVICES`.
- `RUN_INTEGRATION_STABLE` (bool): enables the protected integration-stable lane.
- `RUN_NIGHTLY_MATRIX` (bool): enables full nightly-style GPU matrix.

## Credentials

Integration lanes require a Jenkins secret text credential:

- ID: `hf`
- Env var in pipeline: `HF_TOKEN`

## Make Targets Used By CI

- `make ci-smoke`
- `make ci-integration-stable`
- `make ci-integration`
- `make ci-pipeline-integration`
- `make ci-e2e`

## What Is Tested In Each Lane

The pipeline intentionally uses layers of confidence, from fast deterministic checks to broader GPU-heavy validation.

### PR Smoke (`make ci-smoke`)

Scope:

- Runs only the curated smoke test list (`SMOKE_TESTS` in `Makefile`), with `-m smoke`.
- Current smoke list includes:
	- `tests/test_base_ci.py::test_info`
	- `tests/test_embedding_loader.py`
	- `tests/test_embedding_integration.py`
	- `tests/test_services_smoke.py`
	- `tests/test_http_openwebui_smoke.py`
	- `tests/test_cli_smoke.py`
	- `tests/test_jsonapi_helpers_smoke.py`
	- `tests/test_kvstore_smoke.py`
	- `tests/test_logger_smoke.py`
	- `tests/test_jsonapi_service_smoke.py`
	- `tests/test_core_services_smoke.py`

What this validates:

- Basic API/service wiring and core helper paths.
- CLI and JSON API smoke behavior.
- Import-time stability for the smoke dependency set.

What this does not validate:

- Full integration surface.
- End-to-end (`e2e`) marker tests.
- Pipeline contract test for the Python API integration flow.

### Integration Stable (`make ci-integration-stable`)

Scope:

- Runs a pinned subset (`INTEGRATION_STABLE_TESTS`) with `-m integration`:
	- `tests/test_upload.py`
	- `tests/test_multiple_creation.py`
	- `tests/test_logging_payload.py`
	- `tests/test_base_ci.py::test_llamacpp_hf`

What this validates:

- Stable, high-signal integration paths used as a protected optional gate.
- Real integration behavior with `COLETTE_RUN_INTEGRATION=1` and GPU environment variables.

### Nightly Matrix: Integration (`make ci-integration`)

Scope:

- Runs all tests under `tests/` matching marker expression:
	- `-m "integration and not e2e"`

What this validates:

- Broad integration behavior beyond the stable subset.
- Regressions in integration-marked tests that are too heavy for PR default lanes.

### Nightly Matrix: Pipeline Integration (`make ci-pipeline-integration`)

Scope:

- Runs:
	- `tests/test_pipeline_python_api_integration.py -m integration`

What this validates:

- The Python API pipeline contract end-to-end at the integration level.
- CI artifact production for this dedicated contract path.

### Nightly Matrix: E2E (`make ci-e2e`)

Scope:

- Runs all tests under `tests/` marked:
	- `-m e2e`

What this validates:

- Deepest runtime workflows and end-to-end behavior.
- Scenarios expected to be the most environment-sensitive and longest-running.

### JUnit Artifacts

Each lane writes a dedicated JUnit XML file under `.ci-artifacts/`:

- `junit-smoke.xml`
- `junit-integration-stable.xml`
- `junit-integration.xml`
- `junit-pipeline-integration.xml`
- `junit-e2e.xml`

## Troubleshooting

If smoke lane fails during imports:

1. Confirm `ci/requirements-smoke.txt` includes the needed package.
2. Check preflight output from `ci/verify_smoke_imports.py` for the full missing-module list.
3. Re-run build; the hash change forces a smoke cache rebuild.

If CUDA/torch installation fails during setup:

1. Verify `nvcc` or `nvidia-smi` is available on the Jenkins node.
2. Confirm node labels still match a GPU-capable runner.

If `Integration Stable` or nightly lanes fail with missing tooling such as `pytest`:

1. Confirm `Setup Full` ran on the same build.
2. Check whether `ci/requirements-full.txt` changed; the full cache should rebuild automatically when its hash changes.
3. If the cache predates the deterministic full setup, rerun once on the latest branch head or delete `../venv_colette_full_cache` to force a clean rebuild.

If `flash-attn` fails during full setup:

1. Treat it as non-blocking unless a test explicitly requires it.
2. The pipeline installs it separately on purpose so the main integration environment can still come up without it.

If a cache is corrupted:

1. Delete the corresponding cache directory above workspace.
2. Re-run pipeline to trigger clean recreation.
