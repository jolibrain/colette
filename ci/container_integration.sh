#!/usr/bin/env bash
# ci/container_integration.sh
#
# Run selected integration tests inside the GPU base container.
#
# Required env vars:
#   GPU_BASE_IMAGE
#
# Optional env vars:
#   TEST_FILES (default: two pipeline integration files)
#   MODELS_PATH (default: ./models)
#   TESTS_PATH (default: ./tests)
#   COLETTE_PIPELINE_HF_MODEL (default: Qwen/Qwen2-VL-2B-Instruct)
#   HF_TOKEN (passed through if set)
#   PYTEST_ARGS (default: -q)
#   CONTAINER_UID (default: current user id)
#   CONTAINER_GID (default: current group id)

set -euo pipefail

GPU_BASE_IMAGE="${GPU_BASE_IMAGE:-}"
TEST_FILES="${TEST_FILES:-/app/tests/test_pipeline_python_api_integration.py /app/tests/test_pipeline_tantivy_integration.py}"
MODELS_PATH="${MODELS_PATH:-./models}"
TESTS_PATH="${TESTS_PATH:-./tests}"
COLETTE_PIPELINE_HF_MODEL="${COLETTE_PIPELINE_HF_MODEL:-Qwen/Qwen2-VL-2B-Instruct}"
PYTEST_ARGS="${PYTEST_ARGS:--q}"

if command -v id >/dev/null 2>&1; then
    CONTAINER_UID="${CONTAINER_UID:-$(id -u)}"
    CONTAINER_GID="${CONTAINER_GID:-$(id -g)}"
else
    CONTAINER_UID="${CONTAINER_UID:-}"
    CONTAINER_GID="${CONTAINER_GID:-}"
fi

DOCKER_USER_ARGS=()
if [ -n "${CONTAINER_UID}" ] && [ -n "${CONTAINER_GID}" ]; then
    DOCKER_USER_ARGS=(--user "${CONTAINER_UID}:${CONTAINER_GID}")
fi

if [ -z "${GPU_BASE_IMAGE}" ]; then
    echo "ERROR: GPU_BASE_IMAGE must be set." >&2
    exit 1
fi

bash "$(dirname "$0")/container_preflight.sh" integration

echo "=== Container integration tests ==="
echo "image      : ${GPU_BASE_IMAGE}"
echo "models path: ${MODELS_PATH}"
echo "tests path : ${TESTS_PATH}"
echo "tests      : ${TEST_FILES}"

DOCKER_ENV_ARGS=(
    -e "COLETTE_RUN_INTEGRATION=1"
    -e "COLETTE_PIPELINE_HF_MODEL=${COLETTE_PIPELINE_HF_MODEL}"
)

if [ -n "${HF_TOKEN:-}" ]; then
    DOCKER_ENV_ARGS+=( -e "HF_TOKEN=${HF_TOKEN}" )
fi

# shellcheck disable=SC2086
# TEST_FILES and PYTEST_ARGS are intentionally split into multiple args.
docker run --rm --gpus all \
    "${DOCKER_USER_ARGS[@]}" \
    "${DOCKER_ENV_ARGS[@]}" \
    -e "HOME=/tmp" \
    -v "${MODELS_PATH}:/app/models" \
    -v "${TESTS_PATH}:/app/tests" \
    "${GPU_BASE_IMAGE}" \
    python3 -m pytest ${PYTEST_ARGS} ${TEST_FILES}

echo "=== Integration tests passed ==="
