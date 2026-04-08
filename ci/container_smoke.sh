#!/usr/bin/env bash
# ci/container_smoke.sh
#
# Smoke tests for Colette container images.
#
# Modes:
# - quick: verifies key imports and expected entrypoints/files without starting services
# - api: starts the GPU server container and checks /v1/info health endpoint
#
# Required env vars:
#   GPU_BASE_IMAGE
#   GPU_SERVER_IMAGE
#   UI_IMAGE
#
# Optional env vars:
#   SMOKE_MODE=quick|api (default: quick)
#   SMOKE_TIMEOUT_SEC (default: 120)
#   HF_TOKEN (required only for api mode)
#   MODELS_PATH (default: ./models)
#   DATA_PATH (default: ./docs/pdf)
#   APPS_PATH (default: ./apps)
#   CONTAINER_UID (default: current user id)
#   CONTAINER_GID (default: current group id)

set -euo pipefail

GPU_BASE_IMAGE="${GPU_BASE_IMAGE:-}"
GPU_SERVER_IMAGE="${GPU_SERVER_IMAGE:-}"
UI_IMAGE="${UI_IMAGE:-}"
SMOKE_MODE="${SMOKE_MODE:-quick}"
SMOKE_TIMEOUT_SEC="${SMOKE_TIMEOUT_SEC:-120}"
HF_TOKEN="${HF_TOKEN:-}"
MODELS_PATH="${MODELS_PATH:-./models}"
DATA_PATH="${DATA_PATH:-./docs/pdf}"
APPS_PATH="${APPS_PATH:-./apps}"
API_CID=""

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

cleanup_api_container() {
    if [ -n "${API_CID:-}" ]; then
        docker rm -f "${API_CID}" >/dev/null 2>&1 || true
        API_CID=""
    fi
}

trap cleanup_api_container EXIT

if [ -z "${GPU_BASE_IMAGE}" ] || [ -z "${GPU_SERVER_IMAGE}" ] || [ -z "${UI_IMAGE}" ]; then
    echo "ERROR: GPU_BASE_IMAGE, GPU_SERVER_IMAGE and UI_IMAGE must be set." >&2
    exit 1
fi

if [ "${SMOKE_MODE}" != "quick" ] && [ "${SMOKE_MODE}" != "api" ]; then
    echo "ERROR: unsupported SMOKE_MODE='${SMOKE_MODE}'. Use quick or api." >&2
    exit 1
fi

if [ "${SMOKE_MODE}" = "api" ]; then
    bash "$(dirname "$0")/container_preflight.sh" smoke-api
else
    bash "$(dirname "$0")/container_preflight.sh" smoke-quick
fi

echo "=== Container smoke tests ==="
echo "mode          : ${SMOKE_MODE}"
echo "gpu base image: ${GPU_BASE_IMAGE}"
echo "gpu srv image : ${GPU_SERVER_IMAGE}"
echo "ui image      : ${UI_IMAGE}"

quick_checks() {
    echo "[1/3] GPU base import check"
    docker run --rm "${DOCKER_USER_ARGS[@]}" "${GPU_BASE_IMAGE}" python3 -c "import torch, fastapi; print('torch=' + torch.__version__)"

    echo "[2/3] GPU server image structure/import check"
    docker run --rm "${DOCKER_USER_ARGS[@]}" --entrypoint bash "${GPU_SERVER_IMAGE}" -lc "test -x /app/server/run.sh && python3 -c 'import colette; print(\"colette-ok\")'"

    echo "[3/3] UI image import check"
    docker run --rm "${DOCKER_USER_ARGS[@]}" --entrypoint python3 "${UI_IMAGE}" -c "import gradio, nltk; print('ui-imports-ok')"
}

api_checks() {
    if [ -z "${HF_TOKEN}" ]; then
        echo "ERROR: HF_TOKEN is required for SMOKE_MODE=api." >&2
        exit 1
    fi

    mkdir -p "${MODELS_PATH}" "${APPS_PATH}" "${APPS_PATH}/.cache"

    local api_port
    API_CID="$(docker run -d \
        "${DOCKER_USER_ARGS[@]}" \
        --gpus all \
        -e HF_TOKEN="${HF_TOKEN}" \
        -e HOME=/tmp \
        -e PYTHONPATH=/app/src \
        -v "${MODELS_PATH}:/models" \
        -v "${DATA_PATH}:/data" \
        -v "${APPS_PATH}:/rag" \
        -v "${APPS_PATH}/.cache:/cache" \
        -p 127.0.0.1::1873 \
        "${GPU_SERVER_IMAGE}")"

    api_port="$(docker port "${API_CID}" 1873/tcp | tail -n 1 | awk -F: '{print $NF}')"
    if [ -z "${api_port}" ]; then
        echo "ERROR: could not resolve mapped API port for ${API_CID}" >&2
        exit 1
    fi

    echo "API mapped port: ${api_port}"
    echo "Waiting for /v1/info health endpoint..."

    local elapsed=0
    while [ "${elapsed}" -lt "${SMOKE_TIMEOUT_SEC}" ]; do
        if [ "$(docker inspect -f '{{.State.Running}}' "${API_CID}" 2>/dev/null || echo false)" != "true" ]; then
            echo "ERROR: API container exited before health endpoint became ready." >&2
            docker logs --tail=200 "${API_CID}" || true
            exit 1
        fi

        if curl -fsS "http://127.0.0.1:${api_port}/v1/info" >/dev/null; then
            echo "api-health-ok"
            return 0
        fi
        sleep 3
        elapsed=$((elapsed + 3))
    done

    echo "ERROR: API healthcheck timed out after ${SMOKE_TIMEOUT_SEC}s" >&2
    docker logs --tail=200 "${API_CID}" || true
    exit 1
}

quick_checks

if [ "${SMOKE_MODE}" = "api" ]; then
    api_checks
fi

echo "=== Smoke tests passed ==="
