#!/usr/bin/env bash
# ci/container_preflight.sh
#
# Fast host/environment checks before running container build/smoke scripts.
#
# Usage:
#   bash ci/container_preflight.sh build
#   bash ci/container_preflight.sh smoke-quick
#   bash ci/container_preflight.sh smoke-api
#   bash ci/container_preflight.sh integration
#   bash ci/container_preflight.sh all

set -euo pipefail

MODE="${1:-all}"

require_cmd() {
    local cmd="$1"
    if ! command -v "$cmd" >/dev/null 2>&1; then
        echo "ERROR: missing required command: ${cmd}" >&2
        exit 1
    fi
}

check_common() {
    require_cmd docker
    require_cmd git
    require_cmd bash
}

check_build() {
    : "${REGISTRY:?REGISTRY is required for build mode}"
    : "${IMAGE_TAG:?IMAGE_TAG is required for build mode}"
}

check_smoke_quick() {
    : "${GPU_BASE_IMAGE:?GPU_BASE_IMAGE is required for smoke mode}"
    : "${GPU_SERVER_IMAGE:?GPU_SERVER_IMAGE is required for smoke mode}"
    : "${UI_IMAGE:?UI_IMAGE is required for smoke mode}"
}

check_smoke_api() {
    check_smoke_quick
    : "${HF_TOKEN:?HF_TOKEN is required for smoke-api mode}"
    require_cmd curl
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "ERROR: nvidia-smi is required for smoke-api mode" >&2
        exit 1
    fi

    MODELS_PATH="${MODELS_PATH:-./models}"
    DATA_PATH="${DATA_PATH:-./docs/pdf}"
    APPS_PATH="${APPS_PATH:-./apps}"

    mkdir -p "${MODELS_PATH}" "${APPS_PATH}" "${APPS_PATH}/.cache"

    if [ ! -d "${DATA_PATH}" ]; then
        echo "ERROR: DATA_PATH does not exist: ${DATA_PATH}" >&2
        exit 1
    fi
}

check_integration() {
    : "${GPU_BASE_IMAGE:?GPU_BASE_IMAGE is required for integration mode}"
    require_cmd nvidia-smi

    MODELS_PATH="${MODELS_PATH:-./models}"
    TESTS_PATH="${TESTS_PATH:-./tests}"

    if [ ! -d "${MODELS_PATH}" ]; then
        echo "ERROR: MODELS_PATH does not exist: ${MODELS_PATH}" >&2
        exit 1
    fi
    if [ ! -d "${TESTS_PATH}" ]; then
        echo "ERROR: TESTS_PATH does not exist: ${TESTS_PATH}" >&2
        exit 1
    fi
}

check_common

case "${MODE}" in
    build)
        check_build
        ;;
    smoke-quick)
        check_smoke_quick
        ;;
    smoke-api)
        check_smoke_api
        ;;
    integration)
        check_integration
        ;;
    all)
        check_build
        check_smoke_quick
        ;;
    *)
        echo "ERROR: unsupported mode '${MODE}'. Use build|smoke-quick|smoke-api|integration|all." >&2
        exit 1
        ;;
esac

echo "preflight-ok (${MODE})"
