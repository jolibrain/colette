#!/usr/bin/env bash
# ci/build_container_images.sh
#
# Build Colette container images in a reusable way for local runs and Jenkins.
#
# Required env vars:
#   REGISTRY
#   IMAGE_TAG
#
# Optional env vars:
#   TARGET=all|gpu_base|gpu_server|ui (default: all)
#   NO_CACHE=true|false (default: false)
#   IMAGE_GPU_BASE=colette_gpu
#   IMAGE_GPU_SERVER=colette_gpu_server
#   IMAGE_UI=colette_ui

set -euo pipefail

bash "$(dirname "$0")/container_preflight.sh" build

REGISTRY="${REGISTRY:-}"
IMAGE_TAG="${IMAGE_TAG:-}"
TARGET="${TARGET:-all}"
NO_CACHE="${NO_CACHE:-false}"
IMAGE_GPU_BASE="${IMAGE_GPU_BASE:-colette_gpu}"
IMAGE_GPU_SERVER="${IMAGE_GPU_SERVER:-colette_gpu_server}"
IMAGE_UI="${IMAGE_UI:-colette_ui}"

if [ -z "${REGISTRY}" ] || [ -z "${IMAGE_TAG}" ]; then
    echo "ERROR: REGISTRY and IMAGE_TAG must be set." >&2
    exit 1
fi

case "${TARGET}" in
    all|gpu_base|gpu_server|ui)
        ;;
    *)
        echo "ERROR: unsupported TARGET='${TARGET}'. Use all|gpu_base|gpu_server|ui." >&2
        exit 1
        ;;
esac

NO_CACHE_ARG=""
if [ "${NO_CACHE}" = "true" ]; then
    NO_CACHE_ARG="--no-cache"
fi

build_gpu_base() {
    docker build ${NO_CACHE_ARG} \
        -f docker/gpu_build.Dockerfile \
        -t "${REGISTRY}/${IMAGE_GPU_BASE}:${IMAGE_TAG}" \
        .
}

build_gpu_server() {
    docker build ${NO_CACHE_ARG} \
        -f docker/gpu_jb_server.Dockerfile \
        --build-arg GTAG="${IMAGE_TAG}" \
        -t "${REGISTRY}/${IMAGE_GPU_SERVER}:${IMAGE_TAG}" \
        .
}

build_ui() {
    docker build ${NO_CACHE_ARG} \
        -f docker/ui.Dockerfile \
        -t "${REGISTRY}/${IMAGE_UI}:${IMAGE_TAG}" \
        .
}

echo "=== Container build ==="
echo "target   : ${TARGET}"
echo "registry : ${REGISTRY}"
echo "tag      : ${IMAGE_TAG}"
echo "no_cache : ${NO_CACHE}"

if [ "${TARGET}" = "gpu_base" ] || [ "${TARGET}" = "all" ]; then
    echo "[build] gpu_base"
    build_gpu_base
fi

if [ "${TARGET}" = "gpu_server" ] || [ "${TARGET}" = "all" ]; then
    echo "[build] gpu_server"
    build_gpu_server
fi

if [ "${TARGET}" = "ui" ] || [ "${TARGET}" = "all" ]; then
    echo "[build] ui"
    build_ui
fi

echo "=== Build completed ==="
