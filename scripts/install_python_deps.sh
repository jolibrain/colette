#!/usr/bin/env bash
# scripts/install_python_deps.sh
#
# Shared dependency installer for local venv bootstrap and container builds.

set -euo pipefail

if command -v python >/dev/null 2>&1; then
    PYTHON_BIN="python"
elif command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
else
    echo "ERROR: neither python nor python3 is available." >&2
    exit 1
fi

COLETTE_CUDA_SHORT="${COLETTE_CUDA_SHORT:-}"
TORCH_VERSION="${COLETTE_TORCH_VERSION:-2.8.0}"
TORCHVISION_VERSION="${COLETTE_TORCHVISION_VERSION:-0.23.0}"
TORCHAUDIO_VERSION="${COLETTE_TORCHAUDIO_VERSION:-2.8.0}"
FLASH_ATTN_VERSION="${COLETTE_FLASH_ATTN_VERSION:-2.8.3}"
INSTALL_EDITABLE="${COLETTE_INSTALL_EDITABLE:-1}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-${HOME:-/root}/.cache/pip}"

if [ -z "${COLETTE_CUDA_SHORT}" ]; then
    if command -v nvcc >/dev/null 2>&1; then
        cuda_version=$(nvcc --version | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p' | head -n1)
        COLETTE_CUDA_SHORT="${cuda_version/.}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
        cuda_version=$(nvidia-smi | sed -n 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/p' | head -n1)
        COLETTE_CUDA_SHORT="${cuda_version/.}"
    else
        echo "ERROR: COLETTE_CUDA_SHORT is not set and CUDA version could not be auto-detected." >&2
        exit 1
    fi
fi

if ! [[ "${COLETTE_CUDA_SHORT}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid COLETTE_CUDA_SHORT='${COLETTE_CUDA_SHORT}'" >&2
    exit 1
fi

TORCH_INDEX_URL="https://download.pytorch.org/whl/cu${COLETTE_CUDA_SHORT}"

mkdir -p "${PIP_CACHE_DIR}" "${PIP_CACHE_DIR}/tmp"
export TMPDIR="${TMPDIR:-${PIP_CACHE_DIR}/tmp}"

"${PYTHON_BIN}" -m pip install --upgrade pip setuptools wheel packaging ninja
"${PYTHON_BIN}" -m pip install \
    "torch==${TORCH_VERSION}" \
    "torchvision==${TORCHVISION_VERSION}" \
    "torchaudio==${TORCHAUDIO_VERSION}" \
    --index-url "${TORCH_INDEX_URL}"

export MAX_JOBS="${MAX_JOBS:-4}"
PIP_CACHE_DIR="${PIP_CACHE_DIR}" TMPDIR="${TMPDIR}" \
    "${PYTHON_BIN}" -m pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation

if [ "${INSTALL_EDITABLE}" = "1" ]; then
    "${PYTHON_BIN}" -m pip install -e .[dev,trag]
fi

pip_check_output="$(${PYTHON_BIN} -m pip check 2>&1 || true)"
if [ -n "${pip_check_output}" ]; then
    filtered_output="$(printf '%s\n' "${pip_check_output}" | grep -viE '^pygobject .* requires pycairo, which is not installed\.$' || true)"
    if [ -n "${filtered_output}" ]; then
        echo "${pip_check_output}" >&2
        exit 1
    fi
    echo "WARNING: ignoring known system package mismatch from libreoffice/python3-uno: pygobject requires pycairo" >&2
fi
