#!/usr/bin/env bash
# ci/setup_venv.sh
#
# Ensures a persistent Python venv is available for CI.
#
# The venv is stored ONE DIRECTORY ABOVE the Jenkins workspace so it
# survives cleanWs() between builds while remaining per-node and per-job:
#
#   /var/lib/jenkins/workspace/colette-ci/        <- WORKSPACE (cleaned)
#   /var/lib/jenkins/workspace/venv_colette_cache <- VENV (persists)
#
# A symlink venv_colette -> VENV is created inside the workspace so the
# Makefile's autodiscovery ($(wildcard venv_colette/bin/python)) works.
#
# Usage (from repo root):
#   bash ci/setup_venv.sh
#
# On first run this takes several minutes (torch + flash-attn build).
# Subsequent runs reuse the cache and just resync the editable install.

set -euo pipefail

WORKSPACE="${WORKSPACE:-$(pwd)}"
VENV_CACHE="$(realpath "${WORKSPACE}/..")/venv_colette_cache"

echo "=== CI venv setup ==="
echo "    workspace : ${WORKSPACE}"
echo "    venv cache: ${VENV_CACHE}"

needs_full_setup=0
if [ ! -x "${VENV_CACHE}/bin/python" ] || [ ! -x "${VENV_CACHE}/bin/pip" ]; then
    needs_full_setup=1
fi

if [ "${needs_full_setup}" -eq 1 ]; then
    echo ""
    echo ">>> Building full venv (first run or broken cache detected)"
    echo ">>> Requires CUDA drivers and nvcc/nvidia-smi on PATH"
    echo ""

    # Clean corrupted/incomplete cache before recreating.
    rm -rf "${VENV_CACHE}"

    # Detect CUDA version the same way create_venv_colette.sh does
    if command -v nvcc &>/dev/null; then
        cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    elif command -v nvidia-smi &>/dev/null; then
        cuda_version=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
    else
        echo "ERROR: CUDA not found. This node cannot build the colette venv." >&2
        echo "       Run create_venv_colette.sh manually on this node first." >&2
        exit 1
    fi
    cuda_short=$(echo "${cuda_version}" | tr -d '.')
    echo "    CUDA: cu${cuda_short}"
    torch_index_url="https://download.pytorch.org/whl/cu${cuda_short}"

    python3 -m venv "${VENV_CACHE}"
    "${VENV_CACHE}/bin/pip" install --quiet --upgrade pip setuptools wheel

    # Prefer the known-good torch pin when available; fallback for newer CUDA channels
    # (e.g. cu130) where 2.7.0 wheels no longer exist.
    if ! "${VENV_CACHE}/bin/pip" install torch==2.7.0 torchvision torchaudio --index-url "${torch_index_url}"; then
        echo "    torch==2.7.0 not available for cu${cuda_short}; falling back to latest cu${cuda_short} wheels"
        "${VENV_CACHE}/bin/pip" install torch torchvision torchaudio --index-url "${torch_index_url}"
    fi

    "${VENV_CACHE}/bin/pip" install --quiet -e ".[dev,trag]"
    "${VENV_CACHE}/bin/pip" uninstall -y flash-attn 2>/dev/null || true
    # flash-attn wheels/build may lag behind newest torch/CUDA combos.
    # Keep CI moving if this optional acceleration package cannot be installed.
    if ! "${VENV_CACHE}/bin/pip" install flash-attn==2.5.6 --no-build-isolation; then
        echo "    WARNING: flash-attn==2.5.6 install failed; continuing without flash-attn"
    fi
    echo ">>> Full venv created."
else
    echo "    Cached venv found — updating editable install only"

    # Validate that key test tooling exists in the cached environment.
    # Older caches may miss dev dependencies (e.g. pytest), which makes CI fail
    # even though the venv itself exists.
    missing_dev_deps=0
    for module in pytest pytest_asyncio pytest_cov; do
        if ! "${VENV_CACHE}/bin/python" -c "import ${module}" >/dev/null 2>&1; then
            missing_dev_deps=1
            break
        fi
    done

    if [ "${missing_dev_deps}" -eq 1 ]; then
        echo "    Dev test dependencies missing in cache; repairing with full extras install"
        "${VENV_CACHE}/bin/pip" install -e ".[dev,trag]"
    else
        "${VENV_CACHE}/bin/pip" install --quiet -e ".[dev,trag]" --no-deps
    fi
fi

# Symlink into workspace so the Makefile autodiscovers it
ln -sfn "${VENV_CACHE}" "${WORKSPACE}/venv_colette"

echo "    Python: $("${VENV_CACHE}/bin/python" --version)"
echo "=== venv ready ==="
