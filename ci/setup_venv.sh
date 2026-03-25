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
PROFILE="${1:-smoke}"
SMOKE_REQ_FILE="${WORKSPACE}/ci/requirements-smoke.txt"
FULL_REQ_FILE="${WORKSPACE}/ci/requirements-full.txt"
REQUIRE_FLASH_ATTN="${COLETTE_REQUIRE_FLASH_ATTN:-0}"

if [ "${PROFILE}" = "smoke" ]; then
    VENV_CACHE="$(realpath "${WORKSPACE}/..")/venv_colette_smoke_cache"
elif [ "${PROFILE}" = "full" ]; then
    VENV_CACHE="$(realpath "${WORKSPACE}/..")/venv_colette_full_cache"
else
    echo "ERROR: unknown profile '${PROFILE}'. Expected 'smoke' or 'full'." >&2
    exit 1
fi

echo "=== CI venv setup ==="
echo "    workspace : ${WORKSPACE}"
echo "    profile   : ${PROFILE}"
echo "    venv cache: ${VENV_CACHE}"
echo "    require flash-attn: ${REQUIRE_FLASH_ATTN}"

needs_full_setup=0
setup_reason=""
if [ ! -x "${VENV_CACHE}/bin/python" ] || [ ! -x "${VENV_CACHE}/bin/pip" ]; then
    needs_full_setup=1
    setup_reason="missing python/pip in cache"
fi

if [ "${PROFILE}" = "smoke" ]; then
    if [ ! -f "${SMOKE_REQ_FILE}" ]; then
        echo "ERROR: missing smoke requirements file: ${SMOKE_REQ_FILE}" >&2
        exit 1
    fi

    req_hash_file="${VENV_CACHE}/.smoke_requirements.sha256"
    current_hash=$(sha256sum "${SMOKE_REQ_FILE}" | awk '{print $1}')
    cached_hash=""
    if [ -f "${req_hash_file}" ]; then
        cached_hash=$(cat "${req_hash_file}" 2>/dev/null || true)
    fi

    if [ "${needs_full_setup}" -eq 0 ] && [ "${cached_hash}" != "${current_hash}" ]; then
        needs_full_setup=1
        setup_reason="smoke requirements hash changed"
    fi
elif [ "${PROFILE}" = "full" ]; then
    if [ ! -f "${FULL_REQ_FILE}" ]; then
        echo "ERROR: missing full requirements file: ${FULL_REQ_FILE}" >&2
        exit 1
    fi

    req_hash_file="${VENV_CACHE}/.full_requirements.sha256"
    current_hash=$(sha256sum "${FULL_REQ_FILE}" | awk '{print $1}')
    cached_hash=""
    if [ -f "${req_hash_file}" ]; then
        cached_hash=$(cat "${req_hash_file}" 2>/dev/null || true)
    fi

    if [ "${needs_full_setup}" -eq 0 ] && [ "${cached_hash}" != "${current_hash}" ]; then
        needs_full_setup=1
        setup_reason="full requirements hash changed"
    fi

    if [ "${needs_full_setup}" -eq 0 ] && ! "${VENV_CACHE}/bin/python" -c "import pytest" >/dev/null 2>&1; then
        needs_full_setup=1
        setup_reason="pytest missing from full cache"
    fi

    if [ "${REQUIRE_FLASH_ATTN}" = "1" ] && [ "${needs_full_setup}" -eq 0 ] && ! "${VENV_CACHE}/bin/python" -c "import flash_attn" >/dev/null 2>&1; then
        needs_full_setup=1
        setup_reason="flash_attn required but missing from full cache"
    fi
fi

if [ "${needs_full_setup}" -eq 1 ]; then
    echo ""
    echo ">>> Building venv (reason: ${setup_reason:-first run or broken cache detected})"
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

    if [ "${PROFILE}" = "smoke" ]; then
        echo "    Installing smoke profile dependencies from ${SMOKE_REQ_FILE}"
        "${VENV_CACHE}/bin/pip" install -r "${SMOKE_REQ_FILE}"
        "${VENV_CACHE}/bin/pip" install --quiet -e . --no-deps
        sha256sum "${SMOKE_REQ_FILE}" | awk '{print $1}' > "${VENV_CACHE}/.smoke_requirements.sha256"
        echo ">>> Smoke venv created."
    else
        echo "    Installing full profile dependencies from ${FULL_REQ_FILE}"
        "${VENV_CACHE}/bin/pip" install -r "${FULL_REQ_FILE}"
        "${VENV_CACHE}/bin/pip" install --quiet -e . --no-deps
        sha256sum "${FULL_REQ_FILE}" | awk '{print $1}' > "${VENV_CACHE}/.full_requirements.sha256"

        # Avoid expensive flash-attn build attempts when torch and host CUDA are clearly mismatched.
        torch_cuda_version=$("${VENV_CACHE}/bin/python" -c 'import torch; print(torch.version.cuda or "")')
        can_try_flash_attn=1
        if [ -z "${torch_cuda_version}" ]; then
            can_try_flash_attn=0
            msg="torch has no CUDA runtime (torch.version.cuda is empty)"
        elif [ "${torch_cuda_version}" != "${cuda_version}" ]; then
            can_try_flash_attn=0
            msg="host CUDA ${cuda_version} != torch CUDA ${torch_cuda_version}"
        fi

        if [ "${can_try_flash_attn}" -eq 1 ]; then
            # flash-attn wheels/build may lag behind newest torch/CUDA combos.
            # Keep CI moving if this optional acceleration package cannot be installed,
            # unless this lane explicitly requires it.
            if ! "${VENV_CACHE}/bin/pip" install flash-attn==2.5.6 --no-build-isolation; then
                if [ "${REQUIRE_FLASH_ATTN}" = "1" ]; then
                    echo "ERROR: flash-attn==2.5.6 install failed but COLETTE_REQUIRE_FLASH_ATTN=1" >&2
                    exit 1
                fi
                echo "    WARNING: flash-attn==2.5.6 install failed; continuing without flash-attn"
            fi
        else
            if [ "${REQUIRE_FLASH_ATTN}" = "1" ]; then
                echo "ERROR: ${msg}; cannot satisfy COLETTE_REQUIRE_FLASH_ATTN=1" >&2
                exit 1
            fi
            echo "    WARNING: skipping flash-attn install (${msg})"
        fi
        echo ">>> Full venv created."
    fi
else
    echo "    Cached venv found — updating editable install only"

    if [ "${PROFILE}" = "full" ]; then
        # Keep full profile stable and separate from smoke profile incremental repairs.
        # Assume first full bootstrap already installed heavy dependencies.
        "${VENV_CACHE}/bin/pip" install --quiet -e . --no-deps
        ln -sfn "${VENV_CACHE}" "${WORKSPACE}/venv_colette"
        echo "    Python: $("${VENV_CACHE}/bin/python" --version)"
        echo "=== venv ready ==="
        exit 0
    fi

    # Smoke profile is deterministic from requirements-smoke.txt and hash tracking.
    # Keep editable metadata synced without mutating dependency versions.
    "${VENV_CACHE}/bin/pip" install --quiet -e . --no-deps
fi

if [ "${PROFILE}" = "smoke" ]; then
    echo "    Running smoke import preflight"
    "${VENV_CACHE}/bin/python" "${WORKSPACE}/ci/verify_smoke_imports.py" --workspace "${WORKSPACE}"
fi

# Symlink into workspace so the Makefile autodiscovers it
ln -sfn "${VENV_CACHE}" "${WORKSPACE}/venv_colette"

echo "    Python: $("${VENV_CACHE}/bin/python" --version)"
echo "=== venv ready ==="
