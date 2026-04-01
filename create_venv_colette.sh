#!/bin/bash

set -euo pipefail

# Get CUDA version in cu format (e.g., cu126)
cuda_short=""
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    cuda_short=$(echo $cuda_version | sed 's/\.//')
    echo "cu${cuda_short}"
elif command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
    cuda_short=$(echo $cuda_version | sed 's/\.//')
    echo "CUDA found: cu${cuda_short}"
else
    echo "CUDA not found. Please ensure you have CUDA installed."
    exit 1
fi

python3 -m venv venv_colette
source venv_colette/bin/activate
echo "virtual environment 'venv_colette' activated."
python -m pip install --upgrade pip setuptools wheel packaging ninja
echo "Installing torch with CUDA support based on detected CUDA version..."
python -m pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${cuda_short}

echo "Installing flash-attn first (single build path)..."
# Install flash-attn before editable project install to avoid pip rebuilding it twice.
export MAX_JOBS=${MAX_JOBS:-4}
python -m pip install flash-attn==2.8.3 --no-build-isolation

echo "Installing other dependencies..."
python -m pip install -e .[dev,trag]

echo "Running flash-attn compatibility smoke check..."
python - <<'PY'
import importlib.util

from colette.backends.hf.attention import resolve_attn_implementation

has_flash = importlib.util.find_spec("flash_attn") is not None
attn_impl = resolve_attn_implementation("Qwen/Qwen3.5-9B")
attn_impl_other = resolve_attn_implementation("Qwen/Qwen2-VL-7B-Instruct")
print(f"flash_attn_installed={has_flash}")
print(f"resolved_attn_implementation(Qwen3.5)={attn_impl}")
print(f"resolved_attn_implementation(other)={attn_impl_other}")

# Qwen3.5 uses sdpa (PyTorch native) to avoid flash-attn varlen bug with mrope.
# All other models use flash_attention_2 when flash_attn is installed.
if not has_flash:
    raise SystemExit("flash-attn is not installed")
if attn_impl != "sdpa":
    raise SystemExit(f"Expected sdpa for Qwen3.5, got {attn_impl!r}")
if attn_impl_other != "flash_attention_2":
    raise SystemExit(f"Expected flash_attention_2 for other models, got {attn_impl_other!r}")
PY

python -m pip check
python -m pip cache purge
echo "All dependencies installed."