#!/bin/bash

set -euo pipefail

# Get CUDA version in cu format (e.g., cu126)
cuda_short=""
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    cuda_short=$(echo $cuda_version | sed 's/\.//')
    echo "cu${cuda_short}"
elif command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep -i "CUDA.*Version" | sed 's/.*CUDA[^:]*Version: \([0-9]*\.[0-9]*\).*/\1/')
    cuda_short=$(echo $cuda_version | sed 's/\.//')
    echo "CUDA found: cu${cuda_short}"
else
    echo "CUDA not found. Please ensure you have CUDA installed."
    exit 1
fi

python3 -m venv venv_colette
source venv_colette/bin/activate
echo "virtual environment 'venv_colette' activated."

echo "Installing dependencies using shared installer..."
COLETTE_CUDA_SHORT="${cuda_short}" bash scripts/install_python_deps.sh

echo "Running flash-attn compatibility smoke check..."
python - <<'PY'
from colette.backends.hf.attention import has_flash_attn, resolve_attn_implementation

# Qwen VL models use 3D mrope position IDs, which flash-attn's varlen kernel
# mishandles, so attention.py routes the whole family to sdpa. Everything else
# uses flash_attention_2 when flash-attn is importable.
qwen35 = resolve_attn_implementation("Qwen/Qwen3.5-9B")
qwen2vl = resolve_attn_implementation("Qwen/Qwen2-VL-7B-Instruct")
other = resolve_attn_implementation("meta-llama/Meta-Llama-3-8B")
print(f"has_flash_attn={has_flash_attn()}")
print(f"resolved_attn_implementation(Qwen3.5)={qwen35}")
print(f"resolved_attn_implementation(Qwen2-VL)={qwen2vl}")
print(f"resolved_attn_implementation(non-Qwen-VL)={other}")

if not has_flash_attn():
    raise SystemExit("flash-attn is not installed")
if qwen35 != "sdpa":
    raise SystemExit(f"Expected sdpa for Qwen3.5, got {qwen35!r}")
if qwen2vl != "sdpa":
    raise SystemExit(f"Expected sdpa for Qwen2-VL, got {qwen2vl!r}")
if other != "flash_attention_2":
    raise SystemExit(f"Expected flash_attention_2 for non-Qwen-VL models, got {other!r}")
PY

python -m pip check
python -m pip cache purge
echo "All dependencies installed."