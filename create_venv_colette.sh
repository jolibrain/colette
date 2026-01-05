#!/bin/bash

# Get CUDA version in cu format (e.g., cu126)
cuda_short=""
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    cuda_short=$(echo $cuda_version | sed 's/\.//')
    echo "CUDA found (from NVCC): cu${cuda_short}"
elif command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
    cuda_short=$(echo $cuda_version | sed 's/\.//')
    echo "CUDA found (from nvidia-smi): cu${cuda_short}"
else
    echo "CUDA not found. Please ensure you have CUDA installed."
    exit 1
fi

python3 -m venv venv_colette
source venv_colette/bin/activate
echo "virtual environment 'venv_colette' activated."
pip install packaging wheel
echo "Installing torch with CUDA support based on detected CUDA version..."
pip install torch==2.8.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${cuda_short}
echo "Installing other dependencies..."
pip install -e .[dev,trag,build-system] --no-build-isolation

echo "Installing flash-attention..."
git clone https://github.com/Dao-AILab/flash-attention.git
cd flash-attention
git submodule update --init --recursive
# I had to specify the wheel / tempdir so that the are on the same partition
mkdir -p ~/wheels
TMPDIR=$HOME/.cache/pip-tmp \
python -m pip wheel --no-build-isolation -w ~/wheels .
python -m pip install ~/wheels/flash_attn-*.whl

cd ..
echo "All dependencies installed."