#!/bin/bash

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
pip install packaging wheel
echo "Installing torch with CUDA support based on detected CUDA version..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu${cuda_short}
echo "Installing other dependencies..."
pip install -e .[dev,trag]
pip cache purge
pip uninstall flash-attn -y
pip install flash-attn==2.5.6 --no-build-isolation
echo "All dependencies installed."