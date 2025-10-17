#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
echo "Script directory: $SCRIPT_DIR"

# Get CUDA version in cu format (e.g., cu126)
cuda_short=""
cuda_version=""
if command -v nvcc &> /dev/null; then
    cuda_version=$(nvcc --version | grep "release" | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
    cuda_short=$(echo $cuda_version | sed 's/\.//')
    echo "CUDA found via nvcc: cu${cuda_short} (version ${cuda_version})"
elif command -v nvidia-smi &> /dev/null; then
    cuda_version=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9]*\.[0-9]*\).*/\1/')
    cuda_short=$(echo $cuda_version | sed 's/\.//')
    echo "CUDA found via nvidia-smi: cu${cuda_short} (version ${cuda_version})"
else
    echo "CUDA not found. Please ensure you have CUDA installed."
    exit 1
fi

# Set CUDA environment variables GLOBALLY for the script
export CUDA_HOME=/usr/local/cuda-${cuda_version}
export CUDACXX=${CUDA_HOME}/bin/nvcc
export CUDA_PATH=${CUDA_HOME}
export PATH=${CUDA_HOME}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
export CPATH=${CUDA_HOME}/include:$CPATH

echo "CUDA_HOME set to: $CUDA_HOME"
echo "Checking if nvcc is accessible:"
which nvcc || echo "WARNING: nvcc not found in PATH"

# Check if python3 version is 3.12
python_version=$(python3 --version | awk '{print $2}')
if [[ $python_version != 3.12* ]]; then
    echo "Python 3.12 is required. Current version is $python_version"
    if command -v python3.12 &> /dev/null; then
        echo "Found python3.12, using it."
        PYTHON_CMD="python3.12"
        $PYTHON_CMD --version
    else
        echo "python3.12 not found. Please install Python 3.12."
        exit 1
    fi
else
    PYTHON_CMD="python3"
    echo "Using python3 as it is version 3.12"
fi

# Create virtual environment
$PYTHON_CMD -m venv venv_colette
source venv_colette/bin/activate
echo "Virtual environment 'venv_colette' activated."

# Upgrade pip
pip install --upgrade pip setuptools wheel packaging

# IMPORTANT: Clear pip cache before installing torch to avoid cached CPU version
echo "Clearing pip cache..."
pip cache purge

echo "Installing torch with CUDA support (cu128)."
# Uninstall any existing torch first
pip uninstall -y torch torchvision torchaudio

# Install CUDA-enabled torch
# Ref: https://dev-discuss.pytorch.org/t/pytorch-release-2-7-0-final-rc-is-available/2898
pip3 install torch==2.7.0 torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/test/cu128

# Verify torch installation
echo "Verifying PyTorch installation:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Check if CUDA is actually available
if ! python -c "import torch; exit(0 if torch.cuda.is_available() else 1)"; then
    echo "ERROR: PyTorch was installed but CUDA is not available!"
    echo "This likely means:"
    echo "  1. You don't have an NVIDIA GPU"
    echo "  2. NVIDIA drivers are not installed"
    echo "  3. CUDA toolkit version mismatch"
    exit 1
fi

echo "Installing FAISS from source with GPU support..."
# Install build dependencies
sudo apt-get update
sudo apt-get install -y \
    g++ \
    swig \
    libgflags-dev \
    libgoogle-glog-dev \
    libgtest-dev \
    libblas-dev \
    liblapack-dev \
    libopenblas-dev

# Clone and build Faiss
if [ -d "faiss" ]; then
    echo "Removing existing faiss directory..."
    rm -rf faiss
fi

git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Configure build (CUDA environment variables already set globally)
cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="72;75;80;86;87;89;90" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF

# Build
cmake --build build -j$(nproc)

# Install C++ libraries
sudo cmake --install build

# Install Python bindings
cd "$SCRIPT_DIR/faiss"
pip install build/faiss/python

# CRITICAL FIX: Add FAISS Python library path to LD_LIBRARY_PATH
# The libfaiss_python_callbacks.so is in the Python package, not in system libs
FAISS_PYTHON_LIB="$SCRIPT_DIR/venv_colette/lib/python3.12/site-packages/faiss"
if [ -f "${FAISS_PYTHON_LIB}/libfaiss_python_callbacks.so" ]; then
    echo "Found libfaiss_python_callbacks.so, adding to LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${FAISS_PYTHON_LIB}:${LD_LIBRARY_PATH}"
    
    # Make it permanent for this venv by modifying the activate script
    echo "" >> "$SCRIPT_DIR/venv_colette/bin/activate"
    echo "# FAISS library path" >> "$SCRIPT_DIR/venv_colette/bin/activate"
    echo "export LD_LIBRARY_PATH=\"${FAISS_PYTHON_LIB}:\${LD_LIBRARY_PATH}\"" >> "$SCRIPT_DIR/venv_colette/bin/activate"
    
    echo "✓ FAISS Python library path configured"
else
    echo "✗ WARNING: libfaiss_python_callbacks.so not found in expected location"
fi

# Verify FAISS import works
python -c "import faiss; print(f'✓ FAISS {faiss.__version__} imported successfully')" || echo "✗ FAISS import failed"

# Clean up
cd "$SCRIPT_DIR"
rm -rf faiss
echo "FAISS installation complete."

echo "Installing other dependencies..."

# Backup original pyproject.toml if it exists
if [ -f "$SCRIPT_DIR/pyproject.toml" ]; then
    cp "$SCRIPT_DIR/pyproject.toml" "$SCRIPT_DIR/pyproject.toml.bak"
fi

# Use ARM version if it exists
if [ -f "$SCRIPT_DIR/pyproject_ARM.toml" ]; then
    cp "$SCRIPT_DIR/pyproject_ARM.toml" "$SCRIPT_DIR/pyproject.toml"
fi

# Install with extras
pip install -e "$SCRIPT_DIR[dev,trag]"

# Now install flash-attn with proper environment variables
echo "Installing flash-attn..."
echo "Using CUDA_HOME: $CUDA_HOME"
echo "Using nvcc at: $(which nvcc)"

# Ensure environment variables are still set (they should be from the global export)
pip uninstall -y flash-attn 2>/dev/null || true

# Install flash-attn with --no-build-isolation to use our environment
pip install flash-attn==2.5.6 --no-build-isolation --verbose

# Verify flash-attn installation
if python -c "import flash_attn" 2>/dev/null; then
    echo "✓ flash-attn installed successfully!"
else
    echo "✗ flash-attn installation may have issues"
fi

echo "All dependencies installed."

# Restore original pyproject.toml
if [ -f "$SCRIPT_DIR/pyproject.toml.bak" ]; then
    mv "$SCRIPT_DIR/pyproject.toml.bak" "$SCRIPT_DIR/pyproject.toml"
fi

echo ""
echo "Installation complete!"
echo "To activate the environment in the future, run:"
echo "  source venv_colette/bin/activate"