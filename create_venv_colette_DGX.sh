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
if [ -d "venv_colette" ]; then
    echo "Removing existing virtual environment..."
    rm -rf venv_colette
fi
$PYTHON_CMD -m venv venv_colette
source venv_colette/bin/activate
ACTIVATE_FILE="$SCRIPT_DIR/venv_colette/bin/activate"
echo "Virtual environment 'venv_colette' activated."

# Upgrade pip
pip install --upgrade pip
# Keep toolchain versions compatible with pinned runtime deps.
pip install "setuptools>=77.0.3,<81.0.0" wheel "packaging>=23.2,<26.0.0"

# IMPORTANT: Clear pip cache before installing torch to avoid cached CPU version
echo "Clearing pip cache..."
pip cache purge

# Select torch/flash-attn versions based on detected CUDA major version
cuda_major=$(echo "$cuda_version" | cut -d. -f1)
if [ "$cuda_major" -ge 13 ]; then
    TORCH_VERSION="2.9.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/test/cu130"
    FLASH_ATTN_VERSION="2.8.3"
else
    TORCH_VERSION="2.7.0"
    TORCH_INDEX_URL="https://download.pytorch.org/whl/test/cu128"
    FLASH_ATTN_VERSION="2.5.6"
fi

echo "Installing torch with CUDA support from: $TORCH_INDEX_URL"
# Uninstall any existing torch first
pip uninstall -y torch torchvision torchaudio

# Install CUDA-enabled torch
pip3 install "torch==${TORCH_VERSION}" torchvision torchaudio --extra-index-url "$TORCH_INDEX_URL"

# Verify torch installation
echo "Verifying PyTorch installation:"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Align CUDA toolkit used for native builds with the CUDA version PyTorch was built with
torch_cuda_version=$(python -c "import torch; print(torch.version.cuda or '')")
if [ -n "$torch_cuda_version" ] && [ "$cuda_version" != "$torch_cuda_version" ]; then
    MATCHING_CUDA_HOME="/usr/local/cuda-${torch_cuda_version}"
    if [ -d "$MATCHING_CUDA_HOME" ]; then
        echo "Switching CUDA toolkit to match PyTorch: ${torch_cuda_version}"
        export CUDA_HOME="$MATCHING_CUDA_HOME"
        export CUDACXX=${CUDA_HOME}/bin/nvcc
        export CUDA_PATH=${CUDA_HOME}
        export PATH=${CUDA_HOME}/bin:$PATH
        export LD_LIBRARY_PATH=${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
        export LIBRARY_PATH=${CUDA_HOME}/lib64:$LIBRARY_PATH
        export CPATH=${CUDA_HOME}/include:$CPATH
        cuda_version="$torch_cuda_version"
        cuda_short=$(echo "$cuda_version" | sed 's/\.//')
        echo "Using CUDA toolkit at: $CUDA_HOME"
    else
        echo "ERROR: CUDA toolkit mismatch (system: ${cuda_version}, torch: ${torch_cuda_version})"
        echo "ERROR: ${MATCHING_CUDA_HOME} not found. flash-attn is required and cannot be built."
        exit 1
    fi
fi

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
# Install build dependencies (skip if already present)
BUILD_DEPS="g++ swig libgflags-dev libgoogle-glog-dev libgtest-dev libblas-dev liblapack-dev libopenblas-dev"
MISSING_DEPS=""
for pkg in $BUILD_DEPS; do
    if ! dpkg -s "$pkg" &>/dev/null; then
        MISSING_DEPS="$MISSING_DEPS $pkg"
    fi
done
if [ -n "$MISSING_DEPS" ]; then
    echo "Installing missing packages:$MISSING_DEPS"
    sudo apt-get update
    sudo apt-get install -y $MISSING_DEPS
else
    echo "All build dependencies already installed, skipping apt-get."
fi

# Clone and build Faiss
if [ -d "faiss" ]; then
    echo "Removing existing faiss directory..."
    rm -rf faiss
fi

git clone https://github.com/facebookresearch/faiss.git
cd faiss

# Select CUDA architectures based on CUDA version
# CUDA 13+ dropped sm_72/sm_75; CUDA 12+ dropped sm_70
cuda_major=$(echo "$cuda_version" | cut -d. -f1)
if [ "$cuda_major" -ge 13 ]; then
    CUDA_ARCHS="80;86;87;89;90"
elif [ "$cuda_major" -ge 12 ]; then
    CUDA_ARCHS="75;80;86;87;89;90"
else
    CUDA_ARCHS="72;75;80;86;87;89;90"
fi
echo "Using CUDA architectures: $CUDA_ARCHS"

# Configure build (CUDA environment variables already set globally)
cmake -B build \
  -DFAISS_ENABLE_GPU=ON \
  -DFAISS_ENABLE_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TESTING=OFF

# Build
cmake --build build -j$(nproc)

# Install C++ libraries to user-local prefix (no sudo required)
cmake --install build --prefix "$HOME/.local"
FAISS_LIB_DIR="$HOME/.local/lib"
export LD_LIBRARY_PATH="${FAISS_LIB_DIR}:${LD_LIBRARY_PATH}"

# Install Python bindings
cd "$SCRIPT_DIR/faiss"
pip install build/faiss/python

# CRITICAL FIX: Add FAISS runtime library paths to LD_LIBRARY_PATH.
# The Python package ships libfaiss_python_callbacks.so, while libfaiss.so lives
# under the user-local install prefix. Both must be preferred over /usr/local/lib.
FAISS_PYTHON_LIB="$SCRIPT_DIR/venv_colette/lib/python3.12/site-packages/faiss"
if [ -d "$FAISS_LIB_DIR" ] && [ -f "${FAISS_PYTHON_LIB}/libfaiss_python_callbacks.so" ]; then
    echo "Found FAISS runtime libraries, adding to LD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="${FAISS_LIB_DIR}:${FAISS_PYTHON_LIB}:${LD_LIBRARY_PATH}"

    # Make it permanent for this venv by modifying the activate script.
    echo "" >> "$ACTIVATE_FILE"
    echo "# Colette FAISS library paths" >> "$ACTIVATE_FILE"
    echo "export LD_LIBRARY_PATH=\"${FAISS_LIB_DIR}:${FAISS_PYTHON_LIB}:\${LD_LIBRARY_PATH}\"" >> "$ACTIVATE_FILE"

    echo "✓ FAISS runtime library paths configured"
else
    echo "✗ WARNING: FAISS runtime libraries not found in expected locations"
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

# Use DGX version if it exists
if [ -f "$SCRIPT_DIR/pyproject_DGX.toml" ]; then
    cp "$SCRIPT_DIR/pyproject_DGX.toml" "$SCRIPT_DIR/pyproject.toml"
fi

# Install flash-attn BEFORE colette so pip doesn't try to build it in an
# isolated environment (which lacks torch).
echo "Installing flash-attn..."
echo "Using CUDA_HOME: $CUDA_HOME"
echo "Using nvcc at: $(which nvcc)"
pip uninstall -y flash-attn 2>/dev/null || true
# flash-attn setup.py imports psutil while preparing metadata
pip install psutil ninja
export MAX_JOBS=4
echo "Using MAX_JOBS=$MAX_JOBS for flash-attn build"
if ! pip install "flash-attn==${FLASH_ATTN_VERSION}" --no-build-isolation --verbose; then
    echo "ERROR: flash-attn build failed and is required."
    exit 1
fi

# flash-attn 2.x may require CUDA 12 runtime soname at import time.
# Keep this runtime package and library path when using torch+cu130.
if [ "$cuda_major" -ge 13 ]; then
    pip install nvidia-cuda-runtime-cu12
fi

# Install colette with extras (flash-attn already satisfied above)
pip install -e "$SCRIPT_DIR[dev,trag]"

# Ensure runtime lib paths are available for flash-attn/torch imports.
TORCH_LIB_DIR="$SCRIPT_DIR/venv_colette/lib/python3.12/site-packages/torch/lib"
NVIDIA_CU13_LIB_DIR="$SCRIPT_DIR/venv_colette/lib/python3.12/site-packages/nvidia/cu13/lib"
NVIDIA_CUDA_RUNTIME_LIB_DIR="$SCRIPT_DIR/venv_colette/lib/python3.12/site-packages/nvidia/cuda_runtime/lib"
for lib_dir in "$TORCH_LIB_DIR" "$NVIDIA_CU13_LIB_DIR" "$NVIDIA_CUDA_RUNTIME_LIB_DIR"; do
    if [ -d "$lib_dir" ]; then
        export LD_LIBRARY_PATH="$lib_dir:${LD_LIBRARY_PATH}"
    fi
done

# Persist runtime lib paths in venv activation script
if ! grep -q "Colette CUDA runtime paths" "$ACTIVATE_FILE"; then
    echo "" >> "$ACTIVATE_FILE"
    echo "# Colette CUDA runtime paths" >> "$ACTIVATE_FILE"
    echo "export LD_LIBRARY_PATH=\"$TORCH_LIB_DIR:$NVIDIA_CU13_LIB_DIR:$NVIDIA_CUDA_RUNTIME_LIB_DIR:\${LD_LIBRARY_PATH}\"" >> "$ACTIVATE_FILE"
fi

# Verify flash-attn installation
if python -c "import torch; import flash_attn" 2>/dev/null; then
    echo "✓ flash-attn installed successfully!"
else
    echo "✗ flash-attn is required but failed to import"
    exit 1
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