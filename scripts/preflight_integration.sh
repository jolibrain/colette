#!/usr/bin/env bash
#
# Preflight checks for integration testing
# Validates external service availability before running integration tests
#
# Usage: ./scripts/preflight_integration.sh [--check-gpu] [--strict]
#   --check-gpu : Also validate GPU availability
#   --strict    : Exit with error if any service is unavailable (default: warnings only)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

CHECK_GPU=0
STRICT=0
CHECKS_PASSED=0
CHECKS_FAILED=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --check-gpu) CHECK_GPU=1 ; shift ;;
        --strict) STRICT=1 ; shift ;;
        *) echo "Unknown option: $1" ; exit 1 ;;
    esac
done

# Functions for output
pass() {
    echo -e "${GREEN}✓${NC} $1"
    CHECKS_PASSED=$((CHECKS_PASSED + 1))
}

warn() {
    echo -e "${YELLOW}⚠${NC} $1"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

fail() {
    echo -e "${RED}✗${NC} $1"
    CHECKS_FAILED=$((CHECKS_FAILED + 1))
}

echo "=========================================="
echo "Integration Test Preflight Checks"
echo "=========================================="
echo ""

# Check 1: Python environment
echo "Checking Python environment..."
if [ -f "$PROJECT_ROOT/venv_colette/bin/python" ]; then
    PYTHON="$PROJECT_ROOT/venv_colette/bin/python"
    PY_VERSION=$($PYTHON --version 2>&1 | awk '{print $2}')
    pass "Python environment: $PY_VERSION"
else
    fail "Python venv not found at venv_colette/"
    PYTHON="python3"
fi
echo ""

# Check 2: Required packages
echo "Checking required packages..."
$PYTHON -c "import pytest; import colette" 2>/dev/null && \
    pass "Core packages installed (pytest, colette)" || \
    fail "Missing core packages"
echo ""

# Check 3: Ollama service (optional but recommended for many tests)
echo "Checking external services..."
if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/11434" 2>/dev/null; then
    pass "Ollama service available (localhost:11434)"
else
    warn "Ollama service not reachable (localhost:11434) - some tests will be skipped"
fi

# Check 4: vLLM service (optional)
if timeout 2 bash -c "echo > /dev/tcp/127.0.0.1/8000" 2>/dev/null; then
    pass "vLLM service available (localhost:8000)"
else
    warn "vLLM service not reachable (localhost:8000) - some tests will be skipped"
fi

# Check 5: Model files exist
echo ""
echo "Checking model files..."
MODELS_DIR="$PROJECT_ROOT/models"
if [ -d "$MODELS_DIR" ]; then
    MODEL_COUNT=$(find "$MODELS_DIR" -type f -name "*.pt" -o -name "*.bin" | wc -l)
    if [ "$MODEL_COUNT" -gt 0 ]; then
        pass "Found $MODEL_COUNT model files in ./models/"
    else
        warn "No model files found in ./models/ - download models before running integration tests"
    fi
else
    warn "models/ directory not found - create ./models/ and add required models"
fi
echo ""

# Check 6: GPU availability (if requested)
if [ $CHECK_GPU -eq 1 ]; then
    echo "Checking GPU availability..."
    if command -v nvidia-smi &> /dev/null; then
        GPUS=$($PYTHON -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo "0")
        if [ "$GPUS" -gt 0 ]; then
            pass "GPU detected: $GPUS device(s)"
        else
            warn "nvidia-smi found but no CUDA devices detected"
        fi
    else
        warn "nvidia-smi not found - GPU tests will be skipped"
    fi
    echo ""
fi

# Summary
echo "=========================================="
echo "Preflight Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}$CHECKS_PASSED${NC} | Failed: ${RED}$CHECKS_FAILED${NC}"
echo ""

if [ $CHECKS_FAILED -gt 0 ]; then
    if [ $STRICT -eq 1 ]; then
        echo -e "${RED}STRICT MODE: Failing due to $CHECKS_FAILED issue(s)${NC}"
        exit 1
    else
        echo -e "${YELLOW}Some services unavailable, but proceeding anyway (use --strict to exit on failure)${NC}"
        echo "Set COLETTE_RUN_INTEGRATION=1 to run integration tests (they will be skipped for unavailable services)"
    fi
else
    echo -e "${GREEN}All checks passed! Ready for integration testing.${NC}"
fi
echo ""
echo "To run integration tests:"
echo "  make test-integration"
echo ""
