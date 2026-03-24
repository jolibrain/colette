# Integration Testing Guide

This document describes how to run integration tests for Colette.

## Overview

Colette has a two-tier testing strategy:

1. **Smoke Tests** (Default): Fast, deterministic unit tests that require no external services
   - Command: `make test-smoke` or `pytest tests/ -m smoke`
   - Enforced coverage gate: 35%
   - No external dependencies (no GPU, no services, no models)

2. **Integration Tests** (Opt-in): Full pipeline tests with external services
   - Command: `make test-integration` or `COLETTE_RUN_INTEGRATION=1 pytest tests/`
   - Requires: Ollama, vLLM, GPU models, or other backend services
   - Optional: GPU validation with `--check-gpu`

## Quick Start: Smoke Tests Only

```bash
# Run smoke tests (default, no external deps needed)
make test-smoke

# Or with coverage verification
make test-coverage  # Must achieve ≥35% coverage
```

## Full Integration Testing Setup

### Prerequisites

1. **Python environment** (already set up):
   ```bash
   source venv_colette/bin/activate
   ```

2. **Ollama** (for LLM backend integration):
   ```bash
   # Install from https://ollama.ai
   ollama serve  # Runs on localhost:11434
   ```

3. **vLLM** (optional, for high-throughput LLM serving):
   ```bash
   pip install vllm  # In your venv
   python -m vllm.entrypoints.openai.api_server  # Runs on localhost:8000
   ```

4. **GPU & CUDA** (if testing GPU-accelerated models):
   - NVIDIA GPU with CUDA 11.8+ or 12.x
   - PyTorch with CUDA support (`pip install torch --index-url https://download.pytorch.org/whl/cu118`)
   - Verify: `python -c "import torch; print(torch.cuda.is_available())"`

5. **Model Files**:
   - Place models in `./models/` directory
   - Common models:
     - Qwen3-VL-4B or 8B (multimodal vision-language)
     - Embeddings: Alibaba's GME or Qwen embeddings
   - Download: Models are typically downloaded via Hugging Face on first use

### Preflight Checks

Before running integration tests, validate your environment:

```bash
# Basic checks (Ollama, vLLM, models)
./scripts/preflight_integration.sh

# With GPU validation
./scripts/preflight_integration.sh --check-gpu

# Strict mode (fails if any service is unavailable)
./scripts/preflight_integration.sh --strict
```

### Running Integration Tests

```bash
# Run all tests (smoke + integration)
make test-integration

# Or manually with env var
COLETTE_RUN_INTEGRATION=1 pytest tests/ -v

# Specific test file
COLETTE_RUN_INTEGRATION=1 pytest tests/test_base.py -v

# Run the stable pipeline integration lane only (recommended first)
make test-integration-pipeline
# with a custom GPU:
make test-integration-pipeline GPU_ID=0

# Full e2e suite (GPU required; see Known Issues for vLLM environment notes)
make test-e2e

# With coverage
COLETTE_RUN_INTEGRATION=1 pytest tests/ --cov=src/colette --cov-report=html
```

### Pipeline LLM Model Override

The pipeline integration tests use `Qwen/Qwen2-VL-2B-Instruct` as the default
LLM backend but support a runtime override:

```bash
COLETTE_PIPELINE_HF_MODEL=Qwen/Qwen3-VL-8B-Instruct make test-integration-pipeline
```

Set this when the default model is not present in `models/` or when testing against
a different checkpoint.

## Test Markers

Tests are marked to help organize runs:

- `@pytest.mark.smoke` — Fast, no external deps (always runs)
- `@pytest.mark.integration` — Requires external services (skip by default, enable with `COLETTE_RUN_INTEGRATION=1`)
- `@pytest.mark.e2e` — Broad end-to-end tests with heavier runtime requirements (subset of integration)

Examples:
```bash
# Smoke only
pytest tests/ -m smoke

# Integration only
COLETTE_RUN_INTEGRATION=1 pytest tests/ -m integration

# Pipeline integration only (stable, fastest)
make test-integration-pipeline

# e2e only
COLETTE_RUN_INTEGRATION=1 pytest tests/ -m e2e
```

## Common Service Setup

### Starting Ollama

```bash
# Terminal 1: Start Ollama service
ollama serve

# Terminal 2: Pull a model (in a different terminal)
ollama pull mistral:latest
# or for multimodal:
ollama pull moondream:latest
```

Verify: `curl http://localhost:11434/api/tags`

### Starting vLLM

```bash
# Terminal: Start vLLM with a model
python -m vllm.entrypoints.openai.api_server \
  --model mistralai/Mistral-7B-Instruct-v0.2 \
  --gpu-memory-utilization 0.8
```

Verify: `curl http://localhost:8000/v1/models`

## Troubleshooting

### Services not available?
- Preflight script will warn but not block (use `--strict` to enforce)
- Integration tests that depend on unavailable services will be skipped gracefully
- Check logs: `tail -f /tmp/ollama.log` or vLLM stdout

### GPU not detected?
- Run `nvidia-smi` to verify GPU driver
- Run `python -c "import torch; print(torch.cuda.is_available())`
- If False, install CUDA-compatible PyTorch: `pip install torch --index-url https://download.pytorch.org/whl/cu118`

### Import errors during integration tests?
- Run integration tests from repo root: `cd /path/to/colette && make test-integration`
- Verify venv is activated: `which python` should point to `venv_colette/bin/python`
- Check `sys.path` in conftest.py for package alias setup

### Coverage drops during integration?
- Integration tests run more paths; coverage may vary
- Compare to baseline: `make test-coverage` reports smoke-only (35% enforced)
- Full integration coverage reported separately: `COLETTE_RUN_INTEGRATION=1 pytest ... --cov=...`

## CI/CD Integration

In CI/CD pipelines:

1. **Smoke phase** (always run, blocks on failure):
   ```bash
   make test-smoke
   make test-coverage  # Must exceed COV_MIN=35
   ```

2. **Integration preflight** (advisory, non-blocking):
   ```bash
   ./scripts/preflight_integration.sh --strict || echo "⚠ Services unavailable"
   ```

3. **Pipeline integration** (recommended stable lane, requires GPU + services):
    ```bash
    if [ $? -eq 0 ]; then
       make ci-pipeline-integration   # outputs .ci-artifacts/junit-pipeline-integration.xml
    else
       echo "Skipping pipeline integration (preflight failed)"
    fi
    ```

4. **Full integration / e2e** (optional, broader coverage):
   ```bash
   if [ $? -eq 0 ]; then
     make test-integration
   else
     echo "Skipping integration tests (preflight failed)"
   fi
   ```

## Known Issues

### vLLM e2e tests: "Cannot re-initialize CUDA in forked subprocess"

Tests in `test_base.py` and `test_base_ci.py` marked `@pytest.mark.e2e` that
exercise vLLM may fail with this error in certain CUDA environments when pytest
forks worker processes. This is a runtime limitation, not a code bug. These
tests are gated behind `COLETTE_RUN_INTEGRATION=1` and are excluded from the
default smoke and pipeline integration lanes.

**Workaround:** Run affected tests in isolation or use the pipeline integration
lane (`make test-integration-pipeline`), which does not exercise vLLM directly.

## Summary

| Task | Command | Requires |
|------|---------|----------|
| Quick test & coverage | `make test-smoke && make test-coverage` | None |
| Pipeline integration | `make test-integration-pipeline` | GPU + Ollama + local models |
| Full integration | `make test-integration` | Ollama + GPU |
| Full e2e suite | `make test-e2e` | GPU + vLLM (see Known Issues) |
| CI pipeline artifacts | `make ci-pipeline-integration` | GPU + Ollama + local models |
| Check services | `./scripts/preflight_integration.sh` | None |

---

For more details, see [CONTRIBUTING.md](../CONTRIBUTING.md).
