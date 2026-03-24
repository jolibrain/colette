import multiprocessing as mp
import os

import pytest

models_repo = os.getenv("MODELS_REPO", "models")
VLLM_E2E_ENABLED = os.getenv("COLETTE_ENABLE_VLLM_E2E") == "1"


def require_vllm_runtime():
    """vLLM e2e is opt-in and can fail under fork-based multiprocessing with CUDA."""
    if os.getenv("COLETTE_ENABLE_VLLM_E2E") != "1":
        pytest.skip("set COLETTE_ENABLE_VLLM_E2E=1 to run vLLM e2e tests")
    start_method = mp.get_start_method(allow_none=True) or "fork"
    if start_method != "spawn":
        pytest.skip(f"vLLM e2e requires multiprocessing start method 'spawn' (got '{start_method}')")
