import os
import shutil
import sys
import gc
import time
from pathlib import Path

import pytest

# ---------------------------------------------------------------------------
# GPU selection: honour COLETTE_GPU_ID (set by Makefile via GPU_ID variable).
# This must happen before any CUDA import so that CUDA_VISIBLE_DEVICES takes
# effect when torch / colette backends are first loaded.
# Override from the command line:  make test-e2e GPU_ID=0
# ---------------------------------------------------------------------------
_colette_gpu_id = os.getenv("COLETTE_GPU_ID")
if _colette_gpu_id is not None and os.getenv("CUDA_VISIBLE_DEVICES") is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = _colette_gpu_id

# Reduce CUDA allocator fragmentation during long e2e runs.
if os.getenv("PYTORCH_CUDA_ALLOC_CONF") is None:
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


@pytest.fixture(scope="session")
def gpu_id() -> int:
    """Return the logical CUDA device index for the current test session.

    With CUDA_VISIBLE_DEVICES set to a single physical GPU, the logical device
    index is always 0 (the only visible device).  Use this fixture instead of
    hard-coding 0 in tests so that the physical GPU can be changed via
    ``make test-e2e GPU_ID=<n>``.
    """
    return 0


@pytest.fixture(scope="module")
def client():
    """Shared FastAPI test client for integration-style tests."""
    from fastapi.testclient import TestClient
    from colette.httpjsonapi import app

    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def temp_dir(request):
    """Provide a clean repository path for tests marked with `repository_path`."""
    marker = request.node.get_closest_marker("repository_path")
    if marker is None or not marker.args:
        pytest.fail("temp_dir fixture requires @pytest.mark.repository_path(...)")

    repo_dir = Path(marker.args[0])
    shutil.rmtree(repo_dir, ignore_errors=True)
    repo_dir.mkdir(parents=True, exist_ok=True)
    yield repo_dir
    shutil.rmtree(repo_dir, ignore_errors=True)


# Ensure test imports that use top-level package names (e.g. `backends`) resolve
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_ROOT = os.path.join(ROOT, "src")
if SRC_ROOT not in sys.path:
    sys.path.insert(0, SRC_ROOT)

# Ensure tests that import top-level names like `backends` or `logger` still work
import importlib
import types

try:
    colette_backends = importlib.import_module("colette.backends")
    # expose a top-level package name `backends` that maps to `colette.backends`
    back_mod = types.ModuleType("backends")
    back_mod.__path__ = colette_backends.__path__
    sys.modules["backends"] = back_mod
except Exception:
    # ignore if colette package not importable yet
    pass

for alias in ("logger", "httpjsonapi"):
    try:
        mod = importlib.import_module(f"colette.{alias}")
        sys.modules[alias] = mod
    except Exception:
        pass


def pytest_collection_modifyitems(config, items):
    """Skip integration tests unless explicitly enabled.

    Set COLETTE_RUN_INTEGRATION=1 to run integration tests.
    """
    if os.getenv("COLETTE_RUN_INTEGRATION") == "1":
        return

    skip_integration = pytest.mark.skip(reason="integration test (set COLETTE_RUN_INTEGRATION=1 to run)")
    for item in items:
        if "integration" in item.keywords:
            item.add_marker(skip_integration)


def pytest_sessionstart(session):
    """Session-level setup for integration testing.

    If integration tests are enabled, log service availability info.
    """
    if os.getenv("COLETTE_RUN_INTEGRATION") != "1":
        return

    import socket

    def _service_available(host, port, timeout=2):
        """Check if a service is reachable."""
        try:
            socket.create_connection((host, port), timeout=timeout)
            return True
        except (socket.timeout, OSError):
            return False

    # Log service availability for integration session
    services = {
        "Ollama": ("127.0.0.1", 11434),
        "vLLM": ("127.0.0.1", 8000),
    }
    
    print("\n" + "=" * 50)
    print("Integration Test Session: Service Availability")
    print("=" * 50)
    for service_name, (host, port) in services.items():
        if _service_available(host, port):
            print(f"  ✓ {service_name} ({host}:{port})")
        else:
            print(f"  ⚠ {service_name} ({host}:{port}) - unavailable")
    _phys_gpu = os.getenv("CUDA_VISIBLE_DEVICES", "unset (all GPUs visible)")
    print(f"  GPU: CUDA_VISIBLE_DEVICES={_phys_gpu} (override: make test-e2e GPU_ID=<n>)")
    print("=" * 50 + "\n")


def _gc_cuda() -> None:
    """Reclaim GPU memory. Two gc passes break more reference cycles than one."""
    gc.collect()
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    except Exception:
        pass


def _drain_indexing_queue(timeout_s: int = 60) -> None:
    """Wait until all background indexing jobs reach a terminal state.

    When a test times out during indexing, the background thread spawned by
    asyncio.to_thread(self.index, ...) keeps running and holds model references.
    Waiting here before we attempt model deletion prevents silent GC failures
    where del/None-assignment has no effect because the thread still owns a ref.
    """
    try:
        from colette.httpjsonapi import http_json_api

        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            running = [
                s
                for s, st in http_json_api.indexing_status.items()
                if st in ("running", "queued")
            ]
            if not running:
                break
            time.sleep(1.0)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def release_test_resources(request):
    """Best-effort cleanup to limit cumulative memory usage across e2e tests."""

    # Pre-test: reclaim GPU memory from background threads that completed since
    # the previous test's teardown (e.g. a background indexing thread that was
    # still running when the previous test timed out and has since finished).
    if "integration" in request.keywords or "e2e" in request.keywords:
        _gc_cuda()

    yield

    # Keep cleanup focused on heavy lanes to avoid slowing unit/smoke paths.
    if "integration" not in request.keywords and "e2e" not in request.keywords:
        return

    # Wait for any in-progress background indexing to finish before we try to
    # free models.  Without this, del/None-assignment is a no-op because the
    # thread pool worker still holds a reference to the service/model objects.
    _drain_indexing_queue()

    try:
        from colette.httpjsonapi import http_json_api

        # Force-remove any leftover services even when a test exits early.
        for sname in list(http_json_api.services.keys()):
            try:
                http_json_api.remove_service(sname)
            except Exception:
                pass
    except Exception:
        pass

    try:
        from colette.backends.hf.model_cache import ModelCache

        ModelCache.clear()
    except Exception:
        pass

    _gc_cuda()

