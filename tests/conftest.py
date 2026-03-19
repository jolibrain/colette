import os
import sys
import pytest

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
