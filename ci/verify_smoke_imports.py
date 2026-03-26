#!/usr/bin/env python3
"""Validate smoke test imports and report missing modules in one pass.

This script imports a curated set of smoke test files directly from disk.
If import-time dependencies are missing, it prints all missing module names
and exits non-zero so CI fails early with actionable output.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import traceback
from pathlib import Path


SMOKE_TEST_FILES = [
    "tests/test_base_ci.py",
    "tests/test_embedding_loader.py",
    "tests/test_embedding_integration.py",
    "tests/test_services_smoke.py",
    "tests/test_http_openwebui_smoke.py",
    "tests/test_cli_smoke.py",
    "tests/test_jsonapi_helpers_smoke.py",
    "tests/test_kvstore_smoke.py",
    "tests/test_logger_smoke.py",
    "tests/test_jsonapi_service_smoke.py",
    "tests/test_core_services_smoke.py",
]


def import_file(module_name: str, file_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to create import spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", required=True, help="Repository root path")
    args = parser.parse_args()

    workspace = Path(args.workspace).resolve()

    # Match pytest collection import context for test-local helpers.
    # Several smoke tests import from tests/utils.py as `import utils`.
    repo_src = workspace / "src"
    tests_dir = workspace / "tests"
    for p in (workspace, repo_src, tests_dir):
        p_str = str(p)
        if p.exists() and p_str not in sys.path:
            sys.path.insert(0, p_str)

    missing_modules: set[str] = set()
    failures: list[str] = []

    for idx, rel_path in enumerate(SMOKE_TEST_FILES):
        abs_path = workspace / rel_path
        if not abs_path.exists():
            failures.append(f"{rel_path}: file not found")
            continue

        try:
            import_file(f"_ci_smoke_import_{idx}", abs_path)
        except ModuleNotFoundError as exc:
            missing = (exc.name or "<unknown>").split(".")[0]
            missing_modules.add(missing)
            failures.append(f"{rel_path}: missing module '{exc.name}'")
        except Exception:
            tb = traceback.format_exc(limit=2)
            failures.append(f"{rel_path}: import failed\\n{tb}")

    if failures:
        print("Smoke import preflight found issues:", file=sys.stderr)
        for failure in failures:
            print(f"  - {failure}", file=sys.stderr)

    if missing_modules:
        mods = ", ".join(sorted(missing_modules))
        print("", file=sys.stderr)
        print(f"Missing Python modules detected: {mods}", file=sys.stderr)
        print(
            "Update ci/requirements-smoke.txt with the provider package(s), then rerun setup.",
            file=sys.stderr,
        )
        return 1

    if failures:
        return 1

    print("Smoke import preflight passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
