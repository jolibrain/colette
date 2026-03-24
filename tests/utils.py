import socket
import shutil
import time
from copy import deepcopy
from pprint import pprint


def pretty_print_response(response: dict):
    truncated_response = deepcopy(response)
    nodes = [truncated_response]

    while nodes:
        current = nodes.pop()
        if isinstance(current, dict):
            for key, value in current.items():
                if isinstance(value, str) and value.startswith("data:image/"):
                    current[key] = value[:45]
                elif isinstance(value, (dict, list)):
                    nodes.append(value)
        elif isinstance(current, list):
            for i in range(len(current)):
                item = current[i]
                if isinstance(item, str) and item.startswith("data:image/"):
                    current[i] = item[:45]
                elif isinstance(item, (dict, list)):
                    nodes.append(item)

    pprint(truncated_response)


def compare_dicts(dict1: dict, dict2:dict, path:str=""):
    differences = []

    for key in dict1:
        current_path = f"{path}/{key}" if path else key

        if key not in dict2:
            differences.append(f"Key '{current_path}' found in 'dict1' but missing in 'dict2'.")
        else:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                differences.extend(compare_dicts(dict1[key], dict2[key], current_path))
            elif dict1[key] != dict2[key]:
                differences.append(
                    f"Value mismatch at '{current_path}': 'dict1' has {dict1[key]}, 'dict2' has {dict2[key]}"
                )

    return differences


def ensure_service_deleted(client, sname: str):
    """Best-effort service cleanup to avoid cross-test leakage."""
    try:
        client.delete(f"/v1/app/{sname}")
    except Exception:
        pass


def ensure_repo_deleted(repo_path):
    """Best-effort repository cleanup for test artifacts."""
    shutil.rmtree(repo_path, ignore_errors=True)


def wait_for_index_status(
    client,
    sname: str,
    terminal_tokens=("finished",),
    in_progress_tokens=("queued", "running", "started", "indexing"),
    poll_interval_s: float = 0.5,
    timeout_s: float = 180,
):
    """Poll index status endpoint until one of the terminal tokens is found."""
    response = client.get(f"/v1/index/{sname}/status")
    assert response.status_code == 200

    deadline = time.time() + timeout_s
    message = (response.json().get("message") or "").lower()
    while not any(token in message for token in terminal_tokens):
        if not any(token in message for token in in_progress_tokens):
            return response
        if time.time() >= deadline:
            raise AssertionError(
                f"Timed out waiting for index status for '{sname}'. Last message: {message}"
            )
        time.sleep(poll_interval_s)
        response = client.get(f"/v1/index/{sname}/status")
        assert response.status_code == 200
        message = (response.json().get("message") or "").lower()
    return response


def wait_for_tcp_service(
    host: str,
    port: int,
    timeout_s: float = 180,
    poll_interval_s: float = 1.0,
    process=None,
):
    """Wait until a TCP service is reachable or fail on timeout/process exit."""
    deadline = time.time() + timeout_s
    last_error = "unknown"

    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise AssertionError(
                f"Process exited before service became ready on {host}:{port}. Exit code: {process.returncode}"
            )
        try:
            with socket.create_connection((host, port), timeout=2):
                return True
        except OSError as exc:
            last_error = str(exc)
            time.sleep(poll_interval_s)

    raise AssertionError(
        f"Timed out waiting for TCP service on {host}:{port}. Last error: {last_error}"
    )
