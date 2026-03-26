import os

from utils import pretty_print_response, wait_for_index_status

models_repo = os.getenv("MODELS_REPO", "models")


def generic_index(client, sname, index_json, timeout_s=180):
    response = client.put(f"/v1/index/{sname}", json=index_json)
    assert response.status_code == 200
    response = client.get(f"/v1/index/{sname}/status")
    pretty_print_response(response.json())
    assert response.status_code == 200

    response = wait_for_index_status(client, sname, poll_interval_s=2, timeout_s=timeout_s)
    pretty_print_response(response.json())
    return response
