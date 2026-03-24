import pytest

pytestmark = [pytest.mark.integration, pytest.mark.e2e]


def test_info(client):
    response = client.get("/v1/info")
    assert response.status_code == 200
    assert "commit" in response.json()["version"]
