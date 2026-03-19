import json
from unittest.mock import mock_open, patch

import pytest
from fastapi import Response

from colette.apidata import APIData, APIResponse, ChatCompletionRequest, ChatMessage, StatusObj
from colette.httpjsonapi import HTTPJsonApi, set_response_http_status
from colette.openwebuiapi import NAME_PREFIX, OpenWebUIApi


@pytest.mark.smoke
def test_set_response_http_status_updates_http_response():
    response = Response()
    output = APIResponse(status=StatusObj(code=201, status="Created"))

    set_response_http_status(response, output)

    assert response.status_code == 201


@pytest.mark.smoke
def test_http_info_adds_version(monkeypatch):
    api = HTTPJsonApi()
    monkeypatch.setattr(api, "service_info", lambda: APIResponse(service_name="svc"))

    output = api.info()

    assert output.service_name == "svc"
    assert output.version is not None


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_delete_predict_index_and_streaming(monkeypatch):
    api = HTTPJsonApi()
    response = Response()

    async def _delete(_sname):
        return APIResponse(service_name="deleted")

    monkeypatch.setattr(api, "service_delete", _delete)
    monkeypatch.setattr(api, "service_predict", lambda _sname, _ad: APIResponse(status=StatusObj(code=200), output="ok"))
    monkeypatch.setattr(api, "service_streaming", lambda _sname, _ad: iter([b"chunk"]))

    async def _index(_sname, _ad):
        return APIResponse(status=StatusObj(code=202), service_name="indexed")

    monkeypatch.setattr(api, "service_index_async", _index)

    deleted = await api.delete_service("svc")
    predicted = await api.predict_service("svc", APIData(), response)
    stream = api.streaming_service("svc", APIData())
    indexed = await api.index_service("svc", APIData(), response)

    assert deleted.service_name == "deleted"
    assert predicted.output == "ok"
    assert stream is not None
    assert indexed.service_name == "indexed"
    assert response.status_code == 202


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_upload_parses_json_and_files(monkeypatch, tmp_path):
    api = HTTPJsonApi()
    response = Response()

    class _Service:
        app_repository = tmp_path

    monkeypatch.setattr(api, "get_service", lambda _sname: _Service())

    async def _upload(_sname, _api_data, _files):
        return APIResponse(status=StatusObj(code=200), service_name="uploaded")

    monkeypatch.setattr(api, "service_upload", _upload)

    ad_payload = json.dumps(APIData().model_dump())
    output = await api.upload_service("svc", ad=ad_payload, files=None, response=response)

    assert output.service_name == "uploaded"
    assert response.status_code == 200


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_upload_invalid_json_returns_400():
    api = HTTPJsonApi()
    response = Response()

    output = await api.upload_service("svc", ad="{bad-json", files=None, response=response)

    assert output.status is not None
    assert output.status.code == 400
    assert response.status_code == 400


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_http_create_and_index_status(monkeypatch):
    api = HTTPJsonApi()

    monkeypatch.setattr(api, "service_create", lambda _sname, _ad: APIResponse(service_name="created"))

    async def _index_status(_sname):
        return APIResponse(service_name="status")

    monkeypatch.setattr(api, "service_index_status", _index_status)

    created = await api.create_service("svc", APIData())
    status = await api.index_status("svc")

    assert created.service_name == "created"
    assert status.service_name == "status"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openwebui_get_models(monkeypatch):
    api = OpenWebUIApi()
    monkeypatch.setattr(api, "list_services", lambda: ["a", "b"])

    result = await api.get_models()

    ids = [item["id"] for item in result["data"]]
    assert result["object"] == "list"
    assert f"{NAME_PREFIX}a" in ids
    assert f"{NAME_PREFIX}b" in ids


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_openwebui_chat_completion_formats_sources(monkeypatch):
    api = OpenWebUIApi()

    async def _predict(_service, _api_data, _response):
        return APIResponse(
            output="answer",
            sources={
                "context": [
                    {"distance": 0.9, "source": "a.pdf", "content": "http://img/a"},
                    {"distance": 0.2, "source": "b.pdf", "content": "http://img/b"},
                ]
            },
        )

    monkeypatch.setattr(api, "predict_service", _predict)

    req = ChatCompletionRequest(
        model=f"{NAME_PREFIX}svc",
        messages=[ChatMessage(role="user", content="hello")],
    )

    message_template = {"parameters": {"input": {"message": ""}}}
    with patch("builtins.open", mock_open(read_data="{}")), patch("json.load", return_value=message_template):
        out = await api.chat_completion(req, Response())

    assert out.model == f"{NAME_PREFIX}svc"
    assert out.choices[0].message.role == "assistant"
    assert "answer" in out.choices[0].message.content
    assert "a.pdf" in out.choices[0].message.content
