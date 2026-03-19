import json
import sys
import types

import pytest

from colette import colette_cli


class _DummyResponse:
    def __init__(self, status_code=200, payload=None, text=""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self):
        return self._payload


class _DummyClient:
    def __init__(self, app):
        self._status_calls = 0

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def put(self, url, json=None):
        if "/v1/app/" in url:
            return _DummyResponse(200, {"service_name": "ok"})
        if "/v1/index/" in url:
            return _DummyResponse(200, {"message": "indexing"})
        return _DummyResponse(200)

    def get(self, url):
        if "/status" in url:
            self._status_calls += 1
            if self._status_calls == 1:
                return _DummyResponse(200, {"message": "running"})
            return _DummyResponse(200, {"message": "finished"})
        return _DummyResponse(200, {"info": {"services": []}})

    def post(self, url, json=None):
        return _DummyResponse(200, {"output": "hello"})

    def delete(self, url):
        return _DummyResponse(200)


@pytest.mark.smoke
def test_cli_index_happy_path(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    index_file = tmp_path / "index.json"
    data_dir = tmp_path / "data"
    app_dir = tmp_path / "app"

    data_dir.mkdir()
    app_dir.mkdir()

    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")
    index_file.write_text(json.dumps({"parameters": {"input": {"data": []}}}), encoding="utf-8")

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClient)
    monkeypatch.setattr(colette_cli.time, "sleep", lambda _x: None)

    colette_cli.index(app_dir=app_dir, data_dir=data_dir, models_dir=tmp_path / "models", config_file=config_file, index_file=index_file)


@pytest.mark.smoke
def test_cli_chat_happy_path(monkeypatch, tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    config_file = app_dir / "config.json"
    config_file.write_text(
        json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClient)

    colette_cli.chat(app_dir=app_dir, msg="hello", crop_label=None, models_dir=tmp_path / "models")


@pytest.mark.smoke
def test_cli_ui_happy_path(monkeypatch):
    class _DummyGradioApp:
        def launch(self, server_name, server_port):
            return None

    dummy_module = types.SimpleNamespace(create_gradio_interface=lambda _config: _DummyGradioApp())
    monkeypatch.setitem(sys.modules, "colette.ui.app", dummy_module)

    colette_cli.ui(host="127.0.0.1", port=9999, config=None)
