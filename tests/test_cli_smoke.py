import json
import sys
import types

import pytest
import typer

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


class _DummyClientCreateFail(_DummyClient):
    def put(self, url, json=None):
        if "/v1/app/" in url:
            return _DummyResponse(500, text="create-fail")
        return super().put(url, json)


class _DummyClientIndexFail(_DummyClient):
    def put(self, url, json=None):
        if "/v1/index/" in url:
            return _DummyResponse(500, text="index-fail")
        return super().put(url, json)


class _DummyClientStatusFail(_DummyClient):
    def get(self, url):
        if "/status" in url:
            return _DummyResponse(500, {"message": "running"}, text="status-fail")
        return super().get(url)


class _DummyClientDeleteFail(_DummyClient):
    def delete(self, url):
        return _DummyResponse(500, text="delete-fail")


class _DummyClientChatFail(_DummyClient):
    def post(self, url, json=None):
        return _DummyResponse(500, text="chat-fail")


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


@pytest.mark.smoke
def test_cli_index_fails_on_bad_config(monkeypatch, tmp_path):
    index_file = tmp_path / "index.json"
    data_dir = tmp_path / "data"
    app_dir = tmp_path / "app"
    data_dir.mkdir()
    app_dir.mkdir()
    index_file.write_text(json.dumps({"parameters": {"input": {"data": []}}}), encoding="utf-8")

    with pytest.raises(typer.Exit):
        colette_cli.index(
            app_dir=app_dir,
            data_dir=data_dir,
            models_dir=tmp_path / "models",
            config_file=tmp_path / "missing-config.json",
            index_file=index_file,
        )


@pytest.mark.smoke
def test_cli_index_fails_on_bad_index_file(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    data_dir = tmp_path / "data"
    app_dir = tmp_path / "app"
    data_dir.mkdir()
    app_dir.mkdir()
    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")

    with pytest.raises(typer.Exit):
        colette_cli.index(
            app_dir=app_dir,
            data_dir=data_dir,
            models_dir=tmp_path / "models",
            config_file=config_file,
            index_file=tmp_path / "missing-index.json",
        )


@pytest.mark.smoke
def test_cli_index_fails_on_create(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    index_file = tmp_path / "index.json"
    data_dir = tmp_path / "data"
    app_dir = tmp_path / "app"
    data_dir.mkdir()
    app_dir.mkdir()
    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")
    index_file.write_text(json.dumps({"parameters": {"input": {"data": []}}}), encoding="utf-8")

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClientCreateFail)
    with pytest.raises(typer.Exit):
        colette_cli.index(app_dir=app_dir, data_dir=data_dir, models_dir=tmp_path / "models", config_file=config_file, index_file=index_file)


@pytest.mark.smoke
def test_cli_index_fails_on_index_launch(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    index_file = tmp_path / "index.json"
    data_dir = tmp_path / "data"
    app_dir = tmp_path / "app"
    data_dir.mkdir()
    app_dir.mkdir()
    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")
    index_file.write_text(json.dumps({"parameters": {"input": {"data": []}}}), encoding="utf-8")

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClientIndexFail)
    with pytest.raises(typer.Exit):
        colette_cli.index(app_dir=app_dir, data_dir=data_dir, models_dir=tmp_path / "models", config_file=config_file, index_file=index_file)


@pytest.mark.smoke
def test_cli_index_fails_on_status_poll(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    index_file = tmp_path / "index.json"
    data_dir = tmp_path / "data"
    app_dir = tmp_path / "app"
    data_dir.mkdir()
    app_dir.mkdir()
    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")
    index_file.write_text(json.dumps({"parameters": {"input": {"data": []}}}), encoding="utf-8")

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClientStatusFail)
    with pytest.raises(typer.Exit):
        colette_cli.index(app_dir=app_dir, data_dir=data_dir, models_dir=tmp_path / "models", config_file=config_file, index_file=index_file)


@pytest.mark.smoke
def test_cli_index_fails_on_delete(monkeypatch, tmp_path):
    config_file = tmp_path / "config.json"
    index_file = tmp_path / "index.json"
    data_dir = tmp_path / "data"
    app_dir = tmp_path / "app"
    data_dir.mkdir()
    app_dir.mkdir()
    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")
    index_file.write_text(json.dumps({"parameters": {"input": {"data": []}}}), encoding="utf-8")

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClientDeleteFail)
    with pytest.raises(typer.Exit):
        colette_cli.index(app_dir=app_dir, data_dir=data_dir, models_dir=tmp_path / "models", config_file=config_file, index_file=index_file)


@pytest.mark.smoke
def test_cli_chat_fails_on_missing_config(tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    with pytest.raises(typer.Exit):
        colette_cli.chat(app_dir=app_dir, msg="hello", crop_label=None, models_dir=tmp_path / "models")


@pytest.mark.smoke
def test_cli_chat_fails_on_init(monkeypatch, tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    config_file = app_dir / "config.json"
    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClientCreateFail)
    with pytest.raises(typer.Exit):
        colette_cli.chat(app_dir=app_dir, msg="hello", crop_label=None, models_dir=tmp_path / "models")


@pytest.mark.smoke
def test_cli_chat_fails_on_predict(monkeypatch, tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    config_file = app_dir / "config.json"
    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClientChatFail)
    with pytest.raises(typer.Exit):
        colette_cli.chat(app_dir=app_dir, msg="hello", crop_label=None, models_dir=tmp_path / "models")


@pytest.mark.smoke
def test_cli_chat_fails_on_delete(monkeypatch, tmp_path):
    app_dir = tmp_path / "app"
    app_dir.mkdir()
    config_file = app_dir / "config.json"
    config_file.write_text(json.dumps({"app": {}, "parameters": {"input": {"rag": {}}, "llm": None}}), encoding="utf-8")

    monkeypatch.setattr(colette_cli, "TestClient", _DummyClientDeleteFail)
    with pytest.raises(typer.Exit):
        colette_cli.chat(app_dir=app_dir, msg="hello", crop_label=None, models_dir=tmp_path / "models")


@pytest.mark.smoke
def test_cli_ui_import_error(monkeypatch):
    monkeypatch.delitem(sys.modules, "colette.ui.app", raising=False)

    original_import = __import__

    def _raise_import(name, *args, **kwargs):
        if name == "colette.ui.app":
            raise ImportError("missing")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", _raise_import)
    with pytest.raises(typer.Exit):
        colette_cli.ui(host="127.0.0.1", port=9999, config=None)


@pytest.mark.smoke
def test_cli_ui_runtime_error(monkeypatch):
    class _BadGradioApp:
        def launch(self, server_name, server_port):
            raise RuntimeError("boom")

    dummy_module = types.SimpleNamespace(create_gradio_interface=lambda _config: _BadGradioApp())
    monkeypatch.setitem(sys.modules, "colette.ui.app", dummy_module)

    with pytest.raises(typer.Exit):
        colette_cli.ui(host="127.0.0.1", port=9999, config=None)
