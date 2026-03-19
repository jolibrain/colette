import asyncio
import io
import sys
import types

import pytest

from colette.apidata import APIData, AppObj, InputConnectorObj, LLMInferenceObj, LLMModelObj, ParametersObj
from colette.inputconnector import InputConnectorBadParamException, InputConnectorInternalException
from colette.jsonapi import JSONApi
from colette.llmlib import LLMLibBadParamException, LLMLibInternalException
from colette.llmmodel import LLMModelBadParamException, LLMModelInternalException
from colette.services import ServiceBadParamException


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_service_upload_running_status_returns_error():
    api = JSONApi()
    api.indexing_status["svc"] = "running"

    out = await api.service_upload("svc", APIData(), files=None)

    assert out.status is not None
    assert out.status.code == 500
    assert out.status.colette_code == 1007


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_service_upload_enqueues_when_idle():
    api = JSONApi()

    out = await api.service_upload("svc", APIData(), files=None)

    assert "started" in (out.message or "")
    assert api.indexing_status.get("svc") == "queued"

    queued_sname, queued_ad = await asyncio.wait_for(api.indexing_queue.get(), timeout=1)
    assert queued_sname == "svc"
    assert isinstance(queued_ad, APIData)


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_service_index_async_running_and_idle_paths():
    api = JSONApi()

    api.indexing_status["svc"] = "running"
    out_running = await api.service_index_async("svc", APIData())
    assert out_running.status is not None
    assert out_running.status.code == 500

    api.indexing_status["svc"] = "finished"
    out_idle = await api.service_index_async("svc", APIData())
    assert "started" in (out_idle.message or "")


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_service_index_status_and_delete(monkeypatch):
    api = JSONApi()

    out = await api.service_index_status("svc")
    assert "finished" in (out.message or "")

    monkeypatch.setattr(api, "remove_service", lambda _sname: True)
    deleted = await api.service_delete("svc")
    assert deleted.status is not None
    assert deleted.status.code == 200


@pytest.mark.smoke
def test_service_streaming_exception_branch(monkeypatch):
    api = JSONApi()

    def _raise(_sname, _ad):
        raise RuntimeError("boom")

    monkeypatch.setattr(api, "streaming", _raise)
    assert api.service_streaming("svc", APIData()) is None


@pytest.mark.smoke
def test_service_predict_exception_mapping(monkeypatch):
    api = JSONApi()

    cases = [
        (ServiceBadParamException("x"), 404),
        (InputConnectorBadParamException("x"), 400),
        (InputConnectorInternalException("x"), 500),
        (LLMModelBadParamException("x"), 404),
        (LLMModelInternalException("x"), 500),
        (LLMLibBadParamException("x"), 400),
        (LLMLibInternalException("x"), 500),
    ]

    for exc, expected_code in cases:
        def _raise(_sname, _ad, exc=exc):
            raise exc

        monkeypatch.setattr(api, "predict", _raise)
        out = api.service_predict("svc", APIData())
        assert out.status is not None
        assert out.status.code == expected_code


@pytest.mark.smoke
def test_service_predict_success_sets_ok_status(monkeypatch):
    api = JSONApi()

    class _Resp:
        status = None

    monkeypatch.setattr(api, "predict", lambda _sname, _ad: _Resp())
    out = api.service_predict("svc", APIData())
    assert out.status is not None
    assert out.status.code == 200


def _install_fake_backend_modules(monkeypatch, prefix: str, symbols: dict):
    for suffix, cls_name in symbols.items():
        module_name = f"colette.backends.{prefix}.{suffix}"
        module = types.ModuleType(module_name)
        setattr(module, cls_name, type(cls_name, (), {}))
        monkeypatch.setitem(sys.modules, module_name, module)


def _make_ad(input_lib: str) -> APIData:
    return APIData(
        app=AppObj(),
        parameters=ParametersObj(
            input=InputConnectorObj(lib=input_lib),
            llm=LLMModelObj(inference=LLMInferenceObj(lib="huggingface")),
        ),
    )


@pytest.mark.smoke
def test_service_create_duplicate_returns_bad_request():
    api = JSONApi()
    api.services["svc"] = object()

    out = api.service_create("svc", _make_ad("hf"))

    assert out.status is not None
    assert out.status.code == 400


@pytest.mark.smoke
def test_service_create_langchain_hf_diffusr_paths(monkeypatch):
    api = JSONApi()

    _install_fake_backend_modules(
        monkeypatch,
        "langchain",
        {
            "langchaininputconn": "LangChainInputConn",
            "langchainlib": "LangChainLib",
            "langchainmodel": "LangChainModel",
            "langchainoutputconn": "LangChainOutputConn",
        },
    )
    _install_fake_backend_modules(
        monkeypatch,
        "hf",
        {
            "hfinputconn": "HFInputConn",
            "hflib": "HFLib",
            "hfmodel": "HFModel",
            "hfoutputconn": "HFOutputConn",
        },
    )
    _install_fake_backend_modules(
        monkeypatch,
        "diffusr",
        {
            "diffusrlib": "DiffusrLib",
            "diffusrmodel": "DiffusrModel",
        },
    )

    class _DummyService:
        def __init__(self, *args, **kwargs):
            self.inputc = None
            self.llmmodel = None

        def init(self, ad):
            return None

    monkeypatch.setattr("colette.jsonapi.createLLMService", lambda _base: _DummyService)
    monkeypatch.setattr(api, "add_service", lambda *_args, **_kwargs: None)

    out_langchain = api.service_create("svc-lc", _make_ad("langchain"))
    out_hf = api.service_create("svc-hf", _make_ad("hf"))
    out_diffusr = api.service_create("svc-df", _make_ad("diffusr"))

    assert out_langchain.service_name == "svc-lc"
    assert out_hf.service_name == "svc-hf"
    assert out_diffusr.service_name == "svc-df"


@pytest.mark.smoke
@pytest.mark.asyncio
async def test_service_upload_with_files_branch(monkeypatch, tmp_path):
    api = JSONApi()

    class _Service:
        app_repository = tmp_path

    class _Upload:
        filename = "doc.txt"

        def __init__(self):
            self.file = io.BytesIO(b"hello")

    monkeypatch.setattr(api, "get_service", lambda _sname: _Service())

    out = await api.service_upload("svc", APIData(), files=[_Upload()])

    assert "started" in (out.message or "")
    assert (tmp_path / "uploads" / "doc.txt").exists()
