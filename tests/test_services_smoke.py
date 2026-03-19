import pytest

from colette.apistrategy import APIStrategy
from colette.llmmodel import LLMModel
from colette.services import ServiceBadParamException, Services


class _DummyInputConnector:
    def __init__(self):
        self.deleted = False

    def delete_inputc(self):
        self.deleted = True


class _DummyModel:
    def __init__(self):
        self.deleted = False

    def delete_model(self):
        self.deleted = True


class _DummyService:
    def __init__(self):
        self.inputc = _DummyInputConnector()
        self.llmmodel = _DummyModel()
        self.initialized_with = None

    def init(self, ad):
        self.initialized_with = ad

    def index_job(self, ad, _out):
        return {"indexed": True, "ad": ad}

    def index_job_status(self, ad, _out):
        return {"status": "ok", "ad": ad}

    def predict(self, ad):
        return {"predicted": True, "ad": ad}

    def streaming(self, ad):
        return f"stream::{ad}"


@pytest.mark.smoke
def test_services_lifecycle_and_dispatch():
    services = Services()
    dummy = _DummyService()

    services.add_service(dummy, "svc", ad={"cfg": 1})

    assert services.service_exists("svc") is True
    assert services.get_service("svc") is dummy
    assert list(services.list_services()) == ["svc"]
    assert dummy.initialized_with == {"cfg": 1}

    assert services.index("svc", ad={"idx": 1})["indexed"] is True
    assert services.index_status("svc", ad={"idx": 2})["status"] == "ok"
    assert services.predict("svc", ad={"msg": "hi"})["predicted"] is True
    assert services.streaming("svc", ad="abc") == "stream::abc"

    assert services.remove_service("svc") is True
    assert services.service_exists("svc") is False
    assert services.remove_service("svc") is False


@pytest.mark.smoke
def test_services_missing_service_raises():
    services = Services()

    with pytest.raises(ServiceBadParamException):
        services.index("missing", ad=None)

    with pytest.raises(ServiceBadParamException):
        services.index_status("missing", ad=None)

    with pytest.raises(ServiceBadParamException):
        services.predict("missing", ad=None)

    with pytest.raises(ServiceBadParamException):
        services.streaming("missing", ad=None)


@pytest.mark.smoke
def test_apistrategy_initializes_logger_and_registry():
    api = APIStrategy()
    assert hasattr(api, "logger_api")
    assert isinstance(api.services, dict)


@pytest.mark.smoke
def test_llmmodel_init_with_and_without_inference_lib():
    model = LLMModel()

    class Inference:
        lib = "ollama"

    class Good:
        source = "qwen2.5:0.5b"
        inference = Inference()

    class MissingInference:
        source = "none"

    model.init(Good())
    assert model.llm_lib == "ollama"
    assert model.llm_source == "qwen2.5:0.5b"

    model.init(MissingInference())
    assert model.llm_lib is None
    assert model.llm_source is None
