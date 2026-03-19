import uuid
from pathlib import Path

import pytest

from colette.apidata import APIData, APIResponse, AppObj, InputConnectorObj, LLMModelObj, ParametersObj
from colette.inputconnector import InputConnector
from colette.llmlib import LLMLib
from colette.llmservice import createLLMService
from colette.outputconnector import OutputConnector


class _DummyInputConnector(InputConnector):
    def init(self, ad, kvstore=None):
        super().init(ad)
        self.kvstore = kvstore

    def transform(self, ad):
        return ad

    def get_data(self, ad):
        return super().get_data(ad)


class _DummyOutputConnector(OutputConnector):
    def init(self, ad):
        # Exercise abstract base method line coverage.
        OutputConnector.init(self, ad)
        self.out_ad = ad

    def finalize(self, ad):
        # Exercise abstract base method line coverage.
        OutputConnector.finalize(self, ad)
        return ad


class _DummyModel:
    def init(self, ad, kvstore=None):
        self.model_ad = ad
        self.kvstore = kvstore


class _DummyBaseLib(LLMLib):
    def __init__(self, inputc, outputc, llmmodel):
        super().__init__(inputc, outputc, llmmodel)
        self.initialized = False

    def init(self, ad, kvstore=None):
        self.initialized = True
        self.base_ad = ad
        self.base_kv = kvstore

    def create_index(self):
        return None

    def delete_index(self):
        return None

    def train(self):
        return None

    def status(self, ad=None):
        return APIResponse(message="ok")

    def predict(self, ad):
        return APIResponse(output="ok")

    def update_index(self, ad):
        return APIResponse(message="updated")


@pytest.mark.smoke
def test_inputconnector_get_data_filters_and_extensions(tmp_path):
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("hello", encoding="utf-8")
    (data_dir / "b.pdf").write_text("pdf", encoding="utf-8")
    (data_dir / "ignore.tmp").write_text("tmp", encoding="utf-8")

    ad = InputConnectorObj()
    ad.data = [Path(data_dir)]
    ad.preprocessing.files = ["txt", "pdf"]
    ad.preprocessing.filters = [r"ignore\\.tmp$"]

    conn = _DummyInputConnector()
    conn.logger = type("L", (), {"info": lambda *args, **kwargs: None})()
    conn.init(ad)
    conn.get_data(ad)

    assert any(str(data_dir / "a.txt") == p for p in conn.data)
    assert any(str(data_dir / "b.pdf") == p for p in conn.data)
    assert all("ignore.tmp" not in p for p in conn.data)


@pytest.mark.smoke
def test_llmlib_constructor_and_abstract_methods_are_reachable():
    lib = _DummyBaseLib(None, None, None)
    assert lib.inputc is None
    assert lib.ouputc is None
    assert lib.llmmodel is None


@pytest.mark.smoke
def test_llmservice_init_and_jobs(tmp_path):
    ServiceCls = createLLMService(_DummyBaseLib)

    sname = f"svc-{uuid.uuid4().hex[:8]}"
    service = ServiceCls(
        sname=sname,
        verbose="debug",
        inputc=_DummyInputConnector(),
        outputc=_DummyOutputConnector(),
        llmmodel=_DummyModel(),
    )

    ad = APIData(
        app=AppObj(repository=tmp_path, models_repository="models_rel", log_in_app_dir=False),
        parameters=ParametersObj(input=InputConnectorObj(), llm=LLMModelObj()),
    )

    service.init(ad)

    assert service.initialized is True
    assert service.inputc.app_repository == tmp_path
    assert service.outputc.app_repository == tmp_path
    assert service.llmmodel.app_repository == tmp_path
    assert service.models_repository == Path.cwd() / "models_rel"
    assert (tmp_path / "config.json").exists()
    assert (tmp_path / "kvstore.db").exists()

    job_out = service.index_job(APIData(), None)
    assert job_out.service_name == sname

    status_out = service.index_job_status(APIData(), None)
    assert status_out.service_name == sname

    pred_out = service.predict_job(APIData())
    assert pred_out.service_name == sname

    assert service.index_job_delete(APIData(), None) is None
    assert service.train_job(APIData(), None) is None
    assert service.train_job_status(APIData(), None) is None
    assert service.train_job_delete(APIData(), None) is None


@pytest.mark.smoke
def test_llmservice_init_with_absolute_models_path(tmp_path):
    ServiceCls = createLLMService(_DummyBaseLib)

    models_dir = tmp_path / "abs_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    service = ServiceCls(
        sname=f"svc-{uuid.uuid4().hex[:8]}",
        verbose="info",
        inputc=_DummyInputConnector(),
        outputc=_DummyOutputConnector(),
        llmmodel=_DummyModel(),
    )

    ad = APIData(
        app=AppObj(repository=tmp_path / "app2", models_repository=str(models_dir), log_in_app_dir=True),
        parameters=ParametersObj(input=InputConnectorObj(), llm=LLMModelObj()),
    )

    service.init(ad)
    assert service.models_repository == models_dir
    assert (Path(ad.app.repository) / "app.log").exists()


@pytest.mark.smoke
def test_llmservice_info_string():
    ServiceCls = createLLMService(_DummyBaseLib)
    service = ServiceCls(
        sname="svc-info",
        verbose="info",
        inputc=None,
        outputc=None,
        llmmodel=None,
    )
    service.kvstore = type("KV", (), {"close": lambda self: None})()
    assert "svc-info" in service.info()
