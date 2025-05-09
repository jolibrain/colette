import json
import logging
import os
from pathlib import Path
from typing import Any

from . import logger
from .apidata import APIData, APIResponse, VerboseEnum
from .inputconnector import InputConnector
from .kvstore import ImageStorageFactory
from .llmmodel import LLMModel
from .outputconnector import OutputConnector

## LLM service, managed by services, derives an LLM lib instance


def createLLMService(BaseLLMLib):
    class LLMService(BaseLLMLib):
        def __init__(
            self,
            sname: str,
            verbose: VerboseEnum,
            inputc: InputConnector,
            outputc: OutputConnector,
            llmmodel: LLMModel,
        ):
            super().__init__(inputc, outputc, llmmodel)
            self.sname = sname
            self.logger = logger.get_colette_logger(sname, verbose)
            self.inputc = inputc
            self.outputc = outputc
            self.llmmodel = llmmodel

            # set logger to connectors and model
            if self.inputc:
                self.inputc.logger = self.logger
            if self.outputc:
                self.outputc.logger = self.logger
            if self.llmmodel:
                self.llmmodel.logger = self.logger

        def __del__(self):
            self.kvstore.close()
            del self.logger
            del self.inputc
            del self.outputc
            del self.llmmodel

        def init(self, ad: APIData):
            # application directory
            self.app_repository = ad.app.repository  # app location
            Path(self.app_repository).mkdir(parents=True, exist_ok=True)
            self.logger.info(f"Application directory: {self.app_repository}")

            # add a file handler to the logger
            if ad.app.log_in_app_dir:
                file_handler = logging.FileHandler(f"{str(self.app_repository)}/app.log")
                file_handler.setFormatter(self.logger.handlers[-1].formatter)
                self.logger.addHandler(file_handler)

            # store ad in the app repository
            with open(f"{self.app_repository}/config.json", "w") as f:
                json.dump(ad.model_dump(), f, indent=4)

            if self.inputc:
                self.inputc.app_repository = self.app_repository

            # models directory, if any
            if ad.app.models_repository:
                if os.path.isabs(ad.app.models_repository):
                    self.models_repository = Path(ad.app.models_repository)
                else:
                    self.models_repository = Path(os.getcwd()) / ad.app.models_repository
            else:
                self.models_repository = self.app_repository / "models"
            Path(self.models_repository).mkdir(parents=True, exist_ok=True)
            if self.inputc:
                self.inputc.models_repository = self.models_repository

            if self.outputc:
                self.outputc.app_repository = self.app_repository
                self.outputc.models_repository = self.models_repository

            if self.llmmodel:
                self.llmmodel.app_repository = self.app_repository
                self.llmmodel.models_repository = self.models_repository

            self.kvstore = ImageStorageFactory.create_storage("hdf5", self.app_repository / "kvstore.db", "a")
            params = ad.parameters
            input_params = params.input
            output_params = params.output
            model_params = params.llm
            if self.inputc:
                self.inputc.init(input_params, self.kvstore)
            if self.outputc:
                self.outputc.init(output_params)
            if self.llmmodel:
                self.llmmodel.init(model_params, self.kvstore)
            super().init(ad, self.kvstore)  # llmlib is initialized last

        def info(self) -> dict[str, Any]:
            ##TODO
            return f"{self.sname} service, managed by services, derives an LLM lib instance"

        def index_job(self, ad: APIData, out: APIData) -> int:
            try:
                response = self.update_index(ad)
                response.service_name = self.sname
                return response
            except Exception as e:
                raise e

        def index_job_status(self, ad: APIData, out: APIData) -> int:
            response = self.status(ad)
            response.service_name = self.sname
            return response

        def index_job_delete(self, ad: APIData, out: APIData) -> int:
            ##TODO
            pass

        def train_job(self, ad: APIData, out: APIData) -> int:
            ##TODO
            pass

        def train_job_status(self, ad: APIData, out: APIData) -> int:
            ##TODO
            pass

        def train_job_delete(self, ad: APIData, out: APIData) -> int:
            ##TODO
            pass

        def predict_job(self, ad: APIData) -> APIResponse:
            response = self.predict(ad)
            response.service_name = self.sname
            return response

    return LLMService
