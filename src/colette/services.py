# TODO: service custom exceptions

"""
Class for LLM services.
Contains and manages the list of running LLM services.
Each service instanciates an LLM model and associated input and output connectors.
"""

from typing import Any

from .apidata import APIData, APIResponse


class ServiceBadParamException(Exception):
    pass


class Services:
    def __init__(self):
        self.services = {}

    def add_service(self, service: Any, sname: str, ad: APIData):
        service.init(ad)
        self.services[sname] = service

    def get_service(self, sname: str):
        if sname in self.services:
            return self.services[sname]
        return None

    def list_services(self):
        return self.services.keys()

    def remove_service(self, sname: str):
        if sname in self.services:
            serv = self.services[sname]
            if serv.inputc:
                serv.inputc.delete_inputc()
            if serv.llmmodel:
                serv.llmmodel.delete_model()
            del serv
            del self.services[sname]
            return True
        return False

    def service_exists(self, sname: str):
        if sname in self.services:
            return True
        return False

    def index(self, sname: str, ad: APIData):
        service = self.get_service(sname)
        try:
            if service is not None:
                response = service.index_job(ad, None)
            else:
                raise ServiceBadParamException(f"service {sname} not found")
            return response
        except Exception as e:
            raise e

    def index_status(self, sname: str, ad: APIData):
        service = self.get_service(sname)
        if service is not None:
            response = service.index_job_status(ad, None)
        else:
            raise ServiceBadParamException(f"service {sname} not found")
        return response

    def index_delete(self, sname: str):
        ##TODO
        return

    def train(self, sname: str):
        ##TODO
        return

    def train_status(self, sname: str):
        ##TODO
        return

    def train_delete(self, sname: str):
        ##TODO
        return

    def predict(self, sname, ad: APIData) -> APIResponse:
        ##TODO: call on llmservice predict_job
        service = self.get_service(sname)
        if service is not None:
            response = service.predict(ad)
        else:
            raise ServiceBadParamException(f"service {sname} not found")
        return response

    def streaming(self, sname, ad: APIData) -> str:
        service = self.get_service(sname)
        if service is not None:
            streamer = service.streaming(ad)
        else:
            raise ServiceBadParamException(f"service {sname} not found")
        return streamer
