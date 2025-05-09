# LLM library class, used in every LLMService, managed by Services,
# and derived by every backend lib.

from abc import abstractmethod

from .apidata import APIData, APIResponse
from .inputconnector import InputConnector
from .llmmodel import LLMModel
from .outputconnector import OutputConnector


class LLMLibBadParamException(Exception):
    pass


class LLMLibInternalException(Exception):
    pass


class LLMLib:
    def __init__(
        self, inputc: InputConnector, ouputc: OutputConnector, llmmodel: LLMModel
    ):
        self.inputc = inputc
        self.ouputc = ouputc
        self.llmmodel = llmmodel
        pass

    ##TODO: init
    @abstractmethod
    def init(self, ad: APIData):
        pass

    @abstractmethod
    def create_index(self):
        pass

    @abstractmethod
    def delete_index(self):
        pass

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def status(self):
        pass

    @abstractmethod
    def predict(self, ad: APIData) -> APIResponse:
        pass
