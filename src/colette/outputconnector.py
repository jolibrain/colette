# abstract output connector class

from abc import abstractmethod

from .apidata import APIData


class OutputConnector:
    def __init__(self):
        pass

    @abstractmethod
    def init(self, ad: APIData):
        pass

    @abstractmethod
    def finalize(self, ad: APIData):
        pass
