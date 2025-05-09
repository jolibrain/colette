# Langchain as output connector

from colette.apidata import OutputConnectorObj
from colette.outputconnector import OutputConnector


class LangChainOutputConn(OutputConnector):
    def __init__(self):
        super().__init__()

    def init(self, ad: OutputConnectorObj):
        ##TODO
        super().init(ad)
        pass
