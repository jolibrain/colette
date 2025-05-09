# Main API class
from . import logger
from .services import Services


class APIStrategy(Services):
    def __init__(self):
        super().__init__()
        self.logger_api = logger.get_colette_logger("api")
