# Enhanced logger for Colette

import logging

from .apidata import VerboseEnum


class CustomFormatter(logging.Formatter):
    # color table: https://talyian.github.io/ansicolors/

    green = "\x1b[38;5;10m"
    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    orange = "\x1b[38;5;208m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    log_format = (
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"
    )

    FORMATS = {
        logging.DEBUG: orange + log_format + reset,
        logging.INFO: green + log_format + reset,
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset,
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def get_level(level: VerboseEnum):
    if level == "info":
        return logging.INFO
    elif level == "warning":
        return logging.WARNING
    elif level == "error":
        return logging.ERROR
    elif level == "critical":
        return logging.CRITICAL
    elif level == "debug":
        return logging.DEBUG
    else:
        return logging.INFO


def get_colette_logger(name, level: VerboseEnum = VerboseEnum.info):
    llevel = get_level(level)
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.propagate = False
        logger.setLevel(llevel)
        # Set up console handler
        ch = logging.StreamHandler()
        ch.setLevel(llevel)
        ch.setFormatter(CustomFormatter())
        logger.addHandler(ch)
    return logger
