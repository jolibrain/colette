import logging

import pytest

from colette.logger import CustomFormatter, get_colette_logger, get_level


@pytest.mark.smoke
def test_get_level_mappings():
    assert get_level("info") == logging.INFO
    assert get_level("warning") == logging.WARNING
    assert get_level("error") == logging.ERROR
    assert get_level("critical") == logging.CRITICAL
    assert get_level("debug") == logging.DEBUG
    assert get_level("unknown") == logging.INFO


@pytest.mark.smoke
def test_get_colette_logger_reuses_handlers():
    name = "colette-test-logger-smoke"
    logger = logging.getLogger(name)
    logger.handlers.clear()

    first = get_colette_logger(name, "debug")
    second = get_colette_logger(name, "debug")

    assert first is second
    assert len(first.handlers) == 1


@pytest.mark.smoke
def test_custom_formatter_formats_record():
    formatter = CustomFormatter()
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=42,
        msg="hello",
        args=(),
        exc_info=None,
    )
    rendered = formatter.format(record)
    assert "hello" in rendered
    assert "INFO" in rendered
