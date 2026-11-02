"""Structured, human readable logging configuration.

The same configuration is reused by the API process and by Celery workers so
that logs from every layer share one format.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

from app.core.config import settings

LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)s | "
    "%(funcName)s:%(lineno)d | %(message)s"
)
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_CONFIGURED = False


class RequestContextFilter(logging.Filter):
    """Injects the request id / user id into every record when available."""

    def filter(self, record: logging.LogRecord) -> bool:  # noqa: D102
        if not hasattr(record, "request_id"):
            record.request_id = "-"
        if not hasattr(record, "user_id"):
            record.user_id = "-"
        return True


def configure_logging(level: str | None = None, force: bool = False) -> None:
    """Configure root logging once per process."""
    global _CONFIGURED  # noqa: PLW0603
    if _CONFIGURED and not force:
        return

    resolved_level = (level or settings.log_level).upper()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    handler.addFilter(RequestContextFilter())

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(resolved_level)

    # Third-party loggers are noisy at DEBUG level.
    for noisy in ("httpx", "httpcore", "urllib3", "chromadb", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _CONFIGURED = True


def get_logger(name: str) -> logging.Logger:
    """Return a configured logger for ``name``."""
    configure_logging()
    return logging.getLogger(name)


def log_extra(**kwargs: Any) -> dict[str, Any]:
    """Build the ``extra`` mapping understood by :class:`RequestContextFilter`."""
    return {key: value for key, value in kwargs.items() if value is not None}
