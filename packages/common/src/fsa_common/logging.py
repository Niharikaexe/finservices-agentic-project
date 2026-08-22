"""Structured logging.

structlog, configured once per process. Two renderers: key=value for a human at a
terminal, JSON for anything a log shipper will read. The processor chain also stamps
every event with the OTel trace/span id when one is active, which is what lets you
jump from a Grafana alert to the exact Phoenix trace.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Call once, at process start, before anything logs."""
    logging.basicConfig(format="%(message)s", stream=sys.stdout, level=level.upper())

    processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]
    processors.append(
        structlog.processors.JSONRenderer() if json_output else structlog.dev.ConsoleRenderer()
    )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """`log = get_logger(__name__)` at module scope; bind per-request fields at use."""
    return structlog.get_logger(name)  # type: ignore[no-any-return]
