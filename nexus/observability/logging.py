"""
nexus/observability/logging.py
-------------------------------
Structured logging using structlog + rich.
Every log line is a JSON-serialisable dict in production
and a beautiful coloured human-readable line in development.

Call setup_logging() once at startup before anything else.
"""

from __future__ import annotations

import logging
import sys

import structlog
from rich.console import Console
from rich.logging import RichHandler

from nexus.config.settings import get_settings


def setup_logging() -> None:
    """
    Configure structlog + standard library logging.
    Must be called once at application startup.
    """
    cfg = get_settings().observability
    level = getattr(logging, cfg.log_level, logging.INFO)
    is_dev = get_settings().environment == "development"

    # --- Standard library root logger ---
    if is_dev:
        # Rich coloured output in dev
        logging.basicConfig(
            level=level,
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(console=Console(stderr=True), rich_tracebacks=True)],
        )
    else:
        # Plain JSON-line logs in production (pipe to log aggregator)
        logging.basicConfig(
            level=level,
            format="%(message)s",
            stream=sys.stdout,
        )

    # Silence overly verbose third-party loggers
    for noisy in ("neo4j", "httpx", "httpcore", "urllib3", "asyncio"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    # --- structlog chain ---
    shared_processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.stdlib.add_logger_name,
    ]

    if is_dev:
        structlog.configure(
            processors=shared_processors
            + [
                structlog.dev.ConsoleRenderer(colors=True),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
    else:
        structlog.configure(
            processors=shared_processors
            + [
                structlog.processors.dict_tracebacks,
                structlog.processors.JSONRenderer(),
            ],
            wrapper_class=structlog.make_filtering_bound_logger(level),
            context_class=dict,
            logger_factory=structlog.PrintLoggerFactory(),
            cache_logger_on_first_use=True,
        )
