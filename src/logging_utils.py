"""Logging helpers for the CLI app."""

from __future__ import annotations

import logging
import pathlib


LOGGER_NAME = "sql_agent_cli"


def configure_file_logger(log_path: pathlib.Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.propagate = False

    resolved_path = log_path.resolve()
    for handler in logger.handlers:
        if isinstance(handler, logging.FileHandler) and pathlib.Path(handler.baseFilename) == resolved_path:
            return logger

    logger.handlers = []
    handler = logging.FileHandler(resolved_path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger
