"""Configuration helpers for the SQL agent app."""

from __future__ import annotations

import os
import pathlib
from collections.abc import Mapping
from os import PathLike

DEFAULT_MODEL_NAME = "gpt-5.4"
DEFAULT_DATABASE_URI = "postgresql+psycopg://postgres:postgres@localhost:5432/classicmodels"
DEFAULT_DATABASE_SCHEMA = "classicmodels"
DEFAULT_THREAD_ID = "cli-session"


def load_environment(dotenv_path: str | PathLike[str] = ".env", *, override: bool = False) -> bool:
    from dotenv import load_dotenv

    return load_dotenv(dotenv_path=dotenv_path, override=override)


def get_database_uri(env: Mapping[str, str] | None = None) -> str:
    source = env or os.environ
    return source.get("DATABASE_URI", DEFAULT_DATABASE_URI)


def get_database_schema(env: Mapping[str, str] | None = None) -> str | None:
    source = env or os.environ
    return source.get("DATABASE_SCHEMA", DEFAULT_DATABASE_SCHEMA)


def build_log_file_path(env: Mapping[str, str] | None = None) -> pathlib.Path:
    source = env or os.environ
    return pathlib.Path(source.get("LOG_FILE", "logs/sql-agent.log"))


def get_model_name(env: Mapping[str, str] | None = None) -> str:
    source = env or os.environ
    return source.get("OPENAI_MODEL", DEFAULT_MODEL_NAME)


def build_langgraph_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}
