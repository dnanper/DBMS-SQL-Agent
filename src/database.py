"""Database helpers."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from .config import get_database_schema, get_database_uri

CLASSICMODELS_INCLUDE_TABLES = [
    "customers",
    "employees",
    "offices",
    "orderdetails",
    "orders",
    "payments",
    "productlines",
    "products",
]


def normalize_database_uri(database_uri: str) -> str:
    if database_uri.startswith("postgresql://"):
        return database_uri.replace("postgresql://", "postgresql+psycopg://", 1)
    return database_uri


def build_database_kwargs(database_uri: str, env: Mapping[str, str] | None = None) -> dict[str, list[str]]:
    parsed = urlparse(database_uri)
    if parsed.scheme.startswith("postgresql"):
        return {"include_tables": CLASSICMODELS_INCLUDE_TABLES}
    return {}


def build_database_connection_options(
    database_uri: str,
    env: Mapping[str, str] | None = None,
) -> dict[str, dict[str, str]]:
    parsed = urlparse(database_uri)
    if parsed.scheme.startswith("postgresql"):
        schema = get_database_schema(env)
        if schema:
            return {"connect_args": {"options": f"-csearch_path={schema}"}}
    return {}


def build_database_unavailable_message(database_uri: str, error: Exception) -> str:
    parsed = urlparse(database_uri)
    host = parsed.hostname or "localhost"
    port = parsed.port or 5432
    db_name = (parsed.path or "/").lstrip("/")
    return (
        "Unable to connect to PostgreSQL.\n"
        f"- URI: {database_uri}\n"
        f"- Host: {host}\n"
        f"- Port: {port}\n"
        f"- Database: {db_name}\n"
        "Ensure PostgreSQL is running and reachable, then verify DATABASE_URI/DATABASE_SCHEMA in .env.\n"
        f"Original error: {error}"
    )


def build_database(env: Mapping[str, str] | None = None) -> Any:
    source = env or os.environ
    database_uri = normalize_database_uri(get_database_uri(source))
    database_kwargs = build_database_kwargs(database_uri, source)
    engine_args = build_database_connection_options(database_uri, source)

    from langchain_community.utilities import SQLDatabase
    from sqlalchemy.exc import OperationalError

    try:
        return SQLDatabase.from_uri(database_uri, engine_args=engine_args, **database_kwargs)
    except OperationalError as exc:
        raise RuntimeError(build_database_unavailable_message(database_uri, exc)) from exc
