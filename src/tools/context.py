"""Context retrieval tools for the SQL agent."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from collections.abc import Mapping
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


TAVILY_ENDPOINT = "https://api.tavily.com/search"


class RetrieveContextInput(BaseModel):
    question: str = Field(..., description="User question to enrich with database and web context.")


class RetrieveSchemaContextInput(BaseModel):
    question: str = Field(..., description="User question used to retrieve relevant database schema context.")


class SearchWebContextInput(BaseModel):
    query: str = Field(..., description="Search query for Tavily web context.")
    max_results: int = Field(3, ge=1, le=10, description="Maximum Tavily results to include.")


def _database_schema_text(db: Any) -> str:
    parts: list[str] = []
    if hasattr(db, "get_table_info"):
        parts.append(str(db.get_table_info()))
    elif hasattr(db, "table_info"):
        parts.append(str(db.table_info))
    else:
        parts.append("Schema unavailable from database object.")

    relationship_text = _database_relationship_text(db)
    if relationship_text:
        parts.append(relationship_text)

    comment_text = _database_comment_text(db)
    if comment_text:
        parts.append(comment_text)

    return "\n\n".join(parts)


def _database_engine(db: Any) -> Any | None:
    return getattr(db, "_engine", None) or getattr(db, "engine", None)


def _database_table_names(db: Any) -> list[str]:
    if hasattr(db, "get_usable_table_names"):
        return list(db.get_usable_table_names())
    if hasattr(db, "_usable_tables"):
        return list(getattr(db, "_usable_tables"))
    return []


def _database_relationship_text(db: Any) -> str:
    engine = _database_engine(db)
    table_names = _database_table_names(db)
    if engine is None or not table_names:
        return ""

    try:
        from sqlalchemy import inspect

        inspector = inspect(engine)
    except Exception:
        return ""

    lines: list[str] = []
    for table_name in table_names:
        try:
            foreign_keys = inspector.get_foreign_keys(table_name)
        except Exception:
            continue
        for foreign_key in foreign_keys:
            columns = ", ".join(foreign_key.get("constrained_columns") or [])
            referred_table = foreign_key.get("referred_table", "")
            referred_columns = ", ".join(foreign_key.get("referred_columns") or [])
            if columns and referred_table:
                lines.append(f"- {table_name}({columns}) -> {referred_table}({referred_columns})")

    if not lines:
        return ""
    return "Relationships:\n" + "\n".join(lines)


def _database_comment_text(db: Any) -> str:
    engine = _database_engine(db)
    table_names = _database_table_names(db)
    if engine is None or not table_names:
        return ""

    try:
        from sqlalchemy import inspect

        inspector = inspect(engine)
    except Exception:
        return ""

    lines: list[str] = []
    for table_name in table_names:
        try:
            table_comment = inspector.get_table_comment(table_name).get("text")
        except Exception:
            table_comment = None
        if table_comment:
            lines.append(f"- {table_name}: {table_comment}")

        try:
            columns = inspector.get_columns(table_name)
        except Exception:
            columns = []
        for column in columns:
            comment = column.get("comment")
            if comment:
                lines.append(f"- {table_name}.{column.get('name')}: {comment}")

    if not lines:
        return ""
    return "Comments:\n" + "\n".join(lines)


def search_tavily(
    query: str,
    *,
    env: Mapping[str, str] | None = None,
    max_results: int = 3,
    timeout: float = 8.0,
) -> list[str]:
    """Search Tavily directly without LangChain tools."""

    source = env or os.environ
    api_key = source.get("TAVILY_API_KEY", "").strip()
    if not api_key:
        return []

    body = json.dumps(
        {
            "api_key": api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        TAVILY_ENDPOINT,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return []

    results = payload.get("results", [])
    snippets: list[str] = []
    for item in results:
        title = str(item.get("title", "")).strip()
        content = str(item.get("content", "")).strip()
        url = str(item.get("url", "")).strip()
        parts = [part for part in (title, content, url) if part]
        if parts:
            snippets.append(" - ".join(parts))
    return snippets


def retrieve_context(
    question: str,
    *,
    db: Any,
    env: Mapping[str, str] | None = None,
) -> str:
    schema = _database_schema_text(db)
    tavily_results = search_tavily(question, env=env)
    external_context = "\n".join(f"- {item}" for item in tavily_results)
    if not external_context:
        external_context = "- Tavily not configured or no external context found."

    return f"Database schema:\n{schema}\n\nExternal context:\n{external_context}"


def create_retrieve_schema_context_tool(*, db: Any) -> Any:
    @tool(
        "retrieve_schema_context",
        args_schema=RetrieveSchemaContextInput,
        description="Retrieve database schema context relevant to the user question.",
    )
    def retrieve_schema_context(question: str) -> str:
        return f"Database schema:\n{_database_schema_text(db)}"

    return retrieve_schema_context


def create_search_web_context_tool(env: Mapping[str, str] | None = None) -> Any:
    @tool(
        "search_web_context",
        args_schema=SearchWebContextInput,
        description="Search Tavily for external context that can help interpret the user question.",
    )
    def search_web_context(query: str, max_results: int = 3) -> str:
        tavily_results = search_tavily(query, env=env, max_results=max_results)
        if not tavily_results:
            return "External context:\n- Tavily not configured or no external context found."
        return "External context:\n" + "\n".join(f"- {item}" for item in tavily_results)

    return search_web_context


def create_retrieve_context_tool(
    *,
    db: Any,
    env: Mapping[str, str] | None = None,
) -> Any:
    @tool(
        "retrieve_context",
        args_schema=RetrieveContextInput,
        description="Retrieve database schema plus optional Tavily web context for a user question.",
    )
    def retrieve_context_tool(question: str) -> str:
        return retrieve_context(question, db=db, env=env)

    return retrieve_context_tool
