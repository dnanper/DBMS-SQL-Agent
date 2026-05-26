"""SQL validation and execution tools."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field


_MUTATING_SQL = re.compile(
    r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|MERGE|GRANT|REVOKE|CALL|EXEC|VACUUM)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SQLExecutionResult:
    success: bool
    result: str
    error: str


class GenerateSQLDebugExecuteInput(BaseModel):
    sql: str = Field(..., description="Read-only SQL query to validate and execute against the mock database.")


class GenerateSQLInput(BaseModel):
    question: str = Field(..., description="User question to answer with SQL.")
    schema_context: str = Field(..., description="Database schema context from retrieve_schema_context.")
    web_context: str = Field("", description="Optional web context from search_web_context.")
    feedback: str = Field("", description="Optional human feedback to incorporate.")


class ExecuteSQLInput(BaseModel):
    sql: str = Field(..., description="Read-only SQL query to validate and execute.")


class DebugSQLInput(BaseModel):
    question: str = Field(..., description="User question the SQL should answer.")
    schema_context: str = Field(..., description="Database schema context.")
    web_context: str = Field("", description="Optional web context.")
    broken_sql: str = Field(..., description="SQL query that failed validation or execution.")
    error: str = Field(..., description="Validation or execution error to repair.")


class ExecuteReviewedSQLInput(BaseModel):
    sql: str = Field(..., description="Human-reviewed read-only SQL query to execute against the real database.")


def strip_sql_fences(value: str) -> str:
    text = value.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.lower().startswith("sql\n"):
        text = text[4:].strip()
    return text


def validate_read_only_sql(query: str) -> None:
    normalized = strip_sql_fences(query).strip()
    if not normalized:
        raise ValueError("SQL query is empty")
    if _MUTATING_SQL.search(normalized):
        raise ValueError("Only read-only SELECT/WITH SQL is allowed")
    if not re.match(r"^\s*(SELECT|WITH)\b", normalized, re.IGNORECASE):
        raise ValueError("SQL must start with SELECT or WITH")
    if ";" in normalized.rstrip(";"):
        raise ValueError("Multiple SQL statements are not allowed")


def execute_sql_safely(db: Any, query: str) -> SQLExecutionResult:
    try:
        validate_read_only_sql(query)
        result = db.run(strip_sql_fences(query).rstrip(";"))
    except Exception as exc:
        return SQLExecutionResult(success=False, result="", error=str(exc))
    return SQLExecutionResult(success=True, result=str(result), error="")


def _invoke_text(model: Any, messages: list[dict[str, str]]) -> str:
    response = model.invoke(messages)
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


def create_generate_sql_tool(*, model: Any, top_k: int, dialect: str = "postgresql") -> Any:
    @tool(
        "generate_sql",
        args_schema=GenerateSQLInput,
        description="Generate one read-only SQL query from the user question and retrieved context.",
    )
    def generate_sql(question: str, schema_context: str, web_context: str = "", feedback: str = "") -> dict[str, str]:
        feedback_text = f"\nHuman feedback:\n{feedback}" if feedback else ""
        prompt = f"""
Generate one read-only {dialect} SQL query.

Question:
{question}

Schema context:
{schema_context}

Web context:
{web_context}
{feedback_text}

Rules:
- Return SQL only.
- Use SELECT or WITH only.
- Use {dialect} syntax and functions.
- Limit result rows to at most {top_k} unless the user asks otherwise.
""".strip()
        return {"sql": strip_sql_fences(_invoke_text(model, [{"role": "user", "content": prompt}])).rstrip(";")}

    return generate_sql


def create_execute_sql_tool(*, db: Any, name: str = "execute_sql") -> Any:
    @tool(
        name,
        args_schema=ExecuteSQLInput,
        description="Validate and execute read-only SQL against the configured database.",
    )
    def execute_sql(sql: str) -> dict[str, Any]:
        clean_sql = strip_sql_fences(sql).rstrip(";")
        execution = execute_sql_safely(db, clean_sql)
        return {
            "success": execution.success,
            "sql": clean_sql,
            "result": execution.result,
            "error": execution.error,
        }

    return execute_sql


def create_debug_sql_tool(*, model: Any, top_k: int, dialect: str = "postgresql") -> Any:
    @tool(
        "debug_sql",
        args_schema=DebugSQLInput,
        description="Repair a read-only SQL query using the original question, context, and execution error.",
    )
    def debug_sql(
        question: str,
        schema_context: str,
        web_context: str,
        broken_sql: str,
        error: str,
    ) -> dict[str, str]:
        prompt = f"""
Repair the {dialect} SQL query.

Question:
{question}

Schema context:
{schema_context}

Web context:
{web_context}

Broken SQL:
{broken_sql}

Error:
{error}

Rules:
- Return SQL only.
- Use SELECT or WITH only.
- Use {dialect} syntax and functions.
- Limit result rows to at most {top_k} unless the user asks otherwise.
""".strip()
        return {"sql": strip_sql_fences(_invoke_text(model, [{"role": "user", "content": prompt}])).rstrip(";")}

    return debug_sql


def create_generate_sql_debug_execute_tool(*, mock_db: Any) -> Any:
    @tool(
        "generate_sql_debug_execute",
        args_schema=GenerateSQLDebugExecuteInput,
        description="Validate read-only SQL and execute it against the mock database for debugging.",
    )
    def generate_sql_debug_execute(sql: str) -> dict[str, Any]:
        clean_sql = strip_sql_fences(sql).rstrip(";")
        execution = execute_sql_safely(mock_db, clean_sql)
        return {
            "success": execution.success,
            "sql": clean_sql,
            "result": execution.result,
            "error": execution.error,
        }

    return generate_sql_debug_execute


def create_execute_reviewed_sql_tool(*, db: Any) -> Any:
    @tool(
        "execute_reviewed_sql",
        args_schema=ExecuteReviewedSQLInput,
        description="Execute human-reviewed read-only SQL against the real database.",
    )
    def execute_reviewed_sql(sql: str) -> dict[str, Any]:
        clean_sql = strip_sql_fences(sql).rstrip(";")
        execution = execute_sql_safely(db, clean_sql)
        return {
            "success": execution.success,
            "sql": clean_sql,
            "result": execution.result,
            "error": execution.error,
        }

    return execute_reviewed_sql
