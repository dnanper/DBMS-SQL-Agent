"""Compatibility facade for the project-owned LangGraph SQL agent."""

from __future__ import annotations

from typing import Any, Literal

from .agent.graph import build_agent_graph
from .agent.prompts import build_system_prompt
from .agent.review import build_review_request


END_NODE = "__end__"


def build_check_query_system_prompt(dialect: str) -> str:
    return f"""
You are a SQL expert with a strong attention to detail.
Double check the {dialect} query for common mistakes, including:
- Using NOT IN with NULL values
- Using UNION when UNION ALL should have been used
- Using BETWEEN for exclusive ranges
- Data type mismatch in predicates
- Properly quoting identifiers
- Using the correct number of arguments for functions
- Casting to the correct data type
- Using the proper columns for joins

If there are any of the above mistakes, rewrite the query. If there are no mistakes,
just reproduce the original query.
""".strip()


def should_continue(state: dict[str, list[Any]]) -> Literal["run_query", "__end__"]:
    messages = state["messages"]
    last_message = messages[-1]
    if not getattr(last_message, "tool_calls", None):
        return END_NODE
    return "run_query"


def extract_last_sql_execution(messages: list[Any]) -> tuple[str, str]:
    last_query = ""
    last_result = ""

    for message in reversed(messages):
        tool_calls = getattr(message, "tool_calls", None) or []
        if not last_query and tool_calls:
            for tool_call in tool_calls:
                if tool_call.get("name") == "sql_db_query":
                    last_query = tool_call.get("args", {}).get("query", "")
                    break

        if not last_result and getattr(message, "name", None) == "sql_db_query":
            last_result = str(getattr(message, "content", ""))

        if last_query and last_result:
            break

    return last_query, last_result


def format_sql_result_response(query: str, result: str) -> str:
    return f"SQL:\n{query}\n\nResult:\n{result}"


try:
    agent = build_agent_graph()
except Exception:
    agent = None
