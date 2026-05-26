"""State helpers for the SQL agent graph."""

from __future__ import annotations

from typing import Annotated, Any, TypedDict

from langgraph.graph.message import add_messages


class SQLAgentState(TypedDict, total=False):
    messages: Annotated[list[Any], add_messages]
    question: str
    context: str
    plan: str
    candidate_sql: str
    approved_sql: str
    mock_result: str
    final_result: str
    validation_error: str
    repair_attempts: int
    review_feedback: str
    final_response: str


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "role", ""))


def get_latest_user_question(state: dict[str, Any]) -> str:
    if state.get("question"):
        return str(state["question"])

    for message in reversed(state.get("messages", [])):
        role = _message_role(message)
        content = _message_content(message).strip()
        if content and (role in {"", "user"}):
            return content
    return ""
