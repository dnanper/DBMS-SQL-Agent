"""Human review helpers for SQL execution."""

from __future__ import annotations

from typing import Any


def build_review_request(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": tool_name,
        "args": tool_input,
        "description": "Please review the tool call",
    }
