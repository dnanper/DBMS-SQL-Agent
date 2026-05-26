"""LangGraph node factories for the SQL agent."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from src.tools.sql import strip_sql_fences

from .prompts import build_context_tool_prompt
from .review import build_review_request
from .state import SQLAgentState, get_latest_user_question


SQL_REVIEW_ACTION = "sql_db_query"


def _get_ai_message_cls() -> Any:
    try:
        from langchain.messages import AIMessage
    except ImportError:
        from langchain_core.messages import AIMessage

    return AIMessage


def _ai_message(content: str) -> Any:
    return _get_ai_message_cls()(content=content)


def _format_sql_result_response(query: str, result: str, answer: str | None = None) -> str:
    if answer:
        return f"SQL:\n{query}\n\nResult:\n{result}\n\nAnswer:\n{answer}"
    return f"SQL:\n{query}\n\nResult:\n{result}"


def make_understand_query_node(model: Any | None = None) -> Any:
    def understand_query(state: SQLAgentState) -> dict[str, Any]:
        question = get_latest_user_question(state).strip()
        return {"question": question}

    return understand_query


def make_context_agent_node(model: Any, tools: list[Any]) -> Any:
    def context_agent(state: SQLAgentState) -> dict[str, Any]:
        llm_with_tools = model.bind_tools(tools)
        messages = [{"role": "system", "content": build_context_tool_prompt()}] + list(state.get("messages", []))
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    return context_agent


def route_after_context_agent(state: SQLAgentState) -> str:
    messages = state.get("messages", [])
    if messages and getattr(messages[-1], "tool_calls", None):
        return "context_tools"
    return "sql_tool_phase"


def _tool_message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name", ""))
    return str(getattr(message, "name", ""))


def _tool_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _collect_context_from_tool_messages(messages: list[Any]) -> tuple[str, str]:
    schema_parts: list[str] = []
    web_parts: list[str] = []
    for message in messages:
        name = _tool_message_name(message)
        content = _tool_message_content(message)
        if name == "retrieve_schema_context" and content:
            schema_parts.append(content)
        if name == "search_web_context" and content:
            web_parts.append(content)
    return "\n\n".join(schema_parts), "\n\n".join(web_parts)


def make_sql_tool_phase_node(
    *,
    generate_sql_tool: Any,
    execute_sql_tool: Any,
    debug_sql_tool: Any,
    max_attempts: int,
) -> Any:
    def sql_tool_phase(state: SQLAgentState) -> dict[str, Any]:
        question = state.get("question") or get_latest_user_question(state)
        feedback = state.get("review_feedback", "")
        schema_context, web_context = _collect_context_from_tool_messages(list(state.get("messages", [])))
        generated = generate_sql_tool.invoke(
            {
                "question": question,
                "schema_context": schema_context,
                "web_context": web_context,
                "feedback": feedback,
            }
        )
        sql = generated["sql"]
        attempts = 0

        while attempts <= max_attempts:
            execution = execute_sql_tool.invoke({"sql": sql})
            if execution["success"]:
                return {
                    "question": question,
                    "context": f"{schema_context}\n\n{web_context}",
                    "candidate_sql": execution["sql"],
                    "mock_result": execution["result"],
                    "validation_error": "",
                    "repair_attempts": attempts,
                    "review_feedback": "",
                }
            if attempts == max_attempts:
                return {
                    "question": question,
                    "context": f"{schema_context}\n\n{web_context}",
                    "candidate_sql": strip_sql_fences(sql),
                    "validation_error": execution["error"],
                    "repair_attempts": attempts,
                    "review_feedback": "",
                }
            repaired = debug_sql_tool.invoke(
                {
                    "question": question,
                    "schema_context": schema_context,
                    "web_context": web_context,
                    "broken_sql": sql,
                    "error": execution["error"],
                }
            )
            sql = repaired["sql"]
            attempts += 1

        return {"candidate_sql": strip_sql_fences(sql), "repair_attempts": attempts}

    return sql_tool_phase


def human_review_node(state: SQLAgentState) -> dict[str, Any]:
    request = build_review_request(SQL_REVIEW_ACTION, {"query": state.get("candidate_sql", "")})
    response = interrupt([request])

    response_type = response.get("type")
    if response_type == "accept":
        return {"approved_sql": state.get("candidate_sql", ""), "review_feedback": ""}
    if response_type == "edit":
        edited_sql = str(response.get("args", {}).get("query", ""))
        return {"approved_sql": strip_sql_fences(edited_sql), "candidate_sql": strip_sql_fences(edited_sql)}
    if response_type == "response":
        return {"review_feedback": str(response.get("args", "")), "approved_sql": ""}
    raise ValueError(f"Unsupported interrupt response type: {response_type}")


def route_after_validation(state: SQLAgentState) -> str:
    if state.get("validation_error"):
        return "final_response"
    return "human_review"


def route_after_review(state: SQLAgentState) -> str:
    if state.get("approved_sql"):
        return "execute_real_db"
    return "sql_tool_phase"


def make_execute_real_db_node(execute_tool: Any) -> Any:
    def execute_real_db(state: SQLAgentState) -> dict[str, Any]:
        execution = execute_tool.invoke({"sql": state.get("approved_sql", "")})
        if execution["success"]:
            return {"approved_sql": execution["sql"], "final_result": execution["result"], "validation_error": ""}
        return {"final_result": "", "validation_error": execution["error"]}

    return execute_real_db


def make_final_response_node(model: Any | None = None) -> Any:
    def final_response(state: SQLAgentState) -> dict[str, Any]:
        query = state.get("approved_sql") or state.get("candidate_sql", "")
        result = state.get("final_result") or state.get("mock_result", "")
        error = state.get("validation_error", "")
        if error:
            content = _format_sql_result_response(query, f"Error: {error}")
        else:
            content = _format_sql_result_response(query, result)
        return {"final_response": content, "messages": [_ai_message(content)]}

    return final_response
