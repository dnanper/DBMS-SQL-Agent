"""LangGraph SQL agent wiring."""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from src.database import build_database
from src.model import build_model
from src.tools.context import create_retrieve_schema_context_tool, create_search_web_context_tool
from src.tools.sql import (
    create_debug_sql_tool,
    create_execute_reviewed_sql_tool,
    create_execute_sql_tool,
    create_generate_sql_tool,
)

from .nodes import (
    human_review_node,
    make_context_agent_node,
    make_execute_real_db_node,
    make_final_response_node,
    make_sql_tool_phase_node,
    make_understand_query_node,
    route_after_context_agent,
    route_after_review,
    route_after_validation,
)
from .state import SQLAgentState


def _build_checkpointer() -> Any:
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def build_agent_graph(
    *,
    top_k: int = 5,
    model: Any | None = None,
    db: Any | None = None,
    mock_db: Any | None = None,
    env: Mapping[str, str] | None = None,
    max_repair_attempts: int = 2,
) -> Any:
    runtime_model = model or build_model()
    runtime_db = db or build_database()
    runtime_mock_db = mock_db or runtime_db
    runtime_env = env or os.environ
    schema_context_tool = create_retrieve_schema_context_tool(db=runtime_db)
    web_context_tool = create_search_web_context_tool(env=runtime_env)
    context_tools = [schema_context_tool, web_context_tool]
    dialect = getattr(runtime_db, "dialect", "postgresql")
    generate_sql_tool = create_generate_sql_tool(model=runtime_model, top_k=top_k, dialect=dialect)
    execute_sql_tool = create_execute_sql_tool(db=runtime_mock_db, name="execute_sql")
    debug_sql_tool = create_debug_sql_tool(model=runtime_model, top_k=top_k, dialect=dialect)
    execute_reviewed_sql_tool = create_execute_reviewed_sql_tool(db=runtime_db)

    builder = StateGraph(SQLAgentState)
    builder.add_node("understand_query", make_understand_query_node(runtime_model))
    builder.add_node("context_agent", make_context_agent_node(runtime_model, context_tools))
    builder.add_node("context_tools", ToolNode(context_tools))
    builder.add_node(
        "sql_tool_phase",
        make_sql_tool_phase_node(
            generate_sql_tool=generate_sql_tool,
            execute_sql_tool=execute_sql_tool,
            debug_sql_tool=debug_sql_tool,
            max_attempts=max_repair_attempts,
        ),
    )
    builder.add_node("human_review", human_review_node)
    builder.add_node("execute_real_db", make_execute_real_db_node(execute_reviewed_sql_tool))
    builder.add_node("final_response", make_final_response_node(runtime_model))

    builder.add_edge(START, "understand_query")
    builder.add_edge("understand_query", "context_agent")
    builder.add_conditional_edges(
        "context_agent",
        route_after_context_agent,
        {
            "context_tools": "context_tools",
            "sql_tool_phase": "sql_tool_phase",
        },
    )
    builder.add_edge("context_tools", "context_agent")
    builder.add_conditional_edges(
        "sql_tool_phase",
        route_after_validation,
        {
            "human_review": "human_review",
            "final_response": "final_response",
        },
    )
    builder.add_conditional_edges(
        "human_review",
        route_after_review,
        {
            "execute_real_db": "execute_real_db",
            "sql_tool_phase": "sql_tool_phase",
        },
    )
    builder.add_edge("execute_real_db", "final_response")
    builder.add_edge("final_response", END)

    return builder.compile(checkpointer=_build_checkpointer())
