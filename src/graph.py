"""LangGraph SQL agent wiring with in-memory checkpoint persistence."""

from __future__ import annotations

from typing import Any, Literal

from .database import build_database
from .model import build_model

try:
    from langchain_core.runnables import RunnableConfig
except Exception:
    class RunnableConfig(dict):  # type: ignore[no-redef]
        pass


END_NODE = "__end__"


def build_system_prompt(dialect: str, top_k: int = 5) -> str:
    return """
You are an agent designed to interact with a SQL database.
Given an input question, create a syntactically correct {dialect} query to run,
then look at the results of the query and return the answer. Unless the user specifies a specific number of examples they wish to obtain, always limit your query to at most {top_k} results.

You can order the results by a relevant column to return the most interesting
examples in the database. Never query for all the columns from a specific table,
only ask for the relevant columns given the question.

DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the database.
""".strip().format(dialect=dialect, top_k=top_k)


def build_check_query_system_prompt(dialect: str) -> str:
    return """
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

You will call the appropriate tool to execute the query after running this check.
""".strip().format(dialect=dialect)


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


def build_review_request(tool_name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": tool_name,
        "args": tool_input,
        "description": "Please review the tool call",
    }


def _get_ai_message_cls() -> Any:
    try:
        from langchain.messages import AIMessage
    except ImportError:
        from langchain_core.messages import AIMessage

    return AIMessage


def _build_checkpointer() -> Any:
    from langgraph.checkpoint.memory import MemorySaver

    return MemorySaver()


def build_agent_graph(*, top_k: int = 5, model: Any | None = None, db: Any | None = None) -> Any:
    from langchain_community.agent_toolkits import SQLDatabaseToolkit
    from langchain.tools import tool
    from langgraph.graph import END, START, MessagesState, StateGraph
    from langgraph.prebuilt import ToolNode
    from langgraph.types import interrupt

    runtime_model = model or build_model()
    runtime_db = db or build_database()
    toolkit = SQLDatabaseToolkit(db=runtime_db, llm=runtime_model)
    tools = toolkit.get_tools()

    get_schema_tool = next(tool for tool in tools if tool.name == "sql_db_schema")
    get_schema_node = ToolNode([get_schema_tool], name="get_schema")

    run_query_tool = next(tool for tool in tools if tool.name == "sql_db_query")

    @tool(
        run_query_tool.name,
        description=run_query_tool.description,
        args_schema=run_query_tool.args_schema,
    )
    def run_query_tool_with_interrupt(config: RunnableConfig, **tool_input: Any) -> Any:
        request = build_review_request(run_query_tool.name, tool_input)
        response = interrupt([request])

        if response["type"] == "accept":
            return run_query_tool.invoke(tool_input, config)
        if response["type"] == "edit":
            edited_input = response["args"]
            return run_query_tool.invoke(edited_input, config)
        if response["type"] == "response":
            return response["args"]
        raise ValueError(f"Unsupported interrupt response type: {response['type']}")

    run_query_node = ToolNode([run_query_tool_with_interrupt], name="run_query")

    list_tables_tool = next(tool for tool in tools if tool.name == "sql_db_list_tables")
    ai_message_cls = _get_ai_message_cls()
    generate_query_system_prompt = build_system_prompt(runtime_db.dialect, top_k=top_k)

    def list_tables(state: MessagesState) -> dict[str, list[Any]]:
        tool_call = {
            "name": "sql_db_list_tables",
            "args": {},
            "id": "list_tables_call",
            "type": "tool_call",
        }
        tool_call_message = ai_message_cls(content="", tool_calls=[tool_call])
        tool_message = list_tables_tool.invoke(tool_call)
        response = ai_message_cls(content=f"Available tables: {tool_message.content}")
        return {"messages": [tool_call_message, tool_message, response]}

    def call_get_schema(state: MessagesState) -> dict[str, list[Any]]:
        llm_with_tools = runtime_model.bind_tools([get_schema_tool], tool_choice="any")
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    def generate_query(state: MessagesState) -> dict[str, list[Any]]:
        system_message = {"role": "system", "content": generate_query_system_prompt}
        llm_with_tools = runtime_model.bind_tools([run_query_tool], tool_choice="any")
        response = llm_with_tools.invoke([system_message] + state["messages"])
        return {"messages": [response]}

    def format_answer(state: MessagesState) -> dict[str, list[Any]]:
        query, result = extract_last_sql_execution(state["messages"])
        response = ai_message_cls(content=format_sql_result_response(query, result))
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("list_tables", list_tables)
    builder.add_node("call_get_schema", call_get_schema)
    builder.add_node("get_schema", get_schema_node)
    builder.add_node("generate_query", generate_query)
    builder.add_node("run_query", run_query_node)
    builder.add_node("format_answer", format_answer)

    builder.add_edge(START, "list_tables")
    builder.add_edge("list_tables", "call_get_schema")
    builder.add_edge("call_get_schema", "get_schema")
    builder.add_edge("get_schema", "generate_query")
    builder.add_conditional_edges("generate_query", should_continue, {"run_query": "run_query", END_NODE: END})
    builder.add_edge("run_query", "format_answer")
    builder.add_edge("format_answer", END)

    return builder.compile(checkpointer=_build_checkpointer())


try:
    agent = build_agent_graph()
except Exception:
    agent = None
