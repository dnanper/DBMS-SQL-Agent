import unittest

from src.agent.nodes import route_after_context_agent
from src.agent.state import get_latest_user_question
from src.tools.context import (
    RetrieveContextInput,
    RetrieveSchemaContextInput,
    SearchWebContextInput,
    create_retrieve_context_tool,
    create_retrieve_schema_context_tool,
    create_search_web_context_tool,
    retrieve_context,
)
from src.tools.sql import (
    DebugSQLInput,
    ExecuteReviewedSQLInput,
    ExecuteSQLInput,
    GenerateSQLInput,
    GenerateSQLDebugExecuteInput,
    SQLExecutionResult,
    create_debug_sql_tool,
    create_execute_reviewed_sql_tool,
    create_execute_sql_tool,
    create_generate_sql_debug_execute_tool,
    create_generate_sql_tool,
    execute_sql_safely,
    validate_read_only_sql,
)


class _Message:
    def __init__(self, *, content: str = "", tool_calls=None) -> None:
        self.content = content
        self.tool_calls = tool_calls or []


class _Db:
    dialect = "sqlite"

    def __init__(self, *, result: str = "[(1,)]", error: Exception | None = None) -> None:
        self.result = result
        self.error = error
        self.queries: list[str] = []

    def get_table_info(self) -> str:
        return "CREATE TABLE customers (id integer, name text)"

    def run(self, query: str) -> str:
        self.queries.append(query)
        if self.error:
            raise self.error
        return self.result


class _DetailedDb(_Db):
    def get_table_info(self) -> str:
        return (
            "CREATE TABLE customers (id integer primary key, sales_rep_employee_number integer);\n"
            "CREATE TABLE employees (employee_number integer primary key);\n"
            "COMMENT ON TABLE customers IS 'customer master data';\n"
            "FOREIGN KEY(customers.sales_rep_employee_number) REFERENCES employees(employee_number)"
        )


class _Model:
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.messages: list[object] = []

    def invoke(self, messages):
        self.messages.append(messages)
        content = self.responses.pop(0)
        return type("Msg", (), {"content": content})()


class TestAgentState(unittest.TestCase):
    def test_get_latest_user_question_supports_dict_messages(self) -> None:
        question = get_latest_user_question({"messages": [{"role": "user", "content": "Top customer?"}]})

        self.assertEqual(question, "Top customer?")

    def test_get_latest_user_question_supports_message_objects(self) -> None:
        question = get_latest_user_question({"messages": [_Message(content="Revenue by month?")]})

        self.assertEqual(question, "Revenue by month?")


class TestAgentRoutes(unittest.TestCase):
    def test_route_after_context_agent_uses_tool_node_when_tool_calls_exist(self) -> None:
        route = route_after_context_agent({"messages": [_Message(tool_calls=[{"name": "retrieve_schema_context"}])]})

        self.assertEqual(route, "context_tools")

    def test_route_after_context_agent_continues_to_sql_loop_without_tool_calls(self) -> None:
        route = route_after_context_agent({"messages": [_Message(content="No more context needed.")]})

        self.assertEqual(route, "sql_tool_phase")


class TestContextTool(unittest.TestCase):
    def test_retrieve_context_includes_schema_without_tavily_key(self) -> None:
        context = retrieve_context("Top customer?", db=_Db(), env={})

        self.assertIn("Database schema:", context)
        self.assertIn("customers", context)
        self.assertIn("External context:", context)
        self.assertIn("not configured", context)

    def test_retrieve_context_tool_has_schema_and_invokes(self) -> None:
        tool = create_retrieve_context_tool(db=_Db(), env={})

        self.assertIs(tool.args_schema, RetrieveContextInput)
        result = tool.invoke({"question": "Top customer?"})

        self.assertIn("Database schema:", result)
        self.assertIn("customers", result)

    def test_retrieve_schema_context_tool_has_schema_and_invokes(self) -> None:
        tool = create_retrieve_schema_context_tool(db=_DetailedDb())

        self.assertIs(tool.args_schema, RetrieveSchemaContextInput)
        result = tool.invoke({"question": "Top customer?"})

        self.assertIn("Database schema:", result)
        self.assertIn("customers", result)
        self.assertIn("FOREIGN KEY", result)
        self.assertIn("COMMENT ON TABLE", result)

    def test_search_web_context_tool_has_schema_and_invokes_without_key(self) -> None:
        tool = create_search_web_context_tool(env={})

        self.assertIs(tool.args_schema, SearchWebContextInput)
        result = tool.invoke({"query": "classicmodels customer"})

        self.assertIn("not configured", result)


class TestSqlTool(unittest.TestCase):
    def test_validate_read_only_sql_accepts_select_and_with(self) -> None:
        validate_read_only_sql("SELECT id FROM customers LIMIT 5")
        validate_read_only_sql("WITH totals AS (SELECT 1) SELECT * FROM totals")

    def test_validate_read_only_sql_rejects_mutating_statements(self) -> None:
        with self.assertRaises(ValueError):
            validate_read_only_sql("DELETE FROM customers")

    def test_execute_sql_safely_returns_structured_success(self) -> None:
        db = _Db(result="[(42,)]")

        result = execute_sql_safely(db, "SELECT 42")

        self.assertEqual(result, SQLExecutionResult(success=True, result="[(42,)]", error=""))
        self.assertEqual(db.queries, ["SELECT 42"])

    def test_execute_sql_safely_returns_structured_error(self) -> None:
        result = execute_sql_safely(_Db(error=RuntimeError("bad SQL")), "SELECT nope")

        self.assertFalse(result.success)
        self.assertEqual(result.result, "")
        self.assertIn("bad SQL", result.error)

    def test_generate_sql_debug_execute_tool_has_schema_and_invokes(self) -> None:
        tool = create_generate_sql_debug_execute_tool(mock_db=_Db(result="[(7,)]"))

        self.assertIs(tool.args_schema, GenerateSQLDebugExecuteInput)
        result = tool.invoke({"sql": "SELECT 7"})

        self.assertEqual(
            result,
            {
                "success": True,
                "sql": "SELECT 7",
                "result": "[(7,)]",
                "error": "",
            },
        )

    def test_generate_sql_tool_has_schema_and_invokes_model(self) -> None:
        tool = create_generate_sql_tool(model=_Model(["```sql\nSELECT 1;\n```"]), top_k=5)

        self.assertIs(tool.args_schema, GenerateSQLInput)
        result = tool.invoke(
            {
                "question": "How many customers?",
                "schema_context": "customers table",
                "web_context": "",
                "feedback": "",
            }
        )

        self.assertEqual(result, {"sql": "SELECT 1"})

    def test_execute_sql_tool_has_schema_and_invokes_mock_db(self) -> None:
        tool = create_execute_sql_tool(db=_Db(result="[(3,)]"), name="execute_sql")

        self.assertIs(tool.args_schema, ExecuteSQLInput)
        result = tool.invoke({"sql": "SELECT 3"})

        self.assertEqual(result["success"], True)
        self.assertEqual(result["result"], "[(3,)]")

    def test_debug_sql_tool_has_schema_and_invokes_model(self) -> None:
        tool = create_debug_sql_tool(model=_Model(["SELECT 2"]), top_k=5)

        self.assertIs(tool.args_schema, DebugSQLInput)
        result = tool.invoke(
            {
                "question": "Count",
                "schema_context": "schema",
                "web_context": "",
                "broken_sql": "SELECT nope",
                "error": "no such column",
            }
        )

        self.assertEqual(result, {"sql": "SELECT 2"})

    def test_execute_reviewed_sql_tool_has_schema_and_invokes(self) -> None:
        tool = create_execute_reviewed_sql_tool(db=_Db(result="[(9,)]"))

        self.assertIs(tool.args_schema, ExecuteReviewedSQLInput)
        result = tool.invoke({"sql": "SELECT 9"})

        self.assertEqual(
            result,
            {
                "success": True,
                "sql": "SELECT 9",
                "result": "[(9,)]",
                "error": "",
            },
        )


if __name__ == "__main__":
    unittest.main()
