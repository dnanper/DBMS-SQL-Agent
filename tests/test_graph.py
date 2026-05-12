import unittest

from src.graph import (
    END_NODE,
    build_review_request,
    extract_last_sql_execution,
    format_sql_result_response,
    should_continue,
)


class _Message:
    def __init__(self, *, tool_calls=None, content="", name=None) -> None:
        self.tool_calls = tool_calls or []
        self.content = content
        self.name = name


class TestGraphHelpers(unittest.TestCase):
    def test_should_continue_requires_tool_call(self) -> None:
        route = should_continue({"messages": [_Message(tool_calls=[])]})
        self.assertEqual(route, END_NODE)

        route = should_continue({"messages": [_Message(tool_calls=[{"name": "sql_db_query"}])]})
        self.assertEqual(route, "run_query")

    def test_extract_last_sql_execution(self) -> None:
        messages = [
            _Message(tool_calls=[{"name": "sql_db_query", "args": {"query": "SELECT 1"}}]),
            _Message(content="[('ok',)]", name="sql_db_query"),
        ]

        query, result = extract_last_sql_execution(messages)

        self.assertEqual(query, "SELECT 1")
        self.assertEqual(result, "[('ok',)]")

    def test_format_sql_result_response(self) -> None:
        response = format_sql_result_response("SELECT 1", "[('ok',)]")

        self.assertIn("SQL:", response)
        self.assertIn("SELECT 1", response)
        self.assertIn("Result:", response)
        self.assertIn("[('ok',)]", response)

    def test_build_review_request(self) -> None:
        request = build_review_request("sql_db_query", {"query": "SELECT 1"})

        self.assertEqual(
            request,
            {
                "action": "sql_db_query",
                "args": {"query": "SELECT 1"},
                "description": "Please review the tool call",
            },
        )


if __name__ == "__main__":
    unittest.main()
