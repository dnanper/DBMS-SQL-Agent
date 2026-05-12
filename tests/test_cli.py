import io
import logging
import unittest

from src.cli import build_resume_payload, prompt_for_review, run_cli_turn


class _Graph:
    def __init__(self) -> None:
        self.calls = []

    def stream(self, payload, config=None, stream_mode=None):
        self.calls.append((payload, config, stream_mode))
        if len(self.calls) == 1:
            yield {"__interrupt__": [type("Interrupt", (), {"value": [{"action": "sql_db_query", "args": {"query": "SELECT 1"}, "description": "Please review the tool call"}]})()]}
            return
        yield {
            "messages": [
                type(
                    "Msg",
                    (),
                    {"content": "SQL:\nSELECT 1\n\nResult:\n[(1,)]"},
                )()
            ]
        }


class _ListHandler(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.messages: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.messages.append(record.getMessage())


class TestCli(unittest.TestCase):
    def test_prompt_for_review_accept(self) -> None:
        output = io.StringIO()
        response = prompt_for_review(
            {"action": "sql_db_query", "args": {"query": "SELECT 1"}, "description": "Please review the tool call"},
            input_func=lambda _: "a",
            output=output,
        )

        self.assertEqual(response, {"type": "accept"})
        self.assertIn("SELECT 1", output.getvalue())

    def test_build_resume_payload_uses_factory(self) -> None:
        payload = build_resume_payload({"type": "accept"}, command_factory=lambda *, resume: {"resume": resume})
        self.assertEqual(payload, {"resume": {"type": "accept"}})

    def test_run_cli_turn_prints_and_logs_exchange(self) -> None:
        graph = _Graph()
        output = io.StringIO()
        logger = logging.getLogger("test-cli")
        logger.handlers = []
        logger.setLevel(logging.INFO)
        handler = _ListHandler()
        logger.addHandler(handler)
        logger.propagate = False

        run_cli_turn(
            graph=graph,
            thread_id="thread-1",
            user_input="Which customer has the highest total payment amount?",
            output=output,
            logger=logger,
            input_func=lambda _: "a",
            command_factory=lambda *, resume: {"resume": resume},
        )

        self.assertIn("Assistant: SQL:", output.getvalue())
        self.assertEqual(graph.calls[1][0], {"resume": {"type": "accept"}})
        self.assertTrue(any("User: Which customer has the highest total payment amount?" in m for m in handler.messages))
        self.assertTrue(any("Assistant: SQL:" in m for m in handler.messages))


if __name__ == "__main__":
    unittest.main()
