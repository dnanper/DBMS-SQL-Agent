import unittest

from src.webapp import format_sse_event, iter_text_chunks, normalize_review_payload


class TestWebAppHelpers(unittest.TestCase):
    def test_format_sse_event(self) -> None:
        event = format_sse_event("status", {"thread_id": "abc", "state": "running"})
        self.assertIn("event: status", event)
        self.assertIn('"thread_id": "abc"', event)
        self.assertTrue(event.endswith("\n\n"))

    def test_iter_text_chunks(self) -> None:
        chunks = list(iter_text_chunks("abcdefghij", chunk_size=4))
        self.assertEqual(chunks, ["abcd", "efgh", "ij"])

    def test_normalize_review_payload_accept(self) -> None:
        thread_id, response = normalize_review_payload({"thread_id": "t1", "type": "accept"})
        self.assertEqual(thread_id, "t1")
        self.assertEqual(response, {"type": "accept"})

    def test_normalize_review_payload_edit(self) -> None:
        thread_id, response = normalize_review_payload(
            {"thread_id": "t1", "type": "edit", "args": {"query": "SELECT 1"}}
        )
        self.assertEqual(thread_id, "t1")
        self.assertEqual(response, {"type": "edit", "args": {"query": "SELECT 1"}})

    def test_normalize_review_payload_supports_aliases(self) -> None:
        thread_id, response = normalize_review_payload(
            {"threadId": "t2", "decision": "accept"},
            fallback_thread_id=None,
        )
        self.assertEqual(thread_id, "t2")
        self.assertEqual(response, {"type": "accept"})

    def test_normalize_review_payload_uses_fallback_thread(self) -> None:
        thread_id, response = normalize_review_payload(
            {"type": "accept"},
            fallback_thread_id="fallback-thread",
        )
        self.assertEqual(thread_id, "fallback-thread")
        self.assertEqual(response, {"type": "accept"})


if __name__ == "__main__":
    unittest.main()
