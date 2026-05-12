import threading
import time
import unittest

from src.web_state import ReviewStore


class TestReviewStore(unittest.TestCase):
    def test_submit_review_unblocks_waiter(self) -> None:
        store = ReviewStore()
        store.set_pending_review("thread-1", {"action": "sql_db_query"})
        result: dict[str, str] = {}

        def waiter() -> None:
            result.update(store.wait_for_review_response("thread-1", timeout=1.0))

        thread = threading.Thread(target=waiter)
        thread.start()
        time.sleep(0.05)
        store.submit_review_response("thread-1", {"type": "accept"})
        thread.join()

        self.assertEqual(result, {"type": "accept"})
        self.assertIsNone(store.get_pending_review("thread-1"))

    def test_submit_review_requires_pending_request(self) -> None:
        store = ReviewStore()
        self.assertFalse(store.submit_review_response("missing-thread", {"type": "accept"}))

    def test_pending_thread_ids(self) -> None:
        store = ReviewStore()
        store.set_pending_review("thread-a", {"action": "sql_db_query"})
        self.assertEqual(store.pending_thread_ids(), ["thread-a"])


if __name__ == "__main__":
    unittest.main()
