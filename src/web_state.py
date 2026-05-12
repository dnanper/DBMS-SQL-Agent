"""In-memory state for browser-based human review."""

from __future__ import annotations

import threading
from typing import Any


class ReviewStore:
    """Tracks pending review requests and their responses by thread id."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._conditions: dict[str, threading.Condition] = {}
        self._pending: dict[str, dict[str, Any]] = {}
        self._responses: dict[str, dict[str, Any]] = {}

    def set_pending_review(self, thread_id: str, request: dict[str, Any]) -> None:
        with self._lock:
            condition = self._conditions.setdefault(thread_id, threading.Condition(self._lock))
            self._pending[thread_id] = request
            self._responses.pop(thread_id, None)
            condition.notify_all()

    def get_pending_review(self, thread_id: str) -> dict[str, Any] | None:
        with self._lock:
            return self._pending.get(thread_id)

    def pending_thread_ids(self) -> list[str]:
        with self._lock:
            return list(self._pending.keys())

    def submit_review_response(self, thread_id: str, response: dict[str, Any]) -> bool:
        with self._lock:
            if thread_id not in self._pending:
                return False

            condition = self._conditions.setdefault(thread_id, threading.Condition(self._lock))
            self._responses[thread_id] = response
            condition.notify_all()
            return True

    def wait_for_review_response(self, thread_id: str, timeout: float | None = None) -> dict[str, Any]:
        with self._lock:
            condition = self._conditions.setdefault(thread_id, threading.Condition(self._lock))
            if thread_id not in self._pending:
                raise KeyError(f"No pending review for thread_id={thread_id}")

            if thread_id not in self._responses:
                condition.wait(timeout=timeout)

            if thread_id not in self._responses:
                raise TimeoutError(f"Timed out waiting for review response: {thread_id}")

            response = self._responses.pop(thread_id)
            self._pending.pop(thread_id, None)
            return response
