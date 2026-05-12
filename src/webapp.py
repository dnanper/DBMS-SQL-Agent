"""FastAPI app with SSE streaming and browser-based HITL review."""

from __future__ import annotations

import json
import uuid
from collections.abc import Generator
from typing import Any

from fastapi import Request

from .cli import _message_text, build_resume_payload
from .config import build_langgraph_config, build_log_file_path, load_environment
from .graph import build_agent_graph
from .logging_utils import configure_file_logger
from .web_state import ReviewStore


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>SQL Agent SSE</title>
  <style>
    body { font-family: sans-serif; margin: 24px; background: #f6f4ef; color: #1f1f1f; }
    .shell { max-width: 960px; margin: 0 auto; }
    textarea, input, button { font: inherit; }
    textarea { width: 100%; min-height: 90px; padding: 12px; }
    #stream, #sql-review { white-space: pre-wrap; background: #fff; border: 1px solid #d6d0c4; padding: 16px; min-height: 120px; }
    #review-box { display: none; margin-top: 16px; padding: 16px; background: #fff8dc; border: 1px solid #d8c784; }
    .row { display: flex; gap: 12px; margin-top: 12px; }
    button { padding: 10px 16px; cursor: pointer; }
    .muted { color: #6b6b6b; }
  </style>
</head>
<body>
  <div class="shell">
    <h1>SQL Agent</h1>
    <p class="muted">FastAPI + SSE + human review before SQL execution.</p>
    <textarea id="question" placeholder="Ask a database question..."></textarea>
    <div class="row">
      <button id="send">Send</button>
      <input id="thread-id" placeholder="thread id (optional)" />
    </div>
    <h3>Stream</h3>
    <div id="stream"></div>
    <div id="review-box">
      <h3>Review SQL</h3>
      <div id="sql-review"></div>
      <div class="row">
        <button id="accept">Accept</button>
        <button id="edit">Edit</button>
        <button id="feedback">Feedback</button>
      </div>
      <textarea id="review-input" placeholder="Edited SQL or feedback..."></textarea>
    </div>
  </div>
  <script>
    const streamBox = document.getElementById("stream");
    const reviewBox = document.getElementById("review-box");
    const sqlReview = document.getElementById("sql-review");
    const reviewInput = document.getElementById("review-input");
    const threadInput = document.getElementById("thread-id");
    let currentThreadId = "";
    let currentRequest = null;
    let source = null;

    function append(text) {
      streamBox.textContent += text;
    }

    function resetStream() {
      streamBox.textContent = "";
      reviewBox.style.display = "none";
      reviewInput.value = "";
      currentRequest = null;
    }

    document.getElementById("send").onclick = () => {
      const message = document.getElementById("question").value.trim();
      if (!message) return;
      resetStream();
      currentThreadId = threadInput.value.trim() || crypto.randomUUID();
      threadInput.value = currentThreadId;
      if (source) source.close();
      source = new EventSource(`/api/stream?thread_id=${encodeURIComponent(currentThreadId)}&message=${encodeURIComponent(message)}`);
      source.addEventListener("status", (event) => {
        const payload = JSON.parse(event.data);
        append(`[status] ${payload.state}\\n`);
      });
      source.addEventListener("chunk", (event) => {
        const payload = JSON.parse(event.data);
        append(payload.content);
      });
      source.addEventListener("interrupt", (event) => {
        const payload = JSON.parse(event.data);
        currentRequest = payload.request;
        reviewBox.style.display = "block";
        sqlReview.textContent = payload.request.args.query || "";
        append("\\n[review required]\\n");
      });
      source.addEventListener("done", (event) => {
        const payload = JSON.parse(event.data);
        append(`\\n\\n[done] thread=${payload.thread_id}\\n`);
        source.close();
      });
      source.addEventListener("error", () => {
        append("\\n[stream closed]\\n");
      });
    };

    async function sendReview(type) {
      if (!currentThreadId) return;
      const body = { thread_id: currentThreadId, type };
      if (type === "edit") body.args = { query: reviewInput.value };
      if (type === "response") body.args = reviewInput.value;
      const response = await fetch("/api/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!response.ok) {
        const detail = await response.text();
        append(`\\n[review error] ${response.status}: ${detail}\\n`);
        return;
      }
      reviewBox.style.display = "none";
      reviewInput.value = "";
    }

    document.getElementById("accept").onclick = () => sendReview("accept");
    document.getElementById("edit").onclick = () => sendReview("edit");
    document.getElementById("feedback").onclick = () => sendReview("response");
  </script>
</body>
</html>
"""


def format_sse_event(event: str, payload: dict[str, Any]) -> str:
    return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"


def iter_text_chunks(text: str, *, chunk_size: int = 48) -> Generator[str, None, None]:
    for index in range(0, len(text), chunk_size):
        yield text[index : index + chunk_size]


def normalize_review_payload(
    payload: dict[str, Any],
    *,
    fallback_thread_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    thread_id = str(
        payload.get("thread_id")
        or payload.get("threadId")
        or fallback_thread_id
        or ""
    ).strip()
    response_type = str(
        payload.get("type")
        or payload.get("decision")
        or payload.get("review_type")
        or ""
    ).strip()
    if not thread_id:
        raise ValueError("thread_id is required")
    if not response_type:
        raise ValueError("type is required")

    response = {"type": response_type}
    if "args" in payload and payload["args"] is not None:
        response["args"] = payload["args"]
    elif response_type == "edit" and payload.get("query"):
        response["args"] = {"query": payload["query"]}
    return thread_id, response


def create_app() -> Any:
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import HTMLResponse, StreamingResponse

    load_environment()
    logger = configure_file_logger(build_log_file_path())
    graph = build_agent_graph()
    review_store = ReviewStore()
    app = FastAPI(title="SQL Agent Web UI")

    def event_stream(thread_id: str, user_input: str) -> Generator[str, None, None]:
        config = build_langgraph_config(thread_id)
        stream_input: Any = {"messages": [{"role": "user", "content": user_input}]}
        logger.info("Web user(%s): %s", thread_id, user_input)
        yield format_sse_event("status", {"thread_id": thread_id, "state": "started"})

        while True:
            final_step: dict[str, Any] | None = None
            interrupted = False

            for step in graph.stream(stream_input, config=config, stream_mode="values"):
                final_step = step
                if "__interrupt__" in step:
                    request = step["__interrupt__"][0].value[0]
                    review_store.set_pending_review(thread_id, request)
                    logger.info("Web review requested(%s): %s", thread_id, request["action"])
                    yield format_sse_event("interrupt", {"thread_id": thread_id, "request": request})
                    review_response = review_store.wait_for_review_response(thread_id)
                    logger.info("Web review response(%s): %s", thread_id, review_response["type"])
                    stream_input = build_resume_payload(review_response)
                    interrupted = True
                    break

            if interrupted:
                continue

            if final_step and "messages" in final_step:
                assistant_text = _message_text(final_step["messages"][-1])
                logger.info("Web assistant(%s): %s", thread_id, assistant_text)
                for chunk in iter_text_chunks(assistant_text):
                    yield format_sse_event("chunk", {"thread_id": thread_id, "content": chunk})

            yield format_sse_event("done", {"thread_id": thread_id})
            return

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return INDEX_HTML

    @app.get("/api/stream")
    def stream(message: str, thread_id: str | None = None) -> StreamingResponse:
        resolved_thread_id = thread_id or str(uuid.uuid4())
        return StreamingResponse(event_stream(resolved_thread_id, message), media_type="text/event-stream")

    @app.post("/api/review")
    async def review(request: Request) -> dict[str, Any]:
        raw_body = await request.body()
        logger.info("Web review raw body: %s", raw_body.decode("utf-8", errors="replace"))
        payload = await request.json()
        fallback_thread_id = None
        pending_ids = review_store.pending_thread_ids()
        if len(pending_ids) == 1:
            fallback_thread_id = pending_ids[0]
        try:
            thread_id, response = normalize_review_payload(payload, fallback_thread_id=fallback_thread_id)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        if not review_store.submit_review_response(thread_id, response):
            raise HTTPException(status_code=404, detail="No pending review for thread")
        return {"ok": True}

    return app
