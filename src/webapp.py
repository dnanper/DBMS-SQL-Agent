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
  <title>SQL Agent</title>
  <style>
    :root {
      --bg: #f7f8fb;
      --panel: #ffffff;
      --panel-soft: #eef3f8;
      --ink: #17202a;
      --muted: #647084;
      --line: #d9e0ea;
      --accent: #2563eb;
      --accent-strong: #1d4ed8;
      --ok: #117a55;
      --warn: #a15c07;
      --code-bg: #111827;
      --code-ink: #e5e7eb;
      --shadow: 0 18px 50px rgba(31, 41, 55, 0.12);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      background: var(--bg);
      color: var(--ink);
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }

    button, textarea, input { font: inherit; }
    button {
      border: 1px solid transparent;
      border-radius: 8px;
      padding: 10px 14px;
      cursor: pointer;
      font-weight: 650;
    }
    button:disabled { cursor: not-allowed; opacity: 0.55; }

    .app-shell {
      display: grid;
      grid-template-columns: 260px minmax(0, 1fr) 390px;
      min-height: 100vh;
    }
    .sidebar {
      border-right: 1px solid var(--line);
      background: #fbfcfe;
      padding: 22px 18px;
    }
    .brand {
      display: flex;
      align-items: center;
      gap: 11px;
      margin-bottom: 24px;
    }
    .brand-mark {
      width: 38px;
      height: 38px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: white;
      background: linear-gradient(135deg, #2563eb, #0f766e);
      font-weight: 800;
    }
    .brand h1 { margin: 0; font-size: 18px; line-height: 1.2; }
    .brand p { margin: 2px 0 0; color: var(--muted); font-size: 12px; }
    .sidebar-section { margin-top: 20px; }
    .label {
      color: var(--muted);
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0;
      margin-bottom: 8px;
    }
    .thread-input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px 11px;
      background: white;
      color: var(--ink);
    }
    .example-list { display: grid; gap: 8px; }
    .example-button {
      width: 100%;
      text-align: left;
      background: transparent;
      border-color: var(--line);
      color: #334155;
      font-weight: 560;
      line-height: 1.35;
    }

    .main {
      display: grid;
      grid-template-rows: auto minmax(0, 1fr) auto;
      min-width: 0;
      background: var(--panel);
    }
    .topbar {
      min-height: 72px;
      border-bottom: 1px solid var(--line);
      padding: 18px 24px;
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 16px;
    }
    .topbar h2 { margin: 0; font-size: 20px; line-height: 1.2; }
    .status-pill {
      min-width: 130px;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 7px 12px;
      color: var(--muted);
      background: #f8fafc;
      font-size: 13px;
      text-align: center;
    }
    .status-pill.running { color: var(--accent-strong); border-color: #bfdbfe; background: #eff6ff; }
    .status-pill.waiting { color: var(--warn); border-color: #fed7aa; background: #fff7ed; }

    .conversation { overflow: auto; padding: 24px; }
    .empty-state {
      min-height: 100%;
      display: grid;
      align-content: center;
      gap: 14px;
      max-width: 740px;
      margin: 0 auto;
    }
    .empty-state h3 { margin: 0; font-size: 30px; line-height: 1.15; }
    .empty-state p { margin: 0; color: var(--muted); line-height: 1.6; }
    .message {
      display: grid;
      grid-template-columns: 42px minmax(0, 760px);
      gap: 12px;
      margin-bottom: 18px;
    }
    .message.user { justify-content: end; grid-template-columns: minmax(0, 760px) 42px; }
    .avatar {
      width: 42px;
      height: 42px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      background: var(--panel-soft);
      color: #334155;
      font-weight: 800;
    }
    .message.user .avatar {
      grid-column: 2;
      grid-row: 1;
      background: #dbeafe;
      color: var(--accent-strong);
    }
    .bubble {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 15px;
      line-height: 1.55;
      background: white;
      box-shadow: 0 8px 22px rgba(15, 23, 42, 0.04);
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .message.user .bubble { background: #eff6ff; border-color: #bfdbfe; }

    .composer {
      border-top: 1px solid var(--line);
      padding: 16px 24px 20px;
      background: #fbfcfe;
    }
    .composer-form {
      display: grid;
      grid-template-columns: minmax(0, 1fr) auto;
      gap: 12px;
      align-items: end;
    }
    .question-input {
      width: 100%;
      min-height: 58px;
      max-height: 180px;
      resize: vertical;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 13px 14px;
      background: white;
      color: var(--ink);
    }
    .primary { color: white; background: var(--accent); border-color: var(--accent); }

    .review-panel {
      border-left: 1px solid var(--line);
      background: #fbfcfe;
      padding: 22px;
      overflow: auto;
    }
    .review-card {
      position: sticky;
      top: 22px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .review-header {
      padding: 16px;
      border-bottom: 1px solid var(--line);
      display: flex;
      justify-content: space-between;
      gap: 14px;
      align-items: flex-start;
    }
    .review-header h3 { margin: 0; font-size: 17px; }
    .review-header p { margin: 4px 0 0; color: var(--muted); font-size: 13px; line-height: 1.4; }
    .review-state {
      border-radius: 999px;
      background: #f1f5f9;
      color: var(--muted);
      padding: 5px 9px;
      font-size: 12px;
      font-weight: 750;
      white-space: nowrap;
    }
    .review-state.active { color: var(--warn); background: #fff7ed; }
    .review-body { padding: 16px; display: grid; gap: 14px; }
    .sql-editor {
      width: 100%;
      min-height: 240px;
      resize: vertical;
      border: 1px solid #1f2937;
      border-radius: 8px;
      padding: 13px;
      color: var(--code-ink);
      background: var(--code-bg);
      font-family: "Cascadia Code", Consolas, "Liberation Mono", monospace;
      font-size: 13px;
      line-height: 1.45;
    }
    .review-actions { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
    .approve { background: var(--ok); border-color: var(--ok); color: white; }
    .secondary { background: white; border-color: var(--line); color: #334155; }
    .feedback { grid-column: 1 / -1; background: #fff7ed; border-color: #fed7aa; color: #7c2d12; }
    .hint { margin: 0; color: var(--muted); font-size: 12px; line-height: 1.45; }

    @media (max-width: 1120px) {
      .app-shell { grid-template-columns: 220px minmax(0, 1fr); }
      .review-panel { grid-column: 1 / -1; border-left: 0; border-top: 1px solid var(--line); }
      .review-card { position: static; }
    }
    @media (max-width: 760px) {
      .app-shell { display: block; }
      .sidebar { border-right: 0; border-bottom: 1px solid var(--line); }
      .topbar, .composer, .conversation { padding-left: 16px; padding-right: 16px; }
      .composer-form { grid-template-columns: 1fr; }
      .message, .message.user { grid-template-columns: 34px minmax(0, 1fr); justify-content: stretch; }
      .message.user .avatar { grid-column: 1; }
      .avatar { width: 34px; height: 34px; }
    }
  </style>
</head>
<body>
  <div class="app-shell">
    <aside class="sidebar">
      <div class="brand">
        <div class="brand-mark">SQL</div>
        <div>
          <h1>SQL Agent</h1>
          <p>ClassicModels assistant</p>
        </div>
      </div>

      <div class="sidebar-section">
        <div class="label">Thread</div>
        <input class="thread-input" id="thread-id" placeholder="Auto-generated" />
      </div>

      <div class="sidebar-section">
        <div class="label">Try a question</div>
        <div class="example-list">
          <button class="example-button" data-example="List the top 5 customers by total payment amount.">Top customers by payment</button>
          <button class="example-button" data-example="Which product line has the highest total sales?">Best product line</button>
          <button class="example-button" data-example="Show the 5 most recent orders with customer names and order status.">Recent orders</button>
          <button class="example-button" data-example="Find customers who have not placed any orders.">Customers without orders</button>
        </div>
      </div>
    </aside>

    <main class="main">
      <header class="topbar">
        <div>
          <h2>Ask your database</h2>
        </div>
        <div class="status-pill" id="agent-status">Ready</div>
      </header>

      <section class="conversation" id="conversation-log" aria-live="polite">
        <div class="empty-state" id="empty-state">
          <h3>Review every query before it touches the real database.</h3>
          <p>Ask a business question, inspect the generated SQL, approve it, edit it, or send feedback so the agent can regenerate a better query.</p>
        </div>
      </section>

      <footer class="composer">
        <form class="composer-form" id="composer-form">
          <textarea class="question-input" id="question" placeholder="Ask a database question..."></textarea>
          <button class="primary" id="send" type="submit">Send</button>
        </form>
      </footer>
    </main>

    <aside class="review-panel" id="review-panel">
      <div class="review-card">
        <div class="review-header">
          <div>
            <h3>SQL Review</h3>
            <p>Approve the candidate query, edit it directly, or send feedback for regeneration.</p>
          </div>
          <div class="review-state" id="review-state">Idle</div>
        </div>
        <div class="review-body">
          <textarea class="sql-editor" id="review-input" placeholder="Waiting for generated SQL..." disabled></textarea>
          <div class="review-actions">
            <button class="approve" data-review-action="accept" id="accept" disabled>Approve</button>
            <button class="secondary" data-review-action="edit" id="edit" disabled>Run Edit</button>
            <button class="feedback" data-review-action="response" id="feedback" disabled>Send Feedback</button>
          </div>
          <p class="hint">Feedback mode uses the text in the SQL box as reviewer feedback. Edit mode treats it as the exact SQL to execute after validation.</p>
        </div>
      </div>
    </aside>
  </div>
  <script>
    const conversationLog = document.getElementById("conversation-log");
    const emptyState = document.getElementById("empty-state");
    const statusPill = document.getElementById("agent-status");
    const composerForm = document.getElementById("composer-form");
    const questionInput = document.getElementById("question");
    const sendButton = document.getElementById("send");
    const reviewState = document.getElementById("review-state");
    const reviewInput = document.getElementById("review-input");
    const threadInput = document.getElementById("thread-id");
    let currentThreadId = "";
    let currentRequest = null;
    let source = null;
    let activeAssistantBubble = null;

    function setStatus(text, mode = "") {
      statusPill.textContent = text;
      statusPill.className = `status-pill ${mode}`.trim();
    }

    function setReviewEnabled(enabled) {
      reviewInput.disabled = !enabled;
      document.querySelectorAll("[data-review-action]").forEach((button) => {
        button.disabled = !enabled;
      });
      reviewState.textContent = enabled ? "Review" : "Idle";
      reviewState.className = enabled ? "review-state active" : "review-state";
    }

    function ensureConversationStarted() {
      if (emptyState) emptyState.remove();
    }

    function addMessage(role, text = "") {
      ensureConversationStarted();
      const message = document.createElement("article");
      message.className = `message ${role}`;

      const avatar = document.createElement("div");
      avatar.className = "avatar";
      avatar.textContent = role === "user" ? "You" : "AI";

      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = text;

      if (role === "user") {
        message.append(bubble, avatar);
      } else {
        message.append(avatar, bubble);
      }

      conversationLog.appendChild(message);
      conversationLog.scrollTop = conversationLog.scrollHeight;
      return bubble;
    }

    function appendAssistant(text) {
      if (!activeAssistantBubble) activeAssistantBubble = addMessage("assistant");
      activeAssistantBubble.textContent += text;
      conversationLog.scrollTop = conversationLog.scrollHeight;
    }

    function resetStream() {
      activeAssistantBubble = null;
      reviewInput.value = "";
      currentRequest = null;
      setReviewEnabled(false);
    }

    function startStream(message) {
      if (!message) return;
      resetStream();
      addMessage("user", message);
      questionInput.value = "";
      sendButton.disabled = true;
      setStatus("Starting", "running");
      currentThreadId = threadInput.value.trim() || crypto.randomUUID();
      threadInput.value = currentThreadId;
      if (source) source.close();
      source = new EventSource(`/api/stream?thread_id=${encodeURIComponent(currentThreadId)}&message=${encodeURIComponent(message)}`);
      source.addEventListener("status", (event) => {
        const payload = JSON.parse(event.data);
        setStatus(payload.state || "Running", "running");
      });
      source.addEventListener("chunk", (event) => {
        const payload = JSON.parse(event.data);
        appendAssistant(payload.content);
      });
      source.addEventListener("interrupt", (event) => {
        const payload = JSON.parse(event.data);
        currentRequest = payload.request;
        reviewInput.value = payload.request.args.query || "";
        setReviewEnabled(true);
        setStatus("Waiting for review", "waiting");
        activeAssistantBubble = addMessage("assistant", "I generated SQL and need your review before execution.");
      });
      source.addEventListener("done", () => {
        setStatus("Done");
        sendButton.disabled = false;
        activeAssistantBubble = null;
        source.close();
      });
      source.addEventListener("error", () => {
        setStatus("Stream closed");
        sendButton.disabled = false;
      });
    }

    composerForm.addEventListener("submit", (event) => {
      event.preventDefault();
      startStream(questionInput.value.trim());
    });

    document.querySelectorAll("[data-example]").forEach((button) => {
      button.addEventListener("click", () => {
        questionInput.value = button.dataset.example || "";
        questionInput.focus();
      });
    });

    questionInput.addEventListener("keydown", (event) => {
      if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
        event.preventDefault();
        startStream(questionInput.value.trim());
      }
    });

    async function sendReview(type) {
      if (!currentThreadId) return;
      const body = { thread_id: currentThreadId, type };
      if (type === "edit") body.args = { query: reviewInput.value };
      if (type === "response") body.args = reviewInput.value;
      setStatus(type === "accept" ? "Executing" : "Regenerating", "running");
      const response = await fetch("/api/review", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      if (!response.ok) {
        const detail = await response.text();
        addMessage("assistant", `Review error ${response.status}: ${detail}`);
        setStatus("Review error");
        return;
      }
      setReviewEnabled(false);
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
