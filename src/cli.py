"""Simple CLI for chatting with the SQL agent."""

from __future__ import annotations

import sys
import uuid
import logging
from typing import Any, Callable, TextIO

from .config import (
    DEFAULT_THREAD_ID,
    build_langgraph_config,
    build_log_file_path,
    load_environment,
)
from .graph import build_agent_graph
from .logging_utils import configure_file_logger


def is_exit_command(value: str) -> bool:
    return value.strip().lower() in {"exit", "quit", ":q"}


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(str(item) for item in content)
    return str(content)


def build_resume_payload(
    response: dict[str, Any],
    *,
    command_factory: Callable[..., Any] | None = None,
) -> Any:
    if command_factory is None:
        from langgraph.types import Command

        command_factory = Command

    return command_factory(resume=response)


def prompt_for_review(
    request: dict[str, Any],
    *,
    input_func: Callable[[str], str],
    output: TextIO,
) -> dict[str, Any]:
    print("Review required before executing SQL.", file=output)
    print(f"Action: {request['action']}", file=output)
    print("SQL:", file=output)
    print(request["args"].get("query", ""), file=output)
    print("Options: [a]ccept, [e]dit, [f]eedback", file=output)

    choice = input_func("Review choice: ").strip().lower()
    if choice in {"a", "accept", ""}:
        return {"type": "accept"}
    if choice in {"e", "edit"}:
        edited_query = input_func("Edited SQL: ").strip()
        return {"type": "edit", "args": {"query": edited_query}}

    feedback = input_func("Feedback to assistant: ").strip()
    return {"type": "response", "args": feedback}


def run_cli_turn(
    *,
    graph: Any,
    thread_id: str,
    user_input: str,
    output: TextIO,
    logger: logging.Logger,
    input_func: Callable[[str], str] = input,
    command_factory: Callable[..., Any] | None = None,
) -> None:
    config = build_langgraph_config(thread_id)
    logger.info("User: %s", user_input)
    stream_input: Any = {"messages": [{"role": "user", "content": user_input}]}

    while True:
        final_step: dict[str, Any] | None = None
        interrupted = False

        for step in graph.stream(stream_input, config=config, stream_mode="values"):
            final_step = step
            if "__interrupt__" in step:
                request = step["__interrupt__"][0].value[0]
                review_response = prompt_for_review(request, input_func=input_func, output=output)
                logger.info("Review response: %s", review_response["type"])
                stream_input = build_resume_payload(review_response, command_factory=command_factory)
                interrupted = True
                break

        if interrupted:
            continue

        if final_step and "messages" in final_step:
            final_message = final_step["messages"][-1]
            assistant_text = _message_text(final_message)
            logger.info("Assistant: %s", assistant_text)
            print(f"Assistant: {assistant_text}", file=output)
        return


def main() -> int:
    load_environment()
    logger = configure_file_logger(build_log_file_path())
    graph = build_agent_graph()
    thread_id = DEFAULT_THREAD_ID or str(uuid.uuid4())

    print("SQL Agent CLI. Type 'exit' to quit.")
    logger.info("CLI started with thread_id=%s", thread_id)
    while True:
        try:
            user_input = input("You: ")
        except EOFError:
            logger.info("CLI stopped by EOF")
            print()
            return 0

        if is_exit_command(user_input):
            logger.info("CLI stopped by exit command")
            return 0
        if not user_input.strip():
            continue

        run_cli_turn(
            graph=graph,
            thread_id=thread_id,
            user_input=user_input,
            output=sys.stdout,
            logger=logger,
            input_func=input,
        )


if __name__ == "__main__":
    raise SystemExit(main())
