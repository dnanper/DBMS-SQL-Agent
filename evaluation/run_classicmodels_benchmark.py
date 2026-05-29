"""Run the SQL agent on ClassicModels benchmark questions.

This creates predictions for later EX/VES evaluation without calculating scores.
Default behavior stops at the human-review interrupt and records the candidate SQL.
Use `--execute-reviewed` to auto-accept each candidate SQL and finish the graph.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import uuid
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
for bootstrap_path in (ROOT, SITE_PACKAGES):
    bootstrap_path_str = str(bootstrap_path)
    if bootstrap_path.exists() and bootstrap_path_str not in sys.path:
        sys.path.insert(0, bootstrap_path_str)

warnings.simplefilter("ignore")

from src.cli import build_resume_payload
from src.config import build_langgraph_config, load_environment
from src.graph import build_agent_graph
from src.tools.sql import strip_sql_fences


DEFAULT_CASES_PATH = ROOT / "evaluation" / "benchmark" / "classicmodels_eval" / "dev.json"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation" / "benchmark" / "classicmodels_runs" / "latest"
DEFAULT_MAX_TURNS = 5


@dataclass(frozen=True)
class BenchmarkCase:
    question_id: int
    db_id: str
    question: str
    difficulty: str
    category: str


def _message_text(message: Any) -> str:
    content = getattr(message, "content", "")
    return content if isinstance(content, str) else str(content)


def load_cases(path: Path, *, limit: int | None = None) -> list[BenchmarkCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    cases = [
        BenchmarkCase(
            question_id=int(row["question_id"]),
            db_id=str(row["db_id"]),
            question=str(row["question"]),
            difficulty=str(row.get("difficulty", "simple")),
            category=str(row.get("category", "")),
        )
        for row in rows
    ]
    return cases[:limit] if limit is not None else cases


def _extract_interrupt_sql(step: dict[str, Any]) -> str:
    interrupt_value = step["__interrupt__"][0].value[0]
    return strip_sql_fences(str(interrupt_value.get("args", {}).get("query", "")))


def run_case(
    *,
    graph: Any,
    case: BenchmarkCase,
    thread_prefix: str,
    run_id: str,
    execute_reviewed: bool,
    max_turns: int,
    logger: logging.Logger | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    thread_id = f"{thread_prefix}-{run_id}-{case.question_id}"
    config = build_langgraph_config(thread_id)
    config["recursion_limit"] = max_turns
    stream_input: Any = {"messages": [{"role": "user", "content": case.question}]}
    candidate_sql = ""
    final_response = ""
    status = "started"
    error = ""
    turn_count = 0

    if logger:
        logger.info("case_start id=%s difficulty=%s question=%s", case.question_id, case.difficulty, case.question)

    try:
        while True:
            interrupted = False
            final_step: dict[str, Any] | None = None

            for step in graph.stream(stream_input, config=config, stream_mode="values"):
                turn_count += 1
                final_step = step
                if logger:
                    logger.info("case_step id=%s turn=%s keys=%s", case.question_id, turn_count, sorted(step.keys()))
                if step.get("candidate_sql"):
                    candidate_sql = strip_sql_fences(str(step["candidate_sql"]))
                if "__interrupt__" in step:
                    candidate_sql = _extract_interrupt_sql(step)
                    if not execute_reviewed:
                        status = "sql_generated"
                        interrupted = False
                        final_step = None
                        break
                    stream_input = build_resume_payload({"type": "accept"})
                    interrupted = True
                    break
                if turn_count >= max_turns:
                    status = "max_turns_exceeded"
                    final_step = None
                    break

            if status == "sql_generated":
                break
            if status == "max_turns_exceeded":
                error = f"Exceeded max_turns={max_turns}"
                break
            if interrupted:
                continue
            if final_step and final_step.get("messages"):
                final_response = _message_text(final_step["messages"][-1])
            status = "completed"
            break
    except Exception as exc:  # keep batch running and preserve failure evidence
        status = "error"
        error = str(exc)

    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    if logger:
        logger.info(
            "case_end id=%s status=%s turns=%s elapsed_ms=%s sql_chars=%s error=%s",
            case.question_id,
            status,
            turn_count,
            elapsed_ms,
            len(candidate_sql),
            error,
        )
    return {
        "question_id": case.question_id,
        "db_id": case.db_id,
        "question": case.question,
        "thread_id": thread_id,
        "run_id": run_id,
        "difficulty": case.difficulty,
        "category": case.category,
        "candidate_sql": candidate_sql,
        "final_response": final_response,
        "status": status,
        "error": error,
        "turn_count": turn_count,
        "max_turns": max_turns,
        "elapsed_ms": elapsed_ms,
    }


def _prediction_row(result: dict[str, Any]) -> list[Any]:
    sql = result.get("candidate_sql") or " "
    return [result["question_id"], f"{sql}\t----- bird -----\t{result['db_id']}"]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def build_benchmark_logger(path: Path) -> logging.Logger:
    logger = logging.getLogger(f"classicmodels_benchmark.{path}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def close_benchmark_logger(logger: logging.Logger) -> None:
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)


def run_benchmark(
    *,
    graph: Any | None = None,
    graph_factory: Callable[[], Any] | None = None,
    cases: list[BenchmarkCase],
    output_dir: Path,
    execute_reviewed: bool,
    max_turns: int,
    workers: int = 1,
    run_id: str | None = None,
    thread_prefix: str = "classicmodels-benchmark",
) -> dict[str, Any]:
    if workers < 1:
        raise ValueError("workers must be >= 1")
    if graph is None and graph_factory is None:
        raise ValueError("graph or graph_factory is required")
    if workers > 1 and graph_factory is None:
        raise ValueError("workers > 1 requires graph_factory so each worker has isolated graph state")

    output_dir.mkdir(parents=True, exist_ok=True)
    logger = build_benchmark_logger(output_dir / "benchmark.log")
    resolved_run_id = run_id or uuid.uuid4().hex[:12]

    def run_one(case: BenchmarkCase) -> dict[str, Any]:
        runtime_graph = graph if graph is not None and workers == 1 else graph_factory()
        return run_case(
            graph=runtime_graph,
            case=case,
            thread_prefix=thread_prefix,
            run_id=resolved_run_id,
            execute_reviewed=execute_reviewed,
            max_turns=max_turns,
            logger=logger,
        )

    try:
        if workers == 1:
            results = [run_one(case) for case in cases]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(run_one, cases))
        predictions = [_prediction_row(result) for result in results]

        write_json(output_dir / "predicted_sql.json", predictions)
        write_jsonl(output_dir / "run_results.jsonl", results)
        blank_sql_count = sum(1 for result in results if not result["candidate_sql"])
        max_turns_exceeded_count = sum(1 for result in results if result["status"] == "max_turns_exceeded")
        summary = {
            "case_count": len(results),
            "run_id": resolved_run_id,
            "ok_count": sum(1 for result in results if result["candidate_sql"]),
            "error_count": sum(1 for result in results if result["status"] == "error"),
            "blank_sql_count": blank_sql_count,
            "max_turns_exceeded_count": max_turns_exceeded_count,
            "ready_for_eval": blank_sql_count == 0,
            "execute_reviewed": execute_reviewed,
            "max_turns": max_turns,
            "workers": workers,
            "predicted_sql_path": str(output_dir / "predicted_sql.json"),
            "run_results_path": str(output_dir / "run_results.jsonl"),
            "log_path": str(output_dir / "benchmark.log"),
        }
        write_json(output_dir / "summary.json", summary)
        return summary
    finally:
        close_benchmark_logger(logger)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run SQL agent over ClassicModels benchmark cases.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--execute-reviewed", action="store_true")
    parser.add_argument("--thread-prefix", default="classicmodels-benchmark")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    load_environment()
    cases = load_cases(args.cases, limit=args.limit)
    summary = run_benchmark(
        graph_factory=build_agent_graph,
        cases=cases,
        output_dir=args.output_dir,
        execute_reviewed=args.execute_reviewed,
        max_turns=args.max_turns,
        workers=args.workers,
        run_id=args.run_id,
        thread_prefix=args.thread_prefix,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
