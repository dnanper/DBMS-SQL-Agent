"""Evaluate and compare ClassicModels SQL predictions with EX and VES.

The BIRD reference scripts evaluate one prediction file at a time. This module
keeps the same metric semantics, but accepts both MAG-SQL's question-keyed
output and this repo's question_id-keyed benchmark output, then writes one
comparison report.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
for bootstrap_path in (ROOT, SITE_PACKAGES):
    bootstrap_path_str = str(bootstrap_path)
    if bootstrap_path.exists() and bootstrap_path_str not in sys.path:
        sys.path.insert(0, bootstrap_path_str)

DEFAULT_DATASET = ROOT / "evaluation" / "benchmark" / "classicmodels_eval" / "dev.json"
DEFAULT_OUTPUT_DIR = ROOT / "evaluation" / "benchmark" / "classicmodels_eval" / "comparison"
DEFAULT_MAG_PREDICTIONS = ROOT / "evaluation" / "baseline" / "MAG-SQL" / "output" / "predict_dev_new.json"
DEFAULT_AGENT_PREDICTIONS = ROOT / "evaluation" / "benchmark" / "classicmodels_runs" / "latest" / "predicted_sql.json"
BIRD_DELIMITER = "\t----- bird -----\t"
LEVELS = ("simple", "moderate", "challenging", "total")


@dataclass(frozen=True)
class QueryCase:
    question_id: int
    question: str
    gold_sql: str
    difficulty: str
    category: str
    db_id: str


@dataclass(frozen=True)
class Prediction:
    question_id: int
    sql: str
    db_id: str


@dataclass(frozen=True)
class CaseResult:
    question_id: int
    question: str
    difficulty: str
    category: str
    gold_sql: str
    predicted_sql: str
    ex: int
    ves: float
    error: str


@dataclass(frozen=True)
class SystemEvaluation:
    name: str
    summary: dict[str, dict[str, float]]
    items: list[CaseResult]


class SqlExecutor(Protocol):
    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        ...


class PostgresExecutor:
    def __init__(self, database_uri: str, schema: str | None = None, statement_timeout_ms: int = 30_000) -> None:
        import psycopg

        self._psycopg = psycopg
        self.database_uri = normalize_psycopg_uri(database_uri)
        self.schema = schema
        self.statement_timeout_ms = statement_timeout_ms

    def execute(self, sql: str) -> list[tuple[Any, ...]]:
        rewritten_sql = rewrite_mysqlisms_for_postgres(sql)
        with self._psycopg.connect(self.database_uri) as conn:
            with conn.cursor() as cur:
                cur.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
                if self.schema:
                    cur.execute(f"SET search_path TO {quote_identifier(self.schema)}")
                cur.execute(rewritten_sql)
                return cur.fetchall()


def normalize_psycopg_uri(database_uri: str) -> str:
    return database_uri.replace("postgresql+psycopg://", "postgresql://", 1)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().lower()


def normalize_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", sql).strip()


def split_bird_sql(value: Any) -> tuple[str, str]:
    if not isinstance(value, str):
        return "", "classicmodels"
    if BIRD_DELIMITER not in value:
        return value.strip(), "classicmodels"
    sql, db_id = value.split(BIRD_DELIMITER, 1)
    return sql.strip(), db_id.strip()


def load_cases(path: Path) -> list[QueryCase]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    return [
        QueryCase(
            question_id=int(row["question_id"]),
            question=str(row["question"]),
            gold_sql=str(row["SQL"]),
            difficulty=str(row.get("difficulty", "simple")),
            category=str(row.get("category", "")),
            db_id=str(row.get("db_id", "classicmodels")),
        )
        for row in rows
    ]


def load_predictions(path: Path, cases: list[QueryCase]) -> dict[int, Prediction]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    question_to_id = {normalize_question(case.question): case.question_id for case in cases}
    predictions: dict[int, Prediction] = {}

    for row in rows:
        if not isinstance(row, list | tuple) or len(row) < 2:
            continue
        raw_key, raw_value = row[0], row[1]
        sql, db_id = split_bird_sql(raw_value)
        question_id: int | None = None
        if isinstance(raw_key, int):
            question_id = raw_key
        elif isinstance(raw_key, str):
            question_id = question_to_id.get(normalize_question(raw_key))
        if question_id is not None:
            predictions[question_id] = Prediction(question_id=question_id, sql=sql, db_id=db_id)

    return predictions


def _find_matching_paren(text: str, open_index: int) -> int:
    depth = 0
    for index in range(open_index, len(text)):
        char = text[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _split_args(text: str) -> list[str]:
    args: list[str] = []
    depth = 0
    start = 0
    for index, char in enumerate(text):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == "," and depth == 0:
            args.append(text[start:index].strip())
            start = index + 1
    args.append(text[start:].strip())
    return args


def _replace_function_calls(sql: str, name: str, renderer: Any) -> str:
    pattern = re.compile(rf"\b{name}\s*\(", re.IGNORECASE)
    offset = 0
    output = sql
    while True:
        match = pattern.search(output, offset)
        if not match:
            return output
        open_index = match.end() - 1
        close_index = _find_matching_paren(output, open_index)
        if close_index == -1:
            return output
        args = _split_args(output[open_index + 1 : close_index])
        replacement = renderer(args)
        output = output[: match.start()] + replacement + output[close_index + 1 :]
        offset = match.start() + len(replacement)


def rewrite_mysqlisms_for_postgres(sql: str) -> str:
    rewritten = sql.replace("`", "")
    rewritten = re.sub(r"\bCURDATE\(\)", "CURRENT_DATE", rewritten, flags=re.IGNORECASE)
    rewritten = _replace_function_calls(
        rewritten,
        "DATEDIFF",
        lambda args: f"({args[0]}::date - {args[1]}::date)" if len(args) == 2 else f"DATEDIFF({', '.join(args)})",
    )
    rewritten = _replace_function_calls(
        rewritten,
        "MONTHNAME",
        lambda args: f"TO_CHAR({args[0]}, 'Month')" if len(args) == 1 else f"MONTHNAME({', '.join(args)})",
    )
    rewritten = _replace_function_calls(
        rewritten,
        "MONTH",
        lambda args: f"EXTRACT(MONTH FROM {args[0]})" if len(args) == 1 else f"MONTH({', '.join(args)})",
    )
    rewritten = _replace_function_calls(
        rewritten,
        "YEAR",
        lambda args: f"EXTRACT(YEAR FROM {args[0]})" if len(args) == 1 else f"YEAR({', '.join(args)})",
    )
    return rewritten


def _normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, str, bool)):
        return value
    return str(value)


def compare_rows(predicted_rows: list[tuple[Any, ...]], gold_rows: list[tuple[Any, ...]]) -> bool:
    predicted = {tuple(_normalize_value(value) for value in row) for row in predicted_rows}
    gold = {tuple(_normalize_value(value) for value in row) for row in gold_rows}
    return predicted == gold


def clean_abnormal(values: list[float]) -> list[float]:
    if len(values) < 2:
        return values
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    if std == 0:
        return values
    cleaned = [value for value in values if mean - 3 * std < value < mean + 3 * std]
    return cleaned or values


def calculate_ves(predicted_sql: str, gold_sql: str, executor: SqlExecutor, iterate_num: int) -> float:
    ratios: list[float] = []
    for _ in range(iterate_num):
        gold_start = time.perf_counter()
        executor.execute(gold_sql)
        gold_time = max(time.perf_counter() - gold_start, 1e-9)

        pred_start = time.perf_counter()
        executor.execute(predicted_sql)
        pred_time = max(time.perf_counter() - pred_start, 1e-9)
        ratios.append(gold_time / pred_time)
    cleaned = clean_abnormal(ratios)
    return math.sqrt(sum(cleaned) / len(cleaned)) * 100


def _empty_level() -> dict[str, float]:
    return {"count": 0, "ex": 0.0, "ves": 0.0}


def summarize(items: list[CaseResult]) -> dict[str, dict[str, float]]:
    summary = {level: _empty_level() for level in LEVELS}
    for level in LEVELS:
        subset = items if level == "total" else [item for item in items if item.difficulty == level]
        if not subset:
            continue
        summary[level] = {
            "count": len(subset),
            "ex": sum(item.ex for item in subset) / len(subset) * 100,
            "ves": sum(item.ves for item in subset) / len(subset),
        }
    return summary


def evaluate_system(
    *,
    name: str,
    cases: list[QueryCase],
    predictions: dict[int, Prediction],
    executor: SqlExecutor,
    include_ves: bool,
    iterate_num: int = 10,
) -> SystemEvaluation:
    items: list[CaseResult] = []
    for case in cases:
        prediction = predictions.get(case.question_id)
        predicted_sql = prediction.sql if prediction else ""
        ex = 0
        ves = 0.0
        error = ""
        try:
            if not predicted_sql:
                raise ValueError("missing prediction")
            predicted_rows = executor.execute(predicted_sql)
            gold_rows = executor.execute(case.gold_sql)
            ex = int(compare_rows(predicted_rows, gold_rows))
            if include_ves and ex:
                ves = calculate_ves(predicted_sql, case.gold_sql, executor, iterate_num)
        except Exception as exc:
            error = str(exc)
        items.append(
            CaseResult(
                question_id=case.question_id,
                question=case.question,
                difficulty=case.difficulty,
                category=case.category,
                gold_sql=case.gold_sql,
                predicted_sql=predicted_sql,
                ex=ex,
                ves=ves,
                error=error,
            )
        )
    return SystemEvaluation(name=name, summary=summarize(items), items=items)


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, evaluations: list[SystemEvaluation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["system", "difficulty", "count", "ex", "ves"])
        writer.writeheader()
        for evaluation in evaluations:
            for level in LEVELS:
                row = evaluation.summary[level]
                writer.writerow(
                    {
                        "system": evaluation.name,
                        "difficulty": level,
                        "count": int(row["count"]),
                        "ex": f"{row['ex']:.2f}",
                        "ves": f"{row['ves']:.2f}",
                    }
                )


def write_markdown(path: Path, evaluations: list[SystemEvaluation]) -> None:
    lines = [
        "# ClassicModels EX/VES Comparison",
        "",
        "| System | Difficulty | Count | EX | VES |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for evaluation in evaluations:
        for level in LEVELS:
            row = evaluation.summary[level]
            lines.append(
                f"| {evaluation.name} | {level} | {int(row['count'])} | {row['ex']:.2f} | {row['ves']:.2f} |"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_case_csv(path: Path, evaluations: list[SystemEvaluation]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "system",
                "question_id",
                "difficulty",
                "category",
                "ex",
                "ves",
                "error",
                "question",
                "predicted_sql",
                "gold_sql",
            ],
        )
        writer.writeheader()
        for evaluation in evaluations:
            for item in evaluation.items:
                row = asdict(item)
                row["system"] = evaluation.name
                writer.writerow(row)


def parse_system_arg(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("system must be NAME=PATH")
    name, path = value.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("system name cannot be empty")
    return name.strip(), Path(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare ClassicModels SQL systems with EX and VES.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--database-uri", default=None)
    parser.add_argument("--schema", default=None)
    parser.add_argument("--statement-timeout-ms", type=int, default=30_000)
    parser.add_argument("--iterate-num", type=int, default=10)
    parser.add_argument("--skip-ves", action="store_true", help="Only calculate EX.")
    parser.add_argument(
        "--system",
        type=parse_system_arg,
        action="append",
        default=None,
        help="Prediction file as NAME=PATH. Repeat to compare more systems.",
    )
    return parser.parse_args()


def main() -> int:
    from src.config import get_database_schema, get_database_uri, load_environment

    args = parse_args()
    load_environment(ROOT / ".env")
    cases = load_cases(args.dataset)
    systems = args.system or [
        ("mag-sql", DEFAULT_MAG_PREDICTIONS),
        ("agent", DEFAULT_AGENT_PREDICTIONS),
    ]
    database_uri = args.database_uri or get_database_uri()
    schema = args.schema if args.schema is not None else get_database_schema()
    executor = PostgresExecutor(database_uri, schema=schema, statement_timeout_ms=args.statement_timeout_ms)

    evaluations = [
        evaluate_system(
            name=name,
            cases=cases,
            predictions=load_predictions(path, cases),
            executor=executor,
            include_ves=not args.skip_ves,
            iterate_num=args.iterate_num,
        )
        for name, path in systems
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "dataset": str(args.dataset),
        "database_uri": database_uri,
        "schema": schema,
        "include_ves": not args.skip_ves,
        "iterate_num": args.iterate_num,
        "systems": [
            {
                "name": evaluation.name,
                "summary": evaluation.summary,
                "items": [asdict(item) for item in evaluation.items],
            }
            for evaluation in evaluations
        ],
    }
    write_json(args.output_dir / "comparison.json", payload)
    write_csv(args.output_dir / "summary.csv", evaluations)
    write_case_csv(args.output_dir / "cases.csv", evaluations)
    write_markdown(args.output_dir / "summary.md", evaluations)
    print((args.output_dir / "summary.md").read_text(encoding="utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
