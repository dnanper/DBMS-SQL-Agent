"""Create ClassicModels evaluation files for EX and VES metrics.

The MAG-SQL BIRD evaluators need:
- a JSON file with SQL and difficulty metadata (`diff_json_path`)
- a ground-truth SQL text file with `SQL<TAB>db_id` rows
- a predicted SQL JSON file shaped like `[idx, "SQL\t----- bird -----\tdb_id"]`
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("evaluation/benchmark/ClassicModels-SQL-Solutions")
DEFAULT_OUTPUT_DIR = Path("evaluation/benchmark/classicmodels_eval")
QUESTION_PREFIX_RE = re.compile(r"^\s*(?:\d+[\).\s-]*)?(?P<question>.+?)\s*$")
QUERY_FILE_RE = re.compile(r"query_(\d+)\.sql$", re.IGNORECASE)

CHALLENGING_CATEGORIES = {"Correlated-Subqueries", "Spatial-Data"}
MODERATE_CATEGORIES = {"Many-to-Many-Relationship", "Regular-Expressions"}
MODERATE_SQL_RE = re.compile(
    r"\b(join|group\s+by|having|union|intersect|except|regexp|with|over)\b|\(\s*select\b",
    re.IGNORECASE,
)
MUTATING_SQL_RE = re.compile(
    r"\b(update|delete|insert|alter|drop|truncate|call|procedure|function|delimiter)\b",
    re.IGNORECASE,
)
READ_ONLY_START_RE = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)


def normalize_sql(sql: str) -> str:
    return sql.strip()


def one_line_sql(sql: str) -> str:
    return re.sub(r"\s+", " ", normalize_sql(sql))


def split_sql_statements(sql: str) -> list[str]:
    return [normalize_sql(part) + ";" for part in sql.split(";") if normalize_sql(part)]


def strip_sql_comment(line: str) -> str:
    return line.split("--", 1)[0].rstrip()


def select_metric_sql(sql: str) -> str | None:
    if MUTATING_SQL_RE.search(sql):
        return None
    for statement in reversed(split_sql_statements(sql)):
        if READ_ONLY_START_RE.match(statement):
            return statement
    return None


def infer_difficulty(category: str, sql: str) -> str:
    if category in CHALLENGING_CATEGORIES:
        return "challenging"
    if category in MODERATE_CATEGORIES or MODERATE_SQL_RE.search(sql):
        return "moderate"
    return "simple"


def extract_question_and_sql(sql_text: str) -> tuple[str, str]:
    lines = sql_text.replace("\r\n", "\n").split("\n")
    question = ""
    sql_start = 0

    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("--"):
            if not question:
                raw_question = stripped[2:].strip()
                match = QUESTION_PREFIX_RE.match(raw_question)
                question = match.group("question") if match else raw_question
            sql_start = index + 1
            continue
        if stripped == "":
            sql_start = index + 1
            continue
        break

    body_lines = lines[sql_start:]
    for index, line in enumerate(body_lines):
        if re.match(r"^\s*--\s*final\s+query\b", line, re.IGNORECASE):
            body_lines = body_lines[index + 1 :]
            break

    sql = normalize_sql(
        "\n".join(
            stripped
            for line in body_lines
            if not line.strip().startswith("--")
            for stripped in [strip_sql_comment(line)]
            if stripped.strip()
        )
    )
    return question, sql


def natural_query_key(path: Path) -> tuple[str, int, str]:
    match = QUERY_FILE_RE.match(path.name)
    number = int(match.group(1)) if match else 0
    return path.parent.name, number, path.name


def iter_query_files(root: Path) -> list[Path]:
    return sorted(root.glob("*/query_*.sql"), key=natural_query_key)


def build_case(
    sql_path: Path,
    root: Path,
    db_id: str,
    dialect: str,
    schema: str,
    index: int,
    sql_text: str | None = None,
) -> dict[str, Any]:
    text = sql_text if sql_text is not None else sql_path.read_text(encoding="utf-8-sig")
    question, extracted_sql = extract_question_and_sql(text)
    sql = select_metric_sql(extracted_sql)
    if sql is None:
        raise ValueError("No read-only SELECT/WITH statement suitable for EX/VES")
    relative_path = sql_path.relative_to(root).as_posix() if sql_path.is_absolute() else sql_path.as_posix()
    category = sql_path.parent.name

    if not question:
        question = f"ClassicModels SQL task {relative_path}"

    return {
        "question_id": index,
        "db_id": db_id,
        "dialect": dialect,
        "schema": schema,
        "category": category,
        "difficulty": infer_difficulty(category, sql),
        "question": question,
        "SQL": sql,
        "source_path": relative_path,
    }


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def create_dataset(
    root: Path = DEFAULT_ROOT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    db_id: str = "classicmodels",
    dialect: str = "postgresql",
    schema: str = "classicmodels",
) -> dict[str, Any]:
    root = root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    cases = []
    skipped = []
    for path in iter_query_files(root):
        try:
            cases.append(
                build_case(
                    sql_path=path,
                    root=root,
                    db_id=db_id,
                    dialect=dialect,
                    schema=schema,
                    index=len(cases),
                )
            )
        except ValueError as exc:
            skipped.append({"source_path": path.relative_to(root).as_posix(), "reason": str(exc)})

    if not cases:
        raise ValueError(f"No query_*.sql files found under {root}")

    bird_dev = [
        {
            "question_id": case["question_id"],
            "db_id": case["db_id"],
            "question": case["question"],
            "SQL": one_line_sql(case["SQL"]),
            "difficulty": case["difficulty"],
            "category": case["category"],
        }
        for case in cases
    ]
    gold_lines = [f"{one_line_sql(case['SQL'])}\t{case['db_id']}" for case in cases]
    predicted_template = [
        [case["question_id"], f" \t----- bird -----\t{case['db_id']}"] for case in cases
    ]

    write_json(output_dir / "dev.json", bird_dev)
    (output_dir / "dev_gold.sql").write_text("\n".join(gold_lines) + "\n", encoding="utf-8")
    write_json(output_dir / "predicted_sql_template.json", predicted_template)
    write_jsonl(output_dir / "cases.jsonl", cases)
    write_json(output_dir / "skipped.json", skipped)

    return {
        "case_count": len(cases),
        "skipped_count": len(skipped),
        "output_dir": str(output_dir),
        "dev_json": str(output_dir / "dev.json"),
        "ground_truth_sql": str(output_dir / "dev_gold.sql"),
        "predicted_sql_template": str(output_dir / "predicted_sql_template.json"),
        "cases_jsonl": str(output_dir / "cases.jsonl"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ClassicModels EX/VES eval dataset files.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--db-id", default="classicmodels")
    parser.add_argument("--dialect", default="postgresql")
    parser.add_argument("--schema", default="classicmodels")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = create_dataset(
        root=args.root,
        output_dir=args.output_dir,
        db_id=args.db_id,
        dialect=args.dialect,
        schema=args.schema,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
