"""Prepare ClassicModels benchmark files for MAG-SQL.

MAG-SQL expects BIRD-style paths:
- data/bird/dev/dev.json
- data/bird/dev/dev_tables.json
- data/bird/dev/dev_databases/{db_id}/{db_id}.sqlite

This script reads the current ClassicModels Postgres DB configured by `.env`,
exports table data to SQLite, and writes the BIRD metadata files.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SITE_PACKAGES = ROOT / ".venv" / "Lib" / "site-packages"
for bootstrap_path in (ROOT, SITE_PACKAGES):
    bootstrap_path_str = str(bootstrap_path)
    if bootstrap_path.exists() and bootstrap_path_str not in sys.path:
        sys.path.insert(0, bootstrap_path_str)

from sqlalchemy import create_engine, inspect, text

from src.config import get_database_schema, get_database_uri, load_environment
from src.database import normalize_database_uri


DEFAULT_INPUT = ROOT / "evaluation" / "benchmark" / "classicmodels_eval" / "dev.json"
DEFAULT_OUTPUT = ROOT / "evaluation" / "baseline" / "MAG-SQL" / "data" / "bird" / "dev"
DEFAULT_DB_ID = "classicmodels"


def sqlite_type(source_type: Any) -> str:
    normalized = str(source_type).lower()
    if any(token in normalized for token in ("int", "serial")):
        return "INTEGER"
    if any(token in normalized for token in ("numeric", "decimal", "double", "real", "float")):
        return "REAL"
    return "TEXT"


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def load_cases(path: Path) -> list[dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def export_sqlite_database(output_dir: Path, db_id: str) -> dict[str, Any]:
    load_environment(ROOT / ".env")
    engine = create_engine(normalize_database_uri(get_database_uri()))
    schema = get_database_schema()
    inspector = inspect(engine)
    table_names = sorted(inspector.get_table_names(schema=schema))

    sqlite_db_dir = output_dir / "dev_databases" / db_id
    sqlite_db_dir.mkdir(parents=True, exist_ok=True)
    sqlite_path = sqlite_db_dir / f"{db_id}.sqlite"
    if sqlite_path.exists():
        sqlite_path.unlink()

    column_names: list[list[Any]] = [[-1, "*"]]
    column_names_original: list[list[Any]] = [[-1, "*"]]
    column_types: list[str] = ["text"]
    primary_keys: list[int] = []
    foreign_keys: list[list[int]] = []
    column_id_by_table_column: dict[tuple[str, str], int] = {}

    sqlite_conn = sqlite3.connect(sqlite_path)
    try:
        for table_index, table_name in enumerate(table_names):
            columns = inspector.get_columns(table_name, schema=schema)
            create_columns = []
            for column in columns:
                name = column["name"]
                column_type = sqlite_type(column["type"])
                column_index = len(column_names)
                column_names.append([table_index, name.lower()])
                column_names_original.append([table_index, name])
                column_types.append(column_type.lower())
                column_id_by_table_column[(table_name, name)] = column_index
                create_columns.append(f"{quote_identifier(name)} {column_type}")

            sqlite_conn.execute(
                f"CREATE TABLE {quote_identifier(table_name)} ({', '.join(create_columns)})"
            )

            pk = inspector.get_pk_constraint(table_name, schema=schema).get("constrained_columns") or []
            for column_name in pk:
                column_index = column_id_by_table_column.get((table_name, column_name))
                if column_index is not None:
                    primary_keys.append(column_index)

            with engine.connect() as source_conn:
                rows = source_conn.execute(text(f"SELECT * FROM {quote_identifier(schema)}.{quote_identifier(table_name)}"))
                column_keys = list(rows.keys())
                placeholders = ", ".join("?" for _ in column_keys)
                insert_sql = (
                    f"INSERT INTO {quote_identifier(table_name)} "
                    f"({', '.join(quote_identifier(column) for column in column_keys)}) "
                    f"VALUES ({placeholders})"
                )
                for row in rows:
                    sqlite_conn.execute(insert_sql, [serialize_sqlite_value(value) for value in row])

        for table_name in table_names:
            for fk in inspector.get_foreign_keys(table_name, schema=schema):
                referred_table = fk.get("referred_table")
                constrained = fk.get("constrained_columns") or []
                referred = fk.get("referred_columns") or []
                for from_column, to_column in zip(constrained, referred):
                    from_index = column_id_by_table_column.get((table_name, from_column))
                    to_index = column_id_by_table_column.get((referred_table, to_column))
                    if from_index is not None and to_index is not None:
                        foreign_keys.append([from_index, to_index])

        sqlite_conn.commit()
    finally:
        sqlite_conn.close()

    return {
        "db_id": db_id,
        "table_names": [name.lower() for name in table_names],
        "table_names_original": table_names,
        "column_names": column_names,
        "column_names_original": column_names_original,
        "column_types": column_types,
        "primary_keys": primary_keys,
        "foreign_keys": foreign_keys,
    }


def serialize_sqlite_value(value: Any) -> Any:
    if value is None or isinstance(value, (int, float, str)):
        return value
    return str(value)


def prepare_magsql_dataset(input_path: Path, output_dir: Path, db_id: str) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = load_cases(input_path)
    for case in cases:
        case.setdefault("evidence", "")
    write_json(output_dir / "dev.json", cases)
    (output_dir / "dev_gold.sql").write_text(
        "".join(f"{case['SQL'].strip()}\t{db_id}\n" for case in cases),
        encoding="utf-8",
    )
    table_metadata = export_sqlite_database(output_dir, db_id)
    write_json(output_dir / "dev_tables.json", [table_metadata])
    return {
        "dev_json": str(output_dir / "dev.json"),
        "dev_gold": str(output_dir / "dev_gold.sql"),
        "tables_json": str(output_dir / "dev_tables.json"),
        "db_path": str(output_dir / "dev_databases"),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare ClassicModels BIRD-style input for MAG-SQL.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--db-id", default=DEFAULT_DB_ID)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = prepare_magsql_dataset(args.input, args.output_dir, args.db_id)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
