import json
import tempfile
import unittest
from pathlib import Path

from evaluation.create_classicmodels_eval_dataset import (
    build_case,
    create_dataset,
    extract_question_and_sql,
    select_metric_sql,
)


class ClassicModelsEvalDatasetTest(unittest.TestCase):
    def test_extracts_question_and_removes_leading_comments(self) -> None:
        text = """-- 7. List the top customers.
-- extra note
SELECT customerName
FROM customers;
"""

        question, sql = extract_question_and_sql(text)

        self.assertEqual(question, "List the top customers.")
        self.assertEqual(sql, "SELECT customerName\nFROM customers;")

    def test_extracts_final_query_when_solution_has_working_queries(self) -> None:
        text = """-- 4. Answer the final task.
SELECT scratch FROM example;

-- Final query
SELECT final_answer
FROM example;
"""

        question, sql = extract_question_and_sql(text)

        self.assertEqual(question, "Answer the final task.")
        self.assertEqual(sql, "SELECT final_answer\nFROM example;")

    def test_extracts_sql_without_inline_comments(self) -> None:
        text = """-- 9. Monthly revenue.
SELECT month, -- needed for ordering
SUM(amount)
FROM payments;
"""

        question, sql = extract_question_and_sql(text)

        self.assertEqual(question, "Monthly revenue.")
        self.assertEqual(sql, "SELECT month,\nSUM(amount)\nFROM payments;")

    def test_select_metric_sql_takes_last_select_and_skips_mutating_scripts(self) -> None:
        self.assertEqual(
            select_metric_sql("CREATE VIEW x AS SELECT 1; SELECT * FROM x;"),
            "SELECT * FROM x;",
        )
        self.assertIsNone(
            select_metric_sql("CREATE PROCEDURE p() BEGIN UPDATE customers SET creditLimit = 1; END; CALL p();")
        )

    def test_build_case_sets_bird_required_fields(self) -> None:
        case = build_case(
            sql_path=Path("General-Queries/query_2.sql"),
            root=Path("."),
            db_id="classicmodels",
            dialect="postgresql",
            schema="classicmodels",
            index=3,
            sql_text="-- 2. Who reports to William Patterson?\nSELECT * FROM employees;",
        )

        self.assertEqual(case["question_id"], 3)
        self.assertEqual(case["db_id"], "classicmodels")
        self.assertEqual(case["difficulty"], "simple")
        self.assertEqual(case["category"], "General-Queries")
        self.assertEqual(case["question"], "Who reports to William Patterson?")
        self.assertEqual(case["SQL"], "SELECT * FROM employees;")

    def test_create_dataset_writes_bird_and_jsonl_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "ClassicModels-SQL-Solutions"
            category = root / "General-Queries"
            category.mkdir(parents=True)
            (category / "query_1.sql").write_text(
                "-- 1. Who is at the top?\nSELECT * FROM employees WHERE reportsTo IS NULL;\n",
                encoding="utf-8",
            )
            (category / "query_2.sql").write_text(
                "-- 2. Mutating task.\nUPDATE customers SET creditLimit = 1;\n",
                encoding="utf-8",
            )
            out = Path(tmp) / "out"

            summary = create_dataset(root=root, output_dir=out)

            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["skipped_count"], 1)
            self.assertTrue((out / "dev.json").exists())
            self.assertTrue((out / "skipped.json").exists())
            self.assertTrue((out / "dev_gold.sql").exists())
            self.assertTrue((out / "predicted_sql_template.json").exists())
            self.assertTrue((out / "cases.jsonl").exists())
            dev = json.loads((out / "dev.json").read_text(encoding="utf-8"))
            self.assertEqual(dev[0]["SQL"], "SELECT * FROM employees WHERE reportsTo IS NULL;")
            self.assertEqual((out / "dev_gold.sql").read_text(encoding="utf-8").strip(), dev[0]["SQL"] + "\tclassicmodels")


if __name__ == "__main__":
    unittest.main()
