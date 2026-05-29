import json
import tempfile
import unittest
from pathlib import Path

from evaluation.evaluate_classicmodels_solutions import (
    QueryCase,
    compare_rows,
    evaluate_system,
    load_predictions,
    normalize_question,
    rewrite_mysqlisms_for_postgres,
)


class _FakeExecutor:
    def __init__(self, rows_by_sql):
        self.rows_by_sql = rows_by_sql
        self.calls = []

    def execute(self, sql: str):
        self.calls.append(sql)
        if sql == "BROKEN":
            raise RuntimeError("boom")
        return self.rows_by_sql[sql]


class ClassicModelsSolutionEvaluatorTest(unittest.TestCase):
    def test_load_predictions_supports_question_keyed_magsql_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.json"
            path.write_text(
                json.dumps(
                    [
                        [
                            "Who is at the top of the organization (i.e., reports to no one).",
                            "SELECT * FROM employees\t----- bird -----\tclassicmodels",
                        ]
                    ]
                ),
                encoding="utf-8",
            )
            cases = [
                QueryCase(
                    question_id=4,
                    question="Who is at the top of the organization (i.e.,  reports to no one).",
                    gold_sql="SELECT 1",
                    difficulty="simple",
                    category="General-Queries",
                    db_id="classicmodels",
                )
            ]

            predictions = load_predictions(path, cases)

            self.assertEqual(predictions[4].sql, "SELECT * FROM employees")
            self.assertEqual(predictions[4].db_id, "classicmodels")

    def test_load_predictions_supports_question_id_keyed_agent_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "pred.json"
            path.write_text(
                json.dumps([[2, "SELECT productname FROM products\t----- bird -----\tclassicmodels"]]),
                encoding="utf-8",
            )

            predictions = load_predictions(path, [])

            self.assertEqual(predictions[2].sql, "SELECT productname FROM products")

    def test_normalizes_mysql_functions_used_by_gold_sql(self) -> None:
        sql = "SELECT DATEDIFF(MAX(orderDate), MIN(orderDate)), MONTH(paymentDate), YEAR(paymentDate), `ordernumber` FROM orders"

        rewritten = rewrite_mysqlisms_for_postgres(sql)

        self.assertIn("(MAX(orderDate)::date - MIN(orderDate)::date)", rewritten)
        self.assertIn("EXTRACT(MONTH FROM paymentDate)", rewritten)
        self.assertIn("EXTRACT(YEAR FROM paymentDate)", rewritten)
        self.assertIn("ordernumber", rewritten)
        self.assertNotIn("`", rewritten)

    def test_compare_rows_ignores_row_order(self) -> None:
        self.assertTrue(compare_rows([(1, "a"), (2, "b")], [(2, "b"), (1, "a")]))
        self.assertFalse(compare_rows([(1, "a")], [(1, "b")]))

    def test_evaluate_system_calculates_ex_by_difficulty_and_preserves_errors(self) -> None:
        cases = [
            QueryCase(0, "Q0", "GOLD 0", "simple", "General", "classicmodels"),
            QueryCase(1, "Q1", "GOLD 1", "moderate", "General", "classicmodels"),
            QueryCase(2, "Q2", "GOLD 2", "challenging", "General", "classicmodels"),
        ]
        predictions = {
            0: type("Prediction", (), {"sql": "PRED 0"})(),
            1: type("Prediction", (), {"sql": "PRED 1"})(),
            2: type("Prediction", (), {"sql": "BROKEN"})(),
        }
        executor = _FakeExecutor(
            {
                "GOLD 0": [(1,)],
                "PRED 0": [(1,)],
                "GOLD 1": [(2,)],
                "PRED 1": [(3,)],
                "GOLD 2": [(4,)],
            }
        )

        result = evaluate_system(
            name="agent",
            cases=cases,
            predictions=predictions,
            executor=executor,
            include_ves=False,
        )

        self.assertEqual(result.summary["total"]["count"], 3)
        self.assertAlmostEqual(result.summary["total"]["ex"], 100 / 3)
        self.assertEqual(result.summary["simple"]["ex"], 100.0)
        self.assertEqual(result.summary["moderate"]["ex"], 0.0)
        self.assertEqual(result.summary["challenging"]["ex"], 0.0)
        self.assertEqual(result.items[2].error, "boom")

    def test_normalize_question_collapses_whitespace(self) -> None:
        self.assertEqual(normalize_question("A  question\nwith space"), "a question with space")


if __name__ == "__main__":
    unittest.main()
