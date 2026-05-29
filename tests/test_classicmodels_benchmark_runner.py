import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_classicmodels_benchmark import (
    BenchmarkCase,
    DEFAULT_CASES_PATH,
    DEFAULT_OUTPUT_DIR,
    load_cases,
    run_benchmark,
    run_case,
)


class _Interrupt:
    def __init__(self, sql: str) -> None:
        self.value = [{"action": "sql_db_query", "args": {"query": sql}}]


class _FakeGraph:
    def __init__(self) -> None:
        self.calls = []
        self.configs = []

    def stream(self, payload, config=None, stream_mode=None):
        self.calls.append(payload)
        self.configs.append(config)
        if isinstance(payload, dict):
            question = payload["messages"][0]["content"]
            sql = f"SELECT '{question}' AS answer"
            yield {"candidate_sql": sql, "__interrupt__": [_Interrupt(sql)]}
            return
        yield {"messages": [type("Message", (), {"content": "SQL:\nSELECT 1\n\nResult:\n[(1,)]"})()]}

class _FakeGraphFactory:
    def __init__(self) -> None:
        self.graphs = []

    def __call__(self) -> _FakeGraph:
        graph = _FakeGraph()
        self.graphs.append(graph)
        return graph


class _LoopingGraph:
    def stream(self, payload, config=None, stream_mode=None):
        for index in range(10):
            yield {"messages": [type("Message", (), {"content": f"step {index}"})()]}


class ClassicModelsBenchmarkRunnerTest(unittest.TestCase):
    def test_default_paths_are_repo_root_anchored(self) -> None:
        self.assertTrue(DEFAULT_CASES_PATH.is_absolute())
        self.assertTrue(DEFAULT_OUTPUT_DIR.is_absolute())
        self.assertEqual(DEFAULT_CASES_PATH.name, "dev.json")

    def test_load_cases_reads_dev_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dev.json"
            path.write_text(
                json.dumps(
                    [
                        {
                            "question_id": 7,
                            "db_id": "classicmodels",
                            "question": "List customers",
                            "SQL": "SELECT 1;",
                            "difficulty": "simple",
                            "category": "General-Queries",
                        }
                    ]
                ),
                encoding="utf-8",
            )

            cases = load_cases(path)

            self.assertEqual(cases, [BenchmarkCase(7, "classicmodels", "List customers", "simple", "General-Queries")])

    def test_run_case_stops_after_candidate_sql_by_default(self) -> None:
        graph = _FakeGraph()
        case = BenchmarkCase(1, "classicmodels", "Who is top?", "simple", "General-Queries")

        result = run_case(graph=graph, case=case, thread_prefix="bench", run_id="run-a", execute_reviewed=False, max_turns=5)

        self.assertEqual(result["candidate_sql"], "SELECT 'Who is top?' AS answer")
        self.assertEqual(result["status"], "sql_generated")
        self.assertEqual(result["thread_id"], "bench-run-a-1")
        self.assertEqual(len(graph.calls), 1)

    def test_run_case_can_auto_accept_and_finish(self) -> None:
        graph = _FakeGraph()
        case = BenchmarkCase(1, "classicmodels", "Who is top?", "simple", "General-Queries")

        result = run_case(graph=graph, case=case, thread_prefix="bench", run_id="run-a", execute_reviewed=True, max_turns=5)

        self.assertEqual(result["status"], "completed")
        self.assertIn("SQL:", result["final_response"])
        self.assertEqual(len(graph.calls), 2)

    def test_run_benchmark_writes_prediction_and_result_files(self) -> None:
        graph = _FakeGraph()
        cases = [BenchmarkCase(2, "classicmodels", "Q", "simple", "General-Queries")]
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_benchmark(
                graph=graph,
                cases=cases,
                output_dir=Path(tmp),
                execute_reviewed=False,
                max_turns=5,
                run_id="run-b",
                workers=1,
            )

            self.assertEqual(summary["case_count"], 1)
            self.assertEqual(summary["max_turns"], 5)
            self.assertEqual(summary["run_id"], "run-b")
            self.assertEqual(summary["blank_sql_count"], 0)
            self.assertEqual(summary["max_turns_exceeded_count"], 0)
            self.assertTrue(summary["ready_for_eval"])
            predictions = json.loads((Path(tmp) / "predicted_sql.json").read_text(encoding="utf-8"))
            self.assertEqual(predictions, [[2, "SELECT 'Q' AS answer\t----- bird -----\tclassicmodels"]])
            result_lines = (Path(tmp) / "run_results.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(result_lines), 1)
            self.assertTrue((Path(tmp) / "benchmark.log").exists())
            self.assertEqual(
                graph.configs[0]["configurable"]["thread_id"],
                "classicmodels-benchmark-run-b-2",
            )

    def test_run_benchmark_parallel_uses_isolated_graphs_and_preserves_order(self) -> None:
        factory = _FakeGraphFactory()
        cases = [
            BenchmarkCase(2, "classicmodels", "Q2", "simple", "General-Queries"),
            BenchmarkCase(1, "classicmodels", "Q1", "simple", "General-Queries"),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_benchmark(
                graph_factory=factory,
                cases=cases,
                output_dir=Path(tmp),
                execute_reviewed=False,
                max_turns=5,
                run_id="run-par",
                workers=2,
            )

            self.assertEqual(summary["workers"], 2)
            self.assertEqual(len(factory.graphs), 2)
            predictions = json.loads((Path(tmp) / "predicted_sql.json").read_text(encoding="utf-8"))
            self.assertEqual([row[0] for row in predictions], [2, 1])
            thread_ids = [
                graph.configs[0]["configurable"]["thread_id"]
                for graph in factory.graphs
            ]
            self.assertEqual(len(set(thread_ids)), 2)

    def test_run_case_stops_after_max_turns(self) -> None:
        case = BenchmarkCase(3, "classicmodels", "Q", "simple", "General-Queries")

        result = run_case(
            graph=_LoopingGraph(),
            case=case,
            thread_prefix="bench",
            run_id="run-c",
            execute_reviewed=False,
            max_turns=5,
        )

        self.assertEqual(result["status"], "max_turns_exceeded")
        self.assertEqual(result["turn_count"], 5)

    def test_summary_marks_blank_sql_not_ready_for_eval(self) -> None:
        cases = [BenchmarkCase(3, "classicmodels", "Q", "simple", "General-Queries")]
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_benchmark(
                graph=_LoopingGraph(),
                cases=cases,
                output_dir=Path(tmp),
                execute_reviewed=False,
                max_turns=5,
                workers=1,
                run_id="run-loop",
            )

            self.assertEqual(summary["blank_sql_count"], 1)
            self.assertEqual(summary["max_turns_exceeded_count"], 1)
            self.assertFalse(summary["ready_for_eval"])


if __name__ == "__main__":
    unittest.main()
