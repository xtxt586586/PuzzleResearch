import json
import tempfile
import unittest
from pathlib import Path

from scripts.compare_research_benchmarks import compare_files, compare_reports


class ResearchBenchmarkComparisonTests(unittest.TestCase):
    def _report(self, mode, *, success, quality, calls, coverage):
        return {
            "mode": mode,
            "execute": True,
            "summary": {
                "executed_cases": 2,
                "quality_passed_cases": quality,
                "success_rate": success,
                "failure_rate": 1 - success,
                "avg_model_calls": calls,
                "avg_expected_field_coverage": coverage,
                "latency_ms": {
                    "first_token_p50": 100,
                    "completion_p50": 800,
                    "completion_p95": 1000,
                },
            },
            "metrics": {
                "avg_query_coverage_score": 0.7,
                "avg_citation_coverage": 0.8,
                "avg_freshness_ratio_30d": 0.6,
            },
        }

    def test_compare_reports_returns_deltas_for_quality_and_cost(self):
        report = compare_reports(
            self._report("linear", success=0.5, quality=1, calls=8, coverage=0.5),
            self._report("tree", success=1.0, quality=2, calls=5, coverage=0.9),
        )

        rows = {item["metric"]: item for item in report["metrics"]}
        self.assertEqual(rows["success_rate"]["delta"], 0.5)
        self.assertEqual(rows["avg_model_calls"]["delta"], -3.0)
        self.assertEqual(rows["avg_expected_field_coverage"]["candidate"], 0.9)

    def test_compare_files_writes_json_without_network(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            candidate = root / "candidate.json"
            output = root / "reports" / "comparison.json"
            baseline.write_text(json.dumps(self._report("linear", success=1, quality=2, calls=8, coverage=0.8)), encoding="utf-8")
            candidate.write_text(json.dumps(self._report("tree", success=1, quality=2, calls=5, coverage=0.9)), encoding="utf-8")

            result = compare_files(baseline, candidate, output)

            self.assertTrue(output.is_file())
            self.assertEqual(json.loads(output.read_text(encoding="utf-8"))["candidate_mode"], "tree")
            self.assertEqual(result["baseline_mode"], "linear")


if __name__ == "__main__":
    unittest.main()
