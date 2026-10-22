import unittest

from scripts.benchmark_agent_dispatch import run_comparison


class AgentDispatchBenchmarkTests(unittest.TestCase):
    def test_offline_comparison_covers_all_topics_and_recovers_transient_failure(self):
        report = run_comparison(delay_ms=0)

        self.assertFalse(report["real_model_calls"])
        self.assertFalse(report["real_search_calls"])
        self.assertTrue(report["comparison"]["coverage_equal"])
        self.assertEqual(
            report["single_worker_baseline"]["covered_topics"],
            sorted(report["topics"]),
        )
        self.assertEqual(
            report["coordinator_agents"]["covered_topics"],
            sorted(report["topics"]),
        )
        self.assertEqual(report["single_worker_baseline"]["retried_tasks"], 1)
        self.assertEqual(report["coordinator_agents"]["retried_tasks"], 1)
        self.assertEqual(report["coordinator_agents"]["failed_attempts"], 1)
        self.assertEqual(report["coordinator_agents"]["max_concurrency"], 3)
        self.assertFalse(report["comparison"]["simulated_latency_is_production_representative"])

    def test_rejects_negative_mock_latency(self):
        with self.assertRaises(ValueError):
            run_comparison(delay_ms=-1)


if __name__ == "__main__":
    unittest.main()
