import unittest

from fastapi.testclient import TestClient

import main
from common.metrics import RunMetricsRegistry


class MetricsTests(unittest.TestCase):
    def test_registry_tracks_run_lifecycle(self):
        registry = RunMetricsRegistry()

        metrics = registry.start(
            "run-001",
            "deepseek-chat",
            "agent",
        )

        metrics.mark_event(
            "on_chain_start",
            "router",
        )
        metrics.mark_event(
            "on_chain_end",
            "router",
        )
        metrics.finish()

        payload = metrics.to_dict()

        self.assertEqual(
            payload["run_id"],
            "run-001",
        )
        self.assertEqual(
            payload["event_count"],
            2,
        )
        self.assertEqual(
            payload["nodes_started"]["router"],
            1,
        )
        self.assertEqual(
            payload["nodes_completed"]["router"],
            1,
        )
        self.assertIsNotNone(
            payload["ended_at"],
        )
        self.assertFalse(
            payload["cancelled"],
        )

    def test_registry_records_errors(self):
        registry = RunMetricsRegistry()

        metrics = registry.start(
            "run-002",
            "deepseek-chat",
        )

        metrics.add_error("测试错误")
        metrics.finish(
            cancelled=True,
        )

        payload = metrics.to_dict()

        self.assertEqual(
            payload["errors"],
            ["测试错误"],
        )
        self.assertTrue(
            payload["cancelled"],
        )

    def test_list_runs_api(self):
        original = main.metrics_registry

        registry = RunMetricsRegistry()
        registry.start(
            "run-api-001",
            "deepseek-chat",
            "direct",
        ).finish()

        main.metrics_registry = registry

        try:
            client = TestClient(main.app)
            response = client.get(
                "/api/runs"
            )
        finally:
            main.metrics_registry = original

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["count"],
            1,
        )
        self.assertEqual(
            payload["runs"][0]["run_id"],
            "run-api-001",
        )

    def test_get_missing_run_returns_404(self):
        client = TestClient(main.app)

        response = client.get(
            "/api/runs/missing-run-001"
        )

        self.assertEqual(
            response.status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()