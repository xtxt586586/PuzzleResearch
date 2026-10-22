import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from common.metrics import RunMetrics, RunMetricsRegistry


class ModelMetricsTests(unittest.TestCase):
    def test_registry_counts_alias_events_once_per_model_run(self):
        metrics = RunMetrics("run-model-count", "deepseek-chat")
        metrics.mark_event("on_chat_model_start", "chat", event_run_id="model-1")
        metrics.mark_event("on_llm_start", "llm", event_run_id="model-1")
        metrics.mark_event("on_chat_model_start", "chat", event_run_id="model-2")

        self.assertEqual(metrics.to_dict()["model_call_count"], 2)

    def test_run_metrics_api_exposes_model_call_count(self):
        metrics = RunMetrics("run-model-api", "deepseek-chat")
        metrics.mark_event("on_chat_model_start", "chat", event_run_id="model-1")
        original = main.metrics_registry
        registry = RunMetricsRegistry()
        registry._runs[metrics.run_id] = metrics
        main.metrics_registry = registry
        try:
            with patch.object(main, "_active_api_key", return_value=""):
                response = TestClient(main.app).get("/api/runs/run-model-api")
        finally:
            main.metrics_registry = original

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["model_call_count"], 1)


if __name__ == "__main__":
    unittest.main()
