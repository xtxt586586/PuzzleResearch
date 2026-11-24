import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

import main
from common.metrics import RunMetrics, RunMetricsRegistry


class MetricsTests(unittest.TestCase):
    def test_prometheus_endpoint_exports_connection_metrics(self):
        response = TestClient(main.app).get("/metrics")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/plain", response.headers["content-type"])
        self.assertIn("puzzleresearch_sse_active_connections", response.text)
        self.assertIn("puzzleresearch_ws_active_connections", response.text)
        self.assertIn("puzzleresearch_browser_ws_frames_total", response.text)

    def test_prometheus_endpoint_does_not_require_api_key(self):
        with patch.object(main, "configured_api_key", "test-api-key"):
            response = TestClient(main.app).get("/metrics")

        self.assertEqual(response.status_code, 200)

    def test_public_config_reports_prometheus_setting(self):
        with patch.object(main.settings, "enable_prometheus", True):
            response = TestClient(main.app).get("/api/config/public")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["features"]["prometheus_enabled"])

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

    def test_registry_bounds_history_without_evicting_active_runs(self):
        registry = RunMetricsRegistry(max_runs=2)
        registry.start("active", "deepseek-chat")
        registry.start("completed-1", "deepseek-chat").finish()
        registry.start("completed-2", "deepseek-chat").finish()

        self.assertIsNotNone(registry.get("active"))
        self.assertIsNone(registry.get("completed-1"))
        self.assertIsNotNone(registry.get("completed-2"))

        registry.start("completed-3", "deepseek-chat").finish()

        self.assertIsNotNone(registry.get("active"))
        self.assertIsNone(registry.get("completed-2"))
        self.assertIsNotNone(registry.get("completed-3"))
        self.assertEqual(len(registry.all()), 2)

    def test_registry_rejects_nonpositive_history_limit(self):
        with self.assertRaises(ValueError):
            RunMetricsRegistry(max_runs=0)

    def test_registry_records_first_token_once(self):
        metrics = RunMetrics(
            "run-first-token",
            "deepseek-chat",
            "agent",
            started_monotonic=10.0,
        )
        with patch("common.metrics.time.monotonic", return_value=10.25):
            metrics.mark_first_token()
            metrics.mark_first_token()

        self.assertEqual(metrics.to_dict()["first_token_ms"], 250.0)

    def test_registry_records_node_and_tool_durations(self):
        metrics = RunMetrics(
            "run-timing",
            "deepseek-chat",
            started_monotonic=0.0,
        )
        with patch(
            "common.metrics.time.monotonic",
            side_effect=(10.0, 10.125, 20.0, 20.375),
        ):
            metrics.mark_event("on_chain_start", "search")
            metrics.mark_event("on_chain_end", "search")
            metrics.mark_event("on_tool_start", "web_search")
            metrics.mark_event("on_tool_end", "web_search")

        payload = metrics.to_dict()
        self.assertEqual(payload["node_durations_ms"], {"search": [125.0]})
        self.assertEqual(payload["tool_durations_ms"], {"web_search": [375.0]})

    def test_registry_pairs_parallel_node_events_by_run_id(self):
        metrics = RunMetrics("run-parallel", "deepseek-chat")
        with patch(
            "common.metrics.time.monotonic",
            side_effect=(10.0, 10.1, 10.2, 10.3),
        ):
            metrics.mark_event("on_chain_start", "search", event_run_id="a")
            metrics.mark_event("on_chain_start", "search", event_run_id="b")
            metrics.mark_event("on_chain_end", "search", event_run_id="b")
            metrics.mark_event("on_chain_end", "search", event_run_id="a")

        self.assertEqual(metrics.to_dict()["node_durations_ms"]["search"], [100.0, 300.0])

    def test_registry_records_fast_path_phase_durations(self):
        metrics = RunMetrics("run-phases", "deepseek-chat")
        metrics.mark_event(
            "on_chain_end",
            "agent_fast_path",
            {
                "output": {
                    "agent_performance": {
                        "search_duration_ms": 82.5,
                        "generation_duration_ms": 310,
                        "first_token_duration_ms": 190,
                        "generation_first_token_ms": 107,
                        "node_duration_ms": 410,
                    }
                }
            },
        )

        self.assertEqual(
            metrics.to_dict()["phase_durations_ms"],
            {
                "search_duration_ms": [82.5],
                "generation_duration_ms": [310.0],
                "first_token_duration_ms": [190.0],
                "generation_first_token_ms": [107.0],
                "node_duration_ms": [410.0],
            },
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
        self.assertIn("first_token_ms", payload["runs"][0])
        self.assertIn("tool_durations_ms", payload["runs"][0])

    def test_get_missing_run_returns_404(self):
        client = TestClient(main.app)

        response = client.get(
            "/api/runs/missing-run-001"
        )

        self.assertEqual(
            response.status_code,
            404,
        )

    def test_run_evidence_summary_includes_freshness_and_verifier_metrics(self):
        with patch.object(
            main.session_manager,
            "get_state",
            return_value={
                "state": {
                    "sources": [{"url": "https://example.com"}],
                    "deepsearch_artifacts": {
                        "claims": [
                            {"status": "verified"},
                            {"status": "unsupported"},
                            {"status": "contradicted"},
                        ],
                        "freshness_summary": {"fresh_30_ratio": 0.75},
                        "quality_summary": {
                            "citation_coverage": 0.8,
                            "query_coverage_score": 0.6,
                            "freshness_warning": "low_freshness_for_time_sensitive_query",
                            "claim_verifier_total": 3,
                            "claim_verifier_verified": 1,
                            "claim_verifier_unsupported": 1,
                            "claim_verifier_contradicted": 1,
                        },
                    },
                }
            },
        ):
            summary = main._run_evidence_summary("thread-evidence")

        self.assertEqual(summary.sources_count, 1)
        self.assertEqual(summary.unsupported_claims_count, 2)
        self.assertEqual(summary.freshness_ratio_30d, 0.75)
        self.assertEqual(summary.citation_coverage, 0.8)
        self.assertEqual(summary.query_coverage_score, 0.6)
        self.assertEqual(
            summary.freshness_warning,
            "low_freshness_for_time_sensitive_query",
        )
        self.assertEqual(summary.claim_verifier_total, 3)
        self.assertEqual(summary.claim_verifier_verified, 1)
        self.assertEqual(summary.claim_verifier_unsupported, 1)
        self.assertEqual(summary.claim_verifier_contradicted, 1)

    def test_run_evidence_summary_falls_back_to_verifier_artifact_counts(self):
        with patch.object(
            main.session_manager,
            "get_state",
            return_value={
                "state": {
                    "deepsearch_artifacts": {
                        "claim_verifier": {
                            "verified_count": 2,
                            "unsupported_count": 1,
                            "contradicted_count": 1,
                        },
                    },
                }
            },
        ):
            summary = main._run_evidence_summary("thread-evidence")

        self.assertEqual(summary.claim_verifier_total, 4)
        self.assertEqual(summary.claim_verifier_verified, 2)
        self.assertEqual(summary.claim_verifier_unsupported, 1)
        self.assertEqual(summary.claim_verifier_contradicted, 1)

    def test_run_evidence_summary_leaves_missing_metrics_empty(self):
        with patch.object(
            main.session_manager,
            "get_state",
            return_value={"state": {"deepsearch_artifacts": {}}},
        ):
            summary = main._run_evidence_summary("thread-without-evidence")

        self.assertIsNone(summary.freshness_ratio_30d)
        self.assertIsNone(summary.freshness_warning)
        self.assertIsNone(summary.claim_verifier_total)
        self.assertIsNone(summary.claim_verifier_verified)
        self.assertIsNone(summary.claim_verifier_unsupported)
        self.assertIsNone(summary.claim_verifier_contradicted)

    def test_run_metrics_api_serializes_extended_evidence_summary(self):
        registry = RunMetricsRegistry()
        registry.start(
            "thread-metrics-api",
            "deepseek-chat",
            "deep",
        ).finish()
        snapshot = {
            "state": {
                "deepsearch_artifacts": {
                    "freshness_summary": {"fresh_30_ratio": 0.5},
                    "quality_summary": {
                        "freshness_warning": "low_freshness_for_time_sensitive_query",
                        "claim_verifier_total": 2,
                        "claim_verifier_verified": 1,
                        "claim_verifier_unsupported": 0,
                        "claim_verifier_contradicted": 1,
                    },
                },
            },
        }

        with patch.object(main, "metrics_registry", registry), patch.object(
            main.session_manager,
            "get_state",
            return_value=snapshot,
        ), patch.object(main, "_active_api_key", return_value=""):
            response = TestClient(main.app).get(
                "/api/runs/thread-metrics-api"
            )

        self.assertEqual(response.status_code, 200)
        evidence = response.json()["evidence_summary"]
        self.assertEqual(evidence["freshness_ratio_30d"], 0.5)
        self.assertEqual(
            evidence["freshness_warning"],
            "low_freshness_for_time_sensitive_query",
        )
        self.assertEqual(evidence["claim_verifier_total"], 2)
        self.assertEqual(evidence["claim_verifier_verified"], 1)
        self.assertEqual(evidence["claim_verifier_unsupported"], 0)
        self.assertEqual(evidence["claim_verifier_contradicted"], 1)


class PrometheusStreamGaugeTests(unittest.IsolatedAsyncioTestCase):
    async def test_sse_connection_gauge_decrements_when_stream_finishes(self):
        gauge = Mock()
        metric = Mock()
        metric.labels.return_value = gauge

        async def source():
            yield "first"
            yield "second"

        with patch.object(main, "sse_active_connections", metric):
            chunks = [
                chunk
                async for chunk in main._track_sse_connection(
                    source(),
                    "test_stream",
                )
            ]

        self.assertEqual(chunks, ["first", "second"])
        metric.labels.assert_called_once_with("test_stream")
        gauge.inc.assert_called_once_with()
        gauge.dec.assert_called_once_with()

    async def test_http_metrics_track_status_and_restore_inprogress(self):
        inprogress = Mock()
        request_counter = Mock()
        labeled_counter = Mock()
        request_counter.labels.return_value = labeled_counter
        request = Mock()
        request.method = "GET"
        request.url.path = "/test"
        response = Mock(status_code=204)

        async def call_next(_request):
            return response

        with patch.object(main, "http_inprogress", inprogress), patch.object(
            main,
            "http_requests_total",
            request_counter,
        ):
            result = await main.prometheus_http_metrics_middleware(
                request,
                call_next,
            )

        self.assertIs(result, response)
        request_counter.labels.assert_called_once_with(
            "GET",
            "/test",
            204,
        )
        labeled_counter.inc.assert_called_once_with()
        inprogress.inc.assert_called_once_with()
        inprogress.dec.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
