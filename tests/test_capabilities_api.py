import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class FakeProvider:
    def __init__(
        self,
        name: str,
        available: bool,
    ) -> None:
        self.name = name
        self._available = available

    def is_available(self) -> bool:
        return self._available


class FakeSearchManager:
    def __init__(self) -> None:
        self.providers = {
            "fake_available": FakeProvider(
                "fake_available",
                True,
            ),
            "fake_unavailable": FakeProvider(
                "fake_unavailable",
                False,
            ),
        }


class CapabilitiesApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def test_tool_registry_returns_builtin_tools(self):
        response = self.client.get(
            "/api/tools/registry"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertGreaterEqual(
            payload["stats"]["total_tools"],
            1,
        )
        self.assertEqual(
            payload["stats"]["total_tools"],
            len(payload["tools"]),
        )

        python_tool = next(
            item
            for item in payload["tools"]
            if item["name"] == "python_executor"
        )

        self.assertTrue(
            python_tool["enabled"]
        )
        self.assertEqual(
            python_tool["source"],
            "builtin",
        )
        self.assertTrue(
            python_tool["parameters"]
        )
        for field in (
            "return_type",
            "module_name",
            "class_name",
            "function_name",
            "version",
            "tags",
            "call_count",
            "success_count",
            "failure_count",
            "last_called",
            "average_duration_ms",
            "deprecated",
            "deprecation_message",
            "success_rate",
        ):
            self.assertIn(field, python_tool)

        self.assertEqual(
            payload["stats"]["total_calls"],
            sum(item["call_count"] for item in payload["tools"]),
        )
        self.assertEqual(
            payload["stats"]["total_successes"],
            sum(item["success_count"] for item in payload["tools"]),
        )

    def test_tool_registry_refresh_returns_current_count(self):
        response = self.client.post(
            "/api/tools/registry/refresh"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["discovered"],
            0,
        )
        self.assertGreaterEqual(
            payload["total_tools"],
            1,
        )

    def test_search_providers_report_availability(self):
        with patch.object(
            main,
            "SearchManager",
            return_value=FakeSearchManager(),
        ), patch.object(
            main.settings,
            "search_engines",
            "fake_available,fake_unavailable",
        ):
            response = self.client.get(
                "/api/search/providers"
            )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()
        providers = {
            item["name"]: item
            for item in payload["providers"]
        }

        self.assertTrue(
            providers["fake_available"]["configured"]
        )
        self.assertTrue(
            providers["fake_available"]["available"]
        )
        self.assertTrue(
            providers["fake_available"]["healthy"]
        )

        self.assertTrue(
            providers["fake_unavailable"]["configured"]
        )
        self.assertFalse(
            providers["fake_unavailable"]["available"]
        )
        self.assertFalse(
            providers["fake_unavailable"]["healthy"]
        )

    def test_search_provider_reset_is_idempotent(self):
        response = self.client.post(
            "/api/search/providers/reset"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertTrue(
            payload["reset"]
        )
        self.assertTrue(
            payload["message"]
        )

    def test_search_providers_expose_multi_search_health(self):
        class FakeOrchestrator:
            def provider_snapshots(self):
                return [{
                    "name": "tavily",
                    "available": True,
                    "healthy": False,
                    "total_calls": 10,
                    "success_count": 7,
                    "error_count": 3,
                    "success_rate": 0.7,
                    "avg_latency_ms": 125.5,
                    "avg_result_quality": 0.8,
                    "last_error": "temporary outage",
                    "last_error_time": "2026-09-25T00:00:00+00:00",
                    "circuit": {
                        "is_open": True,
                        "consecutive_failures": 3,
                        "opened_for_seconds": 2.0,
                        "resets_in_seconds": 58.0,
                    },
                }]

        with patch.object(
            main,
            "get_search_orchestrator",
            return_value=FakeOrchestrator(),
        ), patch.object(
            main,
            "_search_provider_snapshots",
            return_value=[],
        ):
            response = self.client.get("/api/search/providers")

        self.assertEqual(response.status_code, 200)
        provider = response.json()["providers"][0]
        self.assertEqual(provider["success_rate"], 0.7)
        self.assertEqual(provider["avg_latency_ms"], 125.5)
        self.assertTrue(provider["circuit"]["is_open"])


if __name__ == "__main__":
    unittest.main()
