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


if __name__ == "__main__":
    unittest.main()