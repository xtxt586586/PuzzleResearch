import unittest

from fastapi.testclient import TestClient

from main import app


class PublicApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_returns_healthy_service(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(
            payload["service"],
            "PuzzleResearch Research Agent",
        )
        self.assertEqual(payload["version"], "0.1.0")

    def test_health_returns_runtime_snapshot(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(payload["status"], "healthy")
        self.assertIn(
            payload["database"],
            {
                "configured",
                "not configured",
            },
        )
        self.assertIsInstance(
            payload["uptime_seconds"],
            float,
        )
        self.assertTrue(payload["timestamp"])

    def test_agent_health_returns_local_capabilities(self):
        response = self.client.get("/api/health/agent")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(payload["agents_count"], 0)
        self.assertGreaterEqual(
            payload["tool_registry_total_tools"],
            1,
        )
        self.assertIn(
            "search_providers_available",
            payload,
        )

    def test_public_config_does_not_expose_api_keys(self):
        response = self.client.get("/api/config/public")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(
            payload["defaults"]["primary_model"],
            "deepseek-chat",
        )
        self.assertIn(
            "deepseek-chat",
            payload["models"]["options"],
        )
        self.assertFalse(
            payload["features"]["mcp_enabled"],
        )
        self.assertFalse(
            payload["features"]["prometheus_enabled"],
        )

        serialized = str(payload).lower()

        self.assertNotIn(
            "deepseek_api_key",
            serialized,
        )
        self.assertNotIn(
            "tavily_api_key",
            serialized,
        )


if __name__ == "__main__":
    unittest.main()