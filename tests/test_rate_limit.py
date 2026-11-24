import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from main import app


class RateLimitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def setUp(self):
        main._rate_limit_buckets.clear()

    def test_api_response_exposes_rate_limit_headers_when_enabled(self):
        with patch.dict(
            main.settings.__dict__,
            {"rate_limit_enabled": True},
            clear=False,
        ):
            response = self.client.get("/api/config/public")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("X-RateLimit-Limit"), "60")
        self.assertEqual(response.headers.get("X-RateLimit-Remaining"), "59")
        self.assertTrue(response.headers.get("X-RateLimit-Reset"))

    def test_chat_uses_chat_limit(self):
        with patch.dict(
            main.settings.__dict__,
            {
                "rate_limit_enabled": True,
                "rate_limit_chat_per_minute": 7,
            },
            clear=False,
        ):
            response = self.client.post(
                "/api/chat",
                json={"messages": [{"role": "user", "content": "hello"}]},
            )

        self.assertEqual(response.headers.get("X-RateLimit-Limit"), "7")

    def test_excess_requests_return_retry_after(self):
        with patch.dict(
            main.settings.__dict__,
            {
                "rate_limit_enabled": True,
                "rate_limit_general_per_minute": 1,
                "rate_limit_window_seconds": 30,
            },
            clear=False,
        ):
            first = self.client.get("/api/config/public")
            second = self.client.get("/api/config/public")

        self.assertEqual(first.status_code, 200)
        self.assertEqual(second.status_code, 429)
        self.assertTrue(second.headers.get("Retry-After"))
        self.assertEqual(second.headers.get("X-RateLimit-Remaining"), "0")

    def test_bucket_storage_is_capped(self):
        with patch.dict(
            main.settings.__dict__,
            {
                "rate_limit_enabled": True,
                "rate_limit_max_buckets": 3,
            },
            clear=False,
        ):
            for index in range(10):
                self.client.get(
                    "/api/config/public",
                    headers={"X-Forwarded-For": f"10.0.0.{index}"},
                )

        self.assertLessEqual(len(main._rate_limit_buckets), 3)

    def test_readiness_is_exempt_from_rate_limiting(self):
        with patch.dict(
            main.settings.__dict__,
            {"rate_limit_enabled": True},
            clear=False,
        ):
            response = self.client.get("/ready")

        self.assertNotEqual(response.status_code, 429)
        self.assertIsNone(response.headers.get("X-RateLimit-Limit"))


if __name__ == "__main__":
    unittest.main()
