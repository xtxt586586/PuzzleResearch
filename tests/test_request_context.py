import logging
import re
import unittest

from fastapi.testclient import TestClient

from main import app


class RequestContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_generated_request_id_is_returned(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertRegex(
            response.headers.get("X-Request-ID", ""),
            re.compile(r"[0-9a-f]{8}"),
        )

    def test_client_request_id_is_preserved(self):
        response = self.client.get(
            "/health",
            headers={"X-Request-ID": "client-id-123"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers.get("X-Request-ID"), "client-id-123")

    def test_request_start_and_completion_are_logged(self):
        logger = logging.getLogger("puzzleresearch")
        with self.assertLogs(logger, level="INFO") as captured:
            response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            any("Request started" in message and "/health" in message for message in captured.output)
        )
        self.assertTrue(
            any("Request completed" in message and "/health" in message for message in captured.output)
        )


if __name__ == "__main__":
    unittest.main()
