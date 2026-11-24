import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from main import app


class InternalAuthTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def _auth_config(self):
        return patch.dict(
            main.settings.__dict__,
            {"internal_api_key": "test-key"},
            clear=False,
        )

    def test_internal_key_is_required_when_configured_at_runtime(self):
        with self._auth_config(), patch.object(main, "configured_api_key", ""):
            response = self.client.get("/api/memory/status")

        self.assertEqual(response.status_code, 401)
        payload = response.json()
        self.assertEqual(payload.get("status_code"), 401)
        self.assertTrue(payload.get("error"))

    def test_internal_key_allows_access_when_provided(self):
        with self._auth_config(), patch.object(main, "configured_api_key", ""):
            response = self.client.get(
                "/api/memory/status",
                headers={"Authorization": "Bearer test-key"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.json().get("backend"))

    def test_authenticated_user_cannot_read_another_users_run(self):
        with self._auth_config(), patch.object(
            main,
            "configured_api_key",
            "",
        ), patch.dict(main._thread_owners, {"thread_alice": "alice"}, clear=False):
            response = self.client.get(
                "/api/runs/thread_alice",
                headers={
                    "Authorization": "Bearer test-key",
                    "X-Weaver-User": "bob",
                },
            )

        self.assertEqual(response.status_code, 403)


if __name__ == "__main__":
    unittest.main()
