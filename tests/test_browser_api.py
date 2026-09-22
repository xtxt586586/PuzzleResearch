import unittest

from fastapi.testclient import TestClient

import main


class BrowserApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def setUp(self):
        main.browser_sessions.reset(
            "browser-test-thread"
        )

    def tearDown(self):
        main.browser_sessions.reset(
            "browser-test-thread"
        )

    def test_info_returns_inactive_session(self):
        response = self.client.get(
            "/api/browser/browser-test-thread/info"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertFalse(
            payload["active"]
        )
        self.assertEqual(
            payload["thread_id"],
            "browser-test-thread",
        )

    def test_screenshot_returns_stable_inactive_error(self):
        response = self.client.post(
            "/api/browser/browser-test-thread/screenshot"
        )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertFalse(
            response.json()["success"]
        )

    def test_browser_stream_returns_initial_status(self):
        with self.client.websocket_connect(
            "/api/browser/browser-test-thread/stream"
        ) as websocket:
            payload = websocket.receive_json()

            self.assertEqual(
                payload["type"],
                "status",
            )
            self.assertFalse(
                payload["active"]
            )

            websocket.send_json(
                {
                    "action": "stop",
                }
            )

            stopped = websocket.receive_json()

            self.assertEqual(
                stopped["message"],
                "stopped",
            )


if __name__ == "__main__":
    unittest.main()