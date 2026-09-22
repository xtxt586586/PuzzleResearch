import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class FakeBrowserLiveView:
    def __init__(self):
        self.url = None

    async def start(self, url):
        self.url = url
        return {
            "url": url,
            "title": "测试页面",
        }

    async def capture(
        self,
        *,
        full_page=False,
        quality=70,
    ):
        return {
            "data": "ZmFrZS1mcmFtZQ==",
            "metadata": {
                "url": self.url,
                "title": "测试页面",
            },
        }

    async def stop(self):
        return None


class BrowserStreamTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def tearDown(self):
        main.browser_sessions.reset(
            "browser-stream-test"
        )

    def test_active_session_streams_frames(self):
        session = main.browser_sessions.get(
            "browser-stream-test"
        )

        session.current = SimpleNamespace(
            url="https://example.com",
            title="测试页面",
        )

        with patch(
            "main.BrowserLiveView",
            FakeBrowserLiveView,
        ):
            with self.client.websocket_connect(
                "/api/browser/browser-stream-test/stream"
            ) as websocket:
                initial = websocket.receive_json()

                self.assertTrue(
                    initial["active"]
                )

                websocket.send_json(
                    {
                        "action": "start",
                        "max_fps": 10,
                    }
                )

                started = websocket.receive_json()

                self.assertEqual(
                    started["message"],
                    "started",
                )

                frame = websocket.receive_json()

                self.assertEqual(
                    frame["type"],
                    "frame",
                )
                self.assertEqual(
                    frame["source"],
                    "playwright",
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