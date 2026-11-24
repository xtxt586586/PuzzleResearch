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
        main.browser_use_sessions.reset(
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

    def test_stream_can_pause_capture_and_resume_on_same_connection(self):
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
                websocket.receive_json()

                websocket.send_json(
                    {
                        "action": "start",
                        "max_fps": 10,
                    }
                )
                started = websocket.receive_json()
                self.assertEqual(started["message"], "started")

                websocket.send_json({"action": "stop"})
                stopped = None
                for _ in range(10):
                    message = websocket.receive_json()
                    if message.get("type") == "status" and message.get("message") == "stopped":
                        stopped = message
                        break
                self.assertIsNotNone(stopped)

                websocket.send_json({"action": "capture"})
                captured = None
                for _ in range(10):
                    message = websocket.receive_json()
                    if message.get("type") == "frame":
                        captured = message
                        break
                self.assertIsNotNone(captured)

                websocket.send_json(
                    {
                        "action": "start",
                        "max_fps": 10,
                    }
                )
                resumed = websocket.receive_json()
                self.assertEqual(resumed["message"], "started")

                websocket.send_json({"action": "stop"})
                for _ in range(10):
                    message = websocket.receive_json()
                    if message.get("type") == "status" and message.get("message") == "stopped":
                        break
                else:
                    self.fail("The resumed stream did not pause cleanly")

    def test_browser_use_stream_captures_and_interacts_with_shared_session(self):
        class SharedBrowserUseSession:
            def __init__(self):
                self.url = "https://example.com/agent"
                self.actions = []
                self.captures = []

            def snapshot(self):
                return {"url": self.url, "title": "Agent page"}

            def capture_screenshot(self, *, full_page=True, quality=85):
                self.captures.append((full_page, quality))
                return {
                    "image_data": b"shared-session-frame",
                    "metadata": self.snapshot(),
                }

            def perform_preview_action(self, action, payload):
                self.actions.append((action, payload))
                if action == "navigate":
                    self.url = payload["url"]
                return self.snapshot()

        session = SharedBrowserUseSession()

        with (
            patch.object(main.browser_sessions, "peek", return_value=None),
            patch.object(
                main.browser_use_sessions,
                "peek",
                return_value=session,
            ),
            patch("main.BrowserLiveView", side_effect=AssertionError("second browser opened")),
            self.client.websocket_connect(
                "/api/browser/browser-stream-test/stream"
            ) as websocket,
        ):
            initial = websocket.receive_json()
            self.assertEqual(initial["source"], "browser_use")

            websocket.send_json({"action": "start", "quality": 61, "max_fps": 10})
            started = websocket.receive_json()
            self.assertEqual(started["message"], "started")

            frame = websocket.receive_json()
            self.assertEqual(frame["type"], "frame")
            self.assertEqual(frame["source"], "browser_use")
            self.assertEqual(frame["data"], "c2hhcmVkLXNlc3Npb24tZnJhbWU=")

            websocket.send_json(
                {
                    "action": "navigate",
                    "url": "https://example.org/shared",
                    "id": "navigate-shared",
                }
            )
            ack = None
            for _ in range(10):
                message = websocket.receive_json()
                if message.get("type") == "ack":
                    ack = message
                    break
            self.assertIsNotNone(ack)
            self.assertTrue(ack["ok"])
            self.assertEqual(ack["id"], "navigate-shared")
            self.assertEqual(ack["metadata"]["url"], "https://example.org/shared")

            websocket.send_json({"action": "capture", "quality": 80})
            captured = None
            for _ in range(10):
                message = websocket.receive_json()
                if message.get("type") == "frame":
                    captured = message
                    break
            self.assertIsNotNone(captured)
            self.assertEqual(captured["source"], "browser_use")

            websocket.send_json({"action": "stop"})
            stopped = None
            for _ in range(10):
                message = websocket.receive_json()
                if message.get("type") == "status" and message.get("message") == "stopped":
                    stopped = message
                    break
            self.assertIsNotNone(stopped)

        self.assertIn((False, 61), session.captures)
        self.assertIn((False, 80), session.captures)
        self.assertEqual(session.url, "https://example.org/shared")
        self.assertEqual(session.actions[0][0], "navigate")

    def test_stream_waiting_for_late_browser_use_session_starts_automatically(self):
        class LateBrowserUseSession:
            def snapshot(self):
                return {
                    "url": "https://example.com/late",
                    "title": "Late page",
                }

            def capture_screenshot(self, *, full_page=True, quality=85):
                return {
                    "image_data": b"late-session-frame",
                    "metadata": self.snapshot(),
                }

        available = {"session": None}
        session = LateBrowserUseSession()

        with (
            patch.object(main.browser_sessions, "peek", return_value=None),
            patch.object(
                main.browser_use_sessions,
                "peek",
                side_effect=lambda _thread_id: available["session"],
            ),
            patch("main.BrowserLiveView", side_effect=AssertionError("second browser opened")),
            self.client.websocket_connect(
                "/api/browser/browser-stream-test/stream"
            ) as websocket,
        ):
            initial = websocket.receive_json()
            self.assertFalse(initial["active"])

            websocket.send_json({"action": "start", "quality": 63, "max_fps": 8})
            waiting = websocket.receive_json()
            self.assertEqual(waiting["message"], "waiting_for_browser")

            available["session"] = session
            messages = []
            for _ in range(5):
                message = websocket.receive_json()
                messages.append(message)
                if message.get("type") == "frame":
                    break

            ready = next(
                message
                for message in messages
                if message.get("type") == "status" and message.get("active") is True
            )
            started = next(
                message
                for message in messages
                if message.get("type") == "status" and message.get("message") == "started"
            )
            frame = next(
                message for message in messages if message.get("type") == "frame"
            )
            self.assertEqual(ready["source"], "browser_use")
            self.assertEqual(started["quality"], 63)
            self.assertEqual(frame["source"], "browser_use")
            self.assertEqual(frame["data"], "bGF0ZS1zZXNzaW9uLWZyYW1l")

            websocket.send_json({"action": "stop"})
            stopped = None
            for _ in range(5):
                message = websocket.receive_json()
                if message.get("type") == "status" and message.get("message") == "stopped":
                    stopped = message
                    break
            self.assertIsNotNone(stopped)

    def test_stream_reports_session_loss_and_rebinds_replacement(self):
        class ManagedBrowserUseSession:
            def __init__(self, url, frame_bytes):
                self.url = url
                self.frame_bytes = frame_bytes

            def snapshot(self):
                return {"url": self.url, "title": self.url}

            def capture_screenshot(self, *, full_page=True, quality=85):
                return {
                    "image_data": self.frame_bytes,
                    "metadata": self.snapshot(),
                }

        old_session = ManagedBrowserUseSession(
            "https://example.com/old",
            b"old-frame",
        )
        new_session = ManagedBrowserUseSession(
            "https://example.org/new",
            b"new-frame",
        )
        available = {"session": old_session}

        with (
            patch.object(main.browser_sessions, "peek", return_value=None),
            patch.object(
                main.browser_use_sessions,
                "peek",
                side_effect=lambda _thread_id: available["session"],
            ),
            patch("main.BrowserLiveView", side_effect=AssertionError("second browser opened")),
            self.client.websocket_connect(
                "/api/browser/browser-stream-test/stream"
            ) as websocket,
        ):
            self.assertTrue(websocket.receive_json()["active"])
            websocket.send_json({"action": "start", "max_fps": 1})

            first_frame = None
            for _ in range(4):
                message = websocket.receive_json()
                if message.get("type") == "frame":
                    first_frame = message
                    break
            self.assertIsNotNone(first_frame)
            self.assertEqual(first_frame["data"], "b2xkLWZyYW1l")

            available["session"] = None
            inactive = None
            for _ in range(5):
                message = websocket.receive_json()
                if message.get("type") == "status" and message.get("active") is False:
                    inactive = message
                    break
            self.assertIsNotNone(inactive)
            self.assertEqual(inactive["message"], "waiting_for_browser")

            available["session"] = new_session
            replacement_messages = []
            for _ in range(6):
                message = websocket.receive_json()
                replacement_messages.append(message)
                if message.get("type") == "frame" and message.get("data") == "bmV3LWZyYW1l":
                    break

            rebound = next(
                message
                for message in replacement_messages
                if message.get("type") == "status"
                and message.get("active") is True
                and message.get("current_url") == "https://example.org/new"
            )
            replacement_frame = next(
                message
                for message in replacement_messages
                if message.get("type") == "frame"
                and message.get("data") == "bmV3LWZyYW1l"
            )
            self.assertEqual(rebound["source"], "browser_use")
            self.assertEqual(replacement_frame["metadata"]["url"], "https://example.org/new")

            websocket.send_json({"action": "stop"})
            stopped = None
            for _ in range(5):
                message = websocket.receive_json()
                if message.get("type") == "status" and message.get("message") == "stopped":
                    stopped = message
                    break
            self.assertIsNotNone(stopped)

    def test_http_browser_page_replacement_restarts_preview_on_new_url(self):
        class RecordingLiveView(FakeBrowserLiveView):
            starts = []

            async def start(self, url):
                self.url = url
                self.starts.append(url)
                return {"url": url, "title": "测试页面"}

        RecordingLiveView.starts = []

        first = SimpleNamespace(
            current=SimpleNamespace(
                url="https://example.com/first",
                title="First",
            )
        )
        second = SimpleNamespace(
            current=SimpleNamespace(
                url="https://example.org/second",
                title="Second",
            )
        )
        active = {"session": first}

        with (
            patch.object(
                main.browser_sessions,
                "peek",
                side_effect=lambda _thread_id: active["session"],
            ),
            patch.object(main.browser_use_sessions, "peek", return_value=None),
            patch("main.BrowserLiveView", RecordingLiveView),
            self.client.websocket_connect(
                "/api/browser/browser-stream-test/stream"
            ) as websocket,
        ):
            websocket.receive_json()
            websocket.send_json({"action": "start", "max_fps": 1})

            first_frame = None
            for _ in range(4):
                message = websocket.receive_json()
                if message.get("type") == "frame":
                    first_frame = message
                    break
            self.assertIsNotNone(first_frame)

            active["session"] = second
            messages = []
            for _ in range(6):
                message = websocket.receive_json()
                messages.append(message)
                if message.get("type") == "frame" and message.get("metadata", {}).get("url") == "https://example.org/second":
                    break

            replacement_frame = next(
                message
                for message in messages
                if message.get("type") == "frame"
                and message.get("metadata", {}).get("url") == "https://example.org/second"
            )
            self.assertEqual(replacement_frame["metadata"]["title"], "测试页面")
            self.assertEqual(
                RecordingLiveView.starts,
                ["https://example.com/first", "https://example.org/second"],
            )

            websocket.send_json({"action": "stop"})
            for _ in range(5):
                message = websocket.receive_json()
                if message.get("type") == "status" and message.get("message") == "stopped":
                    break
            else:
                self.fail("The replaced preview did not stop cleanly")


if __name__ == "__main__":
    unittest.main()
