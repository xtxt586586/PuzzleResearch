import asyncio
import base64
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

import main


class BrowserApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def setUp(self):
        main.browser_sessions.reset(
            "browser-test-thread"
        )
        main.browser_use_sessions.reset(
            "browser-test-thread"
        )

    def tearDown(self):
        main.browser_sessions.reset(
            "browser-test-thread"
        )
        main.browser_use_sessions.reset(
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

    def test_info_reads_active_browser_use_session(self):
        snapshot = {
            "url": "https://example.com/agent-page",
            "title": "Agent page",
        }
        browser_use_session = SimpleNamespace(
            snapshot=MagicMock(return_value=snapshot),
        )
        with (
            patch.object(main, "_assert_thread_owner"),
            patch.object(main.browser_sessions, "peek", return_value=None),
            patch.object(
                main.browser_use_sessions,
                "peek",
                return_value=browser_use_session,
            ),
        ):
            response = self.client.get(
                "/api/browser/browser-test-thread/info"
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "active": True,
                "thread_id": "browser-test-thread",
                "mode": "local",
                "source": "browser_use",
                "cdp_endpoint": None,
                "current_url": snapshot["url"],
                "title": snapshot["title"],
            },
        )
        browser_use_session.snapshot.assert_called_once_with()

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

    def test_browser_cleanup_preserves_paused_runs_only(self):
        with patch.object(
            main,
            "_reset_browser_sessions_for_thread",
        ) as reset:
            asyncio.run(
                main._finalize_browser_sessions(
                    "browser-test-thread",
                    preserve=True,
                )
            )
            reset.assert_not_called()

            asyncio.run(
                main._finalize_browser_sessions(
                    "browser-test-thread",
                )
            )

        reset.assert_called_once_with("browser-test-thread")

    def test_result_with_pending_interrupt_is_detected(self):
        self.assertTrue(
            main._result_has_pending_interrupt(
                {"__interrupt__": [{"value": "approval"}]}
            )
        )
        self.assertFalse(
            main._result_has_pending_interrupt(
                {"__interrupt__": [], "interrupts": []}
            )
        )

    def test_manual_screenshot_emits_page_title_and_url(self):
        thread_id = "browser-test-thread"
        page_url = "https://example.com/"
        title = "Example Domain"
        captured = {
            "data": base64.b64encode(b"png-bytes").decode("ascii"),
            "metadata": {"url": page_url, "title": title},
        }
        saved = {
            "url": "/api/screenshots/example.png",
            "filename": "example.png",
            "mime_type": "image/png",
        }
        store = MagicMock()
        store.save_screenshot.return_value = saved
        emitter = MagicMock()
        emitter.emit = AsyncMock()

        with (
            patch.object(main, "_assert_thread_owner"),
            patch.object(
                main.browser_sessions,
                "peek",
                return_value=SimpleNamespace(
                    current=SimpleNamespace(url=page_url),
                ),
            ),
            patch.object(
                main,
                "capture_url_screenshot",
                new=AsyncMock(return_value=captured),
            ),
            patch.object(main, "_screenshot_store", return_value=store),
            patch("main.get_emitter", new=AsyncMock(return_value=emitter)),
        ):
            response = self.client.post(
                f"/api/browser/{thread_id}/screenshot"
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        store.save_screenshot.assert_called_once_with(
            b"png-bytes",
            action="browser_manual",
            thread_id=thread_id,
            page_url=page_url,
            metadata={"url": page_url, "title": title},
        )
        event = emitter.emit.await_args.args[1]
        self.assertEqual(event["title"], title)
        self.assertEqual(event["page_url"], page_url)

    def test_manual_screenshot_captures_browser_use_page(self):
        thread_id = "browser-test-thread"
        page_url = "https://example.com/agent-page"
        title = "Agent page"
        image_data = b"\xff\xd8\xffjpeg-bytes"
        captured = {
            "image_data": image_data,
            "metadata": {"url": page_url, "title": title},
        }
        browser_use_session = SimpleNamespace(
            capture_screenshot=MagicMock(return_value=captured),
        )
        saved = {
            "url": "/api/screenshots/agent-page.jpg",
            "filename": "agent-page.jpg",
            "mime_type": "image/jpeg",
        }
        store = MagicMock()
        store.save_screenshot.return_value = saved
        emitter = MagicMock()
        emitter.emit = AsyncMock()

        with (
            patch.object(main, "_assert_thread_owner"),
            patch.object(main.browser_sessions, "peek", return_value=None),
            patch.object(
                main.browser_use_sessions,
                "peek",
                return_value=browser_use_session,
            ),
            patch.object(main, "_screenshot_store", return_value=store),
            patch("main.get_emitter", new=AsyncMock(return_value=emitter)),
        ):
            response = self.client.post(
                f"/api/browser/{thread_id}/screenshot"
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])
        browser_use_session.capture_screenshot.assert_called_once_with()
        store.save_screenshot.assert_called_once_with(
            image_data,
            action="browser_manual",
            thread_id=thread_id,
            page_url=page_url,
            metadata={"url": page_url, "title": title},
        )
        event = emitter.emit.await_args.args[1]
        self.assertEqual(event["title"], title)
        self.assertEqual(event["page_url"], page_url)

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

    def test_browser_stream_initial_status_reads_browser_use_session(self):
        browser_use_session = SimpleNamespace(
            snapshot=MagicMock(
                return_value={
                    "url": "https://example.com/agent-page",
                    "title": "Agent page",
                }
            ),
        )
        with (
            patch.object(main.browser_sessions, "peek", return_value=None),
            patch.object(
                main.browser_use_sessions,
                "peek",
                return_value=browser_use_session,
            ),
            self.client.websocket_connect(
                "/api/browser/browser-test-thread/stream"
            ) as websocket,
        ):
            payload = websocket.receive_json()
            self.assertTrue(payload["active"])
            self.assertEqual(payload["source"], "browser_use")
            self.assertEqual(
                payload["current_url"],
                "https://example.com/agent-page",
            )
            self.assertEqual(payload["title"], "Agent page")
            websocket.send_json({"action": "stop"})
            self.assertEqual(
                websocket.receive_json()["message"],
                "stopped",
            )

    def test_browser_stream_rejects_missing_internal_api_key(self):
        with (
            patch.dict(
                main.settings.__dict__,
                {"internal_api_key": "ws-test-key"},
                clear=False,
            ),
            patch.object(main, "configured_api_key", ""),
            self.assertRaises(WebSocketDisconnect) as context,
        ):
            with self.client.websocket_connect(
                "/api/browser/browser-test-thread/stream"
            ):
                self.fail("Unauthenticated websocket unexpectedly connected")

        self.assertEqual(
            getattr(context.exception, "code", None),
            4401,
        )

    def test_browser_stream_rejects_request_from_non_owner(self):
        with (
            patch.dict(
                main.settings.__dict__,
                {"internal_api_key": "ws-test-key"},
                clear=False,
            ),
            patch.object(main, "configured_api_key", ""),
            patch.dict(
                main._thread_owners,
                {"browser-test-thread": "alice"},
                clear=False,
            ),
            patch.object(main.browser_sessions, "peek") as browser_peek,
            patch.object(main.browser_use_sessions, "peek") as browser_use_peek,
            self.assertRaises(WebSocketDisconnect) as context,
        ):
            with self.client.websocket_connect(
                "/api/browser/browser-test-thread/stream",
                headers={
                    "Authorization": "Bearer ws-test-key",
                    "X-Weaver-User": "bob",
                },
            ):
                self.fail("Non-owner websocket unexpectedly connected")

        self.assertEqual(
            getattr(context.exception, "code", None),
            4403,
        )
        browser_peek.assert_not_called()
        browser_use_peek.assert_not_called()

    def test_browser_stream_requires_user_identity_when_api_key_is_enabled(self):
        with (
            patch.dict(
                main.settings.__dict__,
                {"internal_api_key": "ws-test-key"},
                clear=False,
            ),
            patch.object(main, "configured_api_key", ""),
            patch.object(main.browser_sessions, "peek") as browser_peek,
            patch.object(main.browser_use_sessions, "peek") as browser_use_peek,
            self.assertRaises(WebSocketDisconnect) as context,
        ):
            with self.client.websocket_connect(
                "/api/browser/browser-test-thread/stream",
                headers={"Authorization": "Bearer ws-test-key"},
            ):
                self.fail("Websocket without user identity unexpectedly connected")

        self.assertEqual(context.exception.code, 4403)
        browser_peek.assert_not_called()
        browser_use_peek.assert_not_called()

    def test_browser_stream_accepts_authenticated_owner(self):
        with (
            patch.dict(
                main.settings.__dict__,
                {"internal_api_key": "ws-test-key"},
                clear=False,
            ),
            patch.object(main, "configured_api_key", ""),
            patch.dict(
                main._thread_owners,
                {"browser-test-thread": "alice"},
                clear=False,
            ),
            self.client.websocket_connect(
                "/api/browser/browser-test-thread/stream",
                headers={
                    "Authorization": "Bearer ws-test-key",
                    "X-Weaver-User": "alice",
                },
            ) as websocket,
        ):
            payload = websocket.receive_json()
            self.assertEqual(payload["type"], "status")
            self.assertFalse(payload["active"])
            websocket.send_json({"action": "stop"})
            self.assertEqual(
                websocket.receive_json()["message"],
                "stopped",
            )


if __name__ == "__main__":
    unittest.main()
