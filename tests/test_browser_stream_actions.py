import unittest
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


class FakeMouse:
    def __init__(self):
        self.calls = []
        self.fail_click = False

    async def click(self, x, y, *, button="left", click_count=1):
        if self.fail_click:
            raise RuntimeError("click failed")
        self.calls.append(("click", x, y, button, click_count))

    async def move(self, x, y):
        self.calls.append(("move", x, y))

    async def down(self, *, button="left"):
        self.calls.append(("down", button))

    async def up(self, *, button="left"):
        self.calls.append(("up", button))

    async def wheel(self, dx, dy):
        self.calls.append(("wheel", dx, dy))


class FakeKeyboard:
    def __init__(self):
        self.calls = []

    async def press(self, key):
        self.calls.append(("press", key))

    async def type(self, text):
        self.calls.append(("type", text))


class FakePage:
    def __init__(self, *, viewport_size=None):
        self.url = "https://example.com"
        self.viewport_size = viewport_size or {"width": 1000, "height": 500}
        self.mouse = FakeMouse()
        self.keyboard = FakeKeyboard()

    async def title(self):
        return "测试页面"

    async def evaluate(self, _expression):
        return {"w": 1000, "h": 500}

    async def goto(self, url, **_kwargs):
        self.url = url

    async def close(self):
        return None


class BrowserStreamActionTests(unittest.TestCase):
    thread_id = "browser-stream-actions-test"

    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def setUp(self):
        session = main.browser_sessions.get(self.thread_id)
        session.current = SimpleNamespace(
            url="https://example.com",
            title="测试页面",
        )

    def tearDown(self):
        main.browser_sessions.reset(self.thread_id)

    @contextmanager
    def connect(self, page):
        async def fake_start(view, url):
            self.assertEqual(url, "https://example.com")
            view._page = page
            return await view.metadata()

        with patch.object(main.BrowserLiveView, "start", fake_start):
            with self.client.websocket_connect(
                f"/api/browser/{self.thread_id}/stream"
            ) as websocket:
                self.assertEqual(websocket.receive_json()["type"], "status")
                yield websocket

    def test_click_normalizes_coordinates_and_returns_ack(self):
        page = FakePage()
        with self.connect(page) as websocket:
            websocket.send_json(
                {
                    "action": "mouse",
                    "type": "click",
                    "x": 0.5,
                    "y": 0.5,
                    "id": "click-1",
                }
            )
            ack = websocket.receive_json()

        self.assertEqual(ack["type"], "ack")
        self.assertTrue(ack["ok"])
        self.assertEqual(ack["id"], "click-1")
        self.assertEqual(ack["action"], "mouse")
        self.assertIn(("click", 500, 250, "left", 1), page.mouse.calls)

    def test_coordinate_falls_back_to_window_viewport(self):
        page = FakePage(viewport_size=None)
        with self.connect(page) as websocket:
            websocket.send_json(
                {"action": "mouse", "type": "move", "x": 1, "y": 1}
            )
            ack = websocket.receive_json()

        self.assertTrue(ack["ok"])
        self.assertIn(("move", 999, 499), page.mouse.calls)

    def test_scroll_mouse_buttons_and_keyboard_are_executed(self):
        page = FakePage()
        actions = [
            {"action": "mouse", "type": "down", "button": "right"},
            {"action": "mouse", "type": "up", "button": "right"},
            {"action": "scroll", "dx": 5, "dy": 240},
            {"action": "keyboard", "type": "press", "key": "Enter"},
            {"action": "keyboard", "type": "type", "text": "hello"},
        ]

        with self.connect(page) as websocket:
            acknowledgements = []
            for action in actions:
                websocket.send_json(action)
                acknowledgements.append(websocket.receive_json())

        self.assertTrue(all(ack["ok"] for ack in acknowledgements))
        self.assertIn(("down", "right"), page.mouse.calls)
        self.assertIn(("up", "right"), page.mouse.calls)
        self.assertIn(("wheel", 5, 240), page.mouse.calls)
        self.assertEqual(page.keyboard.calls, [("press", "Enter"), ("type", "hello")])

    def test_navigate_updates_page_and_session_metadata(self):
        page = FakePage()
        with self.connect(page) as websocket:
            websocket.send_json(
                {
                    "action": "navigate",
                    "url": "https://example.org/research",
                    "id": "nav-1",
                }
            )
            ack = websocket.receive_json()

        self.assertTrue(ack["ok"])
        self.assertEqual(ack["id"], "nav-1")
        self.assertEqual(page.url, "https://example.org/research")
        self.assertEqual(
            main.browser_sessions.peek(self.thread_id).current.url,
            "https://example.org/research",
        )

    def test_invalid_unknown_and_failed_actions_return_error_ack(self):
        page = FakePage()
        page.mouse.fail_click = True
        with self.connect(page) as websocket:
            requests = [
                {"action": "navigate", "url": "javascript:alert(1)", "id": "bad-url"},
                {"action": "not-supported", "id": "unknown"},
                {"action": "mouse", "type": "click", "x": 0.2, "y": 0.3, "id": "failed"},
            ]
            acknowledgements = []
            for request in requests:
                websocket.send_json(request)
                acknowledgements.append(websocket.receive_json())

        self.assertTrue(all(ack["type"] == "ack" for ack in acknowledgements))
        self.assertTrue(all(not ack["ok"] for ack in acknowledgements))
        self.assertEqual([ack["id"] for ack in acknowledgements], ["bad-url", "unknown", "failed"])
        self.assertIn("http(s)", acknowledgements[0]["error"])
        self.assertIn("Unsupported action", acknowledgements[1]["error"])
        self.assertEqual(acknowledgements[2]["error"], "click failed")


if __name__ == "__main__":
    unittest.main()
