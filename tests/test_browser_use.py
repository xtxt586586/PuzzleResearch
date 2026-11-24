import sys
import threading
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.builtin import create_default_registry
from tools.browser_use import BrowserUseSessionManager, browser_use_sessions


class BrowserUseTests(unittest.TestCase):
    def setUp(self):
        self.thread_id = "browser-use-test"
        browser_use_sessions.reset(self.thread_id)

        self.page = MagicMock()
        self.page.url = "https://example.com"
        self.page.title.return_value = "Example"
        self.page.screenshot.return_value = b"\xff\xd8\xfftest"
        self.page.content.return_value = "<html><body>Example content</body></html>"

        self.context = MagicMock()
        self.context.pages = [self.page]
        self.context.new_page.return_value = self.page
        self.browser = MagicMock()
        self.browser.new_context.return_value = self.context
        self.playwright = SimpleNamespace(
            chromium=SimpleNamespace(launch=MagicMock(return_value=self.browser))
        )
        self.playwright_starter = MagicMock()
        self.playwright_starter.start.return_value = self.playwright

        sync_api = ModuleType("playwright.sync_api")
        sync_api.sync_playwright = MagicMock(return_value=self.playwright_starter)
        playwright_module = ModuleType("playwright")
        self.modules = {
            "playwright": playwright_module,
            "playwright.sync_api": sync_api,
        }

        self.saved_screenshot = {
            "url": "/api/screenshots/browser-use.jpg",
            "filename": "browser-use.jpg",
            "page_url": "https://example.com",
            "mime_type": "image/jpeg",
        }

    def tearDown(self):
        browser_use_sessions.reset(self.thread_id)

    def _registry(self):
        return create_default_registry(
            thread_id=self.thread_id,
            include_browser=True,
        )

    def test_navigation_and_screenshot_use_persistent_playwright_session(self):
        with (
            patch.dict(sys.modules, self.modules),
            patch("tools.builtin.settings.enable_browser_use", True),
            patch(
                "tools.browser_use.get_screenshot_store"
            ) as screenshot_store,
        ):
            screenshot_store.return_value.save_screenshot.return_value = self.saved_screenshot
            result = self._registry().invoke(
                "browser_use",
                {"action": "go_to_url", "url": "https://example.com"},
            )
            second = self._registry().invoke(
                "browser_use",
                {"action": "extract_content", "goal": "read page"},
            )

        screenshot_store.return_value.save_screenshot.assert_called_once_with(
            self.page.screenshot.return_value,
            action="browser_use_navigate",
            thread_id=self.thread_id,
            page_url="https://example.com",
            metadata={"url": "https://example.com", "title": "Example"},
        )
        self.assertEqual(result["url"], "https://example.com")
        self.assertEqual(result["screenshot"], "/api/screenshots/browser-use.jpg")
        self.assertIn("Example content", second["content"])
        self.playwright.chromium.launch.assert_called_once_with(headless=True)
        self.assertEqual(self.browser.new_context.call_count, 1)

    def test_session_snapshot_and_manual_screenshot_use_playwright_page(self):
        session = browser_use_sessions.get(self.thread_id)
        session.page = self.page

        snapshot = session.snapshot()
        captured = session.capture_screenshot()

        self.assertEqual(
            snapshot,
            {
                "url": "https://example.com",
                "title": "Example",
            },
        )
        self.assertEqual(
            captured,
            {
                "image_data": self.page.screenshot.return_value,
                "metadata": {
                    "url": "https://example.com",
                    "title": "Example",
                },
            },
        )
        self.page.screenshot.assert_called_once_with(
            full_page=True,
            type="jpeg",
            quality=85,
        )

    def test_context_snapshot_returns_page_text_and_links(self):
        session = browser_use_sessions.get(self.thread_id)
        session.page = self.page
        body = MagicMock()
        body.inner_text.return_value = "Page body"
        links = MagicMock()
        links.evaluate_all.return_value = [
            {"text": "Source", "url": "https://example.com/source"}
        ]
        self.page.locator.side_effect = [body, links]

        snapshot = session.context_snapshot()

        self.assertEqual(
            snapshot,
            {
                "url": "https://example.com",
                "title": "Example",
                "text": "Page body",
                "links": [{"text": "Source", "url": "https://example.com/source"}],
            },
        )
        body.inner_text.assert_called_once_with(timeout=1000)
        links.evaluate_all.assert_called_once()

    def test_preview_actions_operate_on_persistent_playwright_page(self):
        session = browser_use_sessions.get(self.thread_id)
        session.page = self.page
        self.page.viewport_size = {"width": 1280, "height": 720}
        self.page.mouse = MagicMock()
        self.page.keyboard = MagicMock()

        clicked = session.perform_preview_action(
            "mouse",
            {"type": "click", "x": 0.5, "y": 0.25, "clicks": 2},
        )
        session.perform_preview_action(
            "scroll",
            {"dx": 4, "dy": 120},
        )
        session.perform_preview_action(
            "keyboard",
            {"type": "press", "key": "Enter"},
        )
        session.perform_preview_action(
            "navigate",
            {"url": "https://example.org/research"},
        )

        self.assertEqual(clicked["url"], "https://example.com")
        self.page.mouse.click.assert_called_once_with(
            640,
            180,
            button="left",
            click_count=2,
        )
        self.page.mouse.wheel.assert_called_once_with(4, 120)
        self.page.keyboard.press.assert_called_once_with("Enter")
        self.page.goto.assert_called_once_with(
            "https://example.org/research",
            wait_until="domcontentloaded",
            timeout=30000,
        )

    def test_preview_action_rejects_invalid_navigation_url(self):
        session = browser_use_sessions.get(self.thread_id)
        session.page = self.page

        with self.assertRaisesRegex(ValueError, "http\(s\)"):
            session.perform_preview_action(
                "navigate",
                {"url": "javascript:alert(1)"},
            )

    def test_click_input_scroll_and_tabs_use_action_contract(self):
        locator = MagicMock()
        locator.count.return_value = 1
        locator.nth.return_value = locator
        locator.is_visible.return_value = True
        self.page.locator.return_value = locator

        second_page = MagicMock()
        second_page.url = "https://example.com/second"
        second_page.title.return_value = "Second"
        self.context.pages = [self.page, second_page]
        self.context.new_page.return_value = self.page

        with (
            patch.dict(sys.modules, self.modules),
            patch("tools.builtin.settings.enable_browser_use", True),
            patch("tools.browser_use.get_screenshot_store"),
        ):
            registry = self._registry()
            clicked = registry.invoke(
                "browser_use",
                {"action": "click_element", "index": 0},
            )
            typed = registry.invoke(
                "browser_use",
                {"action": "input_text", "index": 0, "text": "PuzzleResearch"},
            )
            scrolled = registry.invoke(
                "browser_use",
                {"action": "scroll_down", "scroll_amount": 400},
            )
            switched = registry.invoke(
                "browser_use",
                {"action": "switch_tab", "tab_id": 1},
            )

        self.assertEqual(clicked["clicked"], 0)
        self.assertEqual(typed["input"], "PuzzleResearch")
        self.assertEqual(scrolled["scrolled"], 400)
        self.assertEqual(switched["tab"], 1)
        locator.click.assert_called_once_with()
        locator.fill.assert_called_once_with("PuzzleResearch")
        self.page.evaluate.assert_called_once()

    def test_invalid_action_returns_structured_error(self):
        with (
            patch.dict(sys.modules, self.modules),
            patch("tools.builtin.settings.enable_browser_use", True),
        ):
            result = self._registry().invoke(
                "browser_use",
                {"action": "unknown_action"},
            )

        self.assertIn("error", result)
        self.assertIn("不支持", result["error"])

    def test_common_navigate_alias_maps_to_go_to_url(self):
        with (
            patch.dict(sys.modules, self.modules),
            patch("tools.builtin.settings.enable_browser_use", True),
            patch("tools.browser_use.get_screenshot_store"),
        ):
            result = self._registry().invoke(
                "browser_use",
                {"action": "navigate", "url": "https://example.com"},
            )

        self.assertEqual(result["status"], "ok")
        self.page.goto.assert_called_once_with(
            "https://example.com",
            wait_until="domcontentloaded",
            timeout=30000,
        )

    def test_tool_is_disabled_by_default(self):
        with patch("tools.builtin.settings.enable_browser_use", False):
            names = {
                tool.name
                for tool in create_default_registry(
                    thread_id=self.thread_id,
                    include_browser=True,
                ).list_tools()
            }

        self.assertNotIn("browser_use", names)

    def test_reset_all_closes_every_playwright_session(self):
        manager = BrowserUseSessionManager()
        first = MagicMock()
        second = MagicMock()
        manager._sessions.update({"one": first, "two": second})

        self.assertEqual(manager.reset_all(), 2)
        first.close.assert_called_once_with()
        second.close.assert_called_once_with()
        self.assertEqual(manager._sessions, {})

    def test_browser_operations_and_close_share_one_worker_thread(self):
        manager = BrowserUseSessionManager()
        session = manager.get("thread-affinity")
        close_threads = []

        class CloseProbe:
            def close(self):
                close_threads.append(threading.get_ident())

        session.context = CloseProbe()
        first_thread = session.run(threading.get_ident)
        second_thread = session.run(threading.get_ident)
        manager.reset("thread-affinity")

        self.assertNotEqual(first_thread, threading.get_ident())
        self.assertEqual(second_thread, first_thread)
        self.assertEqual(close_threads, [first_thread])
        self.assertEqual(manager._sessions, {})


if __name__ == "__main__":
    unittest.main()
