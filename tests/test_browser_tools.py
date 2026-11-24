import sys
import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock, patch

import httpx

from tools.browser_session import BrowserSessionManager, browser_sessions
from tools.builtin import create_default_registry


class BrowserToolTests(unittest.TestCase):
    def tearDown(self):
        browser_sessions.reset(
            "thread-tools-001"
        )

    def test_navigate_tool_uses_thread_session(self):
        response = httpx.Response(
            200,
            text=(
                "<title>工具测试</title>"
                "<p>页面正文</p>"
            ),
            request=httpx.Request(
                "GET",
                "https://example.com",
            ),
        )

        session = browser_sessions.get(
            "thread-tools-001"
        )
        client = MagicMock()
        client.get.return_value = response
        session._client = client

        registry = create_default_registry(
            thread_id="thread-tools-001",
            include_browser=True,
        )

        result = registry.invoke(
            "browser_navigate",
            {
                "url": "https://example.com",
            },
        )

        self.assertEqual(
            result["title"],
            "工具测试",
        )
        self.assertIn(
            "页面正文",
            result["text_excerpt"],
        )

    def test_extract_text_requires_current_page(self):
        registry = create_default_registry(
            thread_id="thread-tools-001",
            include_browser=True,
        )

        with self.assertRaises(ValueError):
            registry.invoke(
                "browser_extract_text",
                {},
            )

    def test_reset_clears_thread_session(self):
        browser_sessions.get(
            "thread-tools-001"
        )

        registry = create_default_registry(
            thread_id="thread-tools-001",
            include_browser=True,
        )

        result = registry.invoke(
            "browser_reset",
            {},
        )

        self.assertEqual(
            result["status"],
            "reset",
        )
        self.assertIsNone(
            browser_sessions.peek(
                "thread-tools-001"
            )
        )

    def test_reset_all_closes_every_http_session(self):
        manager = BrowserSessionManager()
        first = manager.get("one")
        second = manager.get("two")
        first_client = MagicMock()
        second_client = MagicMock()
        first._client = first_client
        second._client = second_client

        self.assertEqual(manager.reset_all(), 2)
        first_client.close.assert_called_once_with()
        second_client.close.assert_called_once_with()
        self.assertIsNone(manager.peek("one"))

    def test_screenshot_waits_for_dom_content_not_network_idle(self):
        page = MagicMock()
        page.url = "https://example.com/final"
        page.title.return_value = "Example Domain"
        page.screenshot.return_value = b"png-data"
        browser = MagicMock()
        browser.new_page.return_value = page
        playwright = SimpleNamespace(
            chromium=SimpleNamespace(
                launch=MagicMock(return_value=browser)
            )
        )
        playwright_context = MagicMock()
        playwright_context.__enter__.return_value = playwright
        sync_api = ModuleType("playwright.sync_api")
        sync_api.sync_playwright = MagicMock(
            return_value=playwright_context
        )
        playwright_module = ModuleType("playwright")

        screenshot_store = MagicMock()
        screenshot_store.save_screenshot.return_value = {
            "url": "/api/screenshots/test.png",
            "filename": "test.png",
            "mime_type": "image/png",
        }

        registry = create_default_registry(
            thread_id="thread-tools-001",
            include_browser=True,
        )

        with (
            patch.dict(
                sys.modules,
                {
                    "playwright": playwright_module,
                    "playwright.sync_api": sync_api,
                },
            ),
            patch(
                "tools.browser_tools.get_screenshot_store",
                return_value=screenshot_store,
            ),
        ):
            result = registry.invoke(
                "browser_screenshot",
                {
                    "url": "https://example.com",
                    "wait_ms": 250,
                },
            )

        page.goto.assert_called_once_with(
            "https://example.com",
            wait_until="domcontentloaded",
            timeout=30000,
        )
        page.wait_for_timeout.assert_called_once_with(250)
        page.screenshot.assert_called_once_with(
            full_page=True,
            type="png",
        )
        self.assertTrue(result["success"])
        self.assertEqual(result["page_url"], "https://example.com/final")
        self.assertEqual(result["title"], "Example Domain")
        screenshot_store.save_screenshot.assert_called_once_with(
            b"png-data",
            action="browser_screenshot",
            thread_id="thread-tools-001",
            page_url="https://example.com/final",
            metadata={
                "url": "https://example.com/final",
                "title": "Example Domain",
            },
        )


if __name__ == "__main__":
    unittest.main()
