import unittest
from unittest.mock import MagicMock

import httpx

from tools.browser_session import browser_sessions
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


if __name__ == "__main__":
    unittest.main()