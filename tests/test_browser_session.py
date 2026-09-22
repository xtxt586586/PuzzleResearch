import unittest
from unittest.mock import MagicMock

import httpx

from tools.browser_session import (
    BrowserSession,
    BrowserSessionManager,
)


class BrowserSessionTests(unittest.TestCase):
    def test_navigate_extracts_page_data(self):
        response = httpx.Response(
            200,
            text=(
                "<html><title>测试页面</title>"
                "<body>正文内容"
                "<a href='/next'>下一页</a>"
                "</body></html>"
            ),
            request=httpx.Request(
                "GET",
                "https://example.com",
            ),
        )

        client = MagicMock()
        client.get.return_value = response

        session = BrowserSession()
        session._client = client

        page = session.navigate(
            "https://example.com"
        )

        self.assertEqual(
            page.title,
            "测试页面",
        )
        self.assertIn(
            "正文内容",
            page.text,
        )
        self.assertEqual(
            page.links[0]["url"],
            "https://example.com/next",
        )

    def test_back_restores_previous_page(self):
        session = BrowserSession()
        first = MagicMock(
            url="https://example.com/one"
        )
        second = MagicMock(
            url="https://example.com/two"
        )

        session.current = first
        session.history.append(first)
        session.current = second

        restored = session.back()

        self.assertIs(
            restored,
            first,
        )

    def test_manager_peek_does_not_create_session(self):
        manager = BrowserSessionManager()

        self.assertIsNone(
            manager.peek("missing-thread")
        )

        manager.get("thread-001")

        self.assertIsNotNone(
            manager.peek("thread-001")
        )


if __name__ == "__main__":
    unittest.main()