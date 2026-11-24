from __future__ import annotations

import unittest
from unittest.mock import patch

import httpx

from agent.workflows.nodes import agent_tool_node
from tools.builtin import create_default_registry
from tools.web_crawler import crawl_url, crawl_urls


class WebCrawlerTests(unittest.TestCase):
    def run_with_response(self, handler, callback):
        def create_client(timeout):
            return httpx.Client(
                transport=httpx.MockTransport(handler),
                follow_redirects=False,
                trust_env=False,
            )

        with (
            patch("tools.web_crawler._create_client", side_effect=create_client),
            patch(
                "tools.web_crawler._resolve_host_addresses",
                return_value={"93.184.216.34"},
            ),
        ):
            return callback()

    def test_crawl_url_extracts_visible_text_and_truncates(self):
        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/html; charset=utf-8"},
                text=(
                    "<html><script>hidden()</script><style>.x{}</style>"
                    f"<main><h1>研究标题</h1><p>{'正文内容需要保留。' * 30}</p>"
                    "</main></html>"
                ),
                request=request,
            )

        result = self.run_with_response(
            handler,
            lambda: crawl_url("https://example.test/page", max_chars=200),
        )

        self.assertEqual(result["url"], "https://example.test/page")
        self.assertIn("研究标题", result["output"])
        self.assertIn("正文内容需要保留。", result["output"])
        self.assertNotIn("hidden()", result["output"])
        self.assertEqual(len(result["output"]), 203)
        self.assertTrue(result["output"].endswith("..."))
        self.assertEqual(result["error"], "")

    def test_crawl_urls_returns_one_result_per_url(self):
        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text=f"抓取自 {request.url.host}",
                request=request,
            )

        result = self.run_with_response(
            handler,
            lambda: crawl_urls(
                ["https://one.example.test", "https://two.example.test"]
            ),
        )

        self.assertEqual(len(result), 2)
        self.assertIn("one.example.test", result[0]["output"])
        self.assertIn("two.example.test", result[1]["output"])

    def test_redirect_to_private_address_is_rejected(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(
                302,
                headers={"location": "http://127.0.0.1/private"},
                request=request,
            )

        with self.assertRaisesRegex(ValueError, "内部网络地址"):
            self.run_with_response(
                handler,
                lambda: crawl_url("https://example.test/start"),
            )
        self.assertEqual(len(requests), 1)

    def test_local_and_non_http_urls_are_rejected(self):
        for url in ("http://127.0.0.1", "http://localhost", "file:///etc/passwd"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    crawl_url(url)

    def test_limits_are_validated_before_network_access(self):
        with patch("tools.web_crawler._create_client") as create_client:
            with self.assertRaises(ValueError):
                crawl_url("https://example.test", timeout=61)
            with self.assertRaises(ValueError):
                crawl_url("https://example.test", max_chars=100)
            with self.assertRaises(ValueError):
                crawl_urls([f"https://{index}.example.test" for index in range(11)])
        create_client.assert_not_called()

    def test_default_registry_exposes_crawl_tools(self):
        registry = create_default_registry()

        single = registry.get("crawl_url")
        multiple = registry.get("crawl_urls")

        self.assertEqual(single.name, "crawl_url")
        self.assertEqual(multiple.name, "crawl_urls")
        self.assertEqual(
            [item["name"] for item in single.argument_schema()],
            ["url", "timeout", "max_chars"],
        )

    def test_agent_tool_node_exposes_crawled_text_as_standard_output(self):
        def handler(request):
            return httpx.Response(
                200,
                headers={"content-type": "text/plain"},
                text="IANA reserves example domains for documentation.",
                request=request,
            )

        registry = create_default_registry()
        result = self.run_with_response(
            handler,
            lambda: agent_tool_node(
                {
                    "pending_tool_calls": [
                        {
                            "tool_name": "crawl_url",
                            "arguments": {"url": "https://example.test"},
                        }
                    ],
                    "tool_approved": True,
                },
                config={"configurable": {"tool_registry": registry}},
            ),
        )

        self.assertEqual(result["status"], "running")
        self.assertIn("IANA reserves example domains", result["final_report"])
        self.assertEqual(
            result["tool_results"][0]["result"]["error"],
            "",
        )
        self.assertEqual(
            result["tool_events"][-1]["event_type"],
            "tool_completed",
        )


if __name__ == "__main__":
    unittest.main()
