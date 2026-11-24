from __future__ import annotations

import unittest
from unittest.mock import patch

from common.config import settings
from tools.builtin import create_default_registry
from tools.search.search import enhanced_tavily_search, tavily_search


class FakeTavilyClient:
    def __init__(self, payload):
        self.payload = payload
        self.calls = []

    def search(self, **kwargs):
        self.calls.append(kwargs)
        return self.payload


class EnhancedSearchTests(unittest.TestCase):
    def test_enhanced_search_keeps_answer_deduplicates_and_summarizes_first_only(self):
        client = FakeTavilyClient(
            {
                "answer": "综合答案",
                "results": [
                    {
                        "title": "第一条",
                        "url": "https://example.com/a?utm_source=x",
                        "content": "摘要一",
                        "raw_content": "正文一",
                        "score": 0.9,
                    },
                    {
                        "title": "第一条副本",
                        "url": "https://example.com/a?utm_medium=x",
                        "content": "摘要副本",
                        "raw_content": "正文副本",
                        "score": 0.8,
                    },
                    {
                        "title": "第二条",
                        "url": "https://example.com/b",
                        "content": "摘要二",
                        "raw_content": "正文二",
                        "score": 0.7,
                    },
                ],
            }
        )
        with patch.object(settings, "tavily_api_key", "test-key"), patch(
            "tools.search.search._tavily_client", return_value=client
        ), patch(
            "tools.search.search._summarize_content",
            side_effect=lambda text: f"摘要：{text}",
        ) as summarize:
            result = enhanced_tavily_search(
                "  人工智能  ",
                max_results=3,
                search_depth="advanced",
                include_answer=True,
            )

        self.assertEqual(len(result["results"]), 2)
        self.assertEqual(result["results"][0]["summary"], "摘要：正文一")
        self.assertEqual(result["results"][1]["summary"], "摘要二")
        self.assertEqual(result["ai_answer"], "综合答案")
        self.assertEqual(summarize.call_count, 1)
        self.assertEqual(client.calls[0]["include_raw_content"], True)
        self.assertEqual(client.calls[0]["search_depth"], "advanced")

    def test_missing_key_disables_enhanced_search_without_network_call(self):
        with patch.object(settings, "tavily_api_key", ""), patch(
            "tools.search.search._tavily_client"
        ) as client:
            result = enhanced_tavily_search("query")

        self.assertEqual(result["results"], [])
        self.assertIn("未配置", result["error"])
        client.assert_not_called()

    def test_legacy_function_returns_result_list(self):
        with patch(
            "tools.search.search.enhanced_tavily_search",
            return_value={"results": [{"url": "https://example.com"}]},
        ):
            self.assertEqual(
                tavily_search("query"),
                [{"url": "https://example.com"}],
            )

    def test_arguments_and_registry_are_valid(self):
        with self.assertRaisesRegex(ValueError, "basic 或 advanced"):
            enhanced_tavily_search("query", search_depth="deep")
        with self.assertRaisesRegex(ValueError, "1 到 20"):
            enhanced_tavily_search("query", max_results=21)

        tool = create_default_registry().get("tavily_search")
        self.assertEqual(
            [item["name"] for item in tool.argument_schema()],
            ["query", "max_results", "search_depth", "include_answer"],
        )


if __name__ == "__main__":
    unittest.main()
