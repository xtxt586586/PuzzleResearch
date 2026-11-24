from __future__ import annotations

import unittest
from unittest.mock import patch

from tools.builtin import create_default_registry
from tools.search.fallback_search import (
    fallback_search,
    run_fallback_search,
    search_multiple_queries,
)


class FakeProvider:
    def __init__(self, name, results=None, error=None, available=True):
        self.name = name
        self.results = results or []
        self.error = error
        self.available = available
        self.calls = []

    def is_available(self):
        return self.available

    def search(self, *, query, max_results):
        self.calls.append((query, max_results))
        if self.error:
            raise RuntimeError(self.error)
        return self.results[:max_results]


class FakeManager:
    def __init__(self, providers):
        self.providers = providers


class FallbackSearchTests(unittest.TestCase):
    def _manager_patch(self, providers):
        return patch(
            "tools.search.fallback_search.SearchManager",
            return_value=FakeManager(providers),
        )

    def test_alias_unknown_empty_and_failure_continue_fallback(self):
        first = FakeProvider("serper", error="provider failed")
        second = FakeProvider(
            "tavily",
            results=[
                {
                    "title": "结果",
                    "url": "https://example.com/a",
                    "snippet": "摘要",
                    "content": "正文",
                    "provider": "tavily",
                }
            ],
        )
        with self._manager_patch({"serper": first, "tavily": second}):
            engine, results = run_fallback_search(
                query="  查询  ",
                engines=["unknown", "serper", "tavily"],
            )

        self.assertEqual(engine, "tavily")
        self.assertEqual(results[0]["url"], "https://example.com/a")
        self.assertEqual(first.calls, [("查询", 5)])
        self.assertNotIn("provider failed", str(results))

    def test_google_alias_is_normalized(self):
        provider = FakeProvider(
            "google_cse",
            results=[{"url": "https://example.com/google", "title": "Google"}],
        )
        with self._manager_patch({"google_cse": provider}):
            engine, _ = run_fallback_search(query="query", engines=["google"])

        self.assertEqual(engine, "google_cse")

    def test_multiple_queries_isolates_failures_and_deduplicates_urls(self):
        provider = FakeProvider(
            "tavily",
            results=[
                {
                    "title": "同一页面",
                    "url": "https://example.com/page/?utm_source=test",
                    "snippet": "结果",
                }
            ],
        )
        with patch(
            "tools.search.fallback_search.run_fallback_search",
            side_effect=[("tavily", provider.results), RuntimeError("boom")],
        ):
            result = search_multiple_queries(["第一个", "第二个"])

        self.assertEqual(len(result["results"]), 1)
        self.assertEqual(result["results"][0]["query"], "第一个")
        self.assertEqual(result["errors"][0]["query"], "第二个")
        self.assertEqual(result["error"], "部分查询未完成。")

    def test_validation_limits_queries_and_engines(self):
        with self.assertRaisesRegex(ValueError, "非空"):
            search_multiple_queries([])
        with self.assertRaisesRegex(ValueError, "不能超过"):
            search_multiple_queries(["query"] * 11)
        with self.assertRaisesRegex(ValueError, "Provider"):
            run_fallback_search(query="query", engines=["tavily"] * 11)

    def test_tool_output_and_registry_definitions_are_available(self):
        provider = FakeProvider(
            "tavily",
            results=[{"title": "结果", "url": "https://example.com"}],
        )
        with self._manager_patch({"tavily": provider}):
            result = fallback_search("query", engines=["tavily"])

        self.assertEqual(result["engine"], "tavily")
        self.assertIn("结果", result["output"])
        registry = create_default_registry()
        self.assertIsNotNone(registry.get("fallback_search"))
        self.assertIsNotNone(registry.get("search_multiple_queries"))


if __name__ == "__main__":
    unittest.main()
