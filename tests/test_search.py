import unittest
from unittest.mock import patch

from agent.workflows.nodes import search_node
from common.config import settings
from tools.research.models import FetchedPage
from tools.search.manager import SearchManager


class FakeProvider:
    def __init__(self, name, results=None, error=None):
        self.name = name
        self.results = results or []
        self.error = error
        self.calls = []

    def is_available(self):
        return True

    def search(self, query, max_results=5):
        self.calls.append((query, max_results))
        if self.error:
            raise self.error
        return self.results


class SearchContractTests(unittest.TestCase):
    def test_manager_falls_back_to_next_provider(self):
        first = FakeProvider("first", error=RuntimeError("temporary"))
        second = FakeProvider(
            "second",
            results=[
                {
                    "title": "Result",
                    "url": "https://example.com",
                    "snippet": "snippet",
                    "content": "content",
                    "provider": "second",
                }
            ],
        )

        with patch.object(settings, "search_engines", "first,second"):
            results = SearchManager(
                {"first": first, "second": second}
            ).search("query", max_results=3)

        self.assertEqual(results[0]["provider"], "second")
        self.assertEqual(first.calls, [("query", 3)])
        self.assertEqual(second.calls, [("query", 3)])

    def test_deep_search_uses_deep_result_limit(self):
        provider = FakeProvider(
            "fake",
            results=[
                {
                    "title": "Result",
                    "url": "https://example.com",
                    "snippet": "snippet",
                    "content": "content",
                    "provider": "fake",
                }
            ],
        )

        with patch.object(settings, "search_engines", "fake"), patch.object(
            settings, "deepsearch_results_per_query", 2
        ), patch(
            "agent.workflows.nodes.SearchManager",
            return_value=SearchManager({"fake": provider}),
        ):
            result = search_node(
                {"query": "deep query", "search_mode": "deep"}
            )

        self.assertEqual(provider.calls, [("deep query", 2)])
        self.assertEqual(len(result["search_results"]), 1)

    def test_deep_search_uses_configured_multi_provider_strategy(self):
        first = FakeProvider(
            "first",
            results=[
                {
                    "title": "First",
                    "url": "https://first.example.com",
                    "snippet": "first result",
                    "content": "first content",
                    "provider": "first",
                    "score": 0.5,
                }
            ],
        )
        second = FakeProvider(
            "second",
            results=[
                {
                    "title": "Second",
                    "url": "https://second.example.com",
                    "snippet": "second result",
                    "content": "second content",
                    "provider": "second",
                    "score": 0.9,
                }
            ],
        )

        with patch.object(settings, "search_engines", "first,second"), patch.object(
            settings, "search_strategy", "parallel", create=True
        ), patch.object(
            settings, "search_reliability_max_retries", 0, create=True
        ), patch(
            "agent.workflows.nodes.SearchManager",
            return_value=SearchManager({"first": first, "second": second}),
        ):
            result = search_node({"query": "deep query", "search_mode": "deep"})

        self.assertEqual(first.calls, [("deep query", 5)])
        self.assertEqual(second.calls, [("deep query", 5)])
        self.assertEqual(
            [item["provider"] for item in result["search_results"]],
            ["second", "first"],
        )

    def test_deep_search_attaches_fetched_pages_when_enabled(self):
        provider = FakeProvider(
            "fake",
            results=[
                {
                    "title": "Result",
                    "url": "https://example.com/source",
                    "snippet": "snippet",
                    "content": "summary",
                    "provider": "fake",
                }
            ],
        )
        fetched_page = FetchedPage(
            url="https://example.com/source",
            raw_url="https://example.com/source",
            method="direct_http",
            text="full source body",
            http_status=200,
        )

        with patch.object(settings, "search_engines", "fake"), patch.object(
            settings, "deepsearch_enable_research_fetcher", True
        ), patch(
            "agent.workflows.nodes.SearchManager",
            return_value=SearchManager({"fake": provider}),
        ), patch("agent.workflows.nodes.ContentFetcher") as fetcher_class:
            fetcher_class.return_value.fetch_many.return_value = [fetched_page]
            result = search_node(
                {"query": "deep query", "search_mode": "deep"}
            )

        self.assertEqual(
            result["fetched_pages"][0]["method"],
            "direct_http",
        )
        self.assertEqual(
            result["search_results"][0]["fetched_content"],
            "full source body",
        )
        fetcher_class.return_value.fetch_many.assert_called_once_with(
            ["https://example.com/source"]
        )

    def test_manager_reuses_cached_result(self):
        provider = FakeProvider(
            "fake",
            results=[
                {
                    "title": "Cached",
                    "url": "https://example.com",
                    "snippet": "snippet",
                    "content": "content",
                    "provider": "fake",
                }
            ],
        )

        manager = SearchManager(
            {"fake": provider}
        )

        with patch.object(
                settings,
                "search_engines",
                "fake",
        ):
            first = manager.search("same query", 3)
            second = manager.search("same query", 3)

        self.assertEqual(first, second)
        self.assertEqual(
            provider.calls,
            [("same query", 3)],
        )

        stats = manager.cache_stats()

        self.assertEqual(stats["entries"], 1)
        self.assertEqual(stats["hits"], 1)
        self.assertEqual(stats["misses"], 1)
        self.assertEqual(stats["hit_rate"], 0.5)

    def test_provider_call_stats_use_api_field_names(self):
        provider = FakeProvider(
            "fake",
            results=[
                {
                    "title": "Result",
                    "url": "https://example.com",
                    "snippet": "snippet",
                    "content": "content",
                    "provider": "fake",
                }
            ],
        )
        manager = SearchManager({"fake": provider})

        with patch.object(settings, "search_engines", "fake"):
            manager.search("stats query")

        self.assertEqual(
            manager.provider_call_stats()["fake"],
            {
                "calls": 1,
                "successes": 1,
                "failures": 0,
                "empty_results": 0,
                "last_error": "",
            },
        )

    def test_clear_cache_forces_provider_call(self):
        provider = FakeProvider(
            "fake",
            results=[
                {
                    "title": "Result",
                    "url": "https://example.com",
                    "snippet": "snippet",
                    "content": "content",
                    "provider": "fake",
                }
            ],
        )

        manager = SearchManager(
            {"fake": provider}
        )

        with patch.object(
                settings,
                "search_engines",
                "fake",
        ):
            manager.search("query", 2)
            self.assertEqual(manager.clear_cache(), 1)
            manager.search("query", 2)

        self.assertEqual(
            provider.calls,
            [
                ("query", 2),
                ("query", 2),
            ],
        )


if __name__ == "__main__":
    unittest.main()
