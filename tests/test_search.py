import unittest
from unittest.mock import patch

from agent.workflows.nodes import search_node
from common.config import settings
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
