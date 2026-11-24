import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from common.config import settings
from tools.search.manager import (
    SearchManager,
    SearchRuntimeState,
    get_default_search_runtime_state,
)


class FakeProvider:
    name = "fake"

    def __init__(self):
        self.calls = []

    def is_available(self):
        return True

    def search(self, query, max_results=5):
        self.calls.append((query, max_results))
        return [{
            "title": query,
            "url": "https://example.com",
            "snippet": "result",
            "content": "result",
            "provider": "fake",
        }]


class SearchCacheTests(unittest.TestCase):
    def manager(self, state=None, provider=None):
        provider = provider or FakeProvider()
        manager = SearchManager(
            {"fake": provider},
            runtime_state=state or SearchRuntimeState(),
        )
        return manager, provider

    def test_cache_is_shared_between_managers(self):
        state = SearchRuntimeState()
        first, first_provider = self.manager(state)
        second, second_provider = self.manager(state)

        with patch.object(settings, "search_engines", "fake"):
            result = first.search("same query")
            self.assertEqual(second.search("same   query"), result)

        self.assertEqual(len(first_provider.calls), 1)
        self.assertEqual(second_provider.calls, [])
        self.assertEqual(first.cache_api_stats()["hits"], 1)

    def test_cache_expires_after_configured_ttl(self):
        manager, provider = self.manager()
        now = [0.0]
        with patch.object(settings, "search_engines", "fake"), patch.object(
            settings, "search_cache_ttl_seconds", 1.0, create=True
        ), patch.object(
            settings, "search_cache_similarity_threshold", 1.0, create=True
        ), patch("tools.search.manager.time.monotonic", side_effect=lambda: now[0]):
            manager.search("query")
            now[0] = 2.0
            manager.search("query")

        self.assertEqual(len(provider.calls), 2)

    def test_cache_evicts_least_recently_used_entry_at_capacity(self):
        manager, provider = self.manager()
        with patch.object(settings, "search_engines", "fake"), patch.object(
            settings, "search_cache_max_size", 2, create=True
        ), patch.object(
            settings, "search_cache_similarity_threshold", 1.0, create=True
        ):
            manager.search("first")
            manager.search("second")
            manager.search("first")
            manager.search("third")
            manager.search("second")

        self.assertEqual(
            [query for query, _ in provider.calls],
            ["first", "second", "third", "second"],
        )
        self.assertEqual(manager.cache_api_stats()["size"], 2)

    def test_similar_queries_hit_but_result_limit_remains_part_of_key(self):
        manager, provider = self.manager()
        with patch.object(settings, "search_engines", "fake"), patch.object(
            settings, "search_cache_similarity_threshold", 0.85, create=True
        ):
            manager.search("latest python release", max_results=5)
            manager.search("latest python releases", max_results=5)
            manager.search("latest python releases", max_results=3)

        self.assertEqual(len(provider.calls), 2)
        self.assertEqual(manager.cache_api_stats()["similar_hits"], 1)

    def test_cache_api_matches_puzzleresearch_response_contract(self):
        state = get_default_search_runtime_state()
        manager, _ = self.manager(state)
        manager.clear_cache()
        client = TestClient(main.app)
        try:
            with patch.object(settings, "search_engines", "fake"):
                manager.search("api cache query")

            stats = client.get("/api/search/cache/stats")
            self.assertEqual(stats.status_code, 200)
            payload = stats.json()
            self.assertEqual(
                set(payload),
                {"stats"},
            )
            self.assertEqual(payload["stats"]["size"], 1)
            self.assertEqual(payload["stats"]["max_size"], 200)
            self.assertIn("similar_hits", payload["stats"])

            cleared = client.post("/api/search/cache/clear")
            self.assertEqual(cleared.status_code, 200)
            self.assertEqual(cleared.json(), {"cleared": True})
            self.assertEqual(
                client.get("/api/search/cache/stats").json()["stats"]["size"],
                0,
            )
        finally:
            manager.clear_cache()


if __name__ == "__main__":
    unittest.main()
