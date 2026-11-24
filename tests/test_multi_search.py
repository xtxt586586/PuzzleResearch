import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from tools.search.multi_search import (
    MultiSearchOrchestrator,
    SearchProvider,
    SearchResult,
    SearchStrategy,
)
from tools.search.reliability import ProviderReliabilityManager, ReliabilityPolicy


class FlakyProvider(SearchProvider):
    def __init__(self, name="flaky", fail_times=0, result_url="https://example.com/result"):
        super().__init__(name)
        self.fail_times = fail_times
        self.calls = 0
        self.result_url = result_url

    def is_available(self):
        return True

    def search(self, query, max_results=10):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("temporary failure")
        return [SearchResult("result", self.result_url, "result", score=0.8, provider=self.name)]


class MultiSearchTests(unittest.TestCase):
    def test_reliability_retries_then_recovers(self):
        provider = FlakyProvider(fail_times=2)
        reliability = ProviderReliabilityManager(
            ReliabilityPolicy(
                max_retries=2,
                retry_backoff_seconds=0,
                circuit_breaker_failures=5,
                circuit_breaker_reset_seconds=60,
            )
        )
        orchestrator = MultiSearchOrchestrator([provider], reliability_manager=reliability)

        result = orchestrator.search("AI research", max_results=3)

        self.assertEqual(provider.calls, 3)
        self.assertEqual(result[0].url, "https://example.com/result")

    def test_parallel_search_merges_and_deduplicates_tracking_urls(self):
        first = FlakyProvider("first", result_url="https://example.com/a?utm_source=x")
        second = FlakyProvider("second", result_url="https://example.com/a?utm_source=y")
        orchestrator = MultiSearchOrchestrator([first, second])

        result = orchestrator.search("research", max_results=5, strategy=SearchStrategy.PARALLEL)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].url, "https://example.com/a")

    def test_time_sensitive_query_prefers_recent_result(self):
        orchestrator = MultiSearchOrchestrator([])
        old = SearchResult(
            "old", "https://example.com/old", "old result", score=0.9,
            published_date=(datetime.now(timezone.utc) - timedelta(days=400)).isoformat(),
        )
        recent = SearchResult(
            "recent", "https://example.com/recent", "recent result", score=0.6,
            published_date=datetime.now(timezone.utc).isoformat(),
        )

        with patch.object(orchestrator, "_freshness_score", side_effect=[0.01, 1.0]):
            ranked = orchestrator._deduplicate_and_rank([old, recent], 2, "latest AI news")

        self.assertEqual(ranked[0].url, recent.url)

    def test_provider_profile_prioritizes_selected_provider_without_blocking_fallback(self):
        selected = FlakyProvider("selected")
        fallback = FlakyProvider("fallback")
        orchestrator = MultiSearchOrchestrator([fallback, selected])

        result = orchestrator.search(
            "profile query",
            provider_profile=["selected"],
        )

        self.assertEqual(result[0].provider, "selected")
        self.assertEqual(selected.calls, 1)
        self.assertEqual(fallback.calls, 0)


if __name__ == "__main__":
    unittest.main()
