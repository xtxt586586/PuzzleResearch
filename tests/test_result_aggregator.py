import unittest

from agent.workflows.result_aggregator import ResultAggregator


class ResultAggregatorTests(unittest.TestCase):
    def test_aggregate_deduplicates_tracking_urls_and_similar_content(self):
        results = ResultAggregator().aggregate(
            [
                {
                    "query": "python",
                    "title": "Python guide",
                    "url": "https://example.com/guide?utm_source=test",
                    "content": "Python is a programming language.",
                    "score": 0.7,
                },
                {
                    "query": "python",
                    "title": "Python guide copy",
                    "url": "https://example.com/guide?utm_medium=test",
                    "content": "Python is a programming language.",
                    "score": 0.6,
                },
                {
                    "query": "python",
                    "title": "Python official documentation",
                    "url": "https://docs.python.org/3/",
                    "content": "Python documentation covers the language reference.",
                    "score": 0.8,
                },
            ],
            original_query="Python language",
        )

        self.assertEqual(results.total_before, 3)
        self.assertEqual(results.total_after, 2)
        self.assertEqual(results.duplicates_removed, 1)
        self.assertEqual(results.all_results()[0].canonical_url, "https://docs.python.org/3/")

    def test_aggregate_preserves_source_metadata_and_context(self):
        results = ResultAggregator().aggregate(
            [
                {
                    "query": "research",
                    "title": "Research Paper",
                    "url": "https://arxiv.org/abs/1234",
                    "snippet": "Research abstract",
                    "provider": "arxiv",
                }
            ],
            original_query="research",
        )

        item = results.to_dicts()[0]
        context, sources = results.to_context()
        self.assertEqual(item["provider"], "arxiv")
        self.assertTrue(item["source_id"].startswith("src_"))
        self.assertIn("主要来源", context)
        self.assertIn("arxiv.org/abs/1234", sources)

    def test_empty_and_invalid_urls_are_ignored(self):
        results = ResultAggregator().aggregate(
            [{"title": "missing url", "content": "ignored"}, None],
            original_query="query",
        )
        self.assertEqual(results.total_before, 0)
        self.assertEqual(results.total_after, 0)


if __name__ == "__main__":
    unittest.main()
