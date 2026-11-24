from __future__ import annotations

from datetime import datetime, timedelta, timezone
import unittest

from agent.workflows.query_strategy import (
    analyze_query_coverage,
    backfill_diverse_queries,
    build_quality_diagnostics,
    is_time_sensitive_topic,
    merge_query_history,
    new_query_history_entries,
    summarize_freshness,
)


class QueryStrategyTests(unittest.TestCase):
    def test_backfill_covers_missing_dimensions(self):
        queries = backfill_diverse_queries(
            "deep research agent",
            ["deep research agent architecture"],
            [],
            5,
        )
        coverage = analyze_query_coverage(queries)
        self.assertEqual(len(queries), 5)
        self.assertGreaterEqual(coverage["score"], 0.8)
        self.assertIn("official", coverage["covered_dimensions"])
        self.assertIn("evidence", coverage["covered_dimensions"])
        self.assertIn("risk", coverage["covered_dimensions"])

    def test_time_sensitive_topics_support_english_and_chinese(self):
        self.assertTrue(is_time_sensitive_topic("latest ai policy updates"))
        self.assertTrue(is_time_sensitive_topic("AI 最新进展"))
        self.assertFalse(is_time_sensitive_topic("history of relational databases"))

    def test_freshness_summary_computes_buckets(self):
        now = datetime.now(timezone.utc)
        result = summarize_freshness(
            [{
                "results": [
                    {"published_date": (now - timedelta(days=2)).isoformat()},
                    {"published_date": (now - timedelta(days=20)).isoformat()},
                    {"published_date": (now - timedelta(days=220)).isoformat()},
                    {"published_date": ""},
                ]
            }]
        )
        self.assertEqual(result["total_results"], 4)
        self.assertEqual(result["known_count"], 3)
        self.assertEqual(result["unknown_count"], 1)
        self.assertEqual(result["fresh_7_count"], 1)
        self.assertEqual(result["fresh_30_count"], 2)
        self.assertEqual(result["stale_180_count"], 1)
        self.assertEqual(result["fresh_30_ratio"], round(2 / 3, 3))

    def test_quality_diagnostics_report_coverage_and_stale_time_sensitive_sources(self):
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        queries = [
            "AI 最新进展 2026",
            "AI 官方文档 release notes",
            "AI benchmark evaluation metrics",
            "AI risks limitations",
            "AI implementation best practices case study",
        ]
        diagnostics = build_quality_diagnostics(
            "AI 最新进展",
            queries,
            [{"results": [{"published_date": old_date} for _ in range(3)]}],
        )

        self.assertEqual(diagnostics["query_coverage_score"], 1.0)
        self.assertEqual(diagnostics["query_dimensions_missing"], [])
        self.assertTrue(diagnostics["time_sensitive_query"])
        self.assertEqual(
            diagnostics["freshness_warning"],
            "low_freshness_for_time_sensitive_query",
        )

    def test_quality_diagnostics_do_not_warn_for_non_time_sensitive_topics(self):
        old_date = (datetime.now(timezone.utc) - timedelta(days=120)).isoformat()
        diagnostics = build_quality_diagnostics(
            "数据库的发展历史",
            [],
            [{"results": [{"published_date": old_date} for _ in range(3)]}],
        )

        self.assertFalse(diagnostics["time_sensitive_query"])
        self.assertEqual(diagnostics["freshness_warning"], "")

    def test_query_history_helpers_preserve_order_and_append_only_new_entries(self):
        history = merge_query_history(
            ["Official docs", "旧查询"],
            ["official DOCS", "新查询", "新查询"],
        )
        additions = new_query_history_entries(
            ["Official docs", "旧查询"],
            ["official DOCS", "新查询", "新查询"],
        )

        self.assertEqual(history, ["Official docs", "旧查询", "新查询"])
        self.assertEqual(additions, ["新查询"])


if __name__ == "__main__":
    unittest.main()
