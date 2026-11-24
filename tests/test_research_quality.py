import unittest

from agent.workflows.knowledge_gap import (
    analyze_research_quality,
    normalize_knowledge_gap_analysis,
)
from agent.workflows.research_tree import (
    ResearchNodeStatus,
    ResearchTree,
)


class ResearchQualityTests(unittest.TestCase):
    def test_knowledge_gap_analysis_normalizes_priority_and_coverage(self):
        result = normalize_knowledge_gap_analysis(
            {
                "overall_coverage": 1.2,
                "confidence": -0.1,
                "gaps": [
                    {
                        "aspect": "市场风险",
                        "importance": "HIGH",
                        "reason": "缺少量化证据",
                    },
                    {"aspect": " 市场风险 ", "importance": "low"},
                    {"aspect": "用户案例", "importance": "unknown"},
                ],
                "suggested_queries": [
                    "官方市场风险数据",
                    "官方市场风险数据",
                ],
                "covered_aspects": ["定义", "定义"],
            },
            coverage_threshold=0.8,
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["overall_coverage"], 1.0)
        self.assertEqual(result["confidence"], 0.0)
        self.assertEqual(
            result["gaps"],
            [
                {
                    "aspect": "市场风险",
                    "importance": "high",
                    "reason": "缺少量化证据",
                },
                {
                    "aspect": "用户案例",
                    "importance": "medium",
                    "reason": "",
                },
            ],
        )
        self.assertEqual(result["suggested_queries"], ["官方市场风险数据"])
        self.assertEqual(result["covered_aspects"], ["定义"])
        self.assertFalse(result["is_sufficient"])

    def test_knowledge_gap_analysis_accepts_coverage_without_high_priority_gaps(self):
        result = normalize_knowledge_gap_analysis(
            {
                "overall_coverage": 0.85,
                "gaps": [{"aspect": "未来趋势", "importance": "medium"}],
            }
        )

        self.assertTrue(result["is_sufficient"])

    def test_tree_records_query_results(self):
        tree = ResearchTree.from_queries(
            "LangGraph",
            [
                "LangGraph 官方文档",
                "LangGraph 状态图教程",
            ],
            max_branches=4,
        )

        tree.record_results(
            [
                {
                    "query": "LangGraph 官方文档",
                    "title": "Official",
                    "url": "https://example.com/docs",
                },
                {
                    "query": "LangGraph 官方文档",
                    "title": "Guide",
                    "url": "https://example.com/guide",
                },
            ]
        )
        tree.finalize()

        payload = tree.to_dict()

        branch_nodes = [
            node
            for node in payload["nodes"]
            if node["depth"] == 1
        ]

        self.assertEqual(
            len(branch_nodes),
            2,
        )

        completed = next(
            node
            for node in branch_nodes
            if node["query"]
            == "LangGraph 官方文档"
        )

        self.assertEqual(
            completed["status"],
            ResearchNodeStatus.COMPLETED.value,
        )
        self.assertEqual(
            completed["result_count"],
            2,
        )

    def test_quality_requires_body_citations(self):
        quality = analyze_research_quality(
            report=(
                "报告正文没有引用。"
                "\n\n参考来源：\n"
                "[来源 1] Official"
            ),
            sources=[
                {
                    "title": "Official",
                    "url": "https://example.com",
                },
            ],
            citation_threshold=0.6,
        )

        self.assertEqual(
            quality.citation_coverage,
            0.0,
        )
        self.assertFalse(
            quality.sufficient
        )
        self.assertIn(
            "报告引用覆盖率不足",
            quality.missing_topics,
        )

    def test_quality_accepts_cited_sources(self):
        quality = analyze_research_quality(
            report=(
                "结论来自 [来源 1]。"
                "\n\n参考来源：\n"
                "[来源 1] Official"
            ),
            sources=[
                {
                    "title": "Official",
                    "url": "https://example.com",
                },
            ],
            citation_threshold=0.6,
        )

        self.assertEqual(
            quality.citation_coverage,
            1.0,
        )
        self.assertTrue(
            quality.sufficient
        )


if __name__ == "__main__":
    unittest.main()
