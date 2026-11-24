import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.workflows.quality_assessor import QualityAssessor
from agent.workflows.quality_assessor import QualityReport
from agent.workflows.nodes import evaluator_node
from common.config import settings


class QualityAssessorTests(unittest.TestCase):
    def test_assess_reports_claim_and_source_dimensions(self):
        report = "According to the 2024 study, revenue increased by 20% [1]."
        scraped = [{
            "results": [{
                "url": "https://www.example.com/report",
                "summary": "The 2024 study says revenue increased by 20%.",
            }]
        }]

        result = QualityAssessor().assess(
            report,
            scraped,
            ["https://www.example.com/report"],
        )

        self.assertEqual(result.claim_support_score, 1.0)
        self.assertEqual(result.unique_domains, ["example.com"])
        self.assertGreaterEqual(result.citation_coverage_score, 0.99)
        self.assertIn("claim_support_score", result.to_dict())

    def test_invalid_citation_and_uncited_claim_are_reported(self):
        result = QualityAssessor().assess(
            "The 2024 report found growth of 30% [9].",
            [{
                "results": [{
                    "url": "https://example.com/source",
                    "summary": "The report discusses the market.",
                }]
            }],
            ["https://example.com/source"],
        )

        self.assertLess(result.citation_accuracy_score, 1.0)
        self.assertLess(result.citation_coverage_score, 1.0)
        self.assertTrue(result.missing_citations)

    def test_internal_opposite_trends_are_detected(self):
        result = QualityAssessor().assess(
            "The study found revenue increased in 2024. "
            "Another study found revenue decreased in 2024.",
            [],
            [],
        )

        self.assertTrue(result.contradictions)
        self.assertLess(result.contradiction_free_score, 1.0)

    def test_evaluator_persists_enhanced_quality_summary(self):
        class FakeModel:
            def invoke(self, messages, config=None):
                return SimpleNamespace(content=json.dumps({
                    "verdict": "pass",
                    "evaluation": "good",
                    "eval_dimensions": {
                        "accuracy": 0.9,
                        "completeness": 0.9,
                        "citation": 0.9,
                    },
                    "missing_topics": [],
                }))

        class FakeAssessor:
            def __init__(self, llm=None, config=None):
                pass

            def assess(self, report, scraped_content, sources=None):
                return QualityReport(
                    claim_support_score=0.9,
                    source_diversity_score=0.8,
                    contradiction_free_score=1.0,
                    citation_accuracy_score=0.9,
                    citation_coverage_score=0.92,
                    overall_score=0.88,
                )

        state = {
            "route": "deep",
            "input": "Summarize AI chip market trends",
            "draft_report": "According to the report, revenue increased in 2024.",
            "search_results": [],
            "sources": [],
            "deepsearch_artifacts": {
                "mode": "linear",
                "quality_summary": {
                    "query_coverage_score": 0.75,
                    "freshness_summary": {
                        "known_count": 4,
                        "fresh_30_ratio": 0.5,
                    },
                },
            },
        }

        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=FakeModel()), \
             patch("agent.workflows.quality_assessor.QualityAssessor", FakeAssessor):
            result = evaluator_node(state, config={})

        self.assertEqual(result["quality_summary"]["overall_score"], 0.88)
        self.assertEqual(result["quality_summary"]["citation_coverage"], 0.92)
        self.assertEqual(result["quality_summary"]["query_coverage_score"], 0.75)
        self.assertEqual(
            result["quality_summary"]["freshness_summary"]["known_count"],
            4,
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["quality_summary"]["query_coverage_score"],
            0.75,
        )

    def test_evaluator_persists_high_priority_knowledge_gaps(self):
        requested_tasks = []

        class FakeModel:
            def invoke(self, messages, config=None):
                return SimpleNamespace(content=json.dumps({
                    "verdict": "pass",
                    "evaluation": "整体较完整",
                    "eval_dimensions": {"accuracy": 0.9},
                    "missing_topics": [],
                    "knowledge_gap_analysis": {
                        "overall_coverage": 0.72,
                        "confidence": 0.86,
                        "gaps": [{
                            "aspect": "不同地区的成本差异",
                            "importance": "high",
                            "reason": "当前证据只覆盖单一地区",
                        }],
                        "suggested_queries": [
                            "不同地区官方成本数据对比",
                        ],
                        "covered_aspects": ["基本定义", "主要机制"],
                        "analysis": "地域证据不足",
                    },
                }))

        def model_for_task(task, **kwargs):
            requested_tasks.append(task)
            return FakeModel()

        class FakeAssessor:
            def __init__(self, llm=None, config=None):
                pass

            def assess(self, report, scraped_content, sources=None):
                return QualityReport(
                    claim_support_score=0.9,
                    source_diversity_score=0.8,
                    contradiction_free_score=1.0,
                    citation_accuracy_score=0.9,
                    citation_coverage_score=0.92,
                    overall_score=0.88,
                )

        state = {
            "route": "deep",
            "deepsearch_epoch": 1,
            "deepsearch_max_epochs": 3,
            "input": "比较不同地区的成本",
            "draft_report": "当前研究报告。",
            "sources": [],
            "deepsearch_artifacts": {"phase": "synthesized"},
        }

        with patch("agent.workflows.nodes.ModelFactory.for_task", side_effect=model_for_task), \
             patch("agent.workflows.quality_assessor.QualityAssessor", FakeAssessor):
            result = evaluator_node(state, config={})

        self.assertEqual(requested_tasks, ["evaluation"])
        self.assertEqual(result["verdict"], "revise")
        self.assertIn("不同地区的成本差异", result["missing_topics"])
        self.assertEqual(
            result["knowledge_gap_analysis"]["overall_coverage"],
            0.72,
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["knowledge_gap_analysis"]["suggested_queries"],
            ["不同地区官方成本数据对比"],
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["phase"],
            "synthesized",
        )

    def test_evaluator_skips_model_call_when_budget_is_exhausted(self):
        requested_tasks = []

        class FakeModel:
            def invoke(self, messages, config=None):
                return SimpleNamespace(content=json.dumps({
                    "verdict": "pass",
                    "evaluation": "complete",
                    "eval_dimensions": {"accuracy": 0.9},
                    "missing_topics": [],
                    "knowledge_gap_analysis": {
                        "overall_coverage": 0.9,
                        "gaps": [],
                        "suggested_queries": [],
                    },
                }))

        def model_for_task(task, **kwargs):
            requested_tasks.append(task)
            return FakeModel()

        class FakeAssessor:
            def __init__(self, llm=None, config=None):
                pass

            def assess(self, report, scraped_content, sources=None):
                return QualityReport(
                    claim_support_score=1.0,
                    source_diversity_score=1.0,
                    contradiction_free_score=1.0,
                    citation_accuracy_score=1.0,
                    citation_coverage_score=1.0,
                    overall_score=1.0,
                )

        with patch("agent.workflows.nodes.ModelFactory.for_task", side_effect=model_for_task), \
             patch("agent.workflows.quality_assessor.QualityAssessor", FakeAssessor), \
             patch.object(settings, "deepsearch_use_gap_analysis", True):
            result = evaluator_node(
                {
                    "route": "deep",
                    "deepsearch_epoch": 1,
                    "deepsearch_max_epochs": 3,
                    "input": "研究主题",
                    "draft_report": "研究报告 [来源 1]。",
                    "sources": [{
                        "title": "来源",
                        "url": "https://example.test/source",
                    }],
                },
                config={"configurable": {"deepsearch_max_tokens": 1}},
            )

        self.assertEqual(requested_tasks, ["evaluation"])
        self.assertEqual(result["verdict"], "budget_exhausted")
        self.assertTrue(result["quality_summary"]["evaluation_skipped"])
        self.assertIsNone(result["knowledge_gap_analysis"])

    def test_evaluator_skips_gap_analysis_when_disabled_or_at_max_epoch(self):
        class FakeModel:
            def invoke(self, messages, config=None):
                return SimpleNamespace(content=json.dumps({
                    "verdict": "pass",
                    "evaluation": "complete",
                    "eval_dimensions": {"accuracy": 0.9},
                    "missing_topics": [],
                    "knowledge_gap_analysis": {
                        "overall_coverage": 0.2,
                        "gaps": [{
                            "aspect": "不应进入后续补查的主题",
                            "importance": "high",
                        }],
                        "suggested_queries": ["不应运行的补查查询"],
                    },
                }))

        class FakeAssessor:
            def __init__(self, llm=None, config=None):
                pass

            def assess(self, report, scraped_content, sources=None):
                return QualityReport(
                    claim_support_score=1.0,
                    source_diversity_score=1.0,
                    contradiction_free_score=1.0,
                    citation_accuracy_score=1.0,
                    citation_coverage_score=1.0,
                    overall_score=1.0,
                )

        base_state = {
            "route": "deep",
            "input": "研究主题",
            "draft_report": "完整报告 [来源 1]。",
            "sources": [{"title": "来源", "url": "https://example.test/source"}],
        }
        cases = (
            (False, 1, 3),
            (True, 3, 3),
        )

        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=FakeModel()), \
             patch("agent.workflows.quality_assessor.QualityAssessor", FakeAssessor):
            for enabled, epoch, max_epochs in cases:
                with self.subTest(enabled=enabled, epoch=epoch), \
                     patch.object(settings, "deepsearch_use_gap_analysis", enabled):
                    result = evaluator_node(
                        {
                            **base_state,
                            "deepsearch_epoch": epoch,
                            "deepsearch_max_epochs": max_epochs,
                        },
                        config={},
                    )

                self.assertIsNone(result["knowledge_gap_analysis"])
                self.assertNotIn(
                    "不应进入后续补查的主题",
                    result["missing_topics"],
                )


if __name__ == "__main__":
    unittest.main()
