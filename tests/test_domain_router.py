import unittest
from unittest.mock import patch

from agent.workflows.domain_router import (
    DomainClassifier,
    ResearchDomain,
    build_provider_profile,
)
from agent.workflows.nodes import router_node, search_node
from common.config import settings


class DomainRouterTests(unittest.TestCase):
    def test_classifier_identifies_medical_query_without_llm(self):
        result = DomainClassifier().classify("比较临床治疗方案和疾病风险")
        self.assertEqual(result.domain, ResearchDomain.MEDICAL)
        self.assertGreaterEqual(result.confidence, 0.55)
        self.assertIn("pubmed.ncbi.nlm.nih.gov", result.suggested_sources)

    def test_classifier_prefers_structured_llm_when_available(self):
        class StructuredModel:
            def invoke(self, prompt, config=None):
                return {
                    "domain": "technical",
                    "confidence": 0.91,
                    "reasoning": "API implementation question",
                    "search_hints": ["API documentation"],
                }

        class FakeLlm:
            def with_structured_output(self, schema):
                self.schema = schema
                return StructuredModel()

        result = DomainClassifier(llm=FakeLlm()).classify("随便一个问题")
        self.assertEqual(result.domain, ResearchDomain.TECHNICAL)
        self.assertEqual(result.confidence, 0.91)

    def test_classifier_falls_back_when_llm_fails(self):
        class FailingLlm:
            def with_structured_output(self, schema):
                raise RuntimeError("structured output unavailable")

        result = DomainClassifier(llm=FailingLlm()).classify("临床疾病治疗")
        self.assertEqual(result.domain, ResearchDomain.MEDICAL)

    def test_provider_profile_prioritizes_academic_sources(self):
        profile = build_provider_profile(
            ["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com"],
            ResearchDomain.SCIENTIFIC,
        )
        self.assertEqual(profile[:3], ["arxiv", "pubmed", "semantic_scholar"])

    def test_router_keeps_domain_routing_opt_in(self):
        with patch.object(settings, "domain_routing_enabled", False), patch(
            "agent.workflows.nodes.smart_route",
            return_value={
                "route": "deep",
                "routing_reasoning": "test",
                "routing_confidence": 1.0,
                "suggested_queries": [],
                "clarification_question": "",
                "needs_clarification": False,
            },
        ):
            result = router_node({"input": "医学研究", "images": []}, {})
        self.assertNotIn("domain", result)

    def test_enabled_router_writes_domain_metadata(self):
        with patch.object(settings, "domain_routing_enabled", True), patch(
            "agent.workflows.nodes.smart_route",
            return_value={
                "route": "deep",
                "routing_reasoning": "test",
                "routing_confidence": 1.0,
                "suggested_queries": [],
                "clarification_question": "",
                "needs_clarification": False,
            },
        ):
            result = router_node({"input": "医学临床研究", "images": []}, {})
        self.assertEqual(result["domain"], "medical")
        self.assertEqual(result["domain_config"]["domain"], "medical")

    def test_deep_search_passes_domain_provider_profile(self):
        captured = {}

        class FakeManager:
            def __init__(self, **kwargs):
                pass

            def search(self, query, **kwargs):
                captured.update(kwargs)
                return [{
                    "title": "result",
                    "url": "https://example.com",
                    "snippet": "snippet",
                    "content": "content",
                    "provider": "arxiv",
                }]

        with patch("agent.workflows.nodes.SearchManager", FakeManager), patch.object(
            settings, "deepsearch_results_per_query", 1
        ), patch.object(settings, "search_strategy", "fallback"):
            search_node({
                "query": "research",
                "search_mode": "deep",
                "domain": "scientific",
                "domain_config": {"suggested_sources": ["arxiv.org"]},
            })

        self.assertEqual(captured["provider_profile"][:2], ["arxiv", "pubmed"])


if __name__ == "__main__":
    unittest.main()
