import json
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.workflows.claim_verifier import ClaimStatus, ClaimVerifier
from agent.workflows.nodes import evaluator_node
from common.config import settings


class ClaimVerifierTests(unittest.TestCase):
    def test_unsupported_claim_is_reported(self):
        checks = ClaimVerifier().verify_report(
            "2024年该公司营收增长了20%，并在海外市场创下历史新高。",
            [{
                "results": [{
                    "url": "https://example.com/product",
                    "summary": "The company launched a new product line for developers.",
                }]
            }],
        )
        self.assertEqual(len(checks), 1)
        self.assertEqual(checks[0].status, ClaimStatus.UNSUPPORTED)

    def test_conflicting_evidence_is_contradicted(self):
        checks = ClaimVerifier().verify_report(
            "The company's revenue increased in 2024 according to the annual report.",
            [{
                "results": [{
                    "url": "https://example.com/earnings?utm_source=test",
                    "summary": "The company's revenue did not increase in 2024 and decreased by 5%.",
                }]
            }],
        )
        self.assertEqual(checks[0].status, ClaimStatus.CONTRADICTED)
        self.assertEqual(checks[0].evidence_urls, ["https://example.com/earnings"])

    def test_passage_evidence_is_attached_to_verified_claim(self):
        checks = ClaimVerifier().verify_report(
            "The company's revenue increased in 2024 according to the annual report.",
            [],
            passages=[{
                "url": "https://example.com/earnings?utm_source=test",
                "text": "In 2024, the company's revenue increased by 5% year over year.",
                "snippet_hash": "passage_123",
                "quote": "In 2024, the company's revenue increased by 5% year over year.",
                "heading_path": ["Results"],
            }],
        )
        self.assertEqual(checks[0].status, ClaimStatus.VERIFIED)
        self.assertEqual(checks[0].evidence_passages[0]["snippet_hash"], "passage_123")

    def test_deep_evaluator_gate_persists_claims_and_forces_revision(self):
        class FakeModel:
            def invoke(self, messages, config=None):
                return SimpleNamespace(
                    content=json.dumps({
                        "verdict": "pass",
                        "evaluation": "good",
                        "eval_dimensions": {
                            "accuracy": 0.9,
                            "completeness": 0.9,
                            "citation": 0.9,
                        },
                        "missing_topics": [],
                    }),
                    usage_metadata={"total_tokens": 31},
                )

        state = {
            "route": "deep",
            "input": "revenue trend",
            "draft_report": "The company's revenue increased in 2024 according to the annual report.",
            "final_report": "The company's revenue increased in 2024 according to the annual report.",
            "search_results": [{
                "url": "https://example.com/earnings",
                "title": "Earnings",
                "snippet": "",
                "content": "The company's revenue did not increase in 2024 and decreased by 5%.",
            }],
            "sources": [{"url": "https://example.com/earnings", "title": "Earnings"}],
            "deepsearch_artifacts": {"phase": "synthesized"},
        }
        with patch.object(settings, "claim_verifier_gate_max_contradicted", 0), patch.object(
            settings, "claim_verifier_gate_max_unsupported", 0
        ), patch("agent.workflows.nodes.ModelFactory.for_task", return_value=FakeModel()):
            result = evaluator_node(
                state,
                config={"configurable": {"deepsearch_max_tokens": 100000}},
            )

        self.assertEqual(result["verdict"], "revise")
        self.assertEqual(
            result["deepsearch_artifacts"]["claims"][0]["status"],
            "contradicted",
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["claim_verifier"]["contradicted_count"],
            1,
        )
        self.assertEqual(result["verifier_agent_result"]["status"], "completed")
        self.assertEqual(
            result["deepsearch_artifacts"]["verifier_agent"]["task_id"],
            result["verifier_agent_result"]["task_id"],
        )
        self.assertGreaterEqual(
            result["deepsearch_artifacts"]["research_budget"]["tokens_used"],
            31,
        )


if __name__ == "__main__":
    unittest.main()
