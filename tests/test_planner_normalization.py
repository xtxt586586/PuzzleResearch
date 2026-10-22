import unittest
from unittest.mock import patch

from agent.workflows.nodes import (
    _parse_plan_response,
    deep_followup_plan_node,
    deep_plan_node,
    web_plan_node,
)


class PlannerNormalizationTests(unittest.TestCase):
    def test_model_plan_collapses_whitespace_and_casefolds_duplicates(self):
        response = type(
            "Response",
            (),
            {
                "content": (
                    '{"queries": ["  Official   Docs  ", '
                    '"official docs", "' + ("A" * 600) + '"]}'
                )
            },
        )()

        queries, _ = _parse_plan_response(response)

        self.assertEqual(queries, ["Official Docs", "A" * 500])

    def test_web_planner_falls_back_when_model_factory_fails(self):
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            side_effect=RuntimeError("model configuration unavailable"),
        ):
            result = web_plan_node({"input": "fallback question"}, {})

        self.assertEqual(result["research_plan"], ["fallback question"])

    def test_deep_planner_falls_back_when_model_factory_fails(self):
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            side_effect=RuntimeError("model configuration unavailable"),
        ):
            result = deep_plan_node({"input": "fallback question"}, {})

        self.assertTrue(result["research_plan"])
        self.assertEqual(result["research_plan"][0], "fallback question")

    def test_exhausted_followup_budget_skips_model_factory(self):
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            side_effect=AssertionError("budget must prevent model initialization"),
        ):
            result = deep_followup_plan_node(
                {
                    "input": "research topic",
                    "deepsearch_epoch": 1,
                    "deepsearch_max_epochs": 3,
                    "missing_topics": ["missing evidence"],
                    "deepsearch_artifacts": {
                        "research_budget": {
                            "tokens_used": 8,
                            "elapsed_seconds": 0.5,
                            "searches_used": 1,
                        }
                    },
                },
                {"configurable": {"deepsearch_max_tokens": 1}},
            )

        self.assertTrue(result["research_plan"])
        self.assertEqual(result["budget_stop_reason"], "token_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
