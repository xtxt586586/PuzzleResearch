import unittest

from agent.workflows.planner_agent import PlannerAgent, PlannerTask


class PlannerAgentTests(unittest.TestCase):
    def test_returns_bounded_structured_prioritized_plan_without_tools(self):
        class Model:
            def invoke(self, messages, config=None):
                return type(
                    "Response",
                    (),
                    {
                        "content": (
                            '{"queries": ["Official docs", " official   DOCS ", '
                            '"Evidence report", "Third query"], "reasoning":"coverage"}'
                        )
                    },
                )()

        model = Model()
        agent = PlannerAgent(lambda *_args, **_kwargs: model)
        result = agent.run(
            PlannerTask(
                task_id="planner-test-1",
                goal="Research product evidence",
                mode="deep",
                instructions="Plan complementary searches",
                max_queries=2,
            ),
        )

        self.assertEqual(agent.allowed_tools, frozenset())
        self.assertEqual(result.status, "completed")
        self.assertEqual([item.query for item in result.queries], ["Official docs", "Evidence report"])
        self.assertEqual([item.priority for item in result.queries], [20, 19])
        self.assertEqual(result.reasoning, "coverage")

    def test_model_or_parse_failure_returns_structured_failure(self):
        result = PlannerAgent(lambda **_: None).run(
            PlannerTask(
                task_id="planner-test-failure",
                goal="Research topic",
                mode="web",
                instructions="Plan searches",
            ),
            type("Model", (), {"invoke": lambda self, *args, **kwargs: "not json"})(),
        )

        self.assertEqual(result.status, "failed")
        self.assertEqual(result.queries, [])
        self.assertIn("JSON", result.error)


if __name__ == "__main__":
    unittest.main()
