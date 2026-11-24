import unittest

from agent.workflows.research_agent import ResearchAgent, ResearchTask


class ResearchAgentTests(unittest.TestCase):
    def test_task_and_result_use_a_structured_contract(self):
        agent = ResearchAgent({
            "web_search": lambda query, max_results: [{
                "title": "Official source",
                "url": "https://example.test/docs",
                "snippet": "Evidence excerpt",
                "provider": "mock",
            }],
        })
        task = ResearchTask(
            task_id="branch-001",
            topic="checkpoint persistence",
            objective="Find official evidence",
            context={"depth": 1},
            allowed_tools={"web_search"},
        )

        result = agent.run(
            task,
            lambda context: (
                context.call_tool(
                    "web_search",
                    query=task.topic,
                    max_results=1,
                ),
                {"summary": "Checkpoint evidence found"},
            )[1],
        )

        self.assertEqual(result.status, "completed")
        self.assertEqual(result.task_id, "branch-001")
        self.assertEqual(result.summary, "Checkpoint evidence found")
        self.assertEqual(result.tool_calls, ["web_search"])
        self.assertEqual(result.sources, ["https://example.test/docs"])
        self.assertEqual(result.findings[0].query, task.topic)
        self.assertEqual(result.findings[0].provider, "mock")
        self.assertEqual(result.model_dump(mode="json")["status"], "completed")

    def test_task_cannot_call_a_tool_outside_its_allowlist(self):
        agent = ResearchAgent({
            "web_search": lambda **_: [{"url": "https://example.test"}],
            "computer_use": lambda **_: "should not run",
        })
        task = ResearchTask(
            task_id="read-only-001",
            topic="Read-only task",
            allowed_tools={"web_search"},
        )

        result = agent.run(
            task,
            lambda context: context.call_tool("computer_use"),
        )

        self.assertEqual(result.status, "failed")
        self.assertIn("not allowed", result.error)
        self.assertEqual(result.tool_calls, [])

    def test_failure_after_evidence_returns_partial_result(self):
        agent = ResearchAgent({
            "web_search": lambda **_: [{
                "title": "Partial source",
                "url": "https://example.test/partial",
                "snippet": "Useful evidence",
            }],
        })
        task = ResearchTask(
            task_id="partial-001",
            topic="Partial task",
            allowed_tools={"web_search"},
        )

        def execute(context):
            context.call_tool("web_search", query=task.topic, max_results=1)
            raise RuntimeError("summary model unavailable")

        result = agent.run(task, execute)

        self.assertEqual(result.status, "partial")
        self.assertEqual(len(result.findings), 1)
        self.assertIn("summary model unavailable", result.error)

    def test_each_run_keeps_its_own_task_context_and_findings(self):
        agent = ResearchAgent({
            "web_search": lambda query, **_: [{
                "query": query,
                "url": f"https://example.test/{query}",
            }],
        })
        results = []
        for task_id, topic in (("a", "alpha"), ("b", "beta")):
            task = ResearchTask(
                task_id=task_id,
                topic=topic,
                context={"topic": topic},
                allowed_tools={"web_search"},
            )
            def execute(context):
                context.call_tool(
                    "web_search",
                    query=context.task.context["topic"],
                    max_results=1,
                )
                return {"summary": context.task.topic}

            results.append(agent.run(task, execute))

        self.assertEqual(results[0].task_id, "a")
        self.assertEqual(results[0].sources, ["https://example.test/alpha"])
        self.assertEqual(results[1].task_id, "b")
        self.assertEqual(results[1].sources, ["https://example.test/beta"])


if __name__ == "__main__":
    unittest.main()
