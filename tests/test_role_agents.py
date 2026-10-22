import unittest

from agent.workflows.role_agents import (
    VerifierAgent,
    VerifierTask,
    WriterAgent,
    WriterTask,
)


class RoleAgentTests(unittest.TestCase):
    def test_writer_receives_only_its_task_and_has_no_tools(self):
        received = []

        def generate(task):
            received.append(task)
            return "Report with evidence [来源1]."

        agent = WriterAgent(generate)
        task = WriterTask(
            task_id="writer-thread-1-1",
            question="Summarize the evidence",
            evidence=[{"url": "https://example.test", "snippet": "evidence"}],
        )
        result = agent.run(task)

        self.assertEqual(agent.allowed_tools, frozenset())
        self.assertIs(received[0], task)
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.report, "Report with evidence [来源1].")
        self.assertEqual(result.tools_used, [])
        self.assertEqual(result.model_dump(mode="json")["task_id"], task.task_id)

    def test_writer_failure_is_structured_and_does_not_claim_success(self):
        result = WriterAgent(
            lambda _: (_ for _ in ()).throw(RuntimeError("model unavailable"))
        ).run(
            WriterTask(task_id="writer-fail", question="Question")
        )
        self.assertEqual(result.status, "failed")
        self.assertIn("model unavailable", result.error)
        self.assertEqual(result.report, "")

    def test_verifier_is_read_only_and_returns_claim_contract(self):
        agent = VerifierAgent()
        task = VerifierTask(
            task_id="verifier-thread-1-1",
            report="The company's revenue increased in 2024 according to its annual report.",
            passages=[{
                "url": "https://example.test/report",
                "text": "The company's revenue increased in 2024 according to its annual report.",
                "quote": "Revenue increased in 2024.",
            }],
        )
        result = agent.run(task)

        self.assertEqual(agent.allowed_tools, frozenset())
        self.assertEqual(result.status, "completed")
        self.assertEqual(result.verified_count, 1)
        self.assertEqual(result.claims[0]["evidence_urls"], ["https://example.test/report"])
        self.assertEqual(result.tools_used, [])


if __name__ == "__main__":
    unittest.main()
