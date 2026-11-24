import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

from agent.core.state import AgentState
from agent.workflows.nodes import human_review_node
from common.config import settings


class HumanReviewTests(unittest.TestCase):
    @staticmethod
    def build_graph():
        workflow = StateGraph(AgentState)

        workflow.add_node(
            "human_review",
            human_review_node,
        )

        workflow.add_edge(
            START,
            "human_review",
        )

        workflow.add_edge(
            "human_review",
            END,
        )

        return workflow.compile(
            checkpointer=InMemorySaver(),
        )

    def test_review_is_skipped_when_disabled(self):
        graph = self.build_graph()

        result = graph.invoke(
            {
                "final_report": "测试报告",
                "status": "running",
            },
            config={
                "configurable": {
                    "thread_id": "human-review-disabled-001",
                    "human_review": False,
                },
            },
        )

        self.assertEqual(
            result["final_report"],
            "测试报告",
        )

        self.assertNotIn(
            "__interrupt__",
            result,
        )

    def test_final_checkpoint_enables_review_without_legacy_flag(self):
        graph = self.build_graph()
        config = {
            "configurable": {
                "thread_id": "human-review-final-checkpoint-001",
                "human_review": False,
            },
        }

        with patch.object(settings, "hitl_checkpoints", "final"):
            paused = graph.invoke(
                {"final_report": "待审核报告", "status": "running"},
                config=config,
            )

        self.assertEqual(paused["__interrupt__"][0].value["checkpoint"], "final")

    def test_review_pauses_and_resumes_with_approval(self):
        graph = self.build_graph()

        config = {
            "configurable": {
                "thread_id": "human-review-enabled-001",
                "human_review": True,
            },
        }

        paused = graph.invoke(
            {
                "final_report": "待审核报告",
                "status": "running",
            },
            config=config,
        )

        self.assertIn(
            "__interrupt__",
            paused,
        )

        resumed = graph.invoke(
            Command(
                resume={
                    "approved": True,
                },
            ),
            config=config,
        )

        self.assertEqual(
            resumed["final_report"],
            "待审核报告",
        )

    def test_review_boolean_approval_completes_with_original_report(self):
        graph = self.build_graph()
        config = {
            "configurable": {
                "thread_id": "human-review-bool-approve-001",
                "human_review": True,
            },
        }

        paused = graph.invoke(
            {
                "final_report": "待审核报告",
                "status": "running",
            },
            config=config,
        )
        self.assertIn("__interrupt__", paused)

        resumed = graph.invoke(
            Command(resume=True),
            config=config,
        )

        self.assertEqual(resumed["final_report"], "待审核报告")
        self.assertEqual(resumed["status"], "running")

    def test_review_text_approval_keeps_original_report(self):
        graph = self.build_graph()
        config = {
            "configurable": {
                "thread_id": "human-review-text-approve-001",
                "human_review": True,
            },
        }

        graph.invoke(
            {
                "final_report": "待审核报告",
                "status": "running",
            },
            config=config,
        )
        resumed = graph.invoke(
            Command(resume="approve"),
            config=config,
        )

        self.assertEqual(resumed["final_report"], "待审核报告")

    def test_review_accepts_edited_content(self):
        graph = self.build_graph()

        config = {
            "configurable": {
                "thread_id": "human-review-edit-001",
                "human_review": True,
            },
        }

        paused = graph.invoke(
            {
                "final_report": "旧报告",
                "status": "running",
            },
            config=config,
        )

        self.assertIn(
            "__interrupt__",
            paused,
        )

        resumed = graph.invoke(
            Command(
                resume={
                    "approved": True,
                    "content": "修改后的报告",
                },
            ),
            config=config,
        )

        self.assertEqual(
            resumed["final_report"],
            "修改后的报告",
        )

    def test_review_accepts_native_text_edit(self):
        graph = self.build_graph()
        config = {
            "configurable": {
                "thread_id": "human-review-text-edit-001",
                "human_review": True,
            },
        }

        graph.invoke(
            {
                "final_report": "旧报告",
                "status": "running",
            },
            config=config,
        )
        resumed = graph.invoke(
            Command(resume="修订后的报告"),
            config=config,
        )

        self.assertEqual(resumed["final_report"], "修订后的报告")

    def test_review_rejection_cancels_execution(self):
        graph = self.build_graph()

        config = {
            "configurable": {
                "thread_id": "human-review-reject-001",
                "human_review": True,
            },
        }

        paused = graph.invoke(
            {
                "final_report": "待拒绝报告",
                "status": "running",
            },
            config=config,
        )

        self.assertIn(
            "__interrupt__",
            paused,
        )

        resumed = graph.invoke(
            Command(
                resume={
                    "approved": False,
                },
            ),
            config=config,
        )

        self.assertEqual(
            resumed["status"],
            "cancelled",
        )

        self.assertFalse(
            resumed["is_complete"],
        )

    def test_review_native_reject_decision_cancels_execution(self):
        graph = self.build_graph()
        config = {
            "configurable": {
                "thread_id": "human-review-native-reject-001",
                "human_review": True,
            },
        }

        paused = graph.invoke(
            {
                "final_report": "待拒绝报告",
                "status": "running",
            },
            config=config,
        )

        self.assertIn("__interrupt__", paused)

        resumed = graph.invoke(
            Command(
                resume={
                    "type": "reject",
                    "message": "审核人拒绝了报告。",
                },
            ),
            config=config,
        )

        self.assertEqual(resumed["status"], "cancelled")
        self.assertFalse(resumed["is_complete"])
        self.assertEqual(
            resumed["final_report"],
            "审核人拒绝了报告。",
        )


if __name__ == "__main__":
    unittest.main()
