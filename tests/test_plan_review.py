import unittest
from unittest.mock import Mock, patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command

import agent.core.graph as research_graph
from agent.core.graph import fan_out_deep_searches, fan_out_searches
from agent.core.state import AgentState
from agent.workflows.nodes import plan_review_node
from agent.workflows.research_tree import ResearchTree
from common.config import settings


class PlanReviewTests(unittest.TestCase):
    @staticmethod
    def build_graph():
        workflow = StateGraph(AgentState)
        workflow.add_node("plan_review", plan_review_node)
        workflow.add_edge(START, "plan_review")
        workflow.add_edge("plan_review", END)
        return workflow.compile(checkpointer=InMemorySaver())

    @staticmethod
    def config(thread_id):
        return {"configurable": {"thread_id": thread_id}}

    def test_review_is_skipped_when_checkpoint_is_disabled(self):
        graph = self.build_graph()
        with patch.object(settings, "hitl_checkpoints", ""):
            result = graph.invoke(
                {
                    "route": "web",
                    "input": "研究主题",
                    "research_plan": ["查询一", "查询一", "查询二"],
                },
                config=self.config("plan-review-disabled"),
            )

        self.assertNotIn("__interrupt__", result)
        self.assertEqual(result["research_plan"], ["查询一", "查询二"])
        self.assertFalse(result.get("plan_reviewed", False))

    def test_review_pauses_and_edited_plan_is_normalized(self):
        graph = self.build_graph()
        config = self.config("plan-review-edit")
        with patch.object(settings, "hitl_checkpoints", "plan"):
            paused = graph.invoke(
                {
                    "route": "web",
                    "input": "研究主题",
                    "research_plan": ["旧查询"],
                },
                config=config,
            )
            self.assertIn("__interrupt__", paused)
            self.assertEqual(
                paused["__interrupt__"][0].value["type"],
                "plan_review",
            )

            resumed = graph.invoke(
                Command(
                    resume={
                        "type": "edit_plan",
                        "queries": ["- 新查询一", "新查询一", "2. 新查询二"],
                    }
                ),
                config=config,
            )

        self.assertEqual(resumed["research_plan"], ["新查询一", "新查询二"])
        self.assertTrue(resumed["plan_reviewed"])

    def test_review_approval_keeps_plan_and_rejection_cancels(self):
        graph = self.build_graph()
        config = self.config("plan-review-approval")
        with patch.object(settings, "hitl_checkpoints", "plan"):
            graph.invoke(
                {"route": "web", "input": "主题", "research_plan": ["查询"]},
                config=config,
            )
            approved = graph.invoke(Command(resume=True), config=config)

            rejected_config = self.config("plan-review-reject")
            graph.invoke(
                {"route": "web", "input": "主题", "research_plan": ["查询"]},
                config=rejected_config,
            )
            rejected = graph.invoke(
                Command(resume={"type": "reject"}),
                config=rejected_config,
            )

        self.assertEqual(approved["research_plan"], ["查询"])
        self.assertTrue(approved["plan_reviewed"])
        self.assertEqual(rejected["status"], "cancelled")
        self.assertTrue(rejected["is_cancelled"])

    def test_invalid_review_response_fails_closed(self):
        graph = self.build_graph()
        config = self.config("plan-review-invalid")
        with patch.object(settings, "hitl_checkpoints", "plan"):
            graph.invoke(
                {"route": "web", "input": "主题", "research_plan": ["查询"]},
                config=config,
            )
            result = graph.invoke(
                Command(resume={"unexpected": True}),
                config=config,
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("无效", result["final_report"])

    def test_reviewed_plan_drives_web_and_tree_search_dispatch(self):
        web_sends = fan_out_searches(
            {"research_plan": ["网页查询一", "网页查询二"]}
        )
        self.assertEqual([send.arg["query"] for send in web_sends], ["网页查询一", "网页查询二"])

        tree = ResearchTree.from_queries(
            "研究主题",
            ["原始查询"],
            epoch=1,
            max_branches=4,
        )
        graph = self.build_graph()
        config = self.config("plan-review-tree-edit")
        state = {
            "route": "deep",
            "input": "研究主题",
            "research_plan": ["原始查询"],
            "research_tree": tree.to_dict(),
            "deepsearch_mode": "tree",
            "deepsearch_epoch": 1,
            "tree_exploration_enabled": True,
        }
        with patch.object(settings, "hitl_checkpoints", "plan"):
            graph.invoke(state, config=config)
            resumed = graph.invoke(
                Command(resume={"type": "edit_plan", "queries": ["审核后的查询"]}),
                config=config,
            )

        sends = fan_out_deep_searches(resumed)
        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "deep_tree_search")
        self.assertEqual(
            [node.query for node in ResearchTree.from_dict(resumed["research_tree"]).get_pending_nodes()],
            ["审核后的查询"],
        )
        self.assertTrue(resumed["plan_reviewed"])

    def test_full_graph_resumes_to_search_with_edited_queries(self):
        search = Mock(
            return_value={
                "search_results": [],
                "sources": [],
                "fetched_pages": [],
                "passages": [],
            }
        )
        config = self.config("plan-review-full-graph")
        with (
            patch.object(research_graph, "router_node", return_value={"route": "web"}),
            patch.object(
                research_graph,
                "web_plan_node",
                return_value={"research_plan": ["初始查询"], "status": "running"},
            ),
            patch.object(research_graph, "search_node", search),
            patch.object(
                research_graph,
                "writer_node",
                return_value={"draft_report": "报告", "final_report": "报告"},
            ),
            patch.object(
                research_graph,
                "evaluator_node",
                return_value={"verdict": "pass", "missing_topics": []},
            ),
            patch.object(settings, "hitl_checkpoints", "plan"),
            patch.object(settings, "human_review", False),
        ):
            graph = research_graph.create_research_graph(
                checkpointer=InMemorySaver(),
            )
            paused = graph.invoke(
                {"input": "研究主题"},
                config=config,
            )
            self.assertIn("__interrupt__", paused)

            graph.invoke(
                Command(
                    resume={"type": "edit_plan", "queries": ["实际执行查询"]}
                ),
                config=config,
            )

            reject_config = self.config("plan-review-full-graph-reject")
            graph.invoke(
                {"input": "另一研究主题"},
                config=reject_config,
            )
            rejected = graph.invoke(
                Command(resume=False),
                config=reject_config,
            )

        self.assertEqual(search.call_count, 1)
        self.assertEqual(search.call_args.args[0]["query"], "实际执行查询")
        self.assertEqual(rejected["status"], "cancelled")


if __name__ == "__main__":
    unittest.main()
