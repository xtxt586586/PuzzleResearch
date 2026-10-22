import unittest
from unittest.mock import Mock, patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

import agent.core.graph as research_graph
from common.config import settings


class ReviewCheckpointGraphTests(unittest.TestCase):
    @staticmethod
    def config(thread_id):
        return {"configurable": {"thread_id": thread_id}}

    @staticmethod
    def build_graph(*, evaluator=None, revise=None):
        search = Mock(
            return_value={
                "search_results": [
                    {
                        "title": "测试来源",
                        "url": "https://example.org/source",
                        "snippet": "来源摘要",
                    }
                ],
                "sources": [],
                "fetched_pages": [],
                "passages": [],
            }
        )
        writer = Mock(
            return_value={
                "draft_report": "模型初稿",
                "final_report": "模型初稿",
            }
        )
        evaluator = evaluator or Mock(
            return_value={"verdict": "pass", "missing_topics": []}
        )
        revise = revise or Mock(
            return_value={
                "draft_report": "自动修订稿",
                "final_report": "自动修订稿",
                "revision_count": 1,
            }
        )
        with (
            patch.object(research_graph, "router_node", return_value={"route": "web"}),
            patch.object(
                research_graph,
                "web_plan_node",
                return_value={"research_plan": ["测试查询"], "status": "running"},
            ),
            patch.object(research_graph, "search_node", search),
            patch.object(research_graph, "writer_node", writer),
            patch.object(research_graph, "evaluator_node", evaluator),
            patch.object(research_graph, "revise_node", revise),
            patch.object(settings, "use_hierarchical_agents", False),
        ):
            graph = research_graph.create_research_graph(
                checkpointer=InMemorySaver(),
            )
        return graph, search, writer, evaluator

    def test_automatic_revision_does_not_repeat_draft_review(self):
        evaluator = Mock(
            side_effect=[
                {"verdict": "revise", "missing_topics": []},
                {"verdict": "pass", "missing_topics": []},
            ]
        )
        revise = Mock(
            return_value={
                "draft_report": "自动修订稿",
                "final_report": "自动修订稿",
                "revision_count": 1,
            }
        )
        graph, _, _, _ = self.build_graph(evaluator=evaluator, revise=revise)
        config = self.config("draft-review-only-before-first-evaluation")

        with (
            patch.object(settings, "hitl_checkpoints", "draft"),
            patch.object(settings, "human_review", False),
        ):
            paused = graph.invoke({"input": "测试主题"}, config=config)
            completed = graph.invoke(Command(resume=True), config=config)

        self.assertEqual(paused["__interrupt__"][0].value["type"], "draft_review")
        self.assertEqual(revise.call_count, 1)
        self.assertEqual(evaluator.call_count, 2)
        self.assertEqual(
            evaluator.call_args_list[1].args[0]["draft_report"],
            "自动修订稿",
        )
        self.assertEqual(completed["status"], "completed")

    def test_source_guidance_and_draft_edit_reach_downstream_nodes(self):
        graph, _, writer, evaluator = self.build_graph()
        config = self.config("source-draft-review-edit")
        with (
            patch.object(settings, "hitl_checkpoints", "sources,draft"),
            patch.object(settings, "human_review", False),
        ):
            paused_sources = graph.invoke({"input": "测试主题"}, config=config)
            self.assertEqual(
                paused_sources["__interrupt__"][0].value["type"],
                "sources_review",
            )
            self.assertIn(
                "https://example.org/source",
                paused_sources["__interrupt__"][0].value["snapshot"],
            )

            paused_draft = graph.invoke(
                Command(resume={"content": "突出说明证据局限"}),
                config=config,
            )
            self.assertEqual(
                paused_draft["__interrupt__"][0].value["type"],
                "draft_review",
            )
            self.assertEqual(
                writer.call_args.args[0]["human_guidance"],
                "突出说明证据局限",
            )

            completed = graph.invoke(
                Command(resume={"content": "人工修订后的初稿"}),
                config=config,
            )

        self.assertEqual(
            evaluator.call_args.args[0]["draft_report"],
            "人工修订后的初稿",
        )
        self.assertEqual(completed["final_report"], "人工修订后的初稿")
        self.assertEqual(completed["status"], "completed")

    def test_rejecting_sources_skips_writer(self):
        graph, _, writer, _ = self.build_graph()
        config = self.config("source-review-reject")
        with (
            patch.object(settings, "hitl_checkpoints", "sources"),
            patch.object(settings, "human_review", False),
        ):
            paused = graph.invoke({"input": "测试主题"}, config=config)
            completed = graph.invoke(Command(resume=False), config=config)

        self.assertEqual(paused["__interrupt__"][0].value["type"], "sources_review")
        self.assertEqual(writer.call_count, 0)
        self.assertEqual(completed["status"], "cancelled")

    def test_approving_source_and_draft_checkpoints_completes(self):
        graph, _, _, evaluator = self.build_graph()
        config = self.config("source-draft-review-approve")
        with (
            patch.object(settings, "hitl_checkpoints", "sources,draft"),
            patch.object(settings, "human_review", False),
        ):
            source_pause = graph.invoke({"input": "测试主题"}, config=config)
            draft_pause = graph.invoke(Command(resume=True), config=config)
            completed = graph.invoke(Command(resume=True), config=config)

        self.assertEqual(source_pause["__interrupt__"][0].value["type"], "sources_review")
        self.assertEqual(draft_pause["__interrupt__"][0].value["type"], "draft_review")
        evaluator.assert_called_once()
        self.assertEqual(completed["status"], "completed")

    def test_rejecting_draft_skips_evaluator(self):
        graph, _, _, evaluator = self.build_graph()
        config = self.config("draft-review-reject")
        with (
            patch.object(settings, "hitl_checkpoints", "draft"),
            patch.object(settings, "human_review", False),
        ):
            paused = graph.invoke({"input": "测试主题"}, config=config)
            completed = graph.invoke(Command(resume=False), config=config)

        self.assertEqual(paused["__interrupt__"][0].value["type"], "draft_review")
        self.assertEqual(evaluator.call_count, 0)
        self.assertEqual(completed["status"], "cancelled")

    def test_hierarchical_checkpoint_resume_preserves_agent_dispatch_results(self):
        observed = {"tree_calls": 0, "writer_state": None}

        def deep_plan(state, config=None):
            return {
                "research_plan": ["市场规模", "风险限制"],
                "deepsearch_epoch": 1,
                "deepsearch_max_epochs": 2,
                "revision_count": 0,
                "max_revisions": 1,
            }

        def tree_search(state, config=None):
            observed["tree_calls"] += 1
            tasks = list(state["research_tasks"])
            findings = [
                {
                    "query": task["topic"],
                    "title": f"{task['topic']}来源",
                    "url": f"https://example.org/{task['task_id']}",
                    "snippet": "checkpoint evidence",
                }
                for task in tasks
            ]
            return {
                "search_results": findings,
                "sources": [{"title": item["title"], "url": item["url"]} for item in findings],
                "research_agent_results": [
                    {
                        "task_id": task["task_id"],
                        "status": "completed",
                        "summary": f"完成 {task['topic']}",
                        "findings": [findings[index]],
                        "sources": [findings[index]["url"]],
                        "tool_calls": ["web_search"],
                    }
                    for index, task in enumerate(tasks)
                ],
                "research_tree": {"topic": "主题", "nodes": []},
                "deepsearch_artifacts": {"phase": "tree_searched"},
                "status": "running",
            }

        def writer(state, config=None):
            observed["writer_state"] = {
                "tasks": state.get("research_tasks"),
                "results": state.get("research_agent_results"),
                "search_results": state.get("search_results"),
            }
            return {
                "draft_report": "checkpoint draft [来源1] [来源2]",
                "final_report": "checkpoint draft [来源1] [来源2]",
                "status": "running",
            }

        with (
            patch.object(settings, "use_hierarchical_agents", True),
            patch.object(settings, "hitl_checkpoints", "sources"),
            patch.object(settings, "human_review", False),
            patch.object(research_graph, "router_node", return_value={"route": "deep", "status": "running"}),
            patch.object(research_graph, "deep_plan_node", side_effect=deep_plan),
            patch.object(research_graph, "deep_tree_search_node", side_effect=tree_search),
            patch.object(research_graph, "deep_writer_node", side_effect=writer),
            patch.object(research_graph, "evaluator_node", return_value={"verdict": "pass", "missing_topics": []}),
            patch.object(research_graph, "human_review_node", return_value={}),
        ):
            graph = research_graph.create_research_graph(checkpointer=InMemorySaver())
            config = self.config("hierarchical-agent-checkpoint-resume")
            paused = graph.invoke({"input": "评估主题"}, config=config)
            self.assertEqual(
                paused["__interrupt__"][0].value["type"],
                "sources_review",
            )
            expected_task_ids = {
                task["task_id"] for task in paused["research_tasks"]
            }
            approved = graph.invoke(Command(resume=True), config=config)

        self.assertEqual(observed["tree_calls"], 1)
        self.assertEqual(approved["status"], "completed")
        self.assertEqual(
            {task["task_id"] for task in observed["writer_state"]["tasks"]},
            expected_task_ids,
        )
        self.assertEqual(
            {item["task_id"] for item in observed["writer_state"]["results"]},
            expected_task_ids,
        )
        self.assertEqual(len(observed["writer_state"]["search_results"]), 2)


if __name__ == "__main__":
    unittest.main()
