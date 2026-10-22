import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

from agent.core.graph import create_research_graph
from agent.workflows.coordinator import (
    coordinator_destination,
    coordinator_node,
)
from common.config import settings


class CoordinatorTests(unittest.TestCase):
    def _reviewed_state(self, **overrides):
        state = {
            "input": "比较两种方案",
            "route": "deep",
            "research_plan": ["方案一", "方案二"],
            "search_results": [{"url": "https://example.com"}],
            "draft_report": "已有报告",
            "final_report": "已有报告",
            "verdict": "revise",
            "evaluation": "还缺少成本对比",
            "missing_topics": ["成本对比"],
            "deepsearch_epoch": 1,
            "deepsearch_max_epochs": 3,
            "revision_count": 0,
            "max_revisions": 2,
        }
        state.update(overrides)
        return state

    def test_initial_state_plans_without_calling_model(self):
        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            side_effect=AssertionError("initial planning is deterministic"),
        ):
            result = coordinator_node(
                {
                    "input": "研究一个主题",
                    "research_plan": [],
                },
                {},
            )

        self.assertEqual(result["coordinator_action"], "plan")
        self.assertEqual(
            coordinator_destination(
                {"coordinator_action": "plan", "research_plan": []}
            ),
            "deep_plan",
        )

    def test_passed_report_completes_without_calling_model(self):
        state = self._reviewed_state(
            verdict="pass",
            missing_topics=[],
        )

        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            side_effect=AssertionError("a passing report needs no decision call"),
        ):
            result = coordinator_node(state, {})

        self.assertEqual(result["coordinator_action"], "complete")
        self.assertEqual(
            coordinator_destination(result),
            "human_review",
        )

    def test_model_can_choose_followup_research_and_prioritize_topics(self):
        model = SimpleNamespace(
            invoke=lambda messages, config=None: SimpleNamespace(
                content=(
                    'Decision: {"action":"research",'
                    '"reasoning":"补足成本证据",'
                    '"priority_topics":["成本对比","长期维护"]}'
                )
            )
        )

        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            return_value=model,
        ):
            result = coordinator_node(self._reviewed_state(), {})

        self.assertEqual(result["coordinator_action"], "research")
        self.assertEqual(
            result["missing_topics"],
            ["成本对比", "长期维护"],
        )
        self.assertEqual(
            coordinator_destination(result),
            "plan_review",
        )
        self.assertEqual(
            [task["topic"] for task in result["research_tasks"]],
            ["成本对比", "长期维护"],
        )
        self.assertEqual(result["research_tasks"][0]["priority"], 5)
        self.assertEqual(result["research_tasks"][0]["allowed_tools"], ["web_search"])
        self.assertEqual(
            result["research_tasks"][0]["context"]["focus_topic"],
            "成本对比",
        )

    def test_duplicate_topic_variants_produce_one_research_task(self):
        model = SimpleNamespace(
            invoke=lambda messages, config=None: SimpleNamespace(
                content=(
                    '{"action":"research","reasoning":"补足成本证据",'
                    '"priority_topics":["Cost Analysis", " cost   analysis "]}'
                )
            )
        )

        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            return_value=model,
        ):
            result = coordinator_node(self._reviewed_state(), {})

        self.assertEqual(len(result["research_tasks"]), 1)
        self.assertEqual(result["research_tasks"][0]["topic"], "Cost Analysis")
        self.assertEqual(len({task["task_id"] for task in result["research_tasks"]}), 1)

    def test_coordinator_model_usage_is_added_to_shared_budget(self):
        model = SimpleNamespace(
            invoke=lambda messages, config=None: SimpleNamespace(
                content=(
                    '{"action":"research","reasoning":"继续补充",'
                    '"priority_topics":["成本对比"]}'
                ),
                usage_metadata={"total_tokens": 19},
            )
        )

        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            return_value=model,
        ):
            result = coordinator_node(self._reviewed_state(), {})

        self.assertGreaterEqual(
            result["deepsearch_artifacts"]["research_budget"]["tokens_used"],
            19,
        )
        self.assertEqual(
            result["deepsearch_tokens_used"],
            result["deepsearch_artifacts"]["research_budget"]["tokens_used"],
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["quality_summary"]["tokens_used"],
            result["deepsearch_tokens_used"],
        )

    def test_followup_dispatch_increments_epoch_and_task_identity(self):
        state = self._reviewed_state()
        model = SimpleNamespace(
            invoke=lambda messages, config=None: SimpleNamespace(
                content=(
                    '{"action":"research","reasoning":"补充",'
                    '"priority_topics":["成本对比"]}'
                )
            )
        )
        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            return_value=model,
        ):
            first = coordinator_node(state, {})
            second = coordinator_node(
                {**state, "deepsearch_epoch": 2},
                {},
            )

        self.assertEqual(first["deepsearch_epoch"], 2)
        self.assertEqual(second["deepsearch_epoch"], 3)
        self.assertNotEqual(
            first["research_tasks"][0]["task_id"],
            second["research_tasks"][0]["task_id"],
        )

    def test_failed_agent_branch_is_retried_once_with_incremented_retry_count(self):
        state = self._reviewed_state(
            research_tasks=[{
                "task_id": "research-failed",
                "topic": "成本对比",
                "context": {"retry_count": 0},
            }],
            research_agent_results=[{
                "task_id": "research-failed",
                "status": "failed",
                "error": "provider timeout",
            }],
        )
        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            side_effect=AssertionError("failed worker retry is a deterministic policy"),
        ):
            result = coordinator_node(state, {})

        self.assertEqual(result["coordinator_action"], "research")
        self.assertEqual(result["research_tasks"][0]["topic"], "成本对比")
        self.assertEqual(result["research_tasks"][0]["context"]["retry_count"], 1)
        self.assertEqual(result["deepsearch_epoch"], 2)

    def test_failed_first_attempt_without_evidence_retries_in_a_new_epoch(self):
        state = self._reviewed_state(
            draft_report="",
            final_report="",
            search_results=[],
            deepsearch_epoch=1,
            research_tasks=[{
                "task_id": "research-cost-epoch-1",
                "topic": "成本对比",
                "context": {"retry_count": 0, "epoch": 1},
            }],
            research_agent_results=[{
                "task_id": "research-cost-epoch-1",
                "status": "failed",
                "error": "search provider unavailable",
            }],
        )
        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            side_effect=AssertionError("worker retry must not call the coordinator model"),
        ):
            result = coordinator_node(state, {})

        self.assertEqual(result["coordinator_action"], "research")
        self.assertEqual(result["deepsearch_epoch"], 2)
        self.assertEqual(result["research_tasks"][0]["topic"], "成本对比")
        self.assertEqual(result["research_tasks"][0]["context"]["retry_count"], 1)
        self.assertNotEqual(
            result["research_tasks"][0]["task_id"],
            "research-cost-epoch-1",
        )

    def test_second_failure_without_evidence_stops_automatic_redispatch(self):
        state = self._reviewed_state(
            draft_report="",
            final_report="",
            search_results=[],
            deepsearch_epoch=2,
            research_tasks=[{
                "task_id": "research-cost-epoch-2",
                "topic": "成本对比",
                "context": {"retry_count": 1, "epoch": 2},
            }],
            research_agent_results=[{
                "task_id": "research-cost-epoch-1",
                "status": "failed",
                "error": "search provider unavailable",
            }, {
                "task_id": "research-cost-epoch-2",
                "status": "failed",
                "error": "search provider unavailable",
            }],
        )

        result = coordinator_node(state, {})

        self.assertEqual(result["coordinator_action"], "complete")
        self.assertIn("重试后仍未产出证据", result["coordinator_reasoning"])
        self.assertNotIn("research_tasks", result)

    def test_failed_agent_branch_is_not_retried_more_than_once(self):
        state = self._reviewed_state(
            research_tasks=[{
                "task_id": "research-failed-again",
                "topic": "成本对比",
                "context": {"retry_count": 1},
            }],
            research_agent_results=[{
                "task_id": "research-failed-again",
                "status": "empty",
            }],
        )
        model = SimpleNamespace(
            invoke=lambda messages, config=None: SimpleNamespace(
                content='{"action":"synthesize","reasoning":"使用现有证据","priority_topics":[]}'
            )
        )
        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            return_value=model,
        ):
            result = coordinator_node(state, {})

        self.assertEqual(result["coordinator_action"], "synthesize")
        self.assertNotIn("research_tasks", result)

    def test_task_count_respects_tree_branch_limit(self):
        state = self._reviewed_state(
            missing_topics=[f"缺口 {index}" for index in range(8)],
        )
        model = SimpleNamespace(
            invoke=lambda messages, config=None: SimpleNamespace(
                content=(
                    '{"action":"research","reasoning":"补充",'
                    '"priority_topics":' + json.dumps(state["missing_topics"], ensure_ascii=False) + "}"
                )
            )
        )
        with (
            patch.object(settings, "deepsearch_tree_max_branches", 2),
            patch("agent.workflows.coordinator.ModelFactory.for_task", return_value=model),
        ):
            result = coordinator_node(state, {})
        self.assertEqual(len(result["research_tasks"]), 2)

    def test_exhausted_budget_routes_existing_evidence_to_writer(self):
        state = self._reviewed_state(
            draft_report="",
            final_report="",
            deepsearch_artifacts={
                "research_budget": {
                    "tokens_used": 50,
                    "elapsed_seconds": 2.0,
                    "searches_used": 1,
                }
            },
        )

        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            side_effect=AssertionError("the exhausted budget skips coordination"),
        ):
            result = coordinator_node(
                state,
                {"configurable": {"deepsearch_max_tokens": 20}},
            )

        self.assertEqual(result["coordinator_action"], "synthesize")
        self.assertEqual(
            coordinator_destination({**state, **result}),
            "deep_writer",
        )

    def test_exhausted_research_budget_does_not_call_revision_model(self):
        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            side_effect=AssertionError("no model call needed after budget exhaustion"),
        ):
            result = coordinator_node(
                self._reviewed_state(
                    deepsearch_epoch=3,
                    deepsearch_max_epochs=3,
                    deepsearch_artifacts={
                        "research_budget": {
                            "stop_reason": "time_budget_exceeded",
                        },
                    },
                ),
                {},
            )

        self.assertEqual(result["coordinator_action"], "complete")
        self.assertEqual(
            coordinator_destination(
                {
                    **self._reviewed_state(
                        deepsearch_epoch=3,
                        deepsearch_max_epochs=3,
                    ),
                    **result,
                }
            ),
            "human_review",
        )

    def test_model_failure_uses_budget_safe_research_fallback(self):
        with patch(
            "agent.workflows.coordinator.ModelFactory.for_task",
            side_effect=RuntimeError("model unavailable"),
        ):
            result = coordinator_node(self._reviewed_state(), {})

        self.assertEqual(result["coordinator_action"], "research")
        self.assertIn(
            "协调器不可用",
            result["coordinator_reasoning"],
        )

    def test_graph_registers_coordinator_only_when_enabled(self):
        with patch.object(settings, "use_hierarchical_agents", False):
            default_graph = create_research_graph()
        self.assertNotIn("coordinator", default_graph.get_graph().nodes)

        with patch.object(settings, "use_hierarchical_agents", True):
            hierarchical_graph = create_research_graph()
        self.assertIn("coordinator", hierarchical_graph.get_graph().nodes)

    def test_hierarchical_graph_coordinates_a_deep_research_run(self):
        def fake_router(state, config=None):
            return {"route": "deep", "status": "running"}

        def fake_plan(state, config=None):
            return {
                "research_plan": ["研究方案"],
                "deepsearch_epoch": 1,
                "deepsearch_max_epochs": 2,
                "revision_count": 0,
                "max_revisions": 1,
            }

        def fake_search(state, config=None):
            return {
                "search_results": [
                    {
                        "title": "来源",
                        "url": "https://example.com/source",
                        "snippet": "研究证据",
                    }
                ],
                "sources": [
                    {
                        "title": "来源",
                        "url": "https://example.com/source",
                    }
                ],
            }

        def fake_writer(state, config=None):
            return {
                "draft_report": "带有来源的研究报告",
                "final_report": "带有来源的研究报告",
            }

        def fake_evaluator(state, config=None):
            return {
                "verdict": "pass",
                "missing_topics": [],
            }

        with (
            patch.object(settings, "use_hierarchical_agents", True),
            patch("agent.core.graph.router_node", side_effect=fake_router),
            patch("agent.core.graph.deep_plan_node", side_effect=fake_plan),
            patch("agent.core.graph.search_node", side_effect=fake_search),
            patch("agent.core.graph.deep_writer_node", side_effect=fake_writer),
            patch("agent.core.graph.evaluator_node", side_effect=fake_evaluator),
            patch("agent.core.graph.human_review_node", return_value={}),
        ):
            result = create_research_graph().invoke(
                {
                    "input": "研究一个主题",
                    "messages": [],
                }
            )

        self.assertEqual(result["route"], "deep")
        self.assertEqual(result["coordinator_action"], "complete")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(
            result["final_report"],
            "带有来源的研究报告",
        )
        self.assertEqual(
            [task["topic"] for task in result["research_tasks"]],
            ["研究方案"],
        )

    def test_evaluation_gap_returns_to_coordinator_for_followup_dispatch(self):
        assigned_topics = []
        evaluation_count = 0

        def fake_router(state, config=None):
            return {"route": "deep", "status": "running"}

        def fake_plan(state, config=None):
            return {
                "research_plan": ["主要方案"],
                "deepsearch_epoch": 1,
                "deepsearch_max_epochs": 2,
                "revision_count": 0,
                "max_revisions": 2,
            }

        def fake_tree(state, config=None):
            tasks = state["research_tasks"]
            assigned_topics.extend(task["topic"] for task in tasks)
            results = [{
                "query": task["topic"],
                "title": task["topic"],
                "url": f"https://example.test/{task['task_id']}",
                "snippet": "evidence",
            } for task in tasks]
            return {
                "search_results": results,
                "sources": [{"title": item["title"], "url": item["url"]} for item in results],
                "research_tree": {"topic": "主题", "nodes": []},
                "research_agent_results": [{"task_id": task["task_id"], "status": "completed"} for task in tasks],
                "deepsearch_artifacts": {"phase": "tree_searched"},
                "deepsearch_mode": "tree",
            }

        def fake_writer(state, config=None):
            return {"draft_report": "综合报告", "final_report": "综合报告"}

        def fake_evaluator(state, config=None):
            nonlocal evaluation_count
            evaluation_count += 1
            if evaluation_count == 1:
                return {"verdict": "revise", "missing_topics": ["成本缺口"]}
            return {"verdict": "pass", "missing_topics": []}

        coordinator_model = SimpleNamespace(
            invoke=lambda messages, config=None: SimpleNamespace(
                content=(
                    '{"action":"research","reasoning":"补足成本",'
                    '"priority_topics":["成本缺口"]}'
                )
            )
        )
        with (
            patch.object(settings, "use_hierarchical_agents", True),
            patch("agent.core.graph.router_node", side_effect=fake_router),
            patch("agent.core.graph.deep_plan_node", side_effect=fake_plan),
            patch("agent.core.graph.deep_tree_search_node", side_effect=fake_tree),
            patch("agent.core.graph.deep_writer_node", side_effect=fake_writer),
            patch("agent.core.graph.evaluator_node", side_effect=fake_evaluator),
            patch("agent.core.graph.human_review_node", return_value={}),
            patch("agent.workflows.coordinator.ModelFactory.for_task", return_value=coordinator_model),
        ):
            result = create_research_graph().invoke({"input": "主题", "messages": []})

        self.assertEqual(evaluation_count, 2)
        self.assertEqual(assigned_topics, ["主要方案", "成本缺口"])
        self.assertEqual(result["deepsearch_epoch"], 2)
        self.assertEqual(result["coordinator_action"], "complete")
        self.assertEqual(result["status"], "completed")


if __name__ == "__main__":
    unittest.main()
