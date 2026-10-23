import threading
import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph

from agent.core.graph import (
    fan_out_research_tasks,
    research_task_aggregate_node,
    research_task_worker_node,
)
from agent.core.state import AgentState
from common.config import settings


class ResearchTaskCheckpointTests(unittest.TestCase):
    @staticmethod
    def build_graph(checkpointer):
        workflow = StateGraph(AgentState)
        workflow.add_node("research_task_worker", research_task_worker_node)
        workflow.add_node(
            "research_task_aggregate",
            research_task_aggregate_node,
            defer=True,
        )
        workflow.add_node("sources_review", lambda state: {})
        workflow.add_conditional_edges(
            START,
            fan_out_research_tasks,
            ["research_task_worker", "research_task_aggregate"],
        )
        workflow.add_edge("research_task_worker", "research_task_aggregate")
        workflow.add_conditional_edges(
            "research_task_aggregate",
            fan_out_research_tasks,
            ["research_task_worker", "research_task_aggregate", "sources_review"],
        )
        workflow.add_edge("sources_review", END)
        return workflow.compile(checkpointer=checkpointer)

    def test_resume_reuses_successful_worker_write_and_retries_only_failed_worker(self):
        first_worker_done = threading.Event()
        calls = {"task-a": 0, "task-b": 0}

        def run_branch(state, config):
            task = state["active_research_task"]
            task_id = task["task_id"]
            calls[task_id] += 1
            if task_id == "task-a":
                first_worker_done.set()
            elif calls[task_id] == 1:
                self.assertTrue(first_worker_done.wait(timeout=5))
                raise RuntimeError("temporary branch failure")
            topic = task["topic"]
            return {
                "search_results": [{
                    "query": topic,
                    "title": f"{topic} source",
                    "url": f"https://example.test/{task_id}",
                    "snippet": f"Evidence for {topic}",
                }],
                "sources": [{
                    "title": f"{topic} source",
                    "url": f"https://example.test/{task_id}",
                }],
                "research_agent_results": [{
                    "task_id": task_id,
                    "status": "completed",
                    "summary": f"Completed {topic}",
                }],
                "research_tree": {},
                "deepsearch_artifacts": {},
                "summary_notes": [],
            }

        graph = self.build_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "research-task-resume-test"}}
        initial_state = {
            "input": "Checkpoint recovery",
            "research_plan": ["Task A", "Task B"],
            "research_tasks": [
                {"task_id": "task-a", "topic": "Task A", "allowed_tools": ["web_search"]},
                {"task_id": "task-b", "topic": "Task B", "allowed_tools": ["web_search"]},
            ],
        }

        with patch("agent.core.graph.deep_tree_search_node", side_effect=run_branch):
            with self.assertRaisesRegex(RuntimeError, "temporary branch failure"):
                graph.invoke(initial_state, config=config)
            result = graph.invoke(None, config=config)

        self.assertEqual(calls, {"task-a": 1, "task-b": 2})
        self.assertEqual(
            {item["task_id"] for item in result["research_agent_results"]},
            {"task-a", "task-b"},
        )
        self.assertEqual(
            {item["url"] for item in result["search_results"]},
            {"https://example.test/task-a", "https://example.test/task-b"},
        )
        self.assertEqual(result.get("research_task_checkpoints", []), [])

    def test_dispatch_omits_task_with_a_checkpointed_result(self):
        sends = fan_out_research_tasks({
            "research_tasks": [
                {"task_id": "done", "topic": "Already done"},
                {"task_id": "pending", "topic": "Needs work"},
            ],
            "research_task_checkpoints": [
                {
                    "task_id": "done",
                    "result": {
                        "research_agent_results": [{
                            "task_id": "done",
                            "status": "completed",
                        }],
                    },
                },
            ],
        })

        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "research_task_worker")
        self.assertEqual(sends[0].arg["active_research_task"]["task_id"], "pending")

    def test_dispatch_returns_failed_attempt_to_coordinator_without_immediate_retry(self):
        sends = fan_out_research_tasks({
            "research_tasks": [
                {"task_id": "failed", "topic": "Retry me"},
                {"task_id": "done", "topic": "Already done"},
            ],
            "research_task_checkpoints": [
                {
                    "task_id": "failed",
                    "result": {
                        "research_agent_results": [{
                            "task_id": "failed",
                            "status": "failed",
                        }],
                    },
                },
                {
                    "task_id": "done",
                    "result": {
                        "research_agent_results": [{
                            "task_id": "done",
                            "status": "completed",
                        }],
                    },
                },
            ],
        })

        self.assertEqual(sends, "sources_review")

    def test_dispatch_stops_when_global_research_budget_is_exhausted(self):
        result = fan_out_research_tasks({
            "research_tasks": [{"task_id": "pending", "topic": "Do not run"}],
            "deepsearch_artifacts": {
                "research_budget": {
                    "stop_reason": "token_budget_exceeded",
                },
            },
        })

        self.assertEqual(result, "sources_review")

    def test_worker_batches_respect_parallel_branch_limit(self):
        calls = []

        def run_branch(state, config):
            task = state["active_research_task"]
            calls.append(task["task_id"])
            return {
                "search_results": [],
                "sources": [],
                "research_agent_results": [{
                    "task_id": task["task_id"],
                    "status": "completed",
                    "summary": task["topic"],
                }],
                "research_tree": {},
                "deepsearch_artifacts": {},
                "summary_notes": [],
            }

        graph = self.build_graph(InMemorySaver())
        config = {"configurable": {"thread_id": "research-task-batches-test"}}
        initial_state = {
            "input": "Bounded batches",
            "research_tasks": [
                {"task_id": f"task-{index}", "topic": f"Task {index}"}
                for index in range(3)
            ],
        }
        with (
            patch.object(settings, "deepsearch_tree_parallel_branches", 1),
            patch("agent.core.graph.deep_tree_search_node", side_effect=run_branch),
        ):
            result = graph.invoke(initial_state, config=config)

        self.assertCountEqual(calls, ["task-0", "task-1", "task-2"])
        self.assertEqual(
            {item["task_id"] for item in result["research_agent_results"]},
            {"task-0", "task-1", "task-2"},
        )

    def test_parallel_batch_is_aggregated_once_before_next_batch_dispatch(self):
        calls = []
        aggregate_calls = []

        def run_branch(state, config):
            task = state["active_research_task"]
            calls.append(task["task_id"])
            return {
                "search_results": [],
                "sources": [],
                "research_agent_results": [{
                    "task_id": task["task_id"],
                    "status": "completed",
                    "summary": task["topic"],
                }],
                "research_tree": {},
                "deepsearch_artifacts": {},
                "summary_notes": [],
            }

        def aggregate(state):
            aggregate_calls.append(tuple(
                item["task_id"]
                for item in state.get("research_task_checkpoints", [])
            ))
            return research_task_aggregate_node(state)

        workflow = StateGraph(AgentState)
        workflow.add_node("research_task_worker", research_task_worker_node)
        workflow.add_node("research_task_aggregate", aggregate, defer=True)
        workflow.add_node("sources_review", lambda state: {})
        workflow.add_conditional_edges(
            START,
            fan_out_research_tasks,
            ["research_task_worker", "research_task_aggregate"],
        )
        workflow.add_edge("research_task_worker", "research_task_aggregate")
        workflow.add_conditional_edges(
            "research_task_aggregate",
            fan_out_research_tasks,
            ["research_task_worker", "research_task_aggregate", "sources_review"],
        )
        workflow.add_edge("sources_review", END)
        graph = workflow.compile(checkpointer=InMemorySaver())
        initial_state = {
            "input": "Batch barrier",
            "research_task_parallel_limit": 2,
            "research_tasks": [
                {"task_id": f"task-{index}", "topic": f"Task {index}"}
                for index in range(5)
            ],
        }

        with patch("agent.core.graph.deep_tree_search_node", side_effect=run_branch):
            result = graph.invoke(
                initial_state,
                config={"configurable": {"thread_id": "research-task-barrier-test"}},
            )

        self.assertCountEqual(calls, [f"task-{index}" for index in range(5)])
        self.assertEqual(len(aggregate_calls), 3)
        self.assertEqual(
            [len(batch) for batch in aggregate_calls],
            [2, 2, 1],
        )
        self.assertEqual(len(result["research_agent_results"]), 5)

    def test_attempted_task_ledger_prevents_redispatch_without_agent_result_ids(self):
        calls = []

        def run_branch(state, config):
            task = state["active_research_task"]
            calls.append(task["task_id"])
            return {
                "search_results": [],
                "sources": [],
                "research_tree": {},
                "deepsearch_artifacts": {},
                "summary_notes": [],
            }

        graph = self.build_graph(InMemorySaver())
        with (
            patch.object(settings, "deepsearch_tree_parallel_branches", 1),
            patch("agent.core.graph.deep_tree_search_node", side_effect=run_branch),
        ):
            result = graph.invoke(
                {
                    "input": "A branch may produce no structured result",
                    "research_tasks": [
                        {"task_id": "task-a", "topic": "Task A"},
                        {"task_id": "task-b", "topic": "Task B"},
                    ],
                },
                config={"configurable": {"thread_id": "research-task-ledger-test"}},
            )

        self.assertCountEqual(calls, ["task-a", "task-b"])
        self.assertEqual(result["research_task_attempted_ids"], ["task-a", "task-b"])

    def test_local_worker_budget_stop_does_not_exhaust_global_budget(self):
        task = {"task_id": "task-a", "topic": "One research task"}
        state = {
            "input": "topic",
            "research_tasks": [task],
            "deepsearch_artifacts": {
                "research_budget": {
                    "tokens_used": 100,
                    "max_tokens": 50000,
                    "elapsed_seconds": 10,
                    "max_seconds": 300,
                    "searches_used": 0,
                    "stop_reason": "",
                },
            },
            "research_task_checkpoints": [{
                "task_id": "task-a",
                "result": {
                    "research_agent_results": [{
                        "task_id": "task-a",
                        "status": "completed",
                        "summary": "Branch evidence summary",
                    }],
                    "deepsearch_artifacts": {
                        "research_budget": {
                            "tokens_used": 10000,
                            "max_tokens": 10000,
                            "elapsed_seconds": 20,
                            "max_seconds": 60,
                            "searches_used": 3,
                            "stop_reason": "token_budget_exceeded",
                        },
                    },
                },
            }],
        }

        result = research_task_aggregate_node(state)

        self.assertEqual(result["budget_stop_reason"], "")
        self.assertEqual(
            result["deepsearch_artifacts"]["research_budget"]["stop_reason"],
            "",
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["research_budget"]["branch_stop_reasons"],
            [{"task_id": "task-a", "reason": "token_budget_exceeded"}],
        )

    def test_worker_token_share_reserves_budget_for_non_research_roles(self):
        branch_configs = []
        state = {
            "input": "topic",
            "research_tasks": [
                {"task_id": f"task-{index}", "topic": f"Task {index}"}
                for index in range(4)
            ],
            "deepsearch_artifacts": {
                "research_budget": {
                    "tokens_used": 0,
                    "max_tokens": 10000,
                    "elapsed_seconds": 0,
                    "max_seconds": 300,
                    "searches_used": 0,
                    "stop_reason": "",
                },
            },
            "active_research_task": {"task_id": "task-0", "topic": "Task 0"},
        }

        def capture_branch(state, config):
            branch_configs.append(config)
            return {}

        with (
            patch("agent.core.graph.deep_tree_search_node", side_effect=capture_branch),
            patch.object(settings, "deepsearch_tree_parallel_branches", 4),
        ):
            research_task_worker_node(
                state,
                {"configurable": {"deepsearch_tree_parallel_branches": 4}},
            )

        # 60% of the global budget is divided across all assigned tasks.
        self.assertEqual(
            branch_configs[0]["configurable"]["deepsearch_max_tokens"],
            1500,
        )

    def test_aggregate_stops_when_combined_worker_usage_reaches_global_budget(self):
        task = {"task_id": "task-a", "topic": "One research task"}
        state = {
            "input": "topic",
            "research_tasks": [task],
            "deepsearch_artifacts": {
                "research_budget": {
                    "tokens_used": 100,
                    "max_tokens": 10000,
                    "elapsed_seconds": 10,
                    "max_seconds": 300,
                    "searches_used": 0,
                    "stop_reason": "",
                },
            },
            "research_task_checkpoints": [{
                "task_id": "task-a",
                "result": {
                    "research_agent_results": [{
                        "task_id": "task-a",
                        "status": "completed",
                        "summary": "Branch evidence summary",
                    }],
                    "deepsearch_artifacts": {
                        "research_budget": {
                            "tokens_used": 10000,
                            "max_tokens": 10000,
                            "elapsed_seconds": 20,
                            "max_seconds": 60,
                            "searches_used": 3,
                            "stop_reason": "token_budget_exceeded",
                        },
                    },
                },
            }],
        }

        result = research_task_aggregate_node(state)

        self.assertEqual(result["budget_stop_reason"], "token_budget_exceeded")
        self.assertEqual(
            result["deepsearch_tokens_used"],
            10099,
        )


if __name__ == "__main__":
    unittest.main()
