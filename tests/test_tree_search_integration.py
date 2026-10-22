import unittest
import asyncio
import threading
import uuid
from unittest.mock import patch

from agent.core.graph import create_research_graph, fan_out_deep_searches
from agent.core.event_stream import get_emitter_sync, remove_emitter_sync
from agent.workflows.nodes import deep_followup_plan_node, deep_tree_search_node
from agent.workflows.research_tree import ResearchTree
from agent.workflows.query_strategy import merge_query_history
from common.config import settings
from common.cancellation import cancellation_manager


class _Response:
    def __init__(self, content):
        self.content = content


class _Planner:
    def invoke(self, message, config=None):
        content = str(message[0].get("content", ""))
        if "分解" in content:
            return _Response(
                '{"subtopics":[{"topic":"分支 A","relevance":1.0},'
                '{"topic":"分支 B","relevance":0.9}]}'
            )
        return _Response('["补充查询"]')


class _FollowupPlanner:
    def invoke(self, message, config=None):
        return _Response(
            '{"queries":["ai official docs", "AI 风险评估 report"],'
            '"reasoning":"补充来源和风险证据"}'
        )


class _UnavailableFollowupPlanner:
    def invoke(self, message, config=None):
        raise RuntimeError("planning model unavailable")


class TreeSearchIntegrationTests(unittest.TestCase):
    def test_tree_mode_routes_to_tree_node(self):
        sends = fan_out_deep_searches(
            {
                "input": "研究主题",
                "tree_exploration_enabled": True,
                "deepsearch_mode": "tree",
            }
        )

        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "deep_tree_search")
        self.assertEqual(sends[0].arg["input"], "研究主题")

    def test_assigned_agent_tasks_use_tree_dispatch_even_in_linear_mode(self):
        sends = fan_out_deep_searches(
            {
                "input": "研究主题",
                "research_tasks": [{"task_id": "task-1", "topic": "子问题"}],
                "deepsearch_mode": "linear",
            }
        )
        self.assertEqual([send.node for send in sends], ["deep_tree_search"])

    def test_graph_registers_tree_search_path(self):
        with patch.object(settings, "use_hierarchical_agents", False):
            graph = create_research_graph()

        self.assertIn("deep_tree_search", graph.get_graph().nodes)

    def test_tree_node_returns_branch_results_to_deep_writer_contract(self):
        def fake_search(state):
            query = state["query"]
            slug = query.replace(" ", "-")
            item = {
                "title": query,
                "url": f"https://example.test/{slug}",
                "snippet": f"{query} 的证据",
                "content": f"{query} 的正文",
                "provider": "fake",
                "query": query,
            }
            return {
                "search_results": [item],
                "sources": [{
                    "title": query,
                    "url": item["url"],
                    "provider": "fake",
                    "query": query,
                }],
                "fetched_pages": [],
                "passages": [],
            }

        with (
            patch.object(settings, "deepsearch_tree_max_depth", 1),
            patch.object(settings, "deepsearch_tree_max_branches", 2),
            patch.object(settings, "deepsearch_tree_queries_per_branch", 1),
            patch.object(settings, "deepsearch_tree_parallel_branches", 0),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node(
                {
                    "input": "研究主题",
                    "research_plan": ["研究主题"],
                    "deepsearch_epoch": 1,
                },
                {},
            )

        self.assertEqual(result["deepsearch_mode"], "tree")
        self.assertEqual(
            {node["topic"] for node in result["research_tree"]["nodes"]},
            {"研究主题", "分支 A", "分支 B"},
        )
        self.assertEqual(
            {item["query"] for item in result["search_results"]},
            {"研究主题", "分支 A", "分支 B"},
        )
        tree_nodes = {
            node["topic"]: node
            for node in result["research_tree"]["nodes"]
        }
        for topic in ("研究主题", "分支 A", "分支 B"):
            agent_result = tree_nodes[topic]["agent_result"]
            self.assertEqual(agent_result["task_id"], tree_nodes[topic]["id"])
            self.assertEqual(agent_result["status"], "completed")
            self.assertEqual(agent_result["tool_calls"], ["web_search"])
            self.assertTrue(agent_result["findings"])
        self.assertEqual(result["deepsearch_artifacts"]["phase"], "tree_searched")
        self.assertTrue(result["deepsearch_artifacts"]["tree_summary"])
        self.assertEqual(len(result["summary_notes"]), 1)

    def test_coordinator_tasks_execute_without_tree_replanning(self):
        searched = []

        def fake_search(state):
            query = state["query"]
            searched.append(query)
            return {
                "search_results": [{
                    "title": query,
                    "url": f"https://example.test/{query}",
                    "snippet": f"{query} evidence",
                    "provider": "fake",
                }],
                "sources": [],
                "fetched_pages": [],
                "passages": [],
            }

        tasks = [
            {
                "task_id": "task-cost-001",
                "topic": "成本比较",
                "objective": "收集成本数据",
                "priority": 5,
                "context": {"overall_topic": "方案比较", "focus": "成本"},
                "allowed_tools": ["web_search"],
            },
            {
                "task_id": "task-risk-001",
                "topic": "风险分析",
                "objective": "收集风险证据",
                "priority": 4,
                "context": {"overall_topic": "方案比较", "focus": "风险"},
                "allowed_tools": ["web_search"],
            },
        ]
        with (
            patch.object(settings, "deepsearch_tree_max_depth", 2),
            patch.object(settings, "deepsearch_tree_max_branches", 4),
            patch.object(settings, "deepsearch_tree_queries_per_branch", 3),
            patch.object(settings, "deepsearch_tree_parallel_branches", 0),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node(
                {
                    "input": "方案比较",
                    "research_plan": ["成本比较", "风险分析"],
                    "research_tasks": tasks,
                    "deepsearch_epoch": 1,
                },
                {},
            )

        self.assertCountEqual(searched, ["成本比较", "风险分析"])
        self.assertEqual(result["research_plan"], ["成本比较", "风险分析"])
        self.assertCountEqual(
            result["deepsearch_query_history"],
            ["成本比较", "风险分析"],
        )
        self.assertEqual(
            {item["task_id"] for item in result["research_agent_results"]},
            {"task-cost-001", "task-risk-001"},
        )
        tree_nodes = {
            node["topic"]: node
            for node in result["research_tree"]["nodes"]
        }
        for task in tasks:
            agent_result = tree_nodes[task["topic"]]["agent_result"]
            self.assertEqual(agent_result["task_id"], task["task_id"])
            self.assertEqual(agent_result["tool_calls"], ["web_search"])

    def test_followup_coordinator_tasks_extend_checkpoint_tree(self):
        searched = []

        def fake_search(state):
            query = state["query"]
            searched.append(query)
            return {"search_results": [], "sources": [], "fetched_pages": [], "passages": []}

        previous = ResearchTree.from_queries(
            "研究主题",
            ["已有研究方向"],
            epoch=1,
            max_branches=2,
        ).to_dict()
        with (
            patch.object(settings, "deepsearch_tree_parallel_branches", 0),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node(
                {
                    "input": "研究主题",
                    "research_plan": ["新研究方向"],
                    "research_tasks": [{
                        "task_id": "task-followup-002",
                        "topic": "新研究方向",
                        "objective": "填补缺口",
                        "priority": 5,
                        "context": {"epoch": 2},
                        "allowed_tools": ["web_search"],
                    }],
                    "research_tree": previous,
                    "deepsearch_epoch": 2,
                },
                {},
            )

        self.assertEqual(searched, ["新研究方向"])
        nodes = result["research_tree"]["nodes"]
        self.assertEqual(
            {node["topic"] for node in nodes if node["depth"] == 1},
            {"已有研究方向", "新研究方向"},
        )
        self.assertEqual(
            next(item for item in result["research_agent_results"] if item["task_id"] == "task-followup-002")["task_id"],
            "task-followup-002",
        )

    def test_failed_research_task_does_not_discard_successful_sibling(self):
        def fake_search(state):
            query = state["query"]
            if query == "失败分支":
                return {"search_results": [], "errors": ["provider unavailable"]}
            item = {
                "title": query,
                "url": "https://example.test/success",
                "snippet": "成功分支证据",
                "provider": "fake",
            }
            return {"search_results": [item], "sources": [], "fetched_pages": [], "passages": []}

        tasks = [
            {"task_id": "task-failed", "topic": "失败分支", "allowed_tools": ["web_search"]},
            {"task_id": "task-success", "topic": "成功分支", "allowed_tools": ["web_search"]},
        ]
        with (
            patch.object(settings, "deepsearch_tree_parallel_branches", 0),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node(
                {
                    "input": "研究主题",
                    "research_plan": [task["topic"] for task in tasks],
                    "research_tasks": tasks,
                    "deepsearch_epoch": 1,
                },
                {},
            )

        by_id = {item["task_id"]: item for item in result["research_agent_results"]}
        self.assertEqual(by_id["task-failed"]["status"], "failed")
        self.assertEqual(by_id["task-success"]["status"], "completed")
        self.assertTrue(any(item.get("url") == "https://example.test/success" for item in result["search_results"]))

    def test_assigned_agents_run_with_configured_concurrency_limit(self):
        barrier = threading.Barrier(2, timeout=3)
        lock = threading.Lock()
        active = 0
        max_active = 0

        def fake_search(state):
            nonlocal active, max_active
            query = state["query"]
            with lock:
                active += 1
                max_active = max(max_active, active)
            barrier.wait()
            with lock:
                active -= 1
            return {
                "search_results": [{
                    "title": query,
                    "url": f"https://example.test/{query}",
                    "snippet": "evidence",
                }],
                "sources": [],
                "fetched_pages": [],
                "passages": [],
            }

        tasks = [
            {"task_id": "parallel-a", "topic": "并行任务 A", "allowed_tools": ["web_search"]},
            {"task_id": "parallel-b", "topic": "并行任务 B", "allowed_tools": ["web_search"]},
        ]
        with (
            patch.object(settings, "deepsearch_tree_parallel_branches", 2),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node(
                {
                    "input": "并行研究",
                    "research_plan": [task["topic"] for task in tasks],
                    "research_tasks": tasks,
                    "deepsearch_epoch": 1,
                },
                {},
            )

        self.assertEqual(max_active, 2)
        self.assertEqual(len(result["research_agent_results"]), 2)

    def test_cancel_token_is_checked_before_assigned_branch_search(self):
        token_id = f"research-cancel-{uuid.uuid4().hex}"
        cancellation_manager.create_token_sync(token_id).cancel("test cancellation")
        task = {"task_id": "cancelled-task", "topic": "子任务", "allowed_tools": ["web_search"]}
        with (
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node") as search,
        ):
            with self.assertRaises(asyncio.CancelledError):
                deep_tree_search_node(
                    {
                        "input": "主题",
                        "research_plan": ["子任务"],
                        "research_tasks": [task],
                        "cancel_token_id": token_id,
                    },
                    {},
                )
        search.assert_not_called()

    def test_cancel_during_search_prevents_branch_summary_and_next_task(self):
        token_id = f"research-cancel-{uuid.uuid4().hex}"
        token = cancellation_manager.create_token_sync(token_id)
        queries = []

        def fake_search(state):
            query = state["query"]
            queries.append(query)
            token.cancel("cancel during provider call")
            return {"search_results": [], "sources": []}

        task = {"task_id": "cancel-in-flight", "topic": "子任务", "allowed_tools": ["web_search"]}
        with (
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            with self.assertRaises(asyncio.CancelledError):
                deep_tree_search_node(
                    {
                        "input": "主题",
                        "research_plan": ["子任务"],
                        "research_tasks": [task],
                        "cancel_token_id": token_id,
                    },
                    {},
                )
        self.assertEqual(queries, ["子任务"])

    def test_tree_node_applies_token_budget_override_and_reports_usage(self):
        search_calls = []

        def fake_search(state):
            search_calls.append(state["query"])
            return {"search_results": [], "sources": []}

        with (
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node(
                {"input": "research topic", "deepsearch_epoch": 1},
                {
                    "configurable": {
                        "deepsearch_tree_max_depth": 0,
                        "deepsearch_tree_parallel_branches": 0,
                        "deepsearch_tree_queries_per_branch": 1,
                        "deepsearch_max_tokens": 1,
                    }
                },
            )

        self.assertEqual(search_calls, [])
        budget = result["deepsearch_artifacts"]["tree_budget"]
        self.assertEqual(budget["stop_reason"], "token_budget_exceeded")
        self.assertGreaterEqual(budget["tokens_used"], 1)

    def test_tree_node_falls_back_to_linear_search_when_models_are_unavailable(self):
        def fake_search(state):
            return {
                "search_results": [{
                    "title": "线性结果",
                    "url": "https://example.test/linear",
                    "snippet": "fallback",
                    "provider": "fake",
                }],
                "sources": [{
                    "title": "线性结果",
                    "url": "https://example.test/linear",
                    "provider": "fake",
                    "query": state["query"],
                }],
            }

        with (
            patch(
                "agent.workflows.nodes.ModelFactory.for_task",
                side_effect=ValueError("DeepSeek API Key 未配置"),
            ),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node(
                {
                    "input": "研究主题",
                    "research_plan": ["线性查询"],
                },
                {},
            )

        self.assertEqual(result["deepsearch_mode"], "linear")
        self.assertEqual(result["search_results"][0]["url"], "https://example.test/linear")
        self.assertEqual(
            result["deepsearch_artifacts"]["phase"],
            "linear_fallback",
        )

    def test_tree_node_publishes_runtime_observability_events(self):
        thread_id = "tree-events-test"

        def fake_search(state):
            return {
                "search_results": [{
                    "title": "来源",
                    "url": "https://example.test/source",
                    "snippet": "证据",
                    "provider": "fake",
                    "published_date": "2000-01-01",
                }],
                "sources": [],
                "fetched_pages": [],
                "passages": [],
            }

        try:
            with (
                patch.object(settings, "deepsearch_tree_max_depth", 0),
                patch.object(settings, "deepsearch_tree_parallel_branches", 0),
                patch.object(settings, "deepsearch_tree_queries_per_branch", 1),
                patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
                patch("agent.workflows.nodes.search_node", side_effect=fake_search),
            ):
                result = deep_tree_search_node(
                    {"input": "研究主题", "deepsearch_epoch": 1},
                    {"configurable": {"thread_id": thread_id}},
                )

            quality = result["deepsearch_artifacts"]["quality_summary"]
            self.assertIn("query_coverage", quality)
            self.assertEqual(quality["freshness_summary"]["total_results"], 1)
            self.assertFalse(quality["time_sensitive_query"])
            self.assertEqual(
                result["deepsearch_tokens_used"],
                quality["tokens_used"],
            )
            self.assertEqual(
                result["budget_stop_reason"],
                quality["budget_stop_reason"],
            )

            events = get_emitter_sync(thread_id).buffered_events()
            quality_event = next(
                event for event in events if event.event_type == "quality_update"
            )
            self.assertIn("query_coverage", quality_event.data)
            self.assertIn("freshness_summary", quality_event.data)
            self.assertIn("budget_stop_reason", quality_event.data)

            event_types = [
                event.event_type for event in events
            ]
            self.assertIn("research_node_start", event_types)
            self.assertIn("search", event_types)
            self.assertIn("research_node_complete", event_types)
            self.assertIn("research_tree_update", event_types)
            self.assertIn("quality_update", event_types)
        finally:
            remove_emitter_sync(thread_id)

    def test_followup_tree_search_only_explores_new_pending_nodes(self):
        tree = ResearchTree(
            topic="研究主题",
            epoch=2,
            max_depth=1,
            max_branches=2,
        )
        root = tree.create_root("研究主题")
        completed = tree.add_child(root.id, "已完成分支")
        completed.mark_complete("旧摘要")
        tree.epoch = 2
        tree.add_queries(
            ["新增验证问题"],
            epoch=2,
            max_branches=2,
        )
        search_calls = []

        def fake_search(state):
            query = state["query"]
            search_calls.append(query)
            return {
                "search_results": [{
                    "title": query,
                    "url": f"https://example.test/{query}",
                    "snippet": "后续证据",
                    "provider": "fake",
                }],
                "sources": [],
                "fetched_pages": [],
                "passages": [],
            }

        with (
            patch.object(settings, "deepsearch_tree_parallel_branches", 0),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Planner()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node(
                {
                    "input": "研究主题",
                    "research_plan": ["新增验证问题"],
                    "deepsearch_epoch": 2,
                    "deepsearch_query_history": ["历史查询", "研究主题"],
                    "research_tree": tree.to_dict(),
                },
                {
                    "configurable": {
                        "deepsearch_tree_queries_per_branch": 1,
                        "deepsearch_tree_max_depth": 1,
                        "deepsearch_tree_max_branches": 2,
                    }
                },
            )

        self.assertEqual(search_calls, ["新增验证问题"])
        self.assertIn("新增验证问题", {
            item["query"] for item in result["search_results"]
        })
        self.assertNotIn("已完成分支", search_calls)
        self.assertEqual(
            result["deepsearch_query_history"],
            ["已完成分支", "新增验证问题"],
        )
        self.assertEqual(
            merge_query_history(
                ["历史查询", "研究主题"],
                result["deepsearch_query_history"],
            ),
            ["历史查询", "研究主题", "已完成分支", "新增验证问题"],
        )

    def test_followup_planner_keeps_history_and_deduplicates_case_insensitively(self):
        tree = ResearchTree.from_queries(
            "研究主题",
            ["此前官方文档查询"],
            max_branches=4,
        )
        prior_history = ["此前官方文档查询", "AI OFFICIAL DOCS"]

        with (
            patch.object(settings, "deepsearch_max_epochs", 3),
            patch.object(settings, "deepsearch_query_num", 2),
            patch.object(settings, "deepsearch_tree_max_branches", 4),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_FollowupPlanner()),
        ):
            result = deep_followup_plan_node(
                {
                    "input": "研究主题",
                    "deepsearch_epoch": 1,
                    "deepsearch_query_history": prior_history,
                    "missing_topics": ["风险与证据"],
                    "research_tree": tree.to_dict(),
                },
                {},
            )

        prior_keys = {query.casefold() for query in prior_history}
        self.assertTrue(result["research_plan"])
        self.assertTrue(
            all(query.casefold() not in prior_keys for query in result["research_plan"])
        )
        self.assertEqual(
            result["deepsearch_query_history"],
            result["research_plan"],
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["query_history_count"],
            len(merge_query_history(prior_history, result["research_plan"])),
        )

    def test_followup_uses_gap_suggestion_and_preserves_gap_artifact(self):
        tree = ResearchTree.from_queries(
            "研究主题",
            ["此前查询"],
            max_branches=4,
        )
        gap_analysis = {
            "overall_coverage": 0.5,
            "suggested_queries": ["知识缺口对应的官方数据查询"],
        }

        with (
            patch.object(settings, "deepsearch_max_epochs", 3),
            patch.object(settings, "deepsearch_query_num", 1),
            patch.object(settings, "deepsearch_tree_max_branches", 4),
            patch(
                "agent.workflows.nodes.ModelFactory.for_task",
                return_value=_UnavailableFollowupPlanner(),
            ),
        ):
            result = deep_followup_plan_node(
                {
                    "input": "研究主题",
                    "deepsearch_epoch": 1,
                    "deepsearch_query_history": ["此前查询"],
                    "missing_topics": ["证据不足的方面"],
                    "knowledge_gap_analysis": gap_analysis,
                    "deepsearch_artifacts": {
                        "knowledge_gap_analysis": gap_analysis,
                        "quality_summary": {"overall_score": 0.7},
                    },
                    "research_tree": tree.to_dict(),
                },
                {},
            )

        self.assertEqual(
            result["research_plan"],
            ["知识缺口对应的官方数据查询"],
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["knowledge_gap_analysis"],
            gap_analysis,
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["quality_summary"]["overall_score"],
            0.7,
        )


if __name__ == "__main__":
    unittest.main()
