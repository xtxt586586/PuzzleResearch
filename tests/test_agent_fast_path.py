import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.core.graph import create_research_graph
from agent.core.profile import AgentProfile
from agent.workflows.nodes import (
    agent_fast_path_node,
    should_use_agent_fast_path,
)
from tools.registry import ToolDefinition, ToolRegistry


class AgentFastPathTests(unittest.TestCase):
    def setUp(self):
        self.search_calls = []

        def search(query, max_results=3):
            self.search_calls.append((query, max_results))
            return {
                "output": "[1] 官方版本说明\n摘要：当前版本为 3.2。",
                "error": "",
                "query": query,
                "results": [{
                    "title": "官方版本说明",
                    "url": "https://example.test/release",
                    "snippet": "当前版本为 3.2。",
                }],
            }

        self.registry = ToolRegistry([
            ToolDefinition(
                name="web_search",
                description="Search web",
                handler=search,
            )
        ])

    def test_selects_only_short_narrow_verification_questions(self):
        self.assertTrue(should_use_agent_fast_path({"input": "查一下 DeepSeek 最新版本"}))
        self.assertFalse(should_use_agent_fast_path({"input": "请深入研究并生成一份完整报告，核实最新趋势"}))
        self.assertFalse(should_use_agent_fast_path({"input": "查一下这个问题", "images": [{"data": "..."}]}))

    def test_searches_then_streams_answer_and_records_phase_timings(self):
        class FakeModel:
            def stream(self, messages, config=None):
                yield SimpleNamespace(content="根据官方信息，")
                yield SimpleNamespace(content="当前版本为 3.2。")

        state = {
            "input": "查一下 DeepSeek 最新版本",
            "enabled_tools": {"web_search": True},
            "tool_call_limit": 4,
        }
        config = {
            "configurable": {
                "tool_registry": self.registry,
                "thread_id": "fast-path-test",
            }
        }

        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=FakeModel()) as model_factory, \
             patch("agent.workflows.nodes._emit_tool_stream_event") as emit:
            result = agent_fast_path_node(state, config)

        model_factory.assert_called_once()
        self.assertEqual(model_factory.call_args.args[0], "writing")
        self.assertEqual(self.search_calls, [("查一下 DeepSeek 最新版本", 3)])
        self.assertEqual(result["final_report"], "根据官方信息，当前版本为 3.2。")
        self.assertTrue(result["agent_fast_path"])
        self.assertGreaterEqual(result["agent_performance"]["node_duration_ms"], 0)
        self.assertIsNotNone(result["agent_performance"]["first_token_duration_ms"])
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(emit.call_args_list[0].args[1], "tool_started")
        self.assertEqual(emit.call_args_list[1].args[1], "tool_result")

    def test_disabled_search_permission_falls_back_without_searching(self):
        config = {
            "configurable": {
                "tool_registry": self.registry,
                "agent_profile": AgentProfile(
                    name="restricted",
                    enabled_tools=frozenset({"python_executor"}),
                ),
            }
        }

        result = agent_fast_path_node(
            {"input": "查一下 DeepSeek 最新版本"},
            config,
        )

        self.assertTrue(result["agent_fast_path_fallback"])
        self.assertEqual(self.search_calls, [])

    def test_empty_search_results_are_recorded_before_regular_agent_fallback(self):
        self.registry = ToolRegistry([
            ToolDefinition(
                name="web_search",
                description="Search web",
                handler=lambda query, max_results=3: {
                    "output": "",
                    "error": "",
                    "query": query,
                    "results": [],
                },
            )
        ])
        result = agent_fast_path_node(
            {
                "input": "查一下 DeepSeek 最新版本",
                "enabled_tools": {"web_search": True},
            },
            {"configurable": {"tool_registry": self.registry}},
        )

        self.assertTrue(result["agent_fast_path_fallback"])
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(len(result["tool_results"]), 1)

    def test_interrupted_stream_does_not_fall_back_and_duplicate_partial_answer(self):
        class BrokenStreamModel:
            def stream(self, messages, config=None):
                yield SimpleNamespace(content="部分回答")
                raise RuntimeError("stream interrupted")

        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=BrokenStreamModel()):
            result = agent_fast_path_node(
                {
                    "input": "查一下 DeepSeek 最新版本",
                    "enabled_tools": {"web_search": True},
                },
                {"configurable": {"tool_registry": self.registry}},
            )

        self.assertFalse(result["agent_fast_path_fallback"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["final_report"], "部分回答")
        self.assertEqual(result["tool_call_count"], 1)

    def test_graph_routes_narrow_verification_to_fast_path_and_review(self):
        with patch("agent.core.graph.router_node", return_value={"route": "agent"}), \
             patch("agent.core.graph.agent_fast_path_node", return_value={
                 "agent_fast_path_fallback": False,
                 "agent_finished": True,
                 "agent_fast_path": True,
                 "final_report": "根据证据完成核验。",
                 "draft_report": "根据证据完成核验。",
                 "status": "running",
             }) as fast_path, \
             patch("agent.core.graph.human_review_node", return_value={}), \
             patch("common.config.settings.use_hierarchical_agents", False):
            graph = create_research_graph()
            result = graph.invoke({"input": "查一下 DeepSeek 最新版本"})

        fast_path.assert_called_once()
        self.assertEqual(result["final_report"], "根据证据完成核验。")

    def test_graph_routes_short_web_verification_to_fast_path(self):
        with patch("agent.core.graph.router_node", return_value={"route": "web"}), \
             patch("agent.core.graph.agent_fast_path_node", return_value={
                 "agent_fast_path_fallback": False,
                 "agent_finished": True,
                 "agent_fast_path": True,
                 "final_report": "根据搜索证据核验完成。",
                 "draft_report": "根据搜索证据核验完成。",
                 "status": "running",
             }) as fast_path, \
             patch("agent.core.graph.human_review_node", return_value={}), \
             patch("common.config.settings.use_hierarchical_agents", False):
            graph = create_research_graph()
            result = graph.invoke({"input": "查一下 Python 3.13 当前最新稳定版本"})

        fast_path.assert_called_once()
        self.assertEqual(result["final_report"], "根据搜索证据核验完成。")


if __name__ == "__main__":
    unittest.main()
