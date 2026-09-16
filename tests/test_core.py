import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.core.graph import (
    create_research_graph,
    evaluation_decision,
    fan_out_deep_searches,
)
from agent.core.smart_router import SmartRouter
from agent.core.state import capped_add_messages, merge_unique_items
from agent.workflows.nodes import (
    _parse_json_object,
    _parse_plan_response,
    agent_tool_node,
    finalize_node,
)
from common.config import settings


class CoreContractTests(unittest.TestCase):
    def test_message_reducer_keeps_head_and_tail(self):
        with patch.object(settings, "trim_messages", True), patch.object(
            settings, "trim_messages_keep_first", 1
        ), patch.object(settings, "trim_messages_keep_last", 2):
            merged = capped_add_messages(
                [SystemMessage(content="system")],
                [
                    HumanMessage(content="old"),
                    AIMessage(content="middle"),
                    HumanMessage(content="latest"),
                ],
            )

        self.assertEqual(
            [message.content for message in merged],
            ["system", "middle", "latest"],
        )

    def test_search_result_reducer_deduplicates_urls(self):
        merged = merge_unique_items(
            [{"url": "https://example.com/"}],
            [
                {"url": "https://example.com"},
                {"url": "https://other.example.com"},
            ],
        )

        self.assertEqual(
            [item["url"] for item in merged],
            ["https://example.com/", "https://other.example.com"],
        )

    def test_explicit_router_intent_is_stable(self):
        with patch(
            "agent.core.smart_router.ModelFactory.for_task",
            side_effect=AssertionError("explicit intent should not call the LLM"),
        ):
            self.assertEqual(
                SmartRouter().route("请联网搜索最新新闻").route,
                "web",
            )

        self.assertEqual(
            SmartRouter._explicit_decision("请联网搜索最新新闻").route,
            "web",
        )
        self.assertEqual(
            SmartRouter._explicit_decision(
                "请比较 PostgreSQL 和 MySQL 的优缺点"
            ).route,
            "deep",
        )
        self.assertEqual(
            SmartRouter._explicit_decision("请执行 Python 代码").route,
            "agent",
        )
        self.assertIsNone(
            SmartRouter._explicit_decision("什么是 LangGraph")
        )

    def test_plan_and_json_parsers_accept_model_text(self):
        self.assertEqual(
            _parse_json_object('说明\n```json\n{"ok": true}\n```'),
            {"ok": True},
        )
        queries, reasoning = _parse_plan_response(
            type(
                "Response",
                (),
                {"content": '{"queries": ["a", "a", "b"], "reasoning": "r"}'},
            )()
        )
        self.assertEqual(queries, ["a", "b"])
        self.assertEqual(reasoning, "r")

    def test_deep_fan_out_carries_search_mode(self):
        sends = fan_out_deep_searches({"research_plan": ["query"]})

        self.assertEqual(len(sends), 1)
        self.assertEqual(sends[0].node, "deep_search")
        self.assertEqual(sends[0].arg["search_mode"], "deep")

    def test_quality_decision_does_not_pass_with_missing_topics(self):
        self.assertEqual(
            evaluation_decision(
                {
                    "route": "web",
                    "verdict": "pass",
                    "missing_topics": ["缺失主题"],
                    "revision_count": 0,
                    "max_revisions": 1,
                }
            ),
            "revise",
        )
        self.assertEqual(
            evaluation_decision(
                {
                    "route": "deep",
                    "verdict": "pass",
                    "missing_topics": ["缺失主题"],
                    "deepsearch_epoch": 1,
                    "deepsearch_max_epochs": 2,
                }
            ),
            "deep_followup",
        )

    def test_finalize_preserves_paused_clarification_state(self):
        result = finalize_node(
            {
                "route": "clarify",
                "final_report": "请补充信息",
                "status": "running",
            }
        )

        self.assertFalse(result["is_complete"])
        self.assertEqual(result["status"], "paused")


    def test_graph_routes_agent_request_to_tool_node(self):
        class FakePlanningModel:
            def __init__(self):
                self.responses = [
                    (
                        '{"action": "tool", '
                        '"tool_name": "python_executor", '
                        '"arguments": {'
                        '"code": "print(2 + 3)"'
                        "}}"
                    ),
                    (
                        '{"action": "finish", '
                        '"final_report": "计算完成，结果是 5。"}'
                    ),
                ]
                self.index = 0

            def invoke(self, messages, config=None):
                content = self.responses[
                    min(
                        self.index,
                        len(self.responses) - 1,
                    )
                ]
                self.index += 1

                return type(
                    "FakeResponse",
                    (),
                    {
                        "content": content,
                    },
                )()

        with patch(
                "agent.workflows.nodes.ModelFactory.for_task",
                return_value=FakePlanningModel(),
        ):
            result = create_research_graph().invoke(
                {
                    "input": "请执行 Python 代码",
                    "messages": [],
                }
            )

        self.assertEqual(result["route"], "agent")
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["is_complete"])
        self.assertEqual(
            result["tool_results"][0]["tool_name"],
            "python_executor",
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"].strip(),
            "5",
        )

    def test_graph_compiles(self):
        graph = create_research_graph()
        self.assertIsNotNone(graph.get_graph())

    def test_graph_pauses_and_resumes_tool_approval(self):
        class FakePlanningModel:
            def __init__(self):
                self.responses = [
                    (
                        '{"action": "tool", '
                        '"tool_name": "python_executor", '
                        '"arguments": {'
                        '"code": "print(2 + 3)"'
                        "}}"
                    ),
                    (
                        '{"action": "finish", '
                        '"final_report": "计算完成，结果是 5。"}'
                    ),
                ]
                self.index = 0

            def invoke(self, messages, config=None):
                content = self.responses[
                    min(
                        self.index,
                        len(self.responses) - 1,
                    )
                ]
                self.index += 1

                return type(
                    "FakeResponse",
                    (),
                    {
                        "content": content,
                    },
                )()

        checkpointer = InMemorySaver()

        graph = create_research_graph(
            checkpointer=checkpointer,
        )

        config = {
            "configurable": {
                "thread_id": "approval-test-001",
            }
        }

        with patch.object(
                settings,
                "tool_approval",
                True,
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=FakePlanningModel(),
        ):
            paused_result = graph.invoke(
                {
                    "input": "请执行 Python 代码",
                    "messages": [],
                },
                config=config,
            )

            self.assertIn(
                "__interrupt__",
                paused_result,
            )
            self.assertEqual(
                paused_result.get("status"),
                "paused",
            )
            self.assertFalse(
                paused_result.get("is_complete")
            )
            self.assertEqual(
                paused_result.get("tool_results", []),
                [],
            )

            resumed_result = graph.invoke(
                Command(
                    resume={
                        "approved": True,
                    }
                ),
                config=config,
            )

        self.assertEqual(
            resumed_result["status"],
            "completed",
        )
        self.assertTrue(
            resumed_result["is_complete"]
        )
        self.assertEqual(
            resumed_result["tool_results"][0]["result"]["output"].strip(),
            "5",
        )


    def test_graph_compiles_with_memory_checkpointer(self):
        checkpointer = InMemorySaver()

        graph = create_research_graph(
            checkpointer=checkpointer,
        )

        self.assertIsNotNone(graph.get_graph())


    def test_graph_replans_after_tool_result(self):
        class SequentialPlanningModel:
            def __init__(self):
                self.responses = [
                    (
                        '{"action": "tool", '
                        '"tool_name": "python_executor", '
                        '"arguments": {'
                        '"code": "print(2 + 3)"'
                        "}}"
                    ),
                    (
                        '{"action": "finish", '
                        '"final_report": "计算完成，结果是 5。"}'
                    ),
                ]
                self.calls = 0

            def invoke(self, messages, config=None):
                response = type(
                    "FakeResponse",
                    (),
                    {
                        "content": self.responses[self.calls],
                    },
                )()

                self.calls += 1
                return response

        model = SequentialPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = create_research_graph().invoke(
                {
                    "input": "请执行 Python 代码计算 2 加 3，然后告诉我结果",
                    "messages": [],
                }
            )

        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            model.calls,
            2,
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            len(result["tool_results"]),
            1,
        )
        self.assertEqual(
            result["final_report"],
            "计算完成，结果是 5。",
        )


    def test_graph_stops_at_tool_call_limit(self):
        class RepeatingPlanningModel:
            def __init__(self):
                self.calls = 0

            def invoke(self, messages, config=None):
                self.calls += 1

                return type(
                    "FakeResponse",
                    (),
                    {
                        "content": (
                            '{"action": "tool", '
                            '"tool_name": "python_executor", '
                            '"arguments": {'
                            '"code": "print(2 + 3)"'
                            "}}"
                        )
                    },
                )()

        model = RepeatingPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = create_research_graph().invoke(
                {
                    "input": "请执行 Python 代码计算 2 加 3",
                    "messages": [],
                    "tool_call_limit": 1,
                }
            )

        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            model.calls,
            1,
        )


if __name__ == "__main__":
    unittest.main()
