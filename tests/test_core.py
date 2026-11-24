import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.core.graph import (
    create_research_graph,
    evaluation_decision,
    fan_out_deep_searches,
)
from agent.core.smart_router import SmartRouter
from agent.core.state import capped_add_messages, merge_unique_items
from agent.core.middleware import clear_tool_results_for_context
from agent.workflows.nodes import (
    _parse_json_object,
    _parse_plan_response,
    agent_tool_node,
    finalize_node,
    router_node,
)
from common.config import settings


class CoreContractTests(unittest.TestCase):
    def test_context_edit_clears_old_tool_outputs_but_keeps_recent_and_inputs(self):
        results = [
            {
                "tool_name": f"tool-{index}",
                "arguments": {"query": f"query-{index}"},
                "result": "x" * 100,
            }
            for index in range(4)
        ]

        with patch.object(settings, "context_edit_trigger_tokens", 1), patch.object(
            settings, "context_edit_keep_tools", 2
        ):
            edited = clear_tool_results_for_context(results)

        self.assertEqual(edited[0]["result"], "[cleared]")
        self.assertEqual(edited[1]["result"], "[cleared]")
        self.assertEqual(edited[2]["result"], results[2]["result"])
        self.assertEqual(edited[3]["result"], results[3]["result"])
        self.assertEqual(edited[0]["arguments"], results[0]["arguments"])
        self.assertEqual(results[0]["result"], "x" * 100)

    def test_context_edit_does_not_clear_when_under_token_threshold(self):
        results = [{"tool_name": "echo", "result": "small"}]

        with patch.object(settings, "context_edit_trigger_tokens", 100), patch.object(
            settings, "context_edit_keep_tools", 0
        ):
            edited = clear_tool_results_for_context(results)

        self.assertEqual(edited, results)

    def test_message_reducer_preserves_tool_messages_by_default(self):
        messages = [
            HumanMessage(content="运行代码"),
            ToolMessage(content="执行结果", tool_call_id="call-001"),
        ]
        with patch.object(settings, "strip_tool_messages", False):
            merged = capped_add_messages([], messages)

        self.assertEqual(len(merged), 2)
        self.assertIsInstance(merged[1], ToolMessage)

    def test_message_reducer_strips_tool_messages_before_trim_check(self):
        messages = [
            HumanMessage(content="运行代码"),
            ToolMessage(content="执行结果", tool_call_id="call-001"),
            AIMessage(content="结果是 5"),
        ]
        with patch.object(settings, "strip_tool_messages", True), patch.object(
            settings, "trim_messages", False
        ):
            merged = capped_add_messages([], messages)

        self.assertEqual(
            [message.content for message in merged],
            ["运行代码", "结果是 5"],
        )

    def test_tool_message_middleware_is_exported_from_agent_core(self):
        from agent.core import maybe_strip_tool_messages

        self.assertTrue(callable(maybe_strip_tool_messages))

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

    def test_message_reducer_summarizes_middle_history_at_threshold(self):
        class FakeModel:
            def invoke(self, messages):
                return AIMessage(content="压缩后的旧对话")

        source = [HumanMessage(content=f"消息{i}") for i in range(7)]
        with patch.object(settings, "trim_messages", True), patch.object(
            settings, "trim_messages_keep_first", 1
        ), patch.object(settings, "trim_messages_keep_last", 2), patch.object(
            settings, "summary_messages", True
        ), patch.object(settings, "summary_messages_trigger", 5), patch(
            "agent.core.message_utils.ModelFactory.create", return_value=FakeModel()
        ) as create_model:
            merged = capped_add_messages([], source)

        self.assertEqual(len(merged), 4)
        self.assertEqual(merged[0].content, "消息0")
        self.assertEqual(merged[-2].content, "消息5")
        self.assertEqual(merged[-1].content, "消息6")
        self.assertIn("压缩后的旧对话", merged[1].content)
        create_model.assert_called_once()

    def test_message_reducer_carries_previous_summary_into_next_compaction(self):
        prompts = []

        class FakeModel:
            def invoke(self, messages):
                prompts.append(messages[1].content)
                return AIMessage(content=f"第{len(prompts)}次摘要")

        initial = [HumanMessage(content=f"旧消息{i}") for i in range(7)]
        additions = [HumanMessage(content=f"新消息{i}") for i in range(5)]
        with patch.object(settings, "trim_messages", True), patch.object(
            settings, "trim_messages_keep_first", 1
        ), patch.object(settings, "trim_messages_keep_last", 2), patch.object(
            settings, "summary_messages", True
        ), patch.object(settings, "summary_messages_trigger", 5), patch(
            "agent.core.message_utils.ModelFactory.create", return_value=FakeModel()
        ):
            compacted = capped_add_messages([], initial)
            compacted = capped_add_messages(compacted, additions)

        self.assertEqual(len(prompts), 2)
        self.assertIn("第1次摘要", prompts[1])
        self.assertIn("旧消息5", prompts[1])
        self.assertEqual(compacted[1].content, "对话历史摘要：\n第2次摘要")
        self.assertEqual(compacted[-2].content, "新消息3")
        self.assertEqual(compacted[-1].content, "新消息4")

    def test_message_reducer_keeps_history_until_summary_threshold(self):
        messages = [HumanMessage(content=f"消息{i}") for i in range(4)]
        with patch.object(settings, "trim_messages", True), patch.object(
            settings, "trim_messages_keep_first", 1
        ), patch.object(settings, "trim_messages_keep_last", 1), patch.object(
            settings, "summary_messages", True
        ), patch.object(settings, "summary_messages_trigger", 5), patch(
            "agent.core.message_utils.ModelFactory.create",
            side_effect=AssertionError("summary threshold should not be reached"),
        ):
            merged = capped_add_messages([], messages)

        self.assertEqual(merged, messages)

    def test_message_reducer_falls_back_when_summary_model_fails(self):
        messages = [HumanMessage(content=f"消息{i}") for i in range(7)]
        with patch.object(settings, "trim_messages", True), patch.object(
            settings, "trim_messages_keep_first", 1
        ), patch.object(settings, "trim_messages_keep_last", 2), patch.object(
            settings, "summary_messages", True
        ), patch.object(settings, "summary_messages_trigger", 5), patch(
            "agent.core.message_utils.ModelFactory.create",
            side_effect=RuntimeError("模型不可用"),
        ):
            merged = capped_add_messages([], messages)

        self.assertEqual(len(merged), 4)
        self.assertIn("消息4", merged[1].content)
        self.assertEqual(merged[-1].content, "消息6")

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

    def test_router_does_not_repeat_request_after_transient_model_error(self):
        class TimeoutModel:
            calls = 0

            def with_structured_output(self, _schema):
                return self

            def invoke(self, _messages, config=None):
                self.calls += 1
                raise TimeoutError("model request timed out")

        model = TimeoutModel()
        with patch(
            "agent.core.smart_router.ModelFactory.for_task",
            return_value=model,
        ):
            decision = SmartRouter().route("解释缓存的作用")

        self.assertEqual(model.calls, 1)
        self.assertEqual(decision.route, "direct")
        self.assertIn("timed out", decision.reasoning)

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

    def test_graph_executes_xml_batch_in_order_before_replanning(self):
        from tools.registry import ToolDefinition, ToolRegistry

        execution_order = []

        def record(value: str) -> dict[str, str]:
            execution_order.append(value)
            return {"output": value, "error": ""}

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="record",
                    description="记录输入值",
                    handler=record,
                ),
            ]
        )

        class SequentialPlanningModel:
            def __init__(self):
                self.responses = [
                    (
                        "<function_calls>"
                        '<invoke name="record"><parameter name="value">first</parameter></invoke>'
                        '<invoke name="record"><parameter name="value">second</parameter></invoke>'
                        "</function_calls>"
                    ),
                    '{"action":"finish","final_report":"已按序执行。"}',
                ]
                self.calls = 0

            def invoke(self, messages, config=None):
                content = self.responses[self.calls]
                self.calls += 1
                return type("FakeResponse", (), {"content": content})()

        model = SequentialPlanningModel()
        graph = create_research_graph()
        with patch.object(settings, "agent_xml_tool_calling", True), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = graph.invoke(
                {"input": "依次记录 first 和 second", "messages": []},
                config={
                    "configurable": {
                        "search_mode": "agent",
                        "tool_registry": registry,
                    }
                },
            )

        self.assertEqual(execution_order, ["first", "second"])
        self.assertEqual(model.calls, 2)
        self.assertEqual(
            [item["arguments"]["value"] for item in result["tool_results"]],
            ["first", "second"],
        )
        self.assertEqual(result["status"], "completed")

    def test_graph_executes_native_tool_call_batch_and_accepts_final_text(self):
        from tools.registry import ToolDefinition, ToolRegistry

        execution_order = []

        def record(value: str) -> dict[str, str]:
            execution_order.append(value)
            return {"output": value, "error": ""}

        registry = ToolRegistry(
            [ToolDefinition("record", "记录输入值", record)]
        )

        class NativePlanningModel:
            def __init__(self):
                self.responses = [
                    {
                        "content": "",
                        "tool_calls": [
                            {"name": "record", "args": {"value": "first"}},
                            {"name": "record", "args": {"value": "second"}},
                        ],
                    },
                    {"content": "两个值都已记录。", "tool_calls": []},
                ]
                self.calls = 0
                self.bound_tool_names = []

            def bind_tools(self, tools):
                self.bound_tool_names.append(
                    [item["function"]["name"] for item in tools]
                )
                return self

            def invoke(self, messages, config=None):
                response = self.responses[self.calls]
                self.calls += 1
                return response

        model = NativePlanningModel()
        with patch.object(settings, "agent_xml_tool_calling", False), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = create_research_graph().invoke(
                {"input": "依次记录 first 和 second", "messages": []},
                config={
                    "configurable": {
                        "search_mode": "agent",
                        "tool_registry": registry,
                    }
                },
            )

        self.assertEqual(execution_order, ["first", "second"])
        self.assertEqual(model.calls, 2)
        self.assertEqual(model.bound_tool_names, [["record"], ["record"]])
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["final_report"], "两个值都已记录。")

    def test_graph_requests_approval_for_each_xml_batch_call(self):
        from tools.registry import ToolDefinition, ToolRegistry

        execution_order = []
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="record",
                    description="记录输入值",
                    handler=lambda value: (
                        execution_order.append(value)
                        or {"output": value, "error": ""}
                    ),
                    requires_approval=True,
                ),
            ]
        )

        class SequentialPlanningModel:
            def __init__(self):
                self.responses = [
                    (
                        "<function_calls>"
                        '<invoke name="record"><parameter name="value">first</parameter></invoke>'
                        '<invoke name="record"><parameter name="value">second</parameter></invoke>'
                        "</function_calls>"
                    ),
                    '{"action":"finish","final_report":"已按序执行。"}',
                ]
                self.calls = 0

            def invoke(self, messages, config=None):
                content = self.responses[self.calls]
                self.calls += 1
                return type("FakeResponse", (), {"content": content})()

        graph = create_research_graph(checkpointer=InMemorySaver())
        config = {
            "configurable": {
                "thread_id": "xml-batch-approval",
                "search_mode": "agent",
                "tool_registry": registry,
            }
        }
        with patch.object(settings, "agent_xml_tool_calling", True), patch.object(
            settings, "tool_approval", True
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=SequentialPlanningModel(),
        ):
            paused_once = graph.invoke(
                {"input": "依次记录 first 和 second", "messages": []},
                config=config,
            )
            self.assertIn("__interrupt__", paused_once)
            self.assertEqual(paused_once.get("tool_call_count", 0), 0)

            paused_twice = graph.invoke(
                Command(resume={"approved": True}),
                config=config,
            )
            self.assertIn("__interrupt__", paused_twice)
            self.assertEqual(paused_twice.get("tool_call_count"), 1)
            self.assertEqual(execution_order, ["first"])

            completed = graph.invoke(
                Command(resume={"approved": True}),
                config=config,
            )

        self.assertEqual(execution_order, ["first", "second"])
        self.assertEqual(completed["status"], "completed")

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


    def test_graph_preserves_cancelled_request(self):
        with patch(
            "agent.core.smart_router.ModelFactory.for_task",
            side_effect=AssertionError("cancelled requests must skip model routing"),
        ):
            result = create_research_graph().invoke(
                {
                    "input": "请执行一个任务",
                    "messages": [],
                    "status": "running",
                    "is_cancelled": True,
                    "cancel_token_id": "cancel-001",
                }
            )

        self.assertEqual(
            result["status"],
            "cancelled",
        )
        self.assertFalse(
            result["is_complete"],
        )


    def test_finalize_preserves_cancelled_status(self):
        result = finalize_node(
            {
                "status": "cancelled",
                "is_cancelled": True,
                "final_report": "任务已取消",
            }
        )

        self.assertEqual(
            result["status"],
            "cancelled",
        )
        self.assertFalse(
            result["is_complete"],
        )
        self.assertEqual(
            result["final_report"],
            "任务已取消",
        )


    def test_router_honors_search_mode_override(self):
        with patch(
            "agent.workflows.nodes.smart_route",
            side_effect=AssertionError(
                "explicit search_mode should bypass LLM routing"
            ),
        ):
            result = router_node(
                {
                    "input": "普通问题",
                },
                {
                    "configurable": {
                        "search_mode": {
                            "mode": "web",
                        },
                    },
                },
            )

        self.assertEqual(result["route"], "web")
        self.assertEqual(
            result["routing_confidence"],
            1.0,
        )


if __name__ == "__main__":
    unittest.main()
