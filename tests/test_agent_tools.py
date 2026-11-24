import unittest
from unittest.mock import patch

from langchain_core.messages import AIMessage

from agent.workflows.nodes import (
    _select_agent_tools,
    agent_plan_node,
    agent_tool_node,
)
from tools.registry import ToolDefinition, ToolRegistry
from agent.core.profile import AgentProfile

import asyncio

from agent.core.event_stream import (
    get_emitter_sync,
    remove_emitter_sync,
)


class FakePlanningModel:
    def __init__(self, content):
        self.content = content

    def invoke(self, messages, config=None):
        return type(
            "FakeResponse",
            (),
            {
                "content": self.content,
            },
        )()

class CapturingPlanningModel(FakePlanningModel):
    def __init__(self, content):
        super().__init__(content)
        self.messages = None

    def invoke(self, messages, config=None):
        self.messages = messages
        return super().invoke(
            messages,
            config=config,
        )


class AgentToolNodeTests(unittest.TestCase):
    def test_xml_batch_plan_validates_all_calls_and_marks_detection(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=lambda text: {"output": text, "error": ""},
                ),
            ]
        )
        model = FakePlanningModel(
            "<function_calls>"
            '<invoke name="echo"><parameter name="text">first</parameter></invoke>'
            '<invoke name="echo"><parameter name="text">second</parameter></invoke>'
            "</function_calls>"
        )

        with patch(
            "agent.workflows.nodes.settings.agent_xml_tool_calling",
            True,
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {"input": "依次返回两段文本"},
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["status"], "running")
        self.assertTrue(result["xml_tool_calls_detected"])
        self.assertEqual(
            result["pending_tool_calls"],
            [
                {"tool_name": "echo", "arguments": {"text": "first"}},
                {"tool_name": "echo", "arguments": {"text": "second"}},
            ],
        )

    def test_native_tool_call_is_bound_and_uses_registered_schema(self):
        def echo(text: str, prefix: str = "") -> dict[str, str]:
            return {"output": f"{prefix}{text}", "error": ""}

        registry = ToolRegistry(
            [
                ToolDefinition("echo", "返回输入文本", echo),
                ToolDefinition(
                    "hidden",
                    "不可用工具",
                    lambda value: {"output": value, "error": ""},
                ),
            ]
        )
        response = type(
            "FakeNativeResponse",
            (),
            {
                "content": "",
                "tool_calls": [
                    {
                        "name": "echo",
                        "args": {"text": "hello", "prefix": "native: "},
                        "id": "call-native-001",
                    }
                ],
            },
        )()

        class NativePlanningModel:
            def __init__(self):
                self.bound_tools = None

            def bind_tools(self, tools):
                self.bound_tools = tools
                return self

            def invoke(self, messages, config=None):
                return response

        model = NativePlanningModel()
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "返回 hello",
                    "enabled_tools": {"echo": True, "hidden": False},
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["status"], "running")
        self.assertEqual(
            result["pending_tool_calls"],
            [
                {
                    "tool_name": "echo",
                    "arguments": {"text": "hello", "prefix": "native: "},
                }
            ],
        )
        self.assertEqual(
            [item["function"]["name"] for item in model.bound_tools],
            ["echo"],
        )
        echo_schema = model.bound_tools[0]["function"]["parameters"]
        self.assertEqual(echo_schema["required"], ["text"])
        self.assertEqual(echo_schema["properties"]["text"]["type"], "string")
        self.assertEqual(echo_schema["properties"]["prefix"]["default"], "")

    def test_native_openai_tool_call_arguments_are_decoded(self):
        response = {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "type": "function",
                                "function": {
                                    "name": "echo",
                                    "arguments": '{"text":"from-openai"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }

        from agent.workflows.nodes import _extract_native_tool_calls

        self.assertEqual(
            _extract_native_tool_calls(response),
            [
                {
                    "tool_name": "echo",
                    "arguments": {"text": "from-openai"},
                }
            ],
        )

    def test_bound_native_model_plain_response_finishes_agent(self):
        response = {
            "choices": [
                {"message": {"content": "任务已经完成。", "tool_calls": []}}
            ]
        }

        class NativePlanningModel:
            def bind_tools(self, tools):
                return self

            def invoke(self, messages, config=None):
                return response

        registry = ToolRegistry(
            [
                ToolDefinition(
                    "echo",
                    "返回输入文本",
                    lambda text: {"output": text, "error": ""},
                ),
            ]
        )
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=NativePlanningModel(),
        ):
            result = agent_plan_node(
                {"input": "完成后回复"},
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertTrue(result["agent_finished"])
        self.assertEqual(result["final_report"], "任务已经完成。")
        self.assertEqual(result["pending_tool_calls"], [])

    def test_xml_batch_over_remaining_call_limit_is_rejected_whole(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=lambda text: {"output": text, "error": ""},
                ),
            ]
        )
        model = FakePlanningModel(
            "<function_calls>"
            '<invoke name="echo"><parameter name="text">first</parameter></invoke>'
            '<invoke name="echo"><parameter name="text">second</parameter></invoke>'
            "</function_calls>"
        )

        with patch(
            "agent.workflows.nodes.settings.agent_xml_tool_calling",
            True,
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "依次返回两段文本",
                    "tool_call_count": 1,
                    "tool_call_limit": 2,
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["status"], "failed")
        self.assertEqual(result.get("pending_tool_calls"), None)
        self.assertIn("已拒绝整批调用", result["final_report"])

    def test_xml_batch_does_not_enqueue_calls_when_later_tool_is_disabled(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=lambda text: {"output": text, "error": ""},
                ),
                ToolDefinition(
                    name="uppercase",
                    description="转换为大写",
                    handler=lambda text: {"output": text.upper(), "error": ""},
                ),
            ]
        )
        model = FakePlanningModel(
            "<function_calls>"
            '<invoke name="echo"><parameter name="text">first</parameter></invoke>'
            '<invoke name="uppercase"><parameter name="text">second</parameter></invoke>'
            "</function_calls>"
        )

        with patch(
            "agent.workflows.nodes.settings.agent_xml_tool_calling",
            True,
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "执行两个工具",
                    "enabled_tools": {"echo": True},
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["status"], "failed")
        self.assertTrue(result["xml_tool_calls_detected"])
        self.assertNotIn("pending_tool_calls", result)
        self.assertIn("工具未启用：uppercase", result["final_report"])

    def test_tool_execution_consumes_one_call_and_keeps_the_rest(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=lambda text: {"output": text, "error": ""},
                ),
            ]
        )
        pending_calls = [
            {"tool_name": "echo", "arguments": {"text": "first"}},
            {"tool_name": "echo", "arguments": {"text": "second"}},
        ]

        result = agent_tool_node(
            {
                "pending_tool_calls": pending_calls,
                "tool_approved": True,
            },
            config={"configurable": {"tool_registry": registry}},
        )

        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(result["tool_results"][0]["arguments"]["text"], "first")
        self.assertEqual(
            result["pending_tool_calls"],
            [{"tool_name": "echo", "arguments": {"text": "second"}}],
        )

    def test_agent_plan_creates_python_tool_call(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(6 * 7)"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请计算 6 乘以 7",
                    "enabled_tools": {
                        "python_executor": True,
                    },
                },
                config={},
            )

        self.assertEqual(
            result["status"],
            "running",
        )
        self.assertFalse(
            result["is_complete"],
        )
        self.assertEqual(
            result["pending_tool_calls"][0]["tool_name"],
            "python_executor",
        )
        self.assertEqual(
            result["pending_tool_calls"][0]["arguments"]["code"],
            "print(6 * 7)",
        )

    def test_disabled_tool_is_rejected(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(42)"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请执行 Python 代码",
                    "enabled_tools": {
                        "python_executor": False,
                    },
                },
                config={},
            )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "工具未启用",
            result["final_report"],
        )

    def test_tool_call_limit_is_enforced(self):
        result = agent_plan_node(
            {
                "input": "请执行 Python 代码",
                "tool_call_count": 1,
                "tool_call_limit": 1,
            },
            config={},
        )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "达到限制",
            result["final_report"],
        )

    def test_approved_tool_executes_after_plan(self):
        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "python_executor",
                        "arguments": {
                            "code": "print(42)",
                        },
                    }
                ],
                "tool_approved": True,
            },
            config={},
        )

        self.assertEqual(
            result["status"],
            "running",
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"].strip(),
            "42",
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            result["pending_tool_calls"],
            [],
        )

    def test_default_enabled_tools_allow_registered_tool(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(42)"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请执行 Python 代码",
                },
                config={},
            )

        self.assertEqual(
            result["status"],
            "running",
        )
        self.assertFalse(
            result["is_complete"],
        )
        self.assertEqual(
            result["pending_tool_calls"][0]["tool_name"],
            "python_executor",
        )

    def test_custom_tool_registry_is_used_for_plan_and_execution(self):
        def echo_tool(text: str) -> dict[str, str]:
            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo_tool,
                ),
            ]
        )

        model = FakePlanningModel(
            '{"tool_name": "echo", '
            '"arguments": {"text": "hello"}}'
        )

        tool_config = {
            "configurable": {
                "tool_registry": registry,
            }
        }

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            plan_result = agent_plan_node(
                {
                    "input": "请返回 hello",
                },
                config=tool_config,
            )

        self.assertEqual(
            plan_result["pending_tool_calls"][0]["tool_name"],
            "echo",
        )

        execution_result = agent_tool_node(
            {
                "pending_tool_calls": plan_result["pending_tool_calls"],
                "tool_approved": True,
            },
            config=tool_config,
        )

        self.assertEqual(
            execution_result["status"],
            "running",
        )
        self.assertEqual(
            execution_result["tool_results"][0]["result"]["output"],
            "hello",
        )

    def test_invalid_tool_registry_is_rejected(self):
        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "python_executor",
                        "arguments": {
                            "code": "print(1)",
                        },
                    }
                ],
            },
            config={
                "configurable": {
                    "tool_registry": object(),
                }
            },
        )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "ToolRegistry",
            result["final_report"],
        )

    def test_profile_restricts_enabled_tools(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(42)"}}'
        )

        profile = AgentProfile(
            name="echo-only",
            enabled_tools=frozenset({"echo"}),
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请执行 Python 代码",
                },
                config={
                    "configurable": {
                        "agent_profile": profile,
                    }
                },
            )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "工具未启用",
            result["final_report"],
        )

    def test_rag_search_is_disabled_without_profile_opt_in(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="rag_search",
                    description="搜索用户知识库",
                    handler=lambda query: query,
                ),
            ]
        )
        config = {
            "configurable": {
                "tool_registry": registry,
            }
        }
        model = FakePlanningModel(
            '{"tool_name": "rag_search", '
            '"arguments": {"query": "私有资料"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            planned = agent_plan_node(
                {"input": "搜索我的知识库"},
                config=config,
            )

        executed = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "rag_search",
                        "arguments": {"query": "私有资料"},
                    }
                ],
                "tool_approved": True,
            },
            config=config,
        )

        self.assertEqual(planned["status"], "failed")
        self.assertIn("工具未启用：rag_search", planned["final_report"])
        self.assertEqual(executed["status"], "failed")
        self.assertIn("工具未启用：rag_search", executed["final_report"])

    def test_rag_profile_alias_enables_search_for_plan_and_execution(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="rag_search",
                    description="搜索用户知识库",
                    handler=lambda query: {"output": query, "error": ""},
                ),
            ]
        )
        profile = AgentProfile(
            name="knowledge-enabled",
            enabled_tools=frozenset({"rag"}),
        )
        config = {
            "configurable": {
                "tool_registry": registry,
                "agent_profile": profile,
            }
        }
        model = FakePlanningModel(
            '{"tool_name": "rag_search", '
            '"arguments": {"query": "私有资料"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            planned = agent_plan_node(
                {"input": "搜索我的知识库"},
                config=config,
            )

        executed = agent_tool_node(
            {
                "pending_tool_calls": planned["pending_tool_calls"],
                "tool_approved": True,
            },
            config=config,
        )

        self.assertEqual(planned["status"], "running")
        self.assertEqual(executed["status"], "running")
        self.assertEqual(
            executed["tool_results"][0]["result"]["output"],
            "私有资料",
        )

    def test_profile_overrides_tool_call_limit(self):
        profile = AgentProfile(
            name="one-call-only",
            tool_call_limit=1,
        )

        result = agent_plan_node(
            {
                "input": "请执行 Python 代码",
                "tool_call_count": 1,
                "tool_call_limit": 20,
            },
            config={
                "configurable": {
                    "agent_profile": profile,
                }
            },
        )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "达到限制",
            result["final_report"],
        )

    def test_profile_enables_tool_approval(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(42)"}}'
        )

        profile = AgentProfile(
            name="approval-required",
            require_tool_approval=True,
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请执行 Python 代码",
                },
                config={
                    "configurable": {
                        "agent_profile": profile,
                    }
                },
            )

        self.assertEqual(
            result["status"],
            "paused",
        )
        self.assertFalse(
            result["is_complete"],
        )


    def test_edited_approval_rejects_non_candidate_tool_in_legacy_state(self):
        executions = []
        registry = ToolRegistry(
            [
                ToolDefinition(
                    "python_executor",
                    "执行 Python 代码",
                    lambda code: executions.append(("python_executor", code)) or {"output": code},
                ),
                ToolDefinition(
                    "echo",
                    "回显文本",
                    lambda text: executions.append(("echo", text)) or {"output": text},
                ),
            ]
        )
        profile = AgentProfile(
            name="legacy-approval-profile",
            require_tool_approval=True,
        )
        state = {
            "pending_tool_calls": [
                {
                    "tool_name": "python_executor",
                    "arguments": {"code": "print(1)"},
                }
            ],
            # Deliberately omit agent_tool_candidates to model an older checkpoint.
        }

        with patch(
            "agent.workflows.nodes.interrupt",
            return_value={
                "type": "edit",
                "edited_action": {
                    "name": "echo",
                    "args": {"text": "unauthorized"},
                },
            },
        ):
            result = agent_tool_node(
                state,
                config={
                    "configurable": {
                        "agent_profile": profile,
                        "tool_registry": registry,
                    }
                },
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("未进入本轮候选集", result["last_error"])
        self.assertEqual(executions, [])


    def test_edited_approval_rejects_invalid_argument_types(self):
        executions = []

        def run_python(code: str) -> dict[str, str]:
            executions.append(code)
            return {"output": code}

        registry = ToolRegistry(
            [ToolDefinition("python_executor", "执行 Python 代码", run_python)]
        )
        profile = AgentProfile(
            name="approval-argument-profile",
            require_tool_approval=True,
        )
        state = {
            "pending_tool_calls": [
                {
                    "tool_name": "python_executor",
                    "arguments": {"code": "print(1)"},
                }
            ],
            "agent_tool_candidates": ["python_executor"],
        }

        with patch(
            "agent.workflows.nodes.interrupt",
            return_value={
                "type": "edit",
                "edited_action": {
                    "name": "python_executor",
                    "args": {"code": 42},
                },
            },
        ):
            result = agent_tool_node(
                state,
                config={
                    "configurable": {
                        "agent_profile": profile,
                        "tool_registry": registry,
                    }
                },
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("参数", result["last_error"])
        self.assertEqual(executions, [])


    def test_profile_retries_transient_tool_error(self):
        attempts = {
            "count": 0,
        }

        def flaky_tool(text: str) -> dict[str, str]:
            attempts["count"] += 1

            if attempts["count"] < 3:
                raise RuntimeError("temporary failure")

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="flaky",
                    description="一个会暂时失败的测试工具",
                    handler=flaky_tool,
                ),
            ]
        )

        profile = AgentProfile(
            name="retry-profile",
            tool_retry_limit=2,
        )

        with patch("agent.workflows.nodes.settings.tool_retry", False), \
             patch("agent.workflows.nodes.settings.tool_retry_initial_delay", 0):
            result = agent_tool_node(
                {
                    "pending_tool_calls": [
                        {
                            "tool_name": "flaky",
                            "arguments": {
                                "text": "success",
                            },
                        }
                    ],
                    "tool_approved": True,
                },
                config={
                    "configurable": {
                        "tool_registry": registry,
                        "agent_profile": profile,
                    }
                },
            )

        self.assertEqual(
            result["status"],
            "running",
        )
        self.assertEqual(
            attempts["count"],
            3,
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            "success",
        )
        self.assertEqual(
            [
                event["event_type"]
                for event in result["tool_events"]
            ],
            [
                "tool_started",
                "tool_retry",
                "tool_retry",
                "tool_completed",
            ],
        )
        self.assertEqual(
            [
                event["attempt"]
                for event in result["tool_events"]
            ],
            [1, 1, 2, 3],
        )
        self.assert_tool_event_contract(
            result,
            "flaky",
            [
                "tool_started",
                "tool_retry",
                "tool_retry",
                "tool_completed",
            ],
        )

        for event in result["tool_events"][1:]:
            self.assertIn("duration_ms", event)
            self.assertGreaterEqual(
                event["duration_ms"],
                0,
            )


    def test_global_retry_retries_transient_tool_error_without_profile(self):
        attempts = {"count": 0}

        def flaky_tool(value: str) -> dict[str, str]:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("temporary failure")
            return {"output": value, "error": ""}

        registry = ToolRegistry([
            ToolDefinition("flaky", "临时失败后恢复的工具", flaky_tool),
        ])
        with patch("agent.workflows.nodes.settings.tool_retry", True), \
             patch("agent.workflows.nodes.settings.tool_retry_max_attempts", 3), \
             patch("agent.workflows.nodes.settings.tool_retry_initial_delay", 0.2), \
             patch("agent.workflows.nodes.settings.tool_retry_backoff", 1.5), \
             patch("agent.workflows.nodes.settings.tool_retry_max_delay", 0.25), \
             patch("agent.workflows.nodes.settings.tool_retry_jitter", False), \
             patch("agent.workflows.nodes.time.sleep") as sleep:
            result = agent_tool_node(
                {
                    "pending_tool_calls": [
                        {"tool_name": "flaky", "arguments": {"value": "ok"}},
                    ],
                    "tool_approved": True,
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(attempts["count"], 3)
        self.assertEqual(result["status"], "running")
        self.assertEqual(result["tool_results"][0]["result"]["output"], "ok")
        self.assertEqual(
            [event["event_type"] for event in result["tool_events"]],
            ["tool_started", "tool_retry", "tool_retry", "tool_completed"],
        )
        self.assertEqual(len(sleep.call_args_list), 2)
        self.assertAlmostEqual(sleep.call_args_list[0].args[0], 0.2)
        self.assertAlmostEqual(sleep.call_args_list[1].args[0], 0.25)

    def test_global_retry_can_be_disabled(self):
        attempts = {"count": 0}

        def failing_tool() -> dict[str, str]:
            attempts["count"] += 1
            raise RuntimeError("permanent failure")

        registry = ToolRegistry([
            ToolDefinition("failing", "总是失败的工具", failing_tool),
        ])
        with patch("agent.workflows.nodes.settings.tool_retry", False):
            result = agent_tool_node(
                {
                    "pending_tool_calls": [
                        {"tool_name": "failing", "arguments": {}},
                    ],
                    "tool_approved": True,
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(attempts["count"], 1)
        self.assertEqual(result["status"], "failed")
        self.assertEqual(
            [event["event_type"] for event in result["tool_events"]],
            ["tool_started", "tool_failed"],
        )

    def test_zero_retry_backoff_uses_a_constant_delay(self):
        attempts = {"count": 0}

        def flaky_tool() -> dict[str, str]:
            attempts["count"] += 1
            if attempts["count"] < 3:
                raise RuntimeError("temporary failure")
            return {"output": "ok", "error": ""}

        registry = ToolRegistry([
            ToolDefinition("flaky", "临时失败后恢复的工具", flaky_tool),
        ])
        with patch("agent.workflows.nodes.settings.tool_retry", True), \
             patch("agent.workflows.nodes.settings.tool_retry_max_attempts", 3), \
             patch("agent.workflows.nodes.settings.tool_retry_initial_delay", 0.2), \
             patch("agent.workflows.nodes.settings.tool_retry_backoff", 0), \
             patch("agent.workflows.nodes.settings.tool_retry_jitter", False), \
             patch("agent.workflows.nodes.time.sleep") as sleep:
            result = agent_tool_node(
                {
                    "pending_tool_calls": [
                        {"tool_name": "flaky", "arguments": {}},
                    ],
                    "tool_approved": True,
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["status"], "running")
        self.assertEqual(len(sleep.call_args_list), 2)
        self.assertAlmostEqual(sleep.call_args_list[0].args[0], 0.2)
        self.assertAlmostEqual(sleep.call_args_list[1].args[0], 0.2)



    def test_profile_blocks_blacklisted_tool(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(42)"}}'
        )

        profile = AgentProfile(
            name="no-python",
            blocked_tools=frozenset({"python_executor"}),
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请执行 Python 代码",
                },
                config={
                    "configurable": {
                        "agent_profile": profile,
                    }
                },
            )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "工具未启用",
            result["final_report"],
        )

    def test_profile_blocks_direct_tool_execution(self):
        profile = AgentProfile(
            name="no-python",
            blocked_tools=frozenset({"python_executor"}),
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "python_executor",
                        "arguments": {
                            "code": "print(42)",
                        },
                    }
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "agent_profile": profile,
                }
            },
        )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "工具被 Agent Profile 禁止",
            result["final_report"],
        )

    def test_tool_events_capture_successful_execution(self):
        def echo_tool(text: str) -> dict[str, str]:
            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo_tool,
                ),
            ]
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "echo",
                        "arguments": {
                            "text": "hello",
                        },
                    }
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "tool_registry": registry,
                }
            },
        )

        self.assertEqual(
            [
                event["event_type"]
                for event in result["tool_events"]
            ],
            [
                "tool_started",
                "tool_completed",
            ],
        )
        self.assertEqual(
            result["tool_events"][0]["tool_name"],
            "echo",
        )
        self.assertEqual(
            result["tool_events"][1]["attempt"],
            1,
        )
        self.assertGreaterEqual(
            result["tool_events"][1]["duration_ms"],
            0,
        )
        self.assert_tool_event_contract(
            result,
            "echo",
            [
                "tool_started",
                "tool_completed",
            ],
        )


    def test_profile_deduplicates_repeated_tool_plan(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"b": 2, "a": 1}}'
        )

        profile = AgentProfile(
            name="deduplicating-profile",
            deduplicate_tool_calls=True,
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请执行工具",
                    "tool_results": [
                        {
                            "tool_name": "python_executor",
                            "arguments": {
                                "a": 1,
                                "b": 2,
                            },
                            "result": {
                                "output": "previous",
                            },
                        }
                    ],
                },
                config={
                    "configurable": {
                        "agent_profile": profile,
                    }
                },
            )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "重复工具调用",
            result["final_report"],
        )


    def test_profile_blocks_direct_duplicate_execution(self):
        attempts = {
            "count": 0,
        }

        def echo_tool(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo_tool,
                ),
            ]
        )

        profile = AgentProfile(
            name="deduplicating-profile",
            deduplicate_tool_calls=True,
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "echo",
                        "arguments": {
                            "text": "hello",
                        },
                    }
                ],
                "tool_results": [
                    {
                        "tool_name": "echo",
                        "arguments": {
                            "text": "hello",
                        },
                        "result": {
                            "output": "hello",
                        },
                    }
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "tool_registry": registry,
                    "agent_profile": profile,
                }
            },
        )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "重复工具调用",
            result["final_report"],
        )
        self.assertEqual(
            attempts["count"],
            0,
        )


    def test_profile_reuses_cached_tool_result(self):
        attempts = {
            "count": 0,
        }

        def echo_tool(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo_tool,
                ),
            ]
        )

        profile = AgentProfile(
            name="cache-profile",
            cache_tool_results=True,
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "echo",
                        "arguments": {
                            "text": "hello",
                        },
                    }
                ],
                "tool_results": [
                    {
                        "tool_name": "echo",
                        "arguments": {
                            "text": "hello",
                        },
                        "result": {
                            "output": "hello",
                            "error": "",
                        },
                    }
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "tool_registry": registry,
                    "agent_profile": profile,
                }
            },
        )

        self.assertEqual(result["status"], "running")
        self.assertEqual(attempts["count"], 0)
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            "hello",
        )
        self.assertEqual(
            result["tool_events"][0]["event_type"],
            "tool_cache_hit",
        )
        self.assertEqual(result["tool_call_count"], 1)
        self.assertEqual(
            result["tool_events"][0]["duration_ms"],
            0.0,
        )
        self.assert_tool_event_contract(
            result,
            "echo",
            [
                "tool_cache_hit",
            ],
        )

        self.assertEqual(
            result["tool_results"][0]["tool_name"],
            "echo",
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            "hello",
        )


    def test_profile_does_not_cache_failed_tool_result(self):
        attempts = {
            "count": 0,
        }

        def echo_tool(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo_tool,
                ),
            ]
        )

        profile = AgentProfile(
            name="cache-profile",
            cache_tool_results=True,
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "echo",
                        "arguments": {
                            "text": "hello",
                        },
                    }
                ],
                "tool_results": [
                    {
                        "tool_name": "echo",
                        "arguments": {
                            "text": "hello",
                        },
                        "result": {
                            "output": "",
                            "error": "previous failure",
                        },
                    }
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "tool_registry": registry,
                    "agent_profile": profile,
                }
            },
        )

        self.assertEqual(result["status"], "running")
        self.assertEqual(attempts["count"], 1)
        self.assertEqual(
            [
                event["event_type"]
                for event in result["tool_events"]
            ],
            [
                "tool_started",
                "tool_completed",
            ],
        )

    def test_profile_limits_tool_result_output(self):
        attempts = {
            "count": 0,
        }

        def large_output_tool() -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": "abcdefghijklmnopqrstuvwxyz",
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="large_output",
                    description="返回较长文本",
                    handler=large_output_tool,
                ),
            ]
        )

        profile = AgentProfile(
            name="limited-result-profile",
            max_tool_result_chars=16,
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "large_output",
                        "arguments": {},
                    }
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "tool_registry": registry,
                    "agent_profile": profile,
                }
            },
        )

        output = result["tool_results"][0]["result"]["output"]

        self.assertEqual(result["status"], "running")
        self.assertEqual(attempts["count"], 1)
        self.assertLessEqual(len(output), 16)
        self.assertIn("工具结果已截断", output)
        self.assertTrue(
            result["tool_results"][0]["result"]["result_truncated"]
        )
        self.assertIn("工具结果已截断", result["final_report"])


    def test_profile_limits_tool_result_error(self):
        def failing_tool() -> dict[str, str]:
            return {
                "output": "",
                "error": "abcdefghijklmnopqrstuvwxyz",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="failing_tool",
                    description="返回较长错误信息",
                    handler=failing_tool,
                ),
            ]
        )

        profile = AgentProfile(
            name="limited-error-profile",
            max_tool_result_chars=16,
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "failing_tool",
                        "arguments": {},
                    }
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "tool_registry": registry,
                    "agent_profile": profile,
                }
            },
        )

        error_text = result["tool_results"][0]["result"]["error"]

        self.assertEqual(result["status"], "failed")
        self.assertLessEqual(len(error_text), 16)
        self.assertIn("工具结果已截断", error_text)
        self.assertTrue(
            result["tool_results"][0]["result"]["result_truncated"]
        )


    def test_profile_rejects_invalid_result_limit(self):
        with self.assertRaises(ValueError):
            AgentProfile(
                name="invalid-result-profile",
                max_tool_result_chars=0,
            )


    def test_agent_plan_rejects_invalid_tool_arguments(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请执行 Python 代码",
                },
                config={},
            )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "参数无效",
            result["final_report"],
        )


    def test_agent_plan_includes_tool_argument_contract(self):
        def echo_tool(text: str) -> dict[str, str]:
            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo_tool,
                ),
            ]
        )

        model = CapturingPlanningModel(
            '{"tool_name": "echo", '
            '"arguments": {"text": "hello"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请返回 hello",
                },
                config={
                    "configurable": {
                        "tool_registry": registry,
                    }
                },
            )

        self.assertEqual(
            result["status"],
            "running",
        )
        self.assertIsNotNone(model.messages)
        self.assertIn(
            '"name": "text"',
            model.messages[0].content,
        )
        self.assertIn(
            '"required": true',
            model.messages[0].content,
        )
        self.assertIn(
            '"annotation": "str"',
            model.messages[0].content,
        )
        self.assertIn(
            "返回输入文本",
            model.messages[0].content,
        )

    def test_agent_plan_clears_old_tool_output_only_in_model_prompt(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=lambda text: {"output": text, "error": ""},
                ),
            ]
        )
        tool_results = [
            {
                "tool_name": f"tool-{index}",
                "arguments": {"query": f"query-{index}"},
                "result": {"output": "x" * 100, "error": ""},
            }
            for index in range(4)
        ]
        model = CapturingPlanningModel(
            '{"action": "finish", "final_report": "完成"}'
        )

        with patch(
            "agent.workflows.nodes.settings.agent_tool_selector_enabled",
            False,
        ), patch(
            "agent.workflows.nodes.settings.context_edit_trigger_tokens",
            1,
        ), patch(
            "agent.workflows.nodes.settings.context_edit_keep_tools",
            2,
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "继续处理",
                    "tool_results": tool_results,
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["final_report"], "完成")
        prompt = model.messages[1].content
        self.assertEqual(prompt.count("[cleared]"), 2)
        self.assertIn("query-0", prompt)
        self.assertEqual(tool_results[0]["result"]["output"], "x" * 100)

    def test_agent_plan_adds_stuck_hint_for_repeated_assistant_results(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=lambda text: {"output": text, "error": ""},
                ),
            ]
        )
        model = CapturingPlanningModel(
            '{"action": "finish", "final_report": "完成"}'
        )

        with patch(
            "agent.workflows.nodes.settings.agent_tool_selector_enabled",
            False,
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "继续",
                    "messages": [
                        AIMessage(content="工具没有返回有效结果。"),
                        AIMessage(content="工具没有返回有效结果。"),
                    ],
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["final_report"], "完成")
        self.assertIn("连续重复", model.messages[1].content)

    def test_agent_plan_includes_task_guidance_and_custom_create_description(self):
        registry = ToolRegistry(
            [
                ToolDefinition("create_tasks", "default create", lambda sections: {}),
                ToolDefinition("view_tasks", "default view", lambda: {}),
                ToolDefinition("update_task", "default update", lambda task_id: {}),
                ToolDefinition("get_next_task", "default next", lambda: {}),
                ToolDefinition("echo", "返回输入文本", lambda text: {"output": text}),
            ]
        )
        model = CapturingPlanningModel(
            '{"action": "finish", "final_report": "完成"}'
        )

        with patch(
            "agent.workflows.nodes.settings.agent_tool_selector_enabled",
            False,
        ), patch(
            "agent.workflows.nodes.settings.enable_todo_middleware",
            True,
        ), patch(
            "agent.workflows.nodes.settings.todo_system_prompt",
            "使用定制任务规则。",
        ), patch(
            "agent.workflows.nodes.settings.todo_tool_description",
            "按定制描述创建当前会话任务。",
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {"input": "完成一个多步骤工作"},
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["final_report"], "完成")
        self.assertIn("使用定制任务规则。", model.messages[0].content)
        self.assertIn("按定制描述创建当前会话任务。", model.messages[0].content)
        self.assertIn('"name": "update_task"', model.messages[0].content)

    def test_agent_plan_disables_task_guidance_and_custom_description(self):
        registry = ToolRegistry(
            [ToolDefinition("create_tasks", "default create", lambda sections: {})]
        )
        model = CapturingPlanningModel(
            '{"action": "finish", "final_report": "完成"}'
        )

        with patch(
            "agent.workflows.nodes.settings.agent_tool_selector_enabled",
            False,
        ), patch(
            "agent.workflows.nodes.settings.enable_todo_middleware",
            False,
        ), patch(
            "agent.workflows.nodes.settings.todo_system_prompt",
            "不应注入的自定义规则。",
        ), patch(
            "agent.workflows.nodes.settings.todo_tool_description",
            "不应注入的自定义描述。",
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {"input": "简单问题"},
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["final_report"], "完成")
        self.assertNotIn("任务清单规则", model.messages[0].content)
        self.assertNotIn("不应注入", model.messages[0].content)

    def test_tool_selector_always_keeps_task_tools_when_enabled(self):
        tools = [
            ToolDefinition(name, f"{name} description", lambda: {})
            for name in (
                "create_tasks",
                "view_tasks",
                "update_task",
                "get_next_task",
                "alpha",
                "beta",
                "gamma",
            )
        ]

        class SelectionResponse:
            def invoke(self, _messages):
                return {"tools": ["alpha"]}

        class SelectorModel:
            def with_structured_output(self, _schema, *, method):
                self.method = method
                return SelectionResponse()

        model = SelectorModel()
        with patch(
            "agent.workflows.nodes.settings.agent_tool_selector_enabled",
            True,
        ), patch(
            "agent.workflows.nodes.settings.agent_tool_selector_max_tools",
            1,
        ), patch(
            "agent.workflows.nodes.settings.agent_tool_selector_always_include",
            "",
        ), patch(
            "agent.workflows.nodes.settings.enable_todo_middleware",
            True,
        ), patch(
            "agent.workflows.nodes.settings.deepseek_base_url",
            "https://api.deepseek.com",
        ):
            selected = _select_agent_tools(
                model,
                tools,
                user_input="复杂任务",
                tool_history="暂无",
            )

        self.assertEqual(
            {tool.name for tool in selected},
            {"create_tasks", "view_tasks", "update_task", "get_next_task", "alpha"},
        )

    def test_tool_selector_does_not_force_task_tools_when_disabled(self):
        tools = [
            ToolDefinition(name, f"{name} description", lambda: {})
            for name in ("create_tasks", "update_task", "alpha", "beta")
        ]

        class SelectionResponse:
            def invoke(self, _messages):
                return {"tools": ["alpha"]}

        class SelectorModel:
            def with_structured_output(self, _schema, *, method):
                return SelectionResponse()

        with patch(
            "agent.workflows.nodes.settings.agent_tool_selector_enabled",
            True,
        ), patch(
            "agent.workflows.nodes.settings.agent_tool_selector_max_tools",
            1,
        ), patch(
            "agent.workflows.nodes.settings.agent_tool_selector_always_include",
            "",
        ), patch(
            "agent.workflows.nodes.settings.enable_todo_middleware",
            False,
        ), patch(
            "agent.workflows.nodes.settings.deepseek_base_url",
            "https://api.deepseek.com",
        ):
            selected = _select_agent_tools(
                SelectorModel(),
                tools,
                user_input="简单请求",
                tool_history="暂无",
            )

        self.assertEqual([tool.name for tool in selected], ["alpha"])


    def test_agent_plan_stops_when_cancelled(self):
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task"
        ) as model_factory:
            result = agent_plan_node(
                {
                    "input": "请执行 Python 代码",
                    "is_cancelled": True,
                },
                config={},
            )

        self.assertEqual(
            result["status"],
            "cancelled",
        )
        self.assertFalse(
            result["is_complete"],
        )
        self.assertEqual(
            result["pending_tool_calls"],
            [],
        )
        model_factory.assert_not_called()


    def test_agent_tool_does_not_execute_when_cancelled(self):
        result = agent_tool_node(
            {
                "is_cancelled": True,
                "pending_tool_calls": [
                    {
                        "tool_name": "python_executor",
                        "arguments": {
                            "code": "print(42)",
                        },
                    }
                ],
            },
            config={},
        )

        self.assertEqual(
            result["status"],
            "cancelled",
        )
        self.assertFalse(
            result["is_complete"],
        )
        self.assertEqual(
            result["pending_tool_calls"],
            [],
        )

    def test_tool_selector_limits_candidates_and_keeps_always_included_tools(self):
        tools = [
            ToolDefinition(name, f"{name} tool", lambda value: value)
            for name in ("alpha", "beta", "gamma", "delta", "required")
        ]

        class StructuredSelection:
            def invoke(self, _messages):
                return {"tools": ["beta", "alpha", "gamma"]}

        class SelectorModel:
            def __init__(self):
                self.schema = None
                self.method = None

            def with_structured_output(self, schema, *, method):
                self.schema = schema
                self.method = method
                return StructuredSelection()

        model = SelectorModel()
        with patch("agent.workflows.nodes.settings.agent_tool_selector_enabled", True), \
             patch("agent.workflows.nodes.settings.agent_tool_selector_max_tools", 2), \
             patch(
                 "agent.workflows.nodes.settings.agent_tool_selector_always_include",
                 "required",
             ):
            selected = _select_agent_tools(
                model,
                tools,
                user_input="用 alpha 或 beta 完成任务",
                tool_history="暂无",
            )

        self.assertEqual(
            [tool.name for tool in selected],
            ["alpha", "beta", "required"],
        )
        self.assertEqual(
            model.schema["properties"]["tools"]["items"]["enum"],
            ["alpha", "beta", "gamma", "delta"],
        )

    def test_tool_selector_falls_back_to_all_tools_on_provider_errors(self):
        tools = [
            ToolDefinition(name, f"{name} tool", lambda value: value)
            for name in ("alpha", "beta", "gamma", "delta")
        ]

        class BrokenSelection:
            def invoke(self, _messages):
                raise RuntimeError("structured output unavailable")

        class SelectorModel:
            def __init__(self):
                self.methods = []

            def with_structured_output(self, _schema, *, method):
                self.methods.append(method)
                return BrokenSelection()

        model = SelectorModel()
        with patch("agent.workflows.nodes.settings.agent_tool_selector_enabled", True), \
             patch("agent.workflows.nodes.settings.agent_tool_selector_max_tools", 2), \
             patch("agent.workflows.nodes.settings.deepseek_base_url", "https://api.deepseek.com"):
            selected = _select_agent_tools(
                model,
                tools,
                user_input="完成任务",
                tool_history="暂无",
            )

        self.assertIs(selected, tools)
        self.assertEqual(
            model.methods,
            ["json_mode", "function_calling", "json_schema"],
        )

    def test_agent_plan_binds_only_selected_native_tool_schemas(self):
        tools = [
            ToolDefinition(name, f"{name} tool", lambda value: value)
            for name in ("needed", "other", "unused", "extra")
        ]

        class SelectionResponse:
            def invoke(self, _messages):
                return {"tools": ["needed"]}

        class SelectedToolModel:
            def __init__(self):
                self.bound_names = []
                self.planner_messages = None

            def with_structured_output(self, _schema, *, method):
                self.selection_method = method
                return SelectionResponse()

            def bind_tools(self, schemas):
                self.bound_names = [item["function"]["name"] for item in schemas]
                return self

            def invoke(self, messages, config=None):
                self.planner_messages = messages
                return type(
                    "FakeNativeResponse",
                    (),
                    {
                        "content": "",
                        "tool_calls": [
                            {"name": "needed", "args": {"value": "ok"}},
                        ],
                    },
                )()

        model = SelectedToolModel()
        with patch("agent.workflows.nodes.settings.agent_tool_selector_enabled", True), \
             patch("agent.workflows.nodes.settings.agent_tool_selector_max_tools", 1), \
             patch(
                 "agent.workflows.nodes.ModelFactory.for_task",
                 return_value=model,
             ):
            result = agent_plan_node(
                {"input": "执行需要的工具"},
                config={
                    "configurable": {
                        "tool_registry": ToolRegistry(tools),
                    }
                },
            )

        self.assertEqual(model.bound_names, ["needed"])
        self.assertIn('"name": "needed"', model.planner_messages[0].content)
        self.assertNotIn('"name": "other"', model.planner_messages[0].content)
        self.assertEqual(
            result["pending_tool_calls"],
            [{"tool_name": "needed", "arguments": {"value": "ok"}}],
        )

    def test_agent_plan_rejects_tool_outside_selected_candidates(self):
        tools = [
            ToolDefinition(name, f"{name} tool", lambda value: value)
            for name in ("needed", "other", "unused", "extra")
        ]

        class SelectionResponse:
            def invoke(self, _messages):
                return {"tools": ["needed"]}

        class UnselectedToolModel:
            def with_structured_output(self, _schema, *, method):
                return SelectionResponse()

            def bind_tools(self, _schemas):
                return self

            def invoke(self, _messages, config=None):
                return type(
                    "FakeNativeResponse",
                    (),
                    {
                        "content": "",
                        "tool_calls": [
                            {"name": "other", "args": {"value": "blocked"}},
                        ],
                    },
                )()

        with patch("agent.workflows.nodes.settings.agent_tool_selector_enabled", True), \
             patch("agent.workflows.nodes.settings.agent_tool_selector_max_tools", 1), \
             patch(
                 "agent.workflows.nodes.ModelFactory.for_task",
                 return_value=UnselectedToolModel(),
             ):
            result = agent_plan_node(
                {"input": "执行需要的工具"},
                config={
                    "configurable": {
                        "tool_registry": ToolRegistry(tools),
                    }
                },
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("工具未进入本轮候选集：other", result["final_report"])

    def test_agent_plan_includes_machine_readable_tool_schema(self):
        def echo_tool(text: str) -> dict[str, str]:
            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo_tool,
                ),
            ]
        )

        model = CapturingPlanningModel(
            '{"tool_name": "echo", '
            '"arguments": {"text": "hello"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {
                    "input": "请返回 hello",
                },
                config={
                    "configurable": {
                        "tool_registry": registry,
                    }
                },
            )

        self.assertEqual(
            result["status"],
            "running",
        )
        self.assertIn(
            '"name": "text"',
            model.messages[0].content,
        )
        self.assertIn(
            '"required": true',
            model.messages[0].content,
        )
        self.assertIn(
            '"annotation": "str"',
            model.messages[0].content,
        )
        self.assertIn(
            '"description": "返回输入文本"',
            model.messages[0].content,
        )


    def test_agent_plan_and_execution_share_optional_argument_contract(self):
        def format_text(
                text: str,
                prefix: str = ">>",
        ) -> dict[str, str]:
            return {
                "output": f"{prefix}{text}",
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="format_text",
                    description="为文本添加前缀",
                    handler=format_text,
                ),
            ]
        )

        model = FakePlanningModel(
            '{"tool_name": "format_text", '
            '"arguments": {"text": "hello"}}'
        )

        tool_config = {
            "configurable": {
                "tool_registry": registry,
            }
        }

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            plan_result = agent_plan_node(
                {
                    "input": "请给 hello 添加前缀",
                },
                config=tool_config,
            )

        self.assertEqual(
            plan_result["status"],
            "running",
        )
        self.assertEqual(
            plan_result["pending_tool_calls"][0]["arguments"],
            {
                "text": "hello",
            },
        )

        execution_result = agent_tool_node(
            {
                "pending_tool_calls": plan_result["pending_tool_calls"],
                "tool_approved": True,
            },
            config=tool_config,
        )

        self.assertEqual(
            execution_result["status"],
            "running",
        )
        self.assertEqual(
            execution_result["tool_results"][0]["result"]["output"],
            ">>hello",
        )


    def test_agent_plan_and_execution_reject_same_invalid_arguments(self):
        attempts = {
            "count": 0,
        }

        def echo_text(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo_text",
                    description="返回输入文本",
                    handler=echo_text,
                ),
            ]
        )

        invalid_arguments = {
            "text": "hello",
            "unexpected": True,
        }

        model = FakePlanningModel(
            '{"tool_name": "echo_text", '
            '"arguments": {"text": "hello", "unexpected": true}}'
        )

        tool_config = {
            "configurable": {
                "tool_registry": registry,
            }
        }

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            plan_result = agent_plan_node(
                {
                    "input": "请返回 hello",
                },
                config=tool_config,
            )

        self.assertEqual(
            plan_result["status"],
            "failed",
        )
        self.assertIn(
            "参数无效",
            plan_result["final_report"],
        )

        execution_result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "echo_text",
                        "arguments": invalid_arguments,
                    },
                ],
                "tool_approved": True,
            },
            config=tool_config,
        )

        self.assertEqual(
            execution_result["status"],
            "failed",
        )
        self.assertIn(
            "参数无效",
            execution_result["final_report"],
        )
        self.assertEqual(
            attempts["count"],
            0,
        )

    def test_agent_plan_and_execution_reject_same_invalid_argument_type(self):
        attempts = {
            "count": 0,
        }

        def echo_text(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo_text",
                    description="返回输入文本",
                    handler=echo_text,
                ),
            ]
        )

        model = FakePlanningModel(
            '{"tool_name": "echo_text", '
            '"arguments": {"text": 42}}'
        )

        tool_config = {
            "configurable": {
                "tool_registry": registry,
            }
        }

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            plan_result = agent_plan_node(
                {
                    "input": "请返回文本",
                },
                config=tool_config,
            )

        self.assertEqual(
            plan_result["status"],
            "failed",
        )
        self.assertIn(
            "类型无效",
            plan_result["final_report"],
        )

        execution_result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "echo_text",
                        "arguments": {
                            "text": 42,
                        },
                    },
                ],
                "tool_approved": True,
            },
            config=tool_config,
        )

        self.assertEqual(
            execution_result["status"],
            "failed",
        )
        self.assertIn(
            "类型无效",
            execution_result["final_report"],
        )
        self.assertEqual(
            attempts["count"],
            0,
        )


    def test_scalar_tool_result_is_normalized(self):
        def multiply(value: int) -> int:
            return value * 2

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="multiply",
                    description="计算两倍数值",
                    handler=multiply,
                ),
            ]
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "multiply",
                        "arguments": {
                            "value": 21,
                        },
                    },
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "tool_registry": registry,
                },
            },
        )

        self.assertEqual(
            result["status"],
            "running",
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            42,
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["error"],
            "",
        )
        self.assertEqual(
            result["tool_events"][-1]["event_type"],
            "tool_completed",
        )


    def test_error_tool_result_sets_failed_status(self):
        def failing_tool() -> dict[str, str]:
            return {
                "output": "",
                "error": "工具内部测试错误",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="failing_tool",
                    description="返回测试错误",
                    handler=failing_tool,
                ),
            ]
        )

        result = agent_tool_node(
            {
                "pending_tool_calls": [
                    {
                        "tool_name": "failing_tool",
                        "arguments": {},
                    },
                ],
                "tool_approved": True,
            },
            config={
                "configurable": {
                    "tool_registry": registry,
                },
            },
        )

        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "工具内部测试错误",
            result["errors"],
        )
        self.assertIn(
            "工具内部测试错误",
            result["final_report"],
        )
        self.assertEqual(
            result["tool_events"][-1]["event_type"],
            "tool_failed",
        )
        self.assert_tool_event_contract(
            result,
            "failing_tool",
            [
                "tool_started",
                "tool_failed",
            ],
        )


    def assert_tool_event_contract(
            self,
            result,
            tool_name,
            expected_event_types,
    ):
        events = result["tool_events"]

        self.assertEqual(
            [
                event["event_type"]
                for event in events
            ],
            expected_event_types,
        )

        for event in events:
            self.assertEqual(
                event["tool_name"],
                tool_name,
            )
            self.assertIsInstance(
                event["call_count"],
                int,
            )
            self.assertGreaterEqual(
                event["call_count"],
                1,
            )
            self.assertIsInstance(
                event["attempt"],
                int,
            )
            self.assertGreaterEqual(
                event["attempt"],
                1,
            )

    def test_tool_execution_publishes_runtime_events(self):
        thread_id = "thread-tool-events-001"

        def echo_tool(text: str) -> dict[str, str]:
            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo_tool,
                ),
            ]
        )

        try:
            result = agent_tool_node(
                {
                    "pending_tool_calls": [
                        {
                            "tool_name": "echo",
                            "arguments": {
                                "text": "hello",
                            },
                        },
                    ],
                    "tool_approved": True,
                },
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "tool_registry": registry,
                    },
                },
            )

            self.assertEqual(
                result["status"],
                "running",
            )

            emitter = get_emitter_sync(thread_id)
            events = emitter.buffered_events()

            self.assertEqual(
                [
                    event.event_type
                    for event in events
                ],
                [
                    "tool_start",
                    "tool_result",
                ],
            )

            self.assertEqual(
                events[0].data["tool"],
                "echo",
            )
            self.assertEqual(
                events[1].data["result"]["output"],
                "hello",
            )

        finally:
            remove_emitter_sync(thread_id)

    def test_failed_tool_publishes_tool_error_event(self):
        thread_id = "thread-tool-events-002"

        def failing_tool() -> dict[str, str]:
            return {
                "output": "",
                "error": "测试工具错误",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="failing_tool",
                    description="返回错误",
                    handler=failing_tool,
                ),
            ]
        )

        try:
            result = agent_tool_node(
                {
                    "pending_tool_calls": [
                        {
                            "tool_name": "failing_tool",
                            "arguments": {},
                        },
                    ],
                    "tool_approved": True,
                },
                config={
                    "configurable": {
                        "thread_id": thread_id,
                        "tool_registry": registry,
                    },
                },
            )

            self.assertEqual(
                result["status"],
                "failed",
            )

            emitter = get_emitter_sync(thread_id)
            events = emitter.buffered_events()

            self.assertEqual(
                events[-1].event_type,
                "tool_error",
            )
            self.assertEqual(
                events[-1].data["tool"],
                "failing_tool",
            )
            self.assertIn(
                "测试工具错误",
                events[-1].data["error"],
            )

        finally:
            remove_emitter_sync(thread_id)


if __name__ == "__main__":
    unittest.main()
