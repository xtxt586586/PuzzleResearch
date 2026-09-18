import unittest
from unittest.mock import patch

from agent.workflows.nodes import (
    agent_plan_node,
    agent_tool_node,
)
from tools.registry import ToolDefinition, ToolRegistry
from agent.core.profile import AgentProfile


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

        for event in result["tool_events"][1:]:
            self.assertIn("duration_ms", event)
            self.assertGreaterEqual(
                event["duration_ms"],
                0,
            )



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



if __name__ == "__main__":
    unittest.main()
