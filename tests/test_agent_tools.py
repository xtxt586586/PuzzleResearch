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


if __name__ == "__main__":
    unittest.main()
