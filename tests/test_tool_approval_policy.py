import unittest
from unittest.mock import patch

from agent.core.profile import AgentProfile
from agent.workflows.nodes import agent_plan_node, agent_tool_node
from common.config import settings
from tools.builtin import create_default_registry
from tools.registry import ToolDefinition, ToolRegistry


class FakePlanningModel:
    def __init__(self, content: str):
        self.content = content

    def invoke(self, messages, config=None):
        return type("FakeResponse", (), {"content": self.content})()


class ToolApprovalPolicyTests(unittest.TestCase):
    def test_builtin_registry_marks_high_risk_and_read_only_tools(self):
        registry = create_default_registry()

        self.assertTrue(registry.get("python_executor").requires_approval)
        self.assertTrue(registry.get("write_file").requires_approval)
        self.assertFalse(registry.get("web_search").requires_approval)
        self.assertFalse(registry.get("read_file").requires_approval)

    def test_global_approval_does_not_pause_read_only_tool_plan(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    "web_search",
                    "只读搜索",
                    lambda query: {"output": query},
                )
            ]
        )
        model = FakePlanningModel(
            '{"tool_name":"web_search","arguments":{"query":"test"}}'
        )

        with patch.object(settings, "tool_approval", True), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {"input": "搜索 test"},
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["status"], "running")

    def test_global_approval_pauses_high_risk_tool_plan(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    "python_executor",
                    "执行代码",
                    lambda code: {"output": code},
                    requires_approval=True,
                )
            ]
        )
        model = FakePlanningModel(
            '{"tool_name":"python_executor","arguments":{"code":"print(1)"}}'
        )

        with patch.object(settings, "tool_approval", True), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_plan_node(
                {"input": "执行代码"},
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["status"], "paused")

    def test_local_command_tools_require_approval_even_when_profile_disables_it(self):
        for tool_name in ("execute_command", "install_package"):
            with self.subTest(tool_name=tool_name):
                registry = ToolRegistry(
                    [
                        ToolDefinition(
                            tool_name,
                            "需要人工确认的本机操作",
                            lambda **kwargs: {"output": "done"},
                        )
                    ]
                )
                model = FakePlanningModel(
                    '{"tool_name":"'
                    + tool_name
                    + '","arguments":{"command":"echo test",'
                    '"packages":"example"}}'
                )

                with patch.object(settings, "tool_approval", False), patch(
                    "agent.workflows.nodes.ModelFactory.for_task",
                    return_value=model,
                ):
                    result = agent_plan_node(
                        {"input": "执行本机操作"},
                        config={
                            "configurable": {
                                "tool_registry": registry,
                                "agent_profile": AgentProfile(
                                    name="no-approval",
                                    require_tool_approval=False,
                                ),
                            }
                        },
                    )

                self.assertEqual(result["status"], "paused")

    def test_global_approval_executes_read_only_tool_without_interrupt(self):
        calls = []
        registry = ToolRegistry(
            [
                ToolDefinition(
                    "web_search",
                    "只读搜索",
                    lambda query: calls.append(query) or {"output": query},
                )
            ]
        )

        with patch.object(settings, "tool_approval", True), patch(
            "agent.workflows.nodes.interrupt",
            side_effect=AssertionError("只读工具不应触发审批"),
        ):
            result = agent_tool_node(
                {
                    "pending_tool_calls": [
                        {
                            "tool_name": "web_search",
                            "arguments": {"query": "test"},
                        }
                    ],
                    "agent_tool_candidates": ["web_search"],
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(calls, ["test"])
        self.assertEqual(result["status"], "running")

    def test_profile_approval_override_keeps_all_or_none_behavior(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    "web_search",
                    "只读搜索",
                    lambda query: {"output": query},
                ),
                ToolDefinition(
                    "python_executor",
                    "执行代码",
                    lambda code: {"output": code},
                    requires_approval=True,
                ),
            ]
        )
        state = {
            "pending_tool_calls": [
                {"tool_name": "web_search", "arguments": {"query": "x"}}
            ],
            "agent_tool_candidates": ["web_search", "python_executor"],
        }

        with patch.object(settings, "tool_approval", False), patch(
            "agent.workflows.nodes.interrupt", return_value=True
        ) as interrupt_mock:
            agent_tool_node(
                state,
                config={
                    "configurable": {
                        "tool_registry": registry,
                        "agent_profile": AgentProfile(
                            name="force-approval",
                            require_tool_approval=True,
                        ),
                    }
                },
            )
            self.assertEqual(interrupt_mock.call_count, 1)

        risky_state = {
            **state,
            "pending_tool_calls": [
                {"tool_name": "python_executor", "arguments": {"code": "1"}}
            ],
        }
        with patch.object(settings, "tool_approval", True), patch(
            "agent.workflows.nodes.interrupt",
            side_effect=AssertionError("Profile false 应覆盖全局审批"),
        ):
            result = agent_tool_node(
                risky_state,
                config={
                    "configurable": {
                        "tool_registry": registry,
                        "agent_profile": AgentProfile(
                            name="skip-approval",
                            require_tool_approval=False,
                        ),
                    }
                },
            )

        self.assertEqual(result["status"], "running")


if __name__ == "__main__":
    unittest.main()
