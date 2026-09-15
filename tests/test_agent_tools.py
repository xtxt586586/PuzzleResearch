import unittest
from unittest.mock import patch

from agent.workflows.nodes import agent_tool_node
from common.config import settings


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
    def test_python_tool_can_be_called(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(6 * 7)"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_tool_node(
                {
                    "input": "请计算 6 乘以 7",
                    "enabled_tools": {
                        "python_executor": True,
                    },
                },
                config={},
            )

        self.assertEqual(result["status"], "running")
        self.assertEqual(
            result["tool_results"][0]["result"]["output"].strip(),
            "42",
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
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
            result = agent_tool_node(
                {
                    "input": "请执行 Python 代码",
                    "enabled_tools": {
                        "python_executor": False,
                    },
                },
                config={},
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("工具未启用", result["final_report"])

    def test_tool_call_limit_is_enforced(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(42)"}}'
        )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_tool_node(
                {
                    "input": "请执行 Python 代码",
                    "tool_call_count": 1,
                    "tool_call_limit": 1,
                },
                config={},
            )

        self.assertEqual(result["status"], "failed")
        self.assertIn("达到限制", result["final_report"])

    def test_tool_approval_pauses_before_execution(self):
        model = FakePlanningModel(
            '{"tool_name": "python_executor", '
            '"arguments": {"code": "print(42)"}}'
        )

        with patch.object(
            settings,
            "tool_approval",
            True,
        ), patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = agent_tool_node(
                {
                    "input": "请执行 Python 代码",
                },
                config={},
            )

        self.assertEqual(result["status"], "paused")
        self.assertFalse(result["is_complete"])
        self.assertEqual(
            len(result["pending_tool_calls"]),
            1,
        )
        self.assertEqual(
            result["pending_tool_calls"][0]["tool_name"],
            "python_executor",
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
            result = agent_tool_node(
                {
                    "input": "请执行 Python 代码",
                },
                config={},
            )

        self.assertEqual(result["status"], "running")
        self.assertEqual(
            result["tool_results"][0]["result"]["output"].strip(),
            "42",
        )




if __name__ == "__main__":
    unittest.main()