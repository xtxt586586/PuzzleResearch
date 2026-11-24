from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from tools.planning_tool import build_planning_tool_definition, plan_steps


class PlanningToolTests(unittest.TestCase):
    def test_plan_steps_parses_json_and_respects_max_steps(self):
        model = MagicMock()
        model.invoke.return_value = SimpleNamespace(
            content='{"steps": ["确认研究问题", "收集证据", "整理结论", "复核引用"]}'
        )

        with patch(
            "tools.planning_tool.ModelFactory.for_task",
            return_value=model,
        ) as for_task:
            result = plan_steps("比较两种储能技术", max_steps=3)

        self.assertEqual(
            result["steps"],
            ["确认研究问题", "收集证据", "整理结论"],
        )
        self.assertIn("3. 整理结论", result["output"])
        for_task.assert_called_once_with("planning", temperature=0.2)

    def test_plan_steps_accepts_numbered_text_fallback(self):
        model = MagicMock()
        model.invoke.return_value = SimpleNamespace(
            content="1. 搜集资料\n2) 对照证据\n3、撰写结论"
        )
        with patch(
            "tools.planning_tool.ModelFactory.for_task",
            return_value=model,
        ):
            result = plan_steps("撰写分析", max_steps=2)

        self.assertEqual(result["steps"], ["搜集资料", "对照证据"])

    def test_invalid_input_is_rejected_before_model_call(self):
        with patch("tools.planning_tool.ModelFactory.for_task") as for_task:
            with self.assertRaises(ValueError):
                plan_steps("", max_steps=5)
            with self.assertRaises(ValueError):
                plan_steps("目标", max_steps=11)

        for_task.assert_not_called()

    def test_tool_definition_is_invokable_from_registry(self):
        model = MagicMock()
        model.invoke.return_value = SimpleNamespace(
            content='{"steps": ["执行第一步"]}'
        )
        with patch(
            "tools.planning_tool.ModelFactory.for_task",
            return_value=model,
        ):
            result = build_planning_tool_definition().invoke(
                {"goal": "完成目标"}
            )

        self.assertEqual(result["steps"], ["执行第一步"])
        self.assertEqual(result["error"], "")


if __name__ == "__main__":
    unittest.main()
