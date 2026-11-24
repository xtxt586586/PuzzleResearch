from __future__ import annotations

import unittest
from unittest.mock import patch

from agent.workflows.nodes import agent_tool_node
from tools.builtin import create_default_registry
from tools.web_search_tool import web_search


class FakeSearchManager:
    def __init__(self, runtime_state=None):
        self.runtime_state = runtime_state
        self.calls = []

    def search(self, query, max_results=5):
        self.calls.append((query, max_results))
        return [
            {
                "title": "官方资料",
                "url": "https://example.org/guide",
                "snippet": "介绍关键概念与使用条件。",
                "content": "完整正文",
                "provider": "duckduckgo",
            }
        ][:max_results]


class WebSearchToolTests(unittest.TestCase):
    def test_web_search_formats_results_and_uses_shared_search_state(self):
        manager = FakeSearchManager()
        with patch(
            "tools.web_search_tool.SearchManager",
            return_value=manager,
        ) as manager_factory, patch(
            "tools.web_search_tool.get_default_search_runtime_state",
            return_value="shared-state",
        ):
            result = web_search("  储能技术  ", max_results=3)

        manager_factory.assert_called_once_with(runtime_state="shared-state")
        self.assertEqual(manager.calls, [("储能技术", 3)])
        self.assertIn("官方资料", result["output"])
        self.assertIn("https://example.org/guide", result["output"])
        self.assertIn("介绍关键概念与使用条件。", result["output"])
        self.assertEqual(result["error"], "")
        self.assertEqual(result["results"][0]["provider"], "duckduckgo")
        self.assertNotIn("content", result["results"][0])

    def test_invalid_query_or_result_limit_is_rejected_before_search(self):
        with patch("tools.web_search_tool.SearchManager") as manager_factory:
            with self.assertRaisesRegex(ValueError, "不能为空"):
                web_search(" ")
            with self.assertRaisesRegex(ValueError, "1 到 10"):
                web_search("query", max_results=11)
            with self.assertRaisesRegex(ValueError, "必须是 1 到 10"):
                web_search("query", max_results=True)

        manager_factory.assert_not_called()

    def test_default_registry_exposes_web_search_schema(self):
        registry = create_default_registry()
        tool = registry.get("web_search")

        self.assertEqual(
            [parameter["name"] for parameter in tool.argument_schema()],
            ["query", "max_results"],
        )

    def test_agent_tool_node_receives_search_output(self):
        manager = FakeSearchManager()
        registry = create_default_registry()

        with patch(
            "tools.web_search_tool.SearchManager",
            return_value=manager,
        ):
            result = agent_tool_node(
                {
                    "pending_tool_calls": [
                        {
                            "tool_name": "web_search",
                            "arguments": {"query": "储能技术"},
                        }
                    ],
                    "tool_approved": True,
                },
                config={"configurable": {"tool_registry": registry}},
            )

        self.assertEqual(result["status"], "running")
        self.assertIn("官方资料", result["final_report"])
        self.assertEqual(
            result["tool_events"][-1]["event_type"],
            "tool_completed",
        )


if __name__ == "__main__":
    unittest.main()
