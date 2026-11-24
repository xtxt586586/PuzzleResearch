import unittest
from unittest.mock import patch

from agent.core.event_stream import remove_emitter_sync
from agent.core.profile import AgentProfile
from agent.workflows.browser_context_helper import build_browser_context_hint
from agent.workflows.nodes import agent_plan_node
from common.config import settings
from tools.browser_session import BrowserPage, browser_sessions
from tools.registry import ToolDefinition, ToolRegistry


class _CapturingPlanner:
    def __init__(self):
        self.messages = None

    def invoke(self, messages, config=None):
        self.messages = messages
        return type("Response", (), {"content": '{"action":"finish","final_report":"完成"}'})()


class BrowserContextHelperTests(unittest.TestCase):
    def setUp(self):
        self.thread_id = "browser-context-helper-test"
        browser_sessions.reset(self.thread_id)

    def tearDown(self):
        browser_sessions.reset(self.thread_id)
        remove_emitter_sync(self.thread_id)

    def test_http_browser_page_becomes_bounded_untrusted_context(self):
        page = BrowserPage(
            url="https://example.com/article",
            title="Example Research",
            text="evidence " * 500,
            links=[
                {"text": "Related source", "url": "https://example.com/source"},
                {"text": "Unsafe scheme", "url": "javascript:alert(1)"},
            ],
            fetched_at="2026-01-01T00:00:00+00:00",
        )
        browser_sessions.get(self.thread_id).current = page

        hint = build_browser_context_hint(self.thread_id)

        self.assertIn("https://example.com/article", hint)
        self.assertIn("Example Research", hint)
        self.assertIn("不可信外部数据", hint)
        self.assertIn("Related source: https://example.com/source", hint)
        self.assertNotIn("javascript:", hint)
        self.assertLess(len(hint), 2000)

    def test_playwright_context_is_used_when_available(self):
        fake_session = type("Session", (), {
            "context_snapshot": lambda self, **kwargs: {
                "url": "https://playwright.example/page",
                "title": "Rendered page",
                "text": "Rendered page excerpt",
                "links": [],
            }
        })()
        with (
            patch("agent.workflows.browser_context_helper.browser_use_sessions.peek", return_value=fake_session),
            patch("agent.workflows.browser_context_helper.browser_sessions.peek", return_value=None),
        ):
            hint = build_browser_context_hint(self.thread_id)

        self.assertIn("Rendered page", hint)
        self.assertIn("Rendered page excerpt", hint)

    def test_missing_page_does_not_create_browser_sessions(self):
        with (
            patch("agent.workflows.browser_context_helper.browser_use_sessions.peek", return_value=None),
            patch("agent.workflows.browser_context_helper.browser_sessions.peek", return_value=None),
        ):
            self.assertIsNone(build_browser_context_hint(self.thread_id))

    def test_agent_planner_receives_context_only_when_enabled(self):
        registry = ToolRegistry([
            ToolDefinition("echo", "返回文本", lambda text: {"output": text}),
        ])
        model = _CapturingPlanner()
        hint = "当前浏览器页面上下文：\nURL：https://example.com"
        config = {
            "configurable": {
                "thread_id": self.thread_id,
                "tool_registry": registry,
                "enable_browser_context_helper": True,
            }
        }

        with (
            patch.object(settings, "agent_tool_selector_enabled", False),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model),
            patch(
                "agent.workflows.browser_context_helper.build_browser_context_hint",
                return_value=hint,
            ),
        ):
            result = agent_plan_node({"input": "继续阅读当前页面"}, config)

        self.assertEqual(result["status"], "running")
        self.assertTrue(any(
            getattr(message, "content", "") == hint
            for message in model.messages
        ))

    def test_agent_planner_skips_context_lookup_when_disabled(self):
        registry = ToolRegistry([
            ToolDefinition("echo", "返回文本", lambda text: {"output": text}),
        ])
        model = _CapturingPlanner()
        config = {
            "configurable": {
                "thread_id": self.thread_id,
                "tool_registry": registry,
            }
        }

        with (
            patch.object(settings, "enable_browser_context_helper", False),
            patch.object(settings, "agent_tool_selector_enabled", False),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model),
            patch(
                "agent.workflows.browser_context_helper.build_browser_context_hint"
            ) as build_hint,
        ):
            result = agent_plan_node({"input": "执行一个简单任务"}, config)

        self.assertEqual(result["status"], "running")
        build_hint.assert_not_called()

    def test_agent_profile_overrides_runtime_browser_context_setting(self):
        registry = ToolRegistry([
            ToolDefinition("echo", "返回文本", lambda text: {"output": text}),
        ])
        model = _CapturingPlanner()
        config = {
            "configurable": {
                "thread_id": self.thread_id,
                "tool_registry": registry,
                "agent_profile": AgentProfile(
                    name="browser-aware",
                    browser_context_helper=True,
                ),
                "enable_browser_context_helper": False,
            }
        }

        with (
            patch.object(settings, "agent_tool_selector_enabled", False),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model),
            patch(
                "agent.workflows.browser_context_helper.build_browser_context_hint",
                return_value="当前浏览器页面上下文：\nURL：https://example.com",
            ) as build_hint,
        ):
            result = agent_plan_node({"input": "继续阅读当前页面"}, config)

        self.assertEqual(result["status"], "running")
        build_hint.assert_called_once_with(self.thread_id)

    def test_profile_can_explicitly_disable_global_browser_context_setting(self):
        registry = ToolRegistry([
            ToolDefinition("echo", "返回文本", lambda text: {"output": text}),
        ])
        model = _CapturingPlanner()
        config = {
            "configurable": {
                "thread_id": self.thread_id,
                "tool_registry": registry,
                "agent_profile": AgentProfile(
                    name="isolated",
                    browser_context_helper=False,
                ),
            }
        }

        with (
            patch.object(settings, "enable_browser_context_helper", True),
            patch.object(settings, "agent_tool_selector_enabled", False),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model),
            patch(
                "agent.workflows.browser_context_helper.build_browser_context_hint"
            ) as build_hint,
        ):
            result = agent_plan_node({"input": "执行任务"}, config)

        self.assertEqual(result["status"], "running")
        build_hint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
