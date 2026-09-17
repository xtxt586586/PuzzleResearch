import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.core.profile import AgentProfile
from agent.core.runtime import AgentRuntime
from tools.registry import ToolRegistry


class AgentRuntimeTests(unittest.TestCase):
    def test_invoke_injects_profile_registry_and_thread(self):
        profile = AgentProfile(
            name="runtime-profile",
        )
        registry = ToolRegistry()

        runtime = AgentRuntime(
            profile=profile,
            registry=registry,
        )

        with patch.object(
            runtime.graph,
            "invoke",
            return_value={
                "status": "completed",
            },
        ) as graph_invoke:
            result = runtime.invoke(
                "测试请求",
                thread_id="thread-001",
            )

        self.assertEqual(
            result["status"],
            "completed",
        )

        payload = graph_invoke.call_args.args[0]
        runtime_config = graph_invoke.call_args.kwargs["config"]

        self.assertEqual(
            payload["input"],
            "测试请求",
        )
        self.assertEqual(
            payload["messages"],
            [],
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "thread-001",
        )
        self.assertIs(
            runtime_config["configurable"]["agent_profile"],
            profile,
        )
        self.assertIs(
            runtime_config["configurable"]["tool_registry"],
            registry,
        )


    def test_invoke_rejects_empty_input(self):
        runtime = AgentRuntime()

        with self.assertRaises(ValueError):
            runtime.invoke("   ")


    def test_checkpointer_requires_thread_id(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "thread_id",
        ):
            runtime.invoke("测试请求")


    def test_resume_requires_checkpointer(self):
        runtime = AgentRuntime()

        with self.assertRaisesRegex(
            ValueError,
            "checkpointer",
        ):
            runtime.resume(
                {"approved": True},
                thread_id="thread-001",
            )


    def test_resume_builds_langgraph_command(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        with patch.object(
            runtime.graph,
            "invoke",
            return_value={
                "status": "completed",
            },
        ) as graph_invoke:
            result = runtime.resume(
                {"approved": True},
                thread_id="thread-001",
            )

        self.assertEqual(
            result["status"],
            "completed",
        )

        command = graph_invoke.call_args.args[0]
        runtime_config = graph_invoke.call_args.kwargs["config"]

        self.assertIsInstance(
            command,
            Command,
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "thread-001",
        )


if __name__ == "__main__":
    unittest.main()