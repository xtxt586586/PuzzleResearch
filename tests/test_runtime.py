import unittest
from unittest.mock import AsyncMock, patch

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



class AgentRuntimeAsyncTests(
        unittest.IsolatedAsyncioTestCase,
):
    async def test_ainvoke_forwards_payload_and_config(self):
        runtime = AgentRuntime()

        with patch.object(
            runtime.graph,
            "ainvoke",
            new_callable=AsyncMock,
            return_value={
                "status": "completed",
            },
        ) as graph_ainvoke:
            result = await runtime.ainvoke(
                "异步测试请求",
                thread_id="async-thread-001",
            )

        self.assertEqual(
            result["status"],
            "completed",
        )

        payload = graph_ainvoke.call_args.args[0]
        runtime_config = graph_ainvoke.call_args.kwargs["config"]

        self.assertEqual(
            payload["input"],
            "异步测试请求",
        )
        self.assertEqual(
            payload["messages"],
            [],
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-thread-001",
        )


    async def test_aresume_forwards_command_and_config(self):
        from langgraph.checkpoint.memory import InMemorySaver

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        with patch.object(
            runtime.graph,
            "ainvoke",
            new_callable=AsyncMock,
            return_value={
                "status": "completed",
            },
        ) as graph_ainvoke:
            result = await runtime.aresume(
                {"approved": True},
                thread_id="async-thread-002",
            )

        self.assertEqual(
            result["status"],
            "completed",
        )

        command = graph_ainvoke.call_args.args[0]
        runtime_config = graph_ainvoke.call_args.kwargs["config"]

        self.assertIsInstance(
            command,
            Command,
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-thread-002",
        )


if __name__ == "__main__":
    unittest.main()