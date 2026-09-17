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


    def test_get_state_forwards_thread_and_subgraphs(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        with patch.object(
            runtime.graph,
            "get_state",
            return_value={
                "values": {
                    "status": "running",
                },
            },
        ) as graph_get_state:
            result = runtime.get_state(
                thread_id="state-thread-001",
                subgraphs=True,
            )

        self.assertEqual(
            result["values"]["status"],
            "running",
        )

        runtime_config = graph_get_state.call_args.args[0]

        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "state-thread-001",
        )
        self.assertTrue(
            graph_get_state.call_args.kwargs["subgraphs"],
        )


    def test_get_state_requires_checkpointer(self):
        runtime = AgentRuntime()

        with self.assertRaisesRegex(
            ValueError,
            "checkpointer",
        ):
            runtime.get_state(
                thread_id="state-thread-002",
            )


    def test_get_state_history_forwards_limit(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        snapshots = [
            {"id": "snapshot-1"},
            {"id": "snapshot-2"},
        ]

        with patch.object(
            runtime.graph,
            "get_state_history",
            return_value=iter(snapshots),
        ) as graph_get_history:
            result = list(
                runtime.get_state_history(
                    thread_id="history-thread-001",
                    limit=2,
                )
            )

        self.assertEqual(
            result,
            snapshots,
        )

        runtime_config = graph_get_history.call_args.args[0]

        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "history-thread-001",
        )
        self.assertEqual(
            graph_get_history.call_args.kwargs["limit"],
            2,
        )


    def test_stream_forwards_payload_and_mode(self):
        runtime = AgentRuntime()

        chunks = [
            {
                "status": "running",
            },
            {
                "status": "completed",
            },
        ]

        with patch.object(
            runtime.graph,
            "stream",
            return_value=iter(chunks),
        ) as graph_stream:
            result = list(
                runtime.stream(
                    "流式测试请求",
                    thread_id="stream-thread-001",
                    stream_mode="updates",
                )
            )

        self.assertEqual(
            result,
            chunks,
        )

        payload = graph_stream.call_args.args[0]
        runtime_config = graph_stream.call_args.kwargs["config"]

        self.assertEqual(
            payload["input"],
            "流式测试请求",
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "stream-thread-001",
        )
        self.assertEqual(
            graph_stream.call_args.kwargs["stream_mode"],
            "updates",
        )


    async def test_aget_state_forwards_thread(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        with patch.object(
            runtime.graph,
            "aget_state",
            new_callable=AsyncMock,
            return_value={
                "values": {
                    "status": "completed",
                },
            },
        ) as graph_aget_state:
            result = await runtime.aget_state(
                thread_id="async-state-thread-001",
            )

        self.assertEqual(
            result["values"]["status"],
            "completed",
        )

        runtime_config = graph_aget_state.call_args.args[0]

        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-state-thread-001",
        )


    async def test_aget_state_history_forwards_limit(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        async def snapshot_stream():
            yield {"id": "async-snapshot-1"}
            yield {"id": "async-snapshot-2"}

        with patch.object(
            runtime.graph,
            "aget_state_history",
            return_value=snapshot_stream(),
        ) as graph_aget_history:
            result = [
                snapshot
                async for snapshot in runtime.aget_state_history(
                    thread_id="async-history-thread-001",
                    limit=2,
                )
            ]

        self.assertEqual(
            result,
            [
                {"id": "async-snapshot-1"},
                {"id": "async-snapshot-2"},
            ],
        )

        runtime_config = graph_aget_history.call_args.args[0]

        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-history-thread-001",
        )
        self.assertEqual(
            graph_aget_history.call_args.kwargs["limit"],
            2,
        )


    async def test_astream_forwards_payload_and_mode(self):
        runtime = AgentRuntime()

        async def chunk_stream():
            yield {
                "status": "running",
            }
            yield {
                "status": "completed",
            }

        with patch.object(
            runtime.graph,
            "astream",
            return_value=chunk_stream(),
        ) as graph_astream:
            result = [
                chunk
                async for chunk in runtime.astream(
                    "异步流式测试请求",
                    thread_id="async-stream-thread-001",
                    stream_mode="updates",
                )
            ]

        self.assertEqual(
            result,
            [
                {
                    "status": "running",
                },
                {
                    "status": "completed",
                },
            ],
        )

        payload = graph_astream.call_args.args[0]
        runtime_config = graph_astream.call_args.kwargs["config"]

        self.assertEqual(
            payload["input"],
            "异步流式测试请求",
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-stream-thread-001",
        )
        self.assertEqual(
            graph_astream.call_args.kwargs["stream_mode"],
            "updates",
        )


    def test_stream_events_forwards_version_and_filters(self):
        runtime = AgentRuntime()

        events = [
            {
                "event": "on_chain_start",
            },
            {
                "event": "on_chain_end",
            },
        ]

        with patch.object(
            runtime.graph,
            "stream_events",
            return_value=iter(events),
        ) as graph_stream_events:
            result = list(
                runtime.stream_events(
                    "事件流测试请求",
                    thread_id="events-thread-001",
                    version="v2",
                    include_names=["agent_tool"],
                )
            )

        self.assertEqual(result, events)

        payload = graph_stream_events.call_args.args[0]
        runtime_config = graph_stream_events.call_args.kwargs["config"]

        self.assertEqual(
            payload["input"],
            "事件流测试请求",
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "events-thread-001",
        )
        self.assertEqual(
            graph_stream_events.call_args.kwargs["version"],
            "v2",
        )
        self.assertEqual(
            graph_stream_events.call_args.kwargs["include_names"],
            ["agent_tool"],
        )


    def test_resume_events_requires_checkpointer(self):
        runtime = AgentRuntime()

        with self.assertRaisesRegex(
            ValueError,
            "checkpointer",
        ):
            runtime.resume_events(
                {"approved": True},
                thread_id="events-thread-002",
            )


    def test_resume_events_forwards_command(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        events = [
            {
                "event": "on_chain_end",
            },
        ]

        with patch.object(
            runtime.graph,
            "stream_events",
            return_value=iter(events),
        ) as graph_stream_events:
            result = list(
                runtime.resume_events(
                    {"approved": True},
                    thread_id="events-thread-003",
                )
            )

        self.assertEqual(result, events)

        command = graph_stream_events.call_args.args[0]
        runtime_config = graph_stream_events.call_args.kwargs["config"]

        self.assertIsInstance(command, Command)
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "events-thread-003",
        )


    async def test_astream_events_forwards_version(self):
        runtime = AgentRuntime()

        async def event_stream():
            yield {
                "event": "on_chain_start",
            }
            yield {
                "event": "on_chain_end",
            }

        with patch.object(
            runtime.graph,
            "astream_events",
            return_value=event_stream(),
        ) as graph_astream_events:
            result = [
                event
                async for event in runtime.astream_events(
                    "异步事件流测试请求",
                    thread_id="async-events-thread-001",
                    version="v2",
                )
            ]

        self.assertEqual(
            result,
            [
                {
                    "event": "on_chain_start",
                },
                {
                    "event": "on_chain_end",
                },
            ],
        )

        payload = graph_astream_events.call_args.args[0]
        runtime_config = graph_astream_events.call_args.kwargs["config"]

        self.assertEqual(
            payload["input"],
            "异步事件流测试请求",
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-events-thread-001",
        )
        self.assertEqual(
            graph_astream_events.call_args.kwargs["version"],
            "v2",
        )


    async def test_aresume_events_forwards_command(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        async def event_stream():
            yield {
                "event": "on_chain_end",
            }

        with patch.object(
            runtime.graph,
            "astream_events",
            return_value=event_stream(),
        ) as graph_astream_events:
            result = [
                event
                async for event in runtime.aresume_events(
                    {"approved": True},
                    thread_id="async-events-thread-002",
                )
            ]

        self.assertEqual(
            result,
            [
                {
                    "event": "on_chain_end",
                },
            ],
        )

        command = graph_astream_events.call_args.args[0]
        runtime_config = graph_astream_events.call_args.kwargs["config"]

        self.assertIsInstance(command, Command)
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-events-thread-002",
        )


if __name__ == "__main__":
    unittest.main()