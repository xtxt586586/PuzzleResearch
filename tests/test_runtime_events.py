import unittest
from unittest.mock import AsyncMock, patch

from agent.core.events import RuntimeEvent
from agent.core.runtime import AgentRuntime


class RuntimeEventTests(unittest.TestCase):
    def test_from_raw_preserves_event_fields(self):
        event = RuntimeEvent.from_raw(
            {
                "event": "on_chain_end",
                "name": "agent_tool",
                "run_id": "run-001",
                "parent_ids": ["parent-001"],
                "tags": ["agent"],
                "metadata": {
                    "thread_id": "thread-001",
                },
                "data": {
                    "status": "completed",
                },
            }
        )

        self.assertEqual(
            event.event,
            "on_chain_end",
        )
        self.assertEqual(
            event.name,
            "agent_tool",
        )
        self.assertEqual(
            event.run_id,
            "run-001",
        )
        self.assertEqual(
            event.parent_ids,
            ("parent-001",),
        )
        self.assertEqual(
            event.tags,
            ("agent",),
        )
        self.assertEqual(
            event.metadata["thread_id"],
            "thread-001",
        )
        self.assertEqual(
            event.data["status"],
            "completed",
        )


    def test_from_raw_handles_non_mapping(self):
        event = RuntimeEvent.from_raw(
            "unexpected-event"
        )

        self.assertEqual(
            event.event,
            "unknown",
        )
        self.assertEqual(
            event.data,
            "unexpected-event",
        )


    def test_to_dict_returns_json_like_data(self):
        event = RuntimeEvent(
            event="on_chain_start",
            name="router",
            run_id="run-002",
            parent_ids=("parent-002",),
            tags=("route",),
            metadata={
                "thread_id": "thread-002",
            },
            data={
                "input": "hello",
            },
        )

        result = event.to_dict()

        self.assertEqual(
            result["event"],
            "on_chain_start",
        )
        self.assertEqual(
            result["parent_ids"],
            ["parent-002"],
        )
        self.assertEqual(
            result["tags"],
            ["route"],
        )


class RuntimeEventStreamTests(unittest.TestCase):
    def test_stream_runtime_events_normalizes_and_filters(self):
        runtime = AgentRuntime()

        raw_events = [
            {
                "event": "on_chain_start",
                "name": "router",
            },
            {
                "event": "on_chain_end",
                "name": "finalize",
            },
        ]

        with patch.object(
            runtime.graph,
            "stream_events",
            return_value=iter(raw_events),
        ):
            result = list(
                runtime.stream_runtime_events(
                    "测试事件",
                    thread_id="runtime-event-001",
                    event_types={
                        "on_chain_end",
                    },
                )
            )

        self.assertEqual(
            len(result),
            1,
        )
        self.assertIsInstance(
            result[0],
            RuntimeEvent,
        )
        self.assertEqual(
            result[0].event,
            "on_chain_end",
        )
        self.assertEqual(
            result[0].name,
            "finalize",
        )


    def test_tool_events_are_read_from_nested_output(self):
        event = RuntimeEvent.from_raw(
            {
                "event": "on_chain_end",
                "name": "agent_tool",
                "data": {
                    "output": {
                        "tool_events": [
                            {
                                "event_type": "tool_started",
                                "tool_name": "python_executor",
                                "call_count": 1,
                                "attempt": 1,
                            },
                            {
                                "event_type": "tool_completed",
                                "tool_name": "python_executor",
                                "call_count": 1,
                                "attempt": 1,
                            },
                        ],
                    },
                },
            }
        )

        self.assertEqual(
            event.tool_events,
            [
                {
                    "event_type": "tool_started",
                    "tool_name": "python_executor",
                    "call_count": 1,
                    "attempt": 1,
                },
                {
                    "event_type": "tool_completed",
                    "tool_name": "python_executor",
                    "call_count": 1,
                    "attempt": 1,
                },
            ],
        )

    def test_stream_runtime_events_exposes_tool_events(self):
        runtime = AgentRuntime()

        raw_events = [
            {
                "event": "on_chain_end",
                "name": "agent_tool",
                "data": {
                    "output": {
                        "tool_events": [
                            {
                                "event_type": "tool_cache_hit",
                                "tool_name": "echo",
                                "call_count": 2,
                                "attempt": 1,
                            },
                        ],
                    },
                },
            },
        ]

        with patch.object(
            runtime.graph,
            "stream_events",
            return_value=iter(raw_events),
        ):
            result = list(
                runtime.stream_runtime_events(
                    "读取工具事件",
                    thread_id="runtime-tool-event-001",
                )
            )

        self.assertEqual(
            len(result),
            1,
        )
        self.assertEqual(
            result[0].tool_events[0]["event_type"],
            "tool_cache_hit",
        )
        self.assertEqual(
            result[0].tool_events[0]["tool_name"],
            "echo",
        )



class AsyncRuntimeEventStreamTests(
        unittest.IsolatedAsyncioTestCase,
):
    async def test_astream_runtime_events_normalizes(self):
        runtime = AgentRuntime()

        async def raw_event_stream():
            yield {
                "event": "on_chain_start",
                "name": "router",
            }
            yield {
                "event": "on_chain_end",
                "name": "finalize",
            }

        with patch.object(
            runtime.graph,
            "astream_events",
            return_value=raw_event_stream(),
        ):
            result = [
                event
                async for event in runtime.astream_runtime_events(
                    "异步测试事件",
                    thread_id="runtime-event-002",
                )
            ]

        self.assertEqual(
            len(result),
            2,
        )
        self.assertTrue(
            all(
                isinstance(event, RuntimeEvent)
                for event in result
            )
        )
        self.assertEqual(
            result[-1].event,
            "on_chain_end",
        )

    def test_resume_runtime_events_normalizes_and_filters(self):
        from langgraph.checkpoint.memory import InMemorySaver

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        raw_events = [
            {
                "event": "on_chain_start",
                "name": "agent_tool",
            },
            {
                "event": "on_chain_end",
                "name": "finalize",
            },
        ]

        with patch.object(
            runtime.graph,
            "stream_events",
            return_value=iter(raw_events),
        ):
            result = list(
                runtime.resume_runtime_events(
                    {"approved": True},
                    thread_id="resume-events-001",
                    event_types={
                        "on_chain_end",
                    },
                )
            )

        self.assertEqual(
            len(result),
            1,
        )
        self.assertIsInstance(
            result[0],
            RuntimeEvent,
        )
        self.assertEqual(
            result[0].event,
            "on_chain_end",
        )
        self.assertEqual(
            result[0].name,
            "finalize",
        )

    async def test_aresume_runtime_events_normalizes(self):
        from langgraph.checkpoint.memory import InMemorySaver

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        async def raw_event_stream():
            yield {
                "event": "on_chain_start",
                "name": "agent_tool",
            }
            yield {
                "event": "on_chain_end",
                "name": "finalize",
            }

        with patch.object(
            runtime.graph,
            "astream_events",
            return_value=raw_event_stream(),
        ):
            result = [
                event
                async for event in runtime.aresume_runtime_events(
                    {"approved": True},
                    thread_id="resume-events-002",
                )
            ]

        self.assertEqual(
            len(result),
            2,
        )
        self.assertTrue(
            all(
                isinstance(event, RuntimeEvent)
                for event in result
            )
        )
        self.assertEqual(
            result[-1].event,
            "on_chain_end",
        )


    def test_tool_events_are_read_from_nested_output(self):
        event = RuntimeEvent.from_raw(
            {
                "event": "on_chain_end",
                "name": "agent_tool",
                "data": {
                    "output": {
                        "tool_events": [
                            {
                                "event_type": "tool_started",
                                "tool_name": "python_executor",
                                "call_count": 1,
                                "attempt": 1,
                            },
                            {
                                "event_type": "tool_completed",
                                "tool_name": "python_executor",
                                "call_count": 1,
                                "attempt": 1,
                            },
                        ],
                    },
                },
            }
        )

        self.assertEqual(
            event.tool_events,
            [
                {
                    "event_type": "tool_started",
                    "tool_name": "python_executor",
                    "call_count": 1,
                    "attempt": 1,
                },
                {
                    "event_type": "tool_completed",
                    "tool_name": "python_executor",
                    "call_count": 1,
                    "attempt": 1,
                },
            ],
        )


    def test_stream_runtime_events_exposes_tool_events(self):
        runtime = AgentRuntime()

        raw_events = [
            {
                "event": "on_chain_end",
                "name": "agent_tool",
                "data": {
                    "output": {
                        "tool_events": [
                            {
                                "event_type": "tool_cache_hit",
                                "tool_name": "echo",
                                "call_count": 2,
                                "attempt": 1,
                            },
                        ],
                    },
                },
            },
        ]

        with patch.object(
            runtime.graph,
            "stream_events",
            return_value=iter(raw_events),
        ):
            result = list(
                runtime.stream_runtime_events(
                    "读取工具事件",
                    thread_id="runtime-tool-event-001",
                )
            )

        self.assertEqual(
            len(result),
            1,
        )
        self.assertEqual(
            result[0].tool_events[0]["event_type"],
            "tool_cache_hit",
        )
        self.assertEqual(
            result[0].tool_events[0]["tool_name"],
            "echo",
        )


    async def test_astream_runtime_events_exposes_tool_events(self):
        runtime = AgentRuntime()

        async def raw_event_stream():
            yield {
                "event": "on_chain_end",
                "name": "agent_tool",
                "data": {
                    "tool_events": [
                        {
                            "event_type": "tool_failed",
                            "tool_name": "python_executor",
                            "call_count": 1,
                            "attempt": 1,
                            "error": "测试错误",
                        },
                    ],
                },
            }

        with patch.object(
            runtime.graph,
            "astream_events",
            return_value=raw_event_stream(),
        ):
            result = [
                event
                async for event in runtime.astream_runtime_events(
                    "读取异步工具事件",
                    thread_id="runtime-tool-event-002",
                )
            ]

        self.assertEqual(
            len(result),
            1,
        )
        self.assertEqual(
            result[0].tool_events[0]["event_type"],
            "tool_failed",
        )
        self.assertEqual(
            result[0].tool_events[0]["error"],
            "测试错误",
        )


    async def test_astream_runtime_events_exposes_tool_events(self):
        runtime = AgentRuntime()

        async def raw_event_stream():
            yield {
                "event": "on_chain_end",
                "name": "agent_tool",
                "data": {
                    "tool_events": [
                        {
                            "event_type": "tool_failed",
                            "tool_name": "python_executor",
                            "call_count": 1,
                            "attempt": 1,
                            "error": "测试错误",
                        },
                    ],
                },
            }

        with patch.object(
            runtime.graph,
            "astream_events",
            return_value=raw_event_stream(),
        ):
            result = [
                event
                async for event in runtime.astream_runtime_events(
                    "读取异步工具事件",
                    thread_id="runtime-tool-event-002",
                )
            ]

        self.assertEqual(
            len(result),
            1,
        )
        self.assertEqual(
            result[0].tool_events[0]["event_type"],
            "tool_failed",
        )
        self.assertEqual(
            result[0].tool_events[0]["error"],
            "测试错误",
        )








if __name__ == "__main__":
    unittest.main()