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


if __name__ == "__main__":
    unittest.main()