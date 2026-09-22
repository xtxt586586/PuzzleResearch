import unittest
from unittest.mock import AsyncMock, patch

import main
from agent.core.events import RuntimeEvent


class RuntimeTriggerEventTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_chat_events_are_forwarded_to_triggers(self):
        async def raw_events():
            yield RuntimeEvent(
                event="on_tool_start",
                name="echo",
                data={
                    "input": {
                        "text": "hello",
                    },
                },
            )

        trigger_emitter = AsyncMock(
            return_value=[]
        )

        with patch.object(
            main.chat_runtime,
            "astream_runtime_events",
            return_value=raw_events(),
        ), patch.object(
            main.trigger_manager,
            "emit_event",
            trigger_emitter,
        ):
            frames = [
                frame
                async for frame in main._iter_chat_events(
                    user_input="执行 echo",
                    history=[],
                    thread_id="trigger-event-test-001",
                    config={
                        "configurable": {
                            "thread_id": (
                                "trigger-event-test-001"
                            ),
                            "user_id": "user-001",
                            "agent_id": "default",
                        },
                    },
                )
            ]

        self.assertEqual(
            [frame["type"] for frame in frames],
            [
                "status",
                "tool_start",
                "done",
            ],
        )

        event_names = [
            call.args[0]
            for call in (
                trigger_emitter.await_args_list
            )
        ]

        self.assertEqual(
            event_names,
            [
                "chat.status",
                "tool.started",
                "chat.completed",
            ],
        )

    async def test_trigger_execution_suppresses_nested_events(
        self,
    ):
        async def raw_events():
            yield RuntimeEvent(
                event="on_chain_end",
                name="agent",
                data={},
            )

        trigger_emitter = AsyncMock(
            return_value=[]
        )

        with patch.object(
            main.chat_runtime,
            "astream_runtime_events",
            return_value=raw_events(),
        ), patch.object(
            main.trigger_manager,
            "emit_event",
            trigger_emitter,
        ):
            frames = [
                frame
                async for frame in main._iter_chat_events(
                    user_input="触发器内部任务",
                    history=[],
                    thread_id="trigger-event-test-002",
                    config={
                        "configurable": {
                            "thread_id": (
                                "trigger-event-test-002"
                            ),
                            "user_id": "user-001",
                            "agent_id": "default",
                            "suppress_trigger_events": True,
                        },
                    },
                )
            ]

        self.assertTrue(frames)
        trigger_emitter.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()