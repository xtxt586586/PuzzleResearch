import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from agent.core.events import RuntimeEvent
from agent.core.profile import AgentProfile
from agent.core.runtime import AgentRuntime
from tools.registry import ToolDefinition, ToolRegistry


class SequentialPlanningModel:
    def __init__(self):
        self.responses = [
            (
                '{"action": "tool", '
                '"tool_name": "python_executor", '
                '"arguments": {'
                '"code": "print(2 + 3)"'
                "}}"
            ),
            (
                '{"action": "finish", '
                '"final_report": "计算完成，结果是 5。"}'
            ),
        ]
        self.calls = 0

    def invoke(self, messages, config=None):
        response = type(
            "FakeResponse",
            (),
            {
                "content": self.responses[self.calls],
            },
        )()

        self.calls += 1
        return response


class FailingToolEventPlanningModel:
    def __init__(self):
        self.calls = 0

    def with_structured_output(self, schema):
        class StructuredRouter:
            def invoke(self, messages, config=None):
                return {
                    "route": "agent",
                    "reasoning": "测试请求需要调用工具",
                    "confidence": 1.0,
                    "suggested_queries": [],
                    "clarification_question": "",
                }

        return StructuredRouter()

    def invoke(self, messages, config=None):
        self.calls += 1

        return type(
            "FakeResponse",
            (),
            {
                "content": (
                    '{"action": "tool", '
                    '"tool_name": "event_failing_tool", '
                    '"arguments": {}}'
                ),
            },
        )()


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


    def test_stream_runtime_events_completes_agent_tool_chain(self):
        runtime = AgentRuntime()
        model = SequentialPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = list(
                runtime.stream_runtime_events(
                    "请执行 Python 代码计算 2 加 3，然后告诉我结果",
                    thread_id="runtime-e2e-sync-001",
                    event_types="on_chain_end",
                )
            )

        self.assertTrue(
            result,
        )
        self.assertTrue(
            all(
                isinstance(event, RuntimeEvent)
                for event in result
            ),
        )
        self.assertTrue(
            all(
                event.event == "on_chain_end"
                for event in result
            ),
        )

        agent_tool_events = [
            event
            for event in result
            if event.name == "agent_tool"
        ]

        self.assertEqual(
            len(agent_tool_events),
            1,
        )

        tool_events = agent_tool_events[0].tool_events

        self.assertTrue(
            tool_events,
        )
        self.assertEqual(
            tool_events[-1]["event_type"],
            "tool_completed",
        )
        self.assertEqual(
            tool_events[-1]["tool_name"],
            "python_executor",
        )
        self.assertEqual(
            tool_events[-1]["call_count"],
            1,
        )
        self.assertEqual(
            model.calls,
            2,
        )


    def test_stream_runtime_events_propagates_tool_failure(self):
        def event_failing_tool() -> dict[str, str]:
            return {
                "output": "",
                "error": "事件流工具测试错误",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="event_failing_tool",
                    description="返回结构化错误的事件流测试工具",
                    handler=event_failing_tool,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-event-failure-profile",
                enabled_tools=frozenset({"event_failing_tool"}),
            ),
            registry=registry,
        )
        model = FailingToolEventPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = list(
                runtime.stream_runtime_events(
                    "请调用 event_failing_tool",
                    thread_id="runtime-failure-event-sync-001",
                    event_types={"on_chain_end"},
                )
            )

        self.assertTrue(result)
        self.assertTrue(
            all(
                isinstance(event, RuntimeEvent)
                for event in result
            )
        )

        agent_tool_events = [
            event
            for event in result
            if event.name == "agent_tool"
        ]

        self.assertEqual(
            len(agent_tool_events),
            1,
        )

        tool_events = agent_tool_events[0].tool_events

        self.assertTrue(tool_events)
        self.assertEqual(
            tool_events[0]["event_type"],
            "tool_started",
        )
        self.assertEqual(
            tool_events[-1]["event_type"],
            "tool_failed",
        )
        self.assertEqual(
            tool_events[-1]["tool_name"],
            "event_failing_tool",
        )
        self.assertEqual(
            tool_events[-1]["error"],
            "事件流工具测试错误",
        )

        finalize_events = [
            event
            for event in result
            if event.name == "finalize"
        ]

        self.assertTrue(finalize_events)
        self.assertEqual(
            finalize_events[-1].data["output"]["status"],
            "failed",
        )
        self.assertFalse(
            finalize_events[-1].data["output"]["is_complete"],
        )
        self.assertEqual(
            model.calls,
            1,
        )


    def test_stream_runtime_events_exposes_retry_failure_sequence(self):
        attempts = {
            "count": 0,
        }

        def always_failing_tool() -> dict[str, str]:
            attempts["count"] += 1
            raise RuntimeError("事件流重试耗尽")

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="event_failing_tool",
                    description="始终失败的事件流测试工具",
                    handler=always_failing_tool,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-retry-event-profile",
                enabled_tools=frozenset({"event_failing_tool"}),
                tool_retry_limit=2,
            ),
            registry=registry,
        )
        model = FailingToolEventPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = list(
                runtime.stream_runtime_events(
                    "请调用 event_failing_tool",
                    thread_id="runtime-retry-event-sync-001",
                    event_types={"on_chain_end"},
                )
            )

        agent_tool_events = [
            event
            for event in result
            if event.name == "agent_tool"
        ]

        self.assertEqual(
            len(agent_tool_events),
            1,
        )

        tool_events = agent_tool_events[0].tool_events

        self.assertEqual(
            [
                item["event_type"]
                for item in tool_events
            ],
            [
                "tool_started",
                "tool_retry",
                "tool_retry",
                "tool_failed",
            ],
        )
        self.assertEqual(
            attempts["count"],
            3,
        )
        self.assertEqual(
            tool_events[-1]["error"],
            "事件流重试耗尽",
        )
        self.assertEqual(
            model.calls,
            1,
        )

        finalize_events = [
            event
            for event in result
            if event.name == "finalize"
        ]

        self.assertTrue(finalize_events)
        self.assertEqual(
            finalize_events[-1].data["output"]["status"],
            "failed",
        )
        self.assertFalse(
            finalize_events[-1].data["output"]["is_complete"],
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

    def test_resume_runtime_events_completes_approved_agent_tool_chain(
            self,
    ):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
            profile=AgentProfile(
                name="event-approval-profile",
                require_tool_approval=True,
            ),
        )
        model = SequentialPlanningModel()
        thread_id = "runtime-event-approval-sync-001"

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            paused_events = list(
                runtime.stream_runtime_events(
                    "请执行 Python 代码计算 2 加 3",
                    thread_id=thread_id,
                    event_types="on_chain_end",
                )
            )

            paused_plan_events = [
                event
                for event in paused_events
                if event.name == "agent_plan"
            ]

            self.assertTrue(
                paused_plan_events,
            )
            self.assertEqual(
                paused_plan_events[-1].data["output"]["status"],
                "paused",
            )
            self.assertEqual(
                model.calls,
                1,
            )

            resumed_events = list(
                runtime.resume_runtime_events(
                    {
                        "approved": True,
                    },
                    thread_id=thread_id,
                    event_types="on_chain_end",
                )
            )

        agent_tool_events = [
            event
            for event in resumed_events
            if event.name == "agent_tool"
        ]

        self.assertEqual(
            len(agent_tool_events),
            1,
        )
        self.assertEqual(
            agent_tool_events[0].tool_events[-1]["event_type"],
            "tool_completed",
        )
        self.assertEqual(
            agent_tool_events[0].tool_events[-1]["tool_name"],
            "python_executor",
        )

        finalize_events = [
            event
            for event in resumed_events
            if event.name == "finalize"
        ]

        self.assertTrue(
            finalize_events,
        )
        self.assertEqual(
            finalize_events[-1].data["output"]["status"],
            "completed",
        )
        self.assertEqual(
            model.calls,
            2,
        )


    def test_from_raw_isolates_nested_data(self):
        raw_event = {
            "event": "on_chain_end",
            "metadata": {
                "labels": [
                    "original",
                ],
            },
            "data": {
                "output": {
                    "tool_events": [
                        {
                            "event_type": "tool_completed",
                            "tool_name": "echo",
                            "details": {
                                "attempt": 1,
                            },
                        },
                    ],
                },
            },
        }

        event = RuntimeEvent.from_raw(raw_event)

        raw_event["metadata"]["labels"].append(
            "mutated",
        )
        raw_event["data"]["output"]["tool_events"][0][
            "details"
        ]["attempt"] = 99

        self.assertEqual(
            event.metadata["labels"],
            ["original"],
        )
        self.assertEqual(
            event.tool_events[0]["details"]["attempt"],
            1,
        )

    def test_tool_events_return_defensive_copies(self):
        event = RuntimeEvent.from_raw(
            {
                "event": "on_chain_end",
                "data": {
                    "tool_events": [
                        {
                            "event_type": "tool_completed",
                            "tool_name": "echo",
                            "details": {
                                "attempt": 1,
                            },
                        },
                    ],
                },
            }
        )

        extracted_events = event.tool_events
        extracted_events[0]["details"]["attempt"] = 99
        extracted_events.append(
            {
                "event_type": "tool_failed",
            }
        )

        self.assertEqual(
            event.tool_events,
            [
                {
                    "event_type": "tool_completed",
                    "tool_name": "echo",
                    "details": {
                        "attempt": 1,
                    },
                },
            ],
        )

    def test_to_dict_returns_defensive_copies(self):
        event = RuntimeEvent.from_raw(
            {
                "event": "on_chain_end",
                "metadata": {
                    "labels": [
                        "original",
                    ],
                },
                "data": {
                    "status": "completed",
                },
            }
        )

        result = event.to_dict()

        result["metadata"]["labels"].append(
            "mutated",
        )
        result["data"]["status"] = "changed"

        self.assertEqual(
            event.metadata["labels"],
            ["original"],
        )
        self.assertEqual(
            event.data["status"],
            "completed",
        )


    def test_stream_runtime_events_forwards_graph_filters(self):
        runtime = AgentRuntime()

        raw_events = [
            {
                "event": "on_chain_end",
                "name": "agent_tool",
            },
        ]

        with patch.object(
            runtime.graph,
            "stream_events",
            return_value=iter(raw_events),
        ) as graph_stream:
            result = list(
                runtime.stream_runtime_events(
                    "测试过滤参数",
                    thread_id="filter-thread-001",
                    event_types={
                        "on_chain_end",
                    },
                    include_names=[
                        "agent_tool",
                    ],
                    exclude_types=[
                        "on_chain_start",
                    ],
                )
            )

        self.assertEqual(
            len(result),
            1,
        )

        forwarded_kwargs = graph_stream.call_args.kwargs

        self.assertEqual(
            forwarded_kwargs["include_names"],
            ["agent_tool"],
        )
        self.assertEqual(
            forwarded_kwargs["exclude_types"],
            ["on_chain_start"],
        )
        self.assertNotIn(
            "event_types",
            forwarded_kwargs,
        )


    def test_resume_runtime_events_forwards_graph_filters(self):
        from langgraph.checkpoint.memory import InMemorySaver

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        raw_events = [
            {
                "event": "on_chain_end",
                "name": "agent_tool",
            },
        ]

        with patch.object(
            runtime.graph,
            "stream_events",
            return_value=iter(raw_events),
        ) as graph_stream:
            result = list(
                runtime.resume_runtime_events(
                    {"approved": True},
                    thread_id="filter-thread-003",
                    event_types={
                        "on_chain_end",
                    },
                    include_names=[
                        "agent_tool",
                    ],
                )
            )

        self.assertEqual(
            len(result),
            1,
        )

        forwarded_kwargs = graph_stream.call_args.kwargs

        self.assertEqual(
            forwarded_kwargs["include_names"],
            ["agent_tool"],
        )
        self.assertNotIn(
            "event_types",
            forwarded_kwargs,
        )


    def test_from_raw_normalizes_malformed_collections(self):
        event = RuntimeEvent.from_raw(
            {
                "event": "on_chain_end",
                "parent_ids": "parent-001",
                "tags": "agent",
                "metadata": [
                    "invalid-metadata",
                ],
            }
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
            event.metadata,
            {},
        )

    def test_tool_events_skip_non_mapping_items(self):
        event = RuntimeEvent.from_raw(
            {
                "event": "on_chain_end",
                "data": {
                    "tool_events": [
                        None,
                        "invalid-event",
                        {
                            "event_type": "tool_completed",
                            "tool_name": "echo",
                        },
                    ],
                },
            }
        )

        self.assertEqual(
            event.tool_events,
            [
                {
                    "event_type": "tool_completed",
                    "tool_name": "echo",
                },
            ],
        )

    def test_tool_events_return_empty_for_invalid_container(self):
        event = RuntimeEvent.from_raw(
            {
                "event": "on_chain_end",
                "data": {
                    "tool_events": {
                        "event_type": "tool_completed",
                    },
                },
            }
        )

        self.assertEqual(
            event.tool_events,
            [],
        )


    def test_stream_runtime_events_accepts_string_event_type(self):
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
                    "字符串事件过滤",
                    thread_id="event-type-string-001",
                    event_types="on_chain_end",
                )
            )

        self.assertEqual(
            len(result),
            1,
        )
        self.assertEqual(
            result[0].event,
            "on_chain_end",
        )


    def test_stream_runtime_events_rejects_invalid_event_types(self):
        runtime = AgentRuntime()

        with self.assertRaisesRegex(
            TypeError,
            "event_types",
        ):
            list(
                runtime.stream_runtime_events(
                    "非法事件过滤",
                    event_types=123,
                )
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



    async def test_astream_runtime_events_completes_agent_tool_chain(self):
        runtime = AgentRuntime()
        model = SequentialPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = [
                event
                async for event in runtime.astream_runtime_events(
                    "请执行 Python 代码计算 2 加 3，然后告诉我结果",
                    thread_id="runtime-e2e-async-001",
                    event_types={
                        "on_chain_end",
                    },
                )
            ]

        self.assertTrue(
            result,
        )
        self.assertTrue(
            all(
                isinstance(event, RuntimeEvent)
                for event in result
            ),
        )
        self.assertTrue(
            all(
                event.event == "on_chain_end"
                for event in result
            ),
        )

        agent_tool_events = [
            event
            for event in result
            if event.name == "agent_tool"
        ]

        self.assertEqual(
            len(agent_tool_events),
            1,
        )

        tool_events = agent_tool_events[0].tool_events

        self.assertTrue(
            tool_events,
        )
        self.assertEqual(
            tool_events[-1]["event_type"],
            "tool_completed",
        )
        self.assertEqual(
            tool_events[-1]["tool_name"],
            "python_executor",
        )
        self.assertEqual(
            tool_events[-1]["call_count"],
            1,
        )
        self.assertEqual(
            model.calls,
            2,
        )


    async def test_astream_runtime_events_propagates_tool_failure(self):
        def event_failing_tool() -> dict[str, str]:
            return {
                "output": "",
                "error": "异步事件流工具测试错误",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="event_failing_tool",
                    description="返回结构化错误的异步事件流测试工具",
                    handler=event_failing_tool,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-async-event-failure-profile",
                enabled_tools=frozenset({"event_failing_tool"}),
            ),
            registry=registry,
        )
        model = FailingToolEventPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = [
                event
                async for event in runtime.astream_runtime_events(
                    "请调用 event_failing_tool",
                    thread_id="runtime-failure-event-async-001",
                    event_types={"on_chain_end"},
                )
            ]

        self.assertTrue(result)
        self.assertTrue(
            all(
                isinstance(event, RuntimeEvent)
                for event in result
            )
        )

        agent_tool_events = [
            event
            for event in result
            if event.name == "agent_tool"
        ]

        self.assertEqual(
            len(agent_tool_events),
            1,
        )

        tool_events = agent_tool_events[0].tool_events

        self.assertTrue(tool_events)
        self.assertEqual(
            tool_events[0]["event_type"],
            "tool_started",
        )
        self.assertEqual(
            tool_events[-1]["event_type"],
            "tool_failed",
        )
        self.assertEqual(
            tool_events[-1]["tool_name"],
            "event_failing_tool",
        )
        self.assertEqual(
            tool_events[-1]["error"],
            "异步事件流工具测试错误",
        )

        finalize_events = [
            event
            for event in result
            if event.name == "finalize"
        ]

        self.assertTrue(finalize_events)
        self.assertEqual(
            finalize_events[-1].data["output"]["status"],
            "failed",
        )
        self.assertFalse(
            finalize_events[-1].data["output"]["is_complete"],
        )
        self.assertEqual(
            model.calls,
            1,
        )


    async def test_astream_runtime_events_exposes_retry_failure_sequence(self):
        attempts = {
            "count": 0,
        }

        def always_failing_tool() -> dict[str, str]:
            attempts["count"] += 1
            raise RuntimeError("异步事件流重试耗尽")

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="event_failing_tool",
                    description="始终失败的异步事件流测试工具",
                    handler=always_failing_tool,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-async-retry-event-profile",
                enabled_tools=frozenset({"event_failing_tool"}),
                tool_retry_limit=2,
            ),
            registry=registry,
        )
        model = FailingToolEventPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = [
                event
                async for event in runtime.astream_runtime_events(
                    "请调用 event_failing_tool",
                    thread_id="runtime-retry-event-async-001",
                    event_types={"on_chain_end"},
                )
            ]

        agent_tool_events = [
            event
            for event in result
            if event.name == "agent_tool"
        ]

        self.assertEqual(
            len(agent_tool_events),
            1,
        )

        tool_events = agent_tool_events[0].tool_events

        self.assertEqual(
            [
                item["event_type"]
                for item in tool_events
            ],
            [
                "tool_started",
                "tool_retry",
                "tool_retry",
                "tool_failed",
            ],
        )
        self.assertEqual(
            attempts["count"],
            3,
        )
        self.assertEqual(
            tool_events[-1]["error"],
            "异步事件流重试耗尽",
        )
        self.assertEqual(
            model.calls,
            1,
        )

        finalize_events = [
            event
            for event in result
            if event.name == "finalize"
        ]

        self.assertTrue(finalize_events)
        self.assertEqual(
            finalize_events[-1].data["output"]["status"],
            "failed",
        )
        self.assertFalse(
            finalize_events[-1].data["output"]["is_complete"],
        )


    async def test_aresume_runtime_events_completes_approved_agent_tool_chain(
            self,
    ):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
            profile=AgentProfile(
                name="async-event-approval-profile",
                require_tool_approval=True,
            ),
        )
        model = SequentialPlanningModel()
        thread_id = "runtime-event-approval-async-001"

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            paused_events = [
                event
                async for event in runtime.astream_runtime_events(
                    "请执行 Python 代码计算 2 加 3",
                    thread_id=thread_id,
                    event_types="on_chain_end",
                )
            ]

            paused_plan_events = [
                event
                for event in paused_events
                if event.name == "agent_plan"
            ]

            self.assertTrue(
                paused_plan_events,
            )
            self.assertEqual(
                paused_plan_events[-1].data["output"]["status"],
                "paused",
            )
            self.assertEqual(
                model.calls,
                1,
            )

            resumed_events = [
                event
                async for event in runtime.aresume_runtime_events(
                    {
                        "approved": True,
                    },
                    thread_id=thread_id,
                    event_types="on_chain_end",
                )
            ]

        agent_tool_events = [
            event
            for event in resumed_events
            if event.name == "agent_tool"
        ]

        self.assertEqual(
            len(agent_tool_events),
            1,
        )
        self.assertEqual(
            agent_tool_events[0].tool_events[-1]["event_type"],
            "tool_completed",
        )
        self.assertEqual(
            agent_tool_events[0].tool_events[-1]["tool_name"],
            "python_executor",
        )

        finalize_events = [
            event
            for event in resumed_events
            if event.name == "finalize"
        ]

        self.assertTrue(
            finalize_events,
        )
        self.assertEqual(
            finalize_events[-1].data["output"]["status"],
            "completed",
        )
        self.assertEqual(
            model.calls,
            2,
        )




    async def test_astream_runtime_events_forwards_graph_filters(self):
        runtime = AgentRuntime()

        async def raw_event_stream():
            yield {
                "event": "on_chain_end",
                "name": "agent_tool",
            }

        with patch.object(
            runtime.graph,
            "astream_events",
            return_value=raw_event_stream(),
        ) as graph_astream_events:
            result = [
                event
                async for event in runtime.astream_runtime_events(
                    "异步测试过滤参数",
                    thread_id="filter-thread-002",
                    event_types={
                        "on_chain_end",
                    },
                    include_names=[
                        "agent_tool",
                    ],
                    exclude_types=[
                        "on_chain_start",
                    ],
                )
            ]

        self.assertEqual(
            len(result),
            1,
        )

        forwarded_kwargs = (
            graph_astream_events.call_args.kwargs
        )

        self.assertEqual(
            forwarded_kwargs["include_names"],
            ["agent_tool"],
        )
        self.assertEqual(
            forwarded_kwargs["exclude_types"],
            ["on_chain_start"],
        )
        self.assertNotIn(
            "event_types",
            forwarded_kwargs,
        )

    async def test_aresume_runtime_events_forwards_graph_filters(self):
        from langgraph.checkpoint.memory import InMemorySaver

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        async def raw_event_stream():
            yield {
                "event": "on_chain_end",
                "name": "agent_tool",
            }

        with patch.object(
            runtime.graph,
            "astream_events",
            return_value=raw_event_stream(),
        ) as graph_astream_events:
            result = [
                event
                async for event in runtime.aresume_runtime_events(
                    {"approved": True},
                    thread_id="filter-thread-004",
                    event_types={
                        "on_chain_end",
                    },
                    include_names=[
                        "agent_tool",
                    ],
                )
            ]

        self.assertEqual(
            len(result),
            1,
        )

        forwarded_kwargs = (
            graph_astream_events.call_args.kwargs
        )

        self.assertEqual(
            forwarded_kwargs["include_names"],
            ["agent_tool"],
        )
        self.assertNotIn(
            "event_types",
            forwarded_kwargs,
        )


    async def test_astream_runtime_events_accepts_sequence_event_types(self):
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
                    "序列事件过滤",
                    thread_id="event-type-sequence-001",
                    event_types=[
                        "on_chain_end",
                    ],
                )
            ]

        self.assertEqual(
            len(result),
            1,
        )
        self.assertEqual(
            result[0].event,
            "on_chain_end",
        )


if __name__ == "__main__":
    unittest.main()