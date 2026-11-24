import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from agent.core.events import RuntimeEvent
import main
from main import (
    _friendly_runtime_error,
    _runtime_event_to_legacy_events,
    _should_publish_tool_event,
    _stable_tool_event_key,
)
from agent.core.runtime import AgentRuntime


class ChatEventMappingTests(unittest.TestCase):
    def test_runtime_snapshot_reads_interrupts_from_tasks(self):
        snapshot = type(
            "Snapshot",
            (),
            {
                "interrupts": (),
                "tasks": [
                    type(
                        "Task",
                        (),
                        {"interrupts": ("approval",)},
                    )(),
                ],
            },
        )()

        self.assertEqual(
            AgentRuntime._snap_interrupts(snapshot),
            ["approval"],
        )

    def test_accumulated_tool_events_only_publish_new_items(self):
        prior_event = {
            "event_type": "tool_failed",
            "tool_name": "python_executor",
            "call_count": 1,
            "attempt": 1,
            "error": "旧错误",
        }
        current_event = {
            "event_type": "tool_completed",
            "tool_name": "python_executor",
            "call_count": 2,
            "attempt": 1,
        }
        prior_counts = {
            _stable_tool_event_key(prior_event): 1,
        }
        emitted_keys = set()

        self.assertFalse(
            _should_publish_tool_event(
                prior_event,
                prior_counts,
                emitted_keys,
            )
        )
        self.assertTrue(
            _should_publish_tool_event(
                current_event,
                prior_counts,
                emitted_keys,
            )
        )
        self.assertFalse(
            _should_publish_tool_event(
                current_event,
                prior_counts,
                emitted_keys,
            )
        )

    def test_search_chain_start_is_published_as_search(self):
        event = RuntimeEvent(
            event="on_chain_start",
            name="search",
            data={
                "input": {
                    "query": "LangGraph runtime events",
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        self.assertEqual(
            result[0]["type"],
            "search",
        )
        self.assertEqual(
            result[0]["data"]["query"],
            "LangGraph runtime events",
        )

    def test_multi_agent_chain_starts_expose_role_and_responsibility(self):
        cases = [
            ("coordinator", "协调器", "分配下一阶段"),
            ("deep_plan", "规划代理", "拆解深度研究问题"),
            ("deep_tree_search", "研究代理", "探索研究分支"),
            ("deep_writer", "报告代理", "综合多来源证据"),
            ("evaluator", "评估代理", "评估报告质量"),
            ("revise", "修订代理", "依据评估结果修订"),
        ]

        for node, expected_role, expected_status in cases:
            with self.subTest(node=node):
                event = RuntimeEvent(
                    event="on_chain_start",
                    name=node,
                    metadata={"langgraph_node": node},
                    data={"input": {}},
                )
                result = _runtime_event_to_legacy_events(event)

                self.assertEqual(len(result), 1)
                self.assertEqual(result[0]["data"]["agent_role"], expected_role)
                self.assertIn(expected_status, result[0]["data"]["text"])

    def test_plan_chain_end_is_published_as_research_plan(self):
        event = RuntimeEvent(
            event="on_chain_end",
            name="deep_plan",
            data={
                "output": {
                    "research_plan": [
                        "LangGraph architecture",
                        "LangGraph runtime events",
                    ],
                    "deepsearch_artifacts": {
                        "reasoning": "覆盖架构和运行时行为",
                    },
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        plan_event = next(
            item
            for item in result
            if item["type"] == "research_plan"
        )
        self.assertEqual(
            plan_event["data"]["queries"],
            [
                "LangGraph architecture",
                "LangGraph runtime events",
            ],
        )
        self.assertEqual(
            plan_event["data"]["reasoning"],
            "覆盖架构和运行时行为",
        )

    def test_budget_artifact_is_published_as_quality_update(self):
        event = RuntimeEvent(
            event="on_chain_end",
            name="coordinator",
            data={
                "output": {
                    "deepsearch_artifacts": {
                        "quality_summary": {
                            "budget_stop_reason": "token_budget_exceeded",
                            "tokens_used": 256,
                            "elapsed_seconds": 8.5,
                            "searches_used": 4,
                            "max_tokens": 256,
                        },
                    },
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)
        quality_event = next(
            item for item in result if item["type"] == "quality_update"
        )

        self.assertEqual(
            quality_event["data"]["budget_stop_reason"],
            "token_budget_exceeded",
        )
        self.assertEqual(quality_event["data"]["tokens_used"], 256)
        self.assertEqual(quality_event["data"]["max_tokens"], 256)

    def test_agent_plan_is_published_with_pending_tool_call(self):
        event = RuntimeEvent(
            event="on_chain_end",
            name="agent_plan",
            data={
                "output": {
                    "pending_tool_calls": [
                        {
                            "tool_name": "python_executor",
                            "arguments": {"code": "print(1)"},
                        },
                    ],
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        plan_event = next(
            item
            for item in result
            if item["type"] == "agent_plan"
        )
        self.assertEqual(
            plan_event["data"]["pending_tool_calls"][0][
                "tool_name"
            ],
            "python_executor",
        )

    def test_planner_stream_is_published_as_thinking(self):
        event = RuntimeEvent(
            event="on_chat_model_stream",
            name="ChatDeepSeek",
            metadata={
                "langgraph_node": "deep_plan",
            },
            data={
                "chunk": {
                    "content": '{"queries": ["LangGraph"]}',
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "thinking")
        self.assertEqual(
            result[0]["data"]["node"],
            "deep_plan",
        )

    def test_writer_stream_is_published_as_text(self):
        event = RuntimeEvent(
            event="on_chat_model_stream",
            name="ChatDeepSeek",
            metadata={
                "langgraph_node": "deep_writer",
            },
            data={
                "chunk": {
                    "content": "这是最终回答。",
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(
            result[0]["data"]["node"],
            "deep_writer",
        )

    def test_agent_fast_path_stream_is_published_as_text(self):
        event = RuntimeEvent(
            event="on_chat_model_stream",
            name="ChatDeepSeek",
            metadata={
                "langgraph_node": "agent_fast_path",
            },
            data={
                "chunk": {
                    "content": "根据搜索证据，当前版本为 3.2。",
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["type"], "text")
        self.assertEqual(
            result[0]["data"]["node"],
            "agent_fast_path",
        )

    def test_insufficient_balance_is_user_friendly(self):
        message = _friendly_runtime_error(
            "Error code: 402 - {'error': "
            "{'message': 'Insufficient Balance'}}"
        )

        self.assertEqual(
            message,
            "模型服务余额不足，请更换模型或补充 API Key。",
        )

    def test_failed_chain_end_is_published_as_error(self):
        event = RuntimeEvent(
            event="on_chain_end",
            name="agent_plan",
            data={
                "output": {
                    "status": "failed",
                    "last_error": "工具调用规划失败：Connection error.",
                    "final_report": "工具调用规划失败：Connection error.",
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        self.assertEqual(
            result,
            [
                {
                    "type": "error",
                    "data": {
                        "message": "工具调用规划失败：Connection error.",
                        "node": "agent_plan",
                    },
                }
            ],
        )

    def test_failed_chain_end_keeps_tool_failure_event(self):
        event = RuntimeEvent(
            event="on_chain_end",
            name="agent_tool",
            data={
                "output": {
                    "status": "failed",
                    "last_error": "工具「update_task」参数无效。",
                    "tool_events": [
                        {
                            "event_type": "tool_started",
                            "tool_name": "update_task",
                            "call_count": 3,
                            "attempt": 1,
                        },
                        {
                            "event_type": "tool_failed",
                            "tool_name": "update_task",
                            "call_count": 3,
                            "attempt": 1,
                            "error": "任务状态无效。",
                        },
                    ],
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        self.assertEqual(
            [item["type"] for item in result],
            ["tool", "tool", "error"],
        )
        self.assertEqual(
            [item["data"]["event_type"] for item in result[:2]],
            ["tool_started", "tool_failed"],
        )
        self.assertEqual(
            result[2]["data"]["message"],
            "工具「update_task」参数无效。",
        )

    def test_cancelled_chain_end_is_published_as_cancelled(self):
        event = RuntimeEvent(
            event="on_chain_end",
            name="agent_plan",
            data={
                "output": {
                    "status": "cancelled",
                    "final_report": "用户取消了任务。",
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        self.assertEqual(
            result,
            [
                {
                    "type": "cancelled",
                    "data": {
                        "message": "用户取消了任务。",
                        "node": "agent_plan",
                    },
                }
            ],
        )

    def test_completed_chain_end_keeps_completion_event(self):
        event = RuntimeEvent(
            event="on_chain_end",
            name="deep_writer",
            data={
                "output": {
                    "status": "completed",
                    "final_report": "这是最终回答。",
                },
            },
        )

        result = _runtime_event_to_legacy_events(event)

        self.assertEqual(
            result,
            [
                {
                    "type": "completion",
                    "data": {
                        "content": "这是最终回答。",
                    },
                }
            ],
        )


class ChatInterruptStreamTests(unittest.IsolatedAsyncioTestCase):
    async def test_paused_stream_emits_interrupt_without_updating_checkpoint(self):
        thread_id = "chat-interrupt-stream-test"
        interrupt_value = {
            "type": "tool_approval",
            "tool_name": "python_executor",
            "arguments": {"code": "print(1 + 1)"},
            "allowed_decisions": ["approve", "edit", "reject"],
        }
        snapshot = SimpleNamespace(
            values={
                "status": "paused",
                "pending_tool_calls": [
                    {
                        "tool_name": "python_executor",
                        "arguments": {"code": "print(1 + 1)"},
                    }
                ],
            },
            interrupts=(),
            tasks=[
                SimpleNamespace(
                    interrupts=(SimpleNamespace(value=interrupt_value),),
                )
            ],
        )

        async def runtime_events(*args, **kwargs):
            yield RuntimeEvent(
                event="on_chain_end",
                name="agent_plan",
                data={
                    "output": {
                        "status": "paused",
                        "pending_tool_calls": snapshot.values[
                            "pending_tool_calls"
                        ],
                    }
                },
            )

        with patch.object(
            main.chat_runtime,
            "aget_state",
            new=AsyncMock(side_effect=[{}, snapshot]),
        ), patch.object(
            main.chat_runtime,
            "aupdate_state",
            new=AsyncMock(),
        ) as update_state, patch.object(
            main.chat_runtime,
            "astream_runtime_events",
            new=runtime_events,
        ):
            frames = [
                frame
                async for frame in main._iter_chat_events(
                    user_input="执行一次 Python 计算",
                    history=[],
                    conversation_history=[
                        {"role": "user", "content": "执行一次 Python 计算"}
                    ],
                    thread_id=thread_id,
                    config={"configurable": {"thread_id": thread_id}},
                )
            ]

        self.assertIn("interrupt", [frame["type"] for frame in frames])
        interrupt_frame = next(
            frame for frame in frames if frame["type"] == "interrupt"
        )
        self.assertEqual(
            interrupt_frame["data"]["tool_name"],
            "python_executor",
        )
        self.assertEqual(frames[-1]["type"], "done")
        update_state.assert_not_awaited()
        self.assertEqual(
            main.cancellation_manager.get_token(thread_id).status.value,
            "paused",
        )


if __name__ == "__main__":
    unittest.main()
