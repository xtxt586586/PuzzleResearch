import unittest
from unittest.mock import AsyncMock, patch


from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agent.core.profile import AgentProfile
from agent.core.runtime import AgentRuntime
from tools.registry import ToolDefinition, ToolRegistry


class ApprovalPlanningModel:
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
        content = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]
        self.calls += 1

        return type(
            "FakeResponse",
            (),
            {
                "content": content,
            },
        )()


class CustomToolPlanningModel:
    def __init__(self):
        self.responses = [
            (
                '{"action": "tool", '
                '"tool_name": "echo", '
                '"arguments": {'
                '"text": "hello"'
                "}}"
            ),
            (
                '{"action": "finish", '
                '"final_report": "echo 执行完成。"}'
            ),
        ]
        self.calls = 0

    def invoke(self, messages, config=None):
        content = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]
        self.calls += 1

        return type(
            "FakeResponse",
            (),
            {
                "content": content,
            },
        )()


class RetryToolPlanningModel:
    def __init__(self):
        self.responses = [
            (
                '{"action": "tool", '
                '"tool_name": "flaky", '
                '"arguments": {'
                '"text": "success"'
                "}}"
            ),
            (
                '{"action": "finish", '
                '"final_report": "重试后执行完成。"}'
            ),
        ]
        self.calls = 0

    def invoke(self, messages, config=None):
        content = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]
        self.calls += 1

        return type(
            "FakeResponse",
            (),
            {
                "content": content,
            },
        )()


class LongOutputPlanningModel:
    def __init__(self):
        self.responses = [
            (
                '{"action": "tool", '
                '"tool_name": "large_output", '
                '"arguments": {}'
                "}"
            ),
            (
                '{"action": "finish", '
                '"final_report": "长文本处理完成。"}'
            ),
        ]
        self.calls = 0

    def invoke(self, messages, config=None):
        content = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]
        self.calls += 1

        return type(
            "FakeResponse",
            (),
            {
                "content": content,
            },
        )()


class RepeatedToolPlanningModel:
    def __init__(self):
        self.responses = [
            (
                '{"action": "tool", '
                '"tool_name": "echo", '
                '"arguments": {'
                '"text": "hello"'
                "}}"
            ),
            (
                '{"action": "tool", '
                '"tool_name": "echo", '
                '"arguments": {'
                '"text": "hello"'
                "}}"
            ),
            (
                '{"action": "finish", '
                '"final_report": "重复调用处理完成。"}'
            ),
        ]
        self.calls = 0

    def invoke(self, messages, config=None):
        content = self.responses[
            min(
                self.calls,
                len(self.responses) - 1,
            )
        ]
        self.calls += 1

        return type(
            "FakeResponse",
            (),
            {
                "content": content,
            },
        )()


class FailingToolPlanningModel:
    def __init__(self):
        self.responses = [
            '{"action": "tool", "tool_name": "failing_tool", "arguments": {}}',
        ]
        self.calls = 0

    def invoke(self, messages, config=None):
        self.calls += 1

        return type(
            "FakeResponse",
            (),
            {
                "content": self.responses[0],
            },
        )()


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


    def test_invoke_completes_agent_tool_loop(self):
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

        model = SequentialPlanningModel()
        runtime = AgentRuntime()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = runtime.invoke(
                "请执行 Python 代码计算 2 加 3，然后告诉我结果",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            model.calls,
            2,
        )
        self.assertEqual(
            len(result["tool_results"]),
            1,
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"].strip(),
            "5",
        )
        self.assertEqual(
            result["final_report"],
            "计算完成，结果是 5。",
        )


    def test_invoke_executes_custom_registered_tool(self):
        def echo(text: str) -> str:
            return text

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="custom-tool-profile",
                enabled_tools=frozenset({"echo"}),
            ),
            registry=registry,
        )
        model = CustomToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = runtime.invoke(
                "请调用工具 echo 返回 hello",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            result["tool_results"][0]["tool_name"],
            "echo",
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            "hello",
        )
        self.assertEqual(
            model.calls,
            2,
        )


    def test_invoke_rejects_tool_disabled_by_profile(self):
        attempts = {
            "count": 0,
        }

        def echo(text: str) -> str:
            attempts["count"] += 1
            return text

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="disabled-echo-profile",
                enabled_tools=frozenset(),
            ),
            registry=registry,
        )
        model = CustomToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = runtime.invoke(
                "请调用工具 echo 返回 hello",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "工具未启用",
            result["final_report"],
        )
        self.assertEqual(
            result.get("tool_results", []),
            [],
        )
        self.assertEqual(
            attempts["count"],
            0,
        )
        self.assertEqual(
            model.calls,
            1,
        )


    def test_invoke_retries_transient_custom_tool_error(self):
        attempts = {
            "count": 0,
        }

        def flaky(text: str) -> dict[str, str]:
            attempts["count"] += 1

            if attempts["count"] < 3:
                raise RuntimeError("temporary failure")

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="flaky",
                    description="一个会暂时失败的测试工具",
                    handler=flaky,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-retry-profile",
                enabled_tools=frozenset({"flaky"}),
                tool_retry_limit=2,
            ),
            registry=registry,
        )
        model = RetryToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = runtime.invoke(
                "请调用工具 flaky 返回 success",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            attempts["count"],
            3,
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            "success",
        )
        self.assertEqual(
            [
                event["event_type"]
                for event in result["tool_events"]
            ],
            [
                "tool_started",
                "tool_retry",
                "tool_retry",
                "tool_completed",
            ],
        )
        self.assertEqual(
            model.calls,
            2,
        )


    def test_invoke_limits_custom_tool_result_length(self):
        attempts = {
            "count": 0,
        }

        def large_output() -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": "abcdefghijklmnopqrstuvwxyz",
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="large_output",
                    description="返回较长文本",
                    handler=large_output,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="limited-runtime-result-profile",
                enabled_tools=frozenset({"large_output"}),
                max_tool_result_chars=16,
            ),
            registry=registry,
        )
        model = LongOutputPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = runtime.invoke(
                "请调用工具 large_output 返回长文本",
            )

        output = result["tool_results"][0]["result"]["output"]

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            attempts["count"],
            1,
        )
        self.assertLessEqual(
            len(output),
            16,
        )
        self.assertIn(
            "工具结果已截断",
            output,
        )
        self.assertTrue(
            result["tool_results"][0]["result"]["result_truncated"],
        )
        self.assertEqual(
            model.calls,
            2,
        )


    def test_invoke_rejects_repeated_custom_tool_call(self):
        attempts = {
            "count": 0,
        }

        def echo(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-deduplicate-profile",
                enabled_tools=frozenset({"echo"}),
                deduplicate_tool_calls=True,
            ),
            registry=registry,
        )
        model = RepeatedToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = runtime.invoke(
                "请调用工具 echo 返回 hello",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "重复工具调用",
            result["final_report"],
        )
        self.assertEqual(
            attempts["count"],
            1,
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            [
                event["event_type"]
                for event in result["tool_events"]
            ],
            [
                "tool_started",
                "tool_completed",
            ],
        )
        self.assertEqual(
            model.calls,
            2,
        )


    def test_invoke_enforces_profile_tool_call_limit(self):
        attempts = {
            "count": 0,
        }

        def echo(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-limit-profile",
                enabled_tools=frozenset({"echo"}),
                tool_call_limit=1,
            ),
            registry=registry,
        )
        model = RepeatedToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = runtime.invoke(
                "请调用工具 echo 返回 hello",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            len(result["tool_results"]),
            1,
        )
        self.assertEqual(
            attempts["count"],
            1,
        )
        self.assertEqual(
            model.calls,
            1,
        )


    def test_invoke_propagates_structured_tool_error(self):
        attempts = {
            "count": 0,
        }

        def failing_tool() -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": "",
                "error": "工具内部测试错误",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="failing_tool",
                    description="返回结构化错误",
                    handler=failing_tool,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-failure-profile",
                enabled_tools=frozenset({"failing_tool"}),
            ),
            registry=registry,
        )
        model = FailingToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = runtime.invoke(
                "请调用工具 failing_tool",
            )

        self.assertEqual(result["route"], "agent")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["is_complete"])
        self.assertEqual(attempts["count"], 1)
        self.assertEqual(model.calls, 1)

        self.assertEqual(
            result["tool_results"][0]["result"]["error"],
            "工具内部测试错误",
        )
        self.assertIn(
            "工具内部测试错误",
            result["errors"],
        )
        self.assertEqual(
            result["tool_events"][-1]["event_type"],
            "tool_failed",
        )


    def test_invoke_pauses_and_resume_completes_tool_approval(self):
        model = ApprovalPlanningModel()

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
            profile=AgentProfile(
                name="approval-profile",
                require_tool_approval=True,
            ),
        )

        thread_id = "runtime-approval-sync-001"

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            paused_result = runtime.invoke(
                "请执行 Python 代码计算 2 加 3",
                thread_id=thread_id,
            )

            self.assertEqual(
                paused_result["route"],
                "agent",
            )
            self.assertEqual(
                paused_result["status"],
                "paused",
            )
            self.assertFalse(
                paused_result["is_complete"],
            )
            self.assertIn(
                "__interrupt__",
                paused_result,
            )
            self.assertEqual(
                paused_result["pending_tool_calls"][0]["tool_name"],
                "python_executor",
            )
            self.assertEqual(
                paused_result["tool_results"],
                [],
            )
            self.assertEqual(
                model.calls,
                1,
            )

            resumed_result = runtime.resume(
                {
                    "approved": True,
                },
                thread_id=thread_id,
            )

        self.assertEqual(
            resumed_result["status"],
            "completed",
        )
        self.assertTrue(
            resumed_result["is_complete"],
        )
        self.assertEqual(
            resumed_result["tool_call_count"],
            1,
        )
        self.assertEqual(
            resumed_result["tool_results"][0]["result"]["output"].strip(),
            "5",
        )
        self.assertEqual(
            model.calls,
            2,
        )


    def test_cancel_stops_paused_tool_thread_and_persists_state(self):
        model = ApprovalPlanningModel()

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
            profile=AgentProfile(
                name="cancel-profile",
                require_tool_approval=True,
            ),
        )

        thread_id = "runtime-cancel-sync-001"

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            paused_result = runtime.invoke(
                "请执行 Python 代码计算 2 加 3",
                thread_id=thread_id,
            )

            self.assertEqual(
                paused_result["status"],
                "paused",
            )
            self.assertEqual(
                model.calls,
                1,
            )

            cancelled_snapshot = runtime.cancel(
                thread_id=thread_id,
                cancel_token_id="cancel-sync-001",
            )

        self.assertEqual(
            cancelled_snapshot.values["status"],
            "cancelled",
        )
        self.assertTrue(
            cancelled_snapshot.values["is_cancelled"],
        )

        summary = runtime.inspect_thread(
            thread_id=thread_id,
        )

        self.assertEqual(
            summary["status"],
            "cancelled",
        )
        self.assertTrue(
            summary["is_cancelled"],
        )
        self.assertEqual(
            summary["cancel_token_id"],
            "cancel-sync-001",
        )
        self.assertEqual(
            summary["pending_tool_count"],
            0,
        )
        self.assertEqual(
            summary["tool_call_count"],
            0,
        )
        self.assertEqual(
            model.calls,
            1,
        )


    def test_get_state_history_contains_paused_and_cancelled_snapshots(
            self,
    ):
        model = ApprovalPlanningModel()

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
            profile=AgentProfile(
                name="history-profile",
                require_tool_approval=True,
            ),
        )

        thread_id = "runtime-history-sync-001"

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            paused_result = runtime.invoke(
                "请执行 Python 代码计算 2 加 3",
                thread_id=thread_id,
            )

            self.assertEqual(
                paused_result["status"],
                "paused",
            )

            runtime.cancel(
                thread_id=thread_id,
                cancel_token_id="history-cancel-sync-001",
            )

        history = list(
            runtime.get_state_history(
                thread_id=thread_id,
                limit=10,
            )
        )

        self.assertGreaterEqual(
            len(history),
            2,
        )

        statuses = [
            snapshot.values.get("status")
            for snapshot in history
        ]

        self.assertIn(
            "paused",
            statuses,
        )
        self.assertIn(
            "cancelled",
            statuses,
        )
        self.assertEqual(
            history[0].values["status"],
            "cancelled",
        )


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

    def test_describe_default_tools(self):
        runtime = AgentRuntime()

        result = runtime.describe_tools()

        self.assertEqual(
            len(result),
            1,
        )
        self.assertEqual(
            result[0]["name"],
            "python_executor",
        )
        self.assertTrue(
            result[0]["enabled"],
        )
        self.assertTrue(
            any(
                item["name"] == "code"
                and item["required"] is True
                for item in result[0]["parameters"]
            )
        )

    def test_describe_tools_applies_profile_policy(self):
        profile = AgentProfile(
            name="restricted-profile",
            enabled_tools=frozenset(),
        )

        runtime = AgentRuntime(
            profile=profile,
        )

        result = runtime.describe_tools()

        self.assertEqual(
            len(result),
            1,
        )
        self.assertFalse(
            result[0]["enabled"],
        )

    def test_describe_tools_uses_custom_registry(self):
        def echo(message: str) -> str:
            return message

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="回显文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            registry=registry,
        )

        result = runtime.describe_tools()

        self.assertEqual(
            result,
            [
                {
                    "name": "echo",
                    "description": "回显文本",
                    "parameters": [
                        {
                            "name": "message",
                            "required": True,
                            "annotation": "str",
                        },
                    ],
                    "enabled": True,
                },
            ],
        )

    def test_describe_tools_blocklist_overrides_enabled_tools(self):
        profile = AgentProfile(
            name="blocked-profile",
            enabled_tools=frozenset({"python_executor"}),
            blocked_tools=frozenset({"python_executor"}),
        )

        runtime = AgentRuntime(
            profile=profile,
        )

        result = runtime.describe_tools()

        self.assertEqual(
            len(result),
            1,
        )
        self.assertFalse(
            result[0]["enabled"],
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

        self.assertEqual(
            result,
            events,
        )

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

        self.assertEqual(
            result,
            events,
        )

        command = graph_stream_events.call_args.args[0]
        runtime_config = graph_stream_events.call_args.kwargs["config"]

        self.assertIsInstance(
            command,
            Command,
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "events-thread-003",
        )

    def test_cancel_persists_cancelled_state(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        cancelled_snapshot = {
            "values": {
                "status": "cancelled",
                "is_cancelled": True,
                "cancel_token_id": "cancel-001",
                "pending_tool_calls": [],
            },
        }

        with patch.object(
            runtime.graph,
            "update_state",
            return_value={},
        ) as graph_update_state, patch.object(
            runtime.graph,
            "get_state",
            return_value=cancelled_snapshot,
        ) as graph_get_state:
            result = runtime.cancel(
                thread_id="cancel-thread-001",
                cancel_token_id="cancel-001",
            )

        self.assertEqual(
            result,
            cancelled_snapshot,
        )

        runtime_config = graph_update_state.call_args.args[0]
        update_values = graph_update_state.call_args.args[1]

        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "cancel-thread-001",
        )
        self.assertEqual(
            update_values["is_cancelled"],
            True,
        )
        self.assertEqual(
            update_values["status"],
            "cancelled",
        )
        self.assertEqual(
            update_values["cancel_token_id"],
            "cancel-001",
        )
        self.assertEqual(
            update_values["pending_tool_calls"],
            [],
        )
        graph_get_state.assert_called_once()

    def test_cancel_generates_token_when_missing(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        with patch.object(
            runtime.graph,
            "update_state",
            return_value={},
        ) as graph_update_state, patch.object(
            runtime.graph,
            "get_state",
            return_value={
                "values": {
                    "status": "cancelled",
                },
            },
        ):
            runtime.cancel(
                thread_id="cancel-thread-002",
            )

        update_values = graph_update_state.call_args.args[1]
        generated_token = update_values["cancel_token_id"]

        self.assertIsInstance(
            generated_token,
            str,
        )
        self.assertTrue(
            generated_token,
        )

    def test_cancel_requires_checkpointer(self):
        runtime = AgentRuntime()

        with self.assertRaisesRegex(
            ValueError,
            "checkpointer",
        ):
            runtime.cancel(
                thread_id="cancel-thread-003",
            )

    def test_cancel_rejects_blank_token(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        with self.assertRaisesRegex(
            ValueError,
            "cancel_token_id",
        ):
            runtime.cancel(
                thread_id="cancel-thread-004",
                cancel_token_id="   ",
            )

    def test_thread_id_is_normalized(self):
        runtime = AgentRuntime()

        with patch.object(
            runtime.graph,
            "invoke",
            return_value={
                "status": "completed",
            },
        ) as graph_invoke:
            runtime.invoke(
                "测试线程 ID",
                thread_id="  thread-normalized  ",
            )

        runtime_config = graph_invoke.call_args.kwargs["config"]

        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "thread-normalized",
        )

    def test_thread_id_rejects_non_string(self):
        runtime = AgentRuntime()

        with self.assertRaises(TypeError):
            runtime.invoke(
                "测试请求",
                thread_id=123,
            )

    def test_inspect_thread_returns_summary(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        snapshot = {
            "values": {
                "status": "paused",
                "is_complete": False,
                "is_cancelled": False,
                "cancel_token_id": None,
                "pending_tool_calls": [
                    {
                        "tool_name": "python_executor",
                    },
                ],
                "tool_call_count": 2,
            },
            "interrupts": [
                {
                    "value": {
                        "type": "tool_approval",
                    },
                },
            ],
        }

        with patch.object(
            runtime.graph,
            "get_state",
            return_value=snapshot,
        ):
            result = runtime.inspect_thread(
                thread_id="  inspect-thread-001  ",
            )

        self.assertEqual(
            result["thread_id"],
            "inspect-thread-001",
        )
        self.assertEqual(
            result["status"],
            "paused",
        )
        self.assertFalse(
            result["is_complete"],
        )
        self.assertFalse(
            result["is_cancelled"],
        )
        self.assertEqual(
            result["pending_tool_count"],
            1,
        )
        self.assertEqual(
            result["tool_call_count"],
            2,
        )
        self.assertEqual(
            result["interrupt_count"],
            1,
        )
        self.assertIs(
            result["snapshot"],
            snapshot,
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


    async def test_ainvoke_executes_custom_registered_tool(self):
        def echo(text: str) -> str:
            return text

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="async-custom-tool-profile",
                enabled_tools=frozenset({"echo"}),
            ),
            registry=registry,
        )
        model = CustomToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = await runtime.ainvoke(
                "请调用工具 echo 返回 hello",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            result["tool_results"][0]["tool_name"],
            "echo",
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            "hello",
        )
        self.assertEqual(
            model.calls,
            2,
        )


    async def test_ainvoke_rejects_tool_blocked_by_profile(self):
        attempts = {
            "count": 0,
        }

        def echo(text: str) -> str:
            attempts["count"] += 1
            return text

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="blocked-echo-profile",
                enabled_tools=frozenset({"echo"}),
                blocked_tools=frozenset({"echo"}),
            ),
            registry=registry,
        )
        model = CustomToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = await runtime.ainvoke(
                "请调用工具 echo 返回 hello",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "failed",
        )
        self.assertIn(
            "工具未启用",
            result["final_report"],
        )
        self.assertEqual(
            result.get("tool_results", []),
            [],
        )
        self.assertEqual(
            attempts["count"],
            0,
        )
        self.assertEqual(
            model.calls,
            1,
        )


    async def test_ainvoke_retries_transient_custom_tool_error(self):
        attempts = {
            "count": 0,
        }

        def flaky(text: str) -> dict[str, str]:
            attempts["count"] += 1

            if attempts["count"] < 3:
                raise RuntimeError("temporary failure")

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="flaky",
                    description="一个会暂时失败的测试工具",
                    handler=flaky,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="async-runtime-retry-profile",
                enabled_tools=frozenset({"flaky"}),
                tool_retry_limit=2,
            ),
            registry=registry,
        )
        model = RetryToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = await runtime.ainvoke(
                "请调用工具 flaky 返回 success",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            attempts["count"],
            3,
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            "success",
        )
        self.assertEqual(
            [
                event["event_type"]
                for event in result["tool_events"]
            ],
            [
                "tool_started",
                "tool_retry",
                "tool_retry",
                "tool_completed",
            ],
        )
        self.assertEqual(
            model.calls,
            2,
        )


    async def test_ainvoke_limits_custom_tool_result_length(self):
        attempts = {
            "count": 0,
        }

        def large_output() -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": "abcdefghijklmnopqrstuvwxyz",
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="large_output",
                    description="返回较长文本",
                    handler=large_output,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="async-limited-runtime-result-profile",
                enabled_tools=frozenset({"large_output"}),
                max_tool_result_chars=16,
            ),
            registry=registry,
        )
        model = LongOutputPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = await runtime.ainvoke(
                "请调用工具 large_output 返回长文本",
            )

        output = result["tool_results"][0]["result"]["output"]

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            attempts["count"],
            1,
        )
        self.assertLessEqual(
            len(output),
            16,
        )
        self.assertIn(
            "工具结果已截断",
            output,
        )
        self.assertTrue(
            result["tool_results"][0]["result"]["result_truncated"],
        )
        self.assertEqual(
            model.calls,
            2,
        )


    async def test_ainvoke_reuses_cached_custom_tool_result(self):
        attempts = {
            "count": 0,
        }

        def echo(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="async-cache-profile",
                enabled_tools=frozenset({"echo"}),
                cache_tool_results=True,
            ),
            registry=registry,
        )
        model = RepeatedToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = await runtime.ainvoke(
                "请调用工具 echo 返回 hello",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            attempts["count"],
            1,
        )
        self.assertEqual(
            result["tool_call_count"],
            2,
        )
        self.assertEqual(
            len(result["tool_results"]),
            2,
        )
        self.assertEqual(
            result["tool_results"][0]["result"]["output"],
            "hello",
        )
        self.assertEqual(
            result["tool_results"][1]["result"]["output"],
            "hello",
        )
        self.assertEqual(
            result["tool_events"][-1]["event_type"],
            "tool_cache_hit",
        )
        self.assertEqual(
            model.calls,
            3,
        )


    async def test_ainvoke_enforces_profile_tool_call_limit(self):
        attempts = {
            "count": 0,
        }

        def echo(text: str) -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": text,
                "error": "",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回输入文本",
                    handler=echo,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="async-runtime-limit-profile",
                enabled_tools=frozenset({"echo"}),
                tool_call_limit=1,
            ),
            registry=registry,
        )
        model = RepeatedToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = await runtime.ainvoke(
                "请调用工具 echo 返回 hello",
            )

        self.assertEqual(
            result["route"],
            "agent",
        )
        self.assertEqual(
            result["status"],
            "completed",
        )
        self.assertTrue(
            result["is_complete"],
        )
        self.assertEqual(
            result["tool_call_count"],
            1,
        )
        self.assertEqual(
            len(result["tool_results"]),
            1,
        )
        self.assertEqual(
            attempts["count"],
            1,
        )
        self.assertEqual(
            model.calls,
            1,
        )


    async def test_ainvoke_propagates_structured_tool_error(self):
        attempts = {
            "count": 0,
        }

        def failing_tool() -> dict[str, str]:
            attempts["count"] += 1

            return {
                "output": "",
                "error": "工具内部测试错误",
            }

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="failing_tool",
                    description="返回结构化错误",
                    handler=failing_tool,
                ),
            ]
        )

        runtime = AgentRuntime(
            profile=AgentProfile(
                name="runtime-failure-profile",
                enabled_tools=frozenset({"failing_tool"}),
            ),
            registry=registry,
        )
        model = FailingToolPlanningModel()

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = await runtime.ainvoke(
                "请调用工具 failing_tool",
            )

        self.assertEqual(result["route"], "agent")
        self.assertEqual(result["status"], "failed")
        self.assertFalse(result["is_complete"])
        self.assertEqual(attempts["count"], 1)
        self.assertEqual(model.calls, 1)

        self.assertEqual(
            result["tool_results"][0]["result"]["error"],
            "工具内部测试错误",
        )
        self.assertIn(
            "工具内部测试错误",
            result["errors"],
        )
        self.assertEqual(
            result["tool_events"][-1]["event_type"],
            "tool_failed",
        )


    async def test_ainvoke_pauses_and_aresume_completes_tool_approval(
            self,
    ):
        model = ApprovalPlanningModel()

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
            profile=AgentProfile(
                name="async-approval-profile",
                require_tool_approval=True,
            ),
        )

        thread_id = "runtime-approval-async-001"

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            paused_result = await runtime.ainvoke(
                "请执行 Python 代码计算 2 加 3",
                thread_id=thread_id,
            )

            self.assertEqual(
                paused_result["route"],
                "agent",
            )
            self.assertEqual(
                paused_result["status"],
                "paused",
            )
            self.assertFalse(
                paused_result["is_complete"],
            )
            self.assertIn(
                "__interrupt__",
                paused_result,
            )
            self.assertEqual(
                paused_result["pending_tool_calls"][0]["tool_name"],
                "python_executor",
            )
            self.assertEqual(
                paused_result["tool_results"],
                [],
            )
            self.assertEqual(
                model.calls,
                1,
            )

            resumed_result = await runtime.aresume(
                {
                    "approved": True,
                },
                thread_id=thread_id,
            )

        self.assertEqual(
            resumed_result["status"],
            "completed",
        )
        self.assertTrue(
            resumed_result["is_complete"],
        )
        self.assertEqual(
            resumed_result["tool_call_count"],
            1,
        )
        self.assertEqual(
            resumed_result["tool_results"][0]["result"]["output"].strip(),
            "5",
        )
        self.assertEqual(
            model.calls,
            2,
        )


    async def test_acancel_stops_paused_tool_thread_and_persists_state(
            self,
    ):
        model = ApprovalPlanningModel()

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
            profile=AgentProfile(
                name="async-cancel-profile",
                require_tool_approval=True,
            ),
        )

        thread_id = "runtime-cancel-async-001"

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            paused_result = await runtime.ainvoke(
                "请执行 Python 代码计算 2 加 3",
                thread_id=thread_id,
            )

            self.assertEqual(
                paused_result["status"],
                "paused",
            )
            self.assertEqual(
                model.calls,
                1,
            )

            cancelled_snapshot = await runtime.acancel(
                thread_id=thread_id,
                cancel_token_id="cancel-async-001",
            )

        self.assertEqual(
            cancelled_snapshot.values["status"],
            "cancelled",
        )
        self.assertTrue(
            cancelled_snapshot.values["is_cancelled"],
        )

        summary = await runtime.ainspect_thread(
            thread_id=thread_id,
        )

        self.assertEqual(
            summary["status"],
            "cancelled",
        )
        self.assertTrue(
            summary["is_cancelled"],
        )
        self.assertEqual(
            summary["cancel_token_id"],
            "cancel-async-001",
        )
        self.assertEqual(
            summary["pending_tool_count"],
            0,
        )
        self.assertEqual(
            summary["tool_call_count"],
            0,
        )
        self.assertEqual(
            model.calls,
            1,
        )


    async def test_aget_state_history_contains_paused_and_cancelled_snapshots(
            self,
    ):
        model = ApprovalPlanningModel()

        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
            profile=AgentProfile(
                name="async-history-profile",
                require_tool_approval=True,
            ),
        )

        thread_id = "runtime-history-async-001"

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            paused_result = await runtime.ainvoke(
                "请执行 Python 代码计算 2 加 3",
                thread_id=thread_id,
            )

            self.assertEqual(
                paused_result["status"],
                "paused",
            )

            await runtime.acancel(
                thread_id=thread_id,
                cancel_token_id="history-cancel-async-001",
            )

        history = [
            snapshot
            async for snapshot in runtime.aget_state_history(
                thread_id=thread_id,
                limit=10,
            )
        ]

        self.assertGreaterEqual(
            len(history),
            2,
        )

        statuses = [
            snapshot.values.get("status")
            for snapshot in history
        ]

        self.assertIn(
            "paused",
            statuses,
        )
        self.assertIn(
            "cancelled",
            statuses,
        )
        self.assertEqual(
            history[0].values["status"],
            "cancelled",
        )


    async def test_aresume_forwards_command_and_config(self):
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

        self.assertIsInstance(
            command,
            Command,
        )
        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-events-thread-002",
        )

    async def test_acancel_persists_cancelled_state(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        cancelled_snapshot = {
            "values": {
                "status": "cancelled",
                "is_cancelled": True,
                "cancel_token_id": "async-cancel-001",
                "pending_tool_calls": [],
            },
        }

        with patch.object(
            runtime.graph,
            "aupdate_state",
            new_callable=AsyncMock,
            return_value={},
        ) as graph_aupdate_state, patch.object(
            runtime.graph,
            "aget_state",
            new_callable=AsyncMock,
            return_value=cancelled_snapshot,
        ) as graph_aget_state:
            result = await runtime.acancel(
                thread_id="async-cancel-thread-001",
                cancel_token_id="async-cancel-001",
            )

        self.assertEqual(
            result,
            cancelled_snapshot,
        )

        runtime_config = graph_aupdate_state.call_args.args[0]
        update_values = graph_aupdate_state.call_args.args[1]

        self.assertEqual(
            runtime_config["configurable"]["thread_id"],
            "async-cancel-thread-001",
        )
        self.assertTrue(
            update_values["is_cancelled"],
        )
        self.assertEqual(
            update_values["status"],
            "cancelled",
        )
        self.assertEqual(
            update_values["cancel_token_id"],
            "async-cancel-001",
        )
        self.assertEqual(
            update_values["pending_tool_calls"],
            [],
        )
        graph_aupdate_state.assert_awaited_once()
        graph_aget_state.assert_awaited_once()

    async def test_ainspect_thread_returns_summary(self):
        runtime = AgentRuntime(
            checkpointer=InMemorySaver(),
        )

        snapshot = {
            "values": {
                "status": "cancelled",
                "is_complete": False,
                "is_cancelled": True,
                "cancel_token_id": "async-cancel-001",
                "pending_tool_calls": [],
                "tool_call_count": 3,
            },
            "interrupts": [],
        }

        with patch.object(
            runtime.graph,
            "aget_state",
            new_callable=AsyncMock,
            return_value=snapshot,
        ):
            result = await runtime.ainspect_thread(
                thread_id="async-inspect-thread-001",
            )

        self.assertEqual(
            result["thread_id"],
            "async-inspect-thread-001",
        )
        self.assertEqual(
            result["status"],
            "cancelled",
        )
        self.assertTrue(
            result["is_cancelled"],
        )
        self.assertEqual(
            result["cancel_token_id"],
            "async-cancel-001",
        )
        self.assertEqual(
            result["tool_call_count"],
            3,
        )
        self.assertEqual(
            result["pending_tool_count"],
            0,
        )
        self.assertEqual(
            result["interrupt_count"],
            0,
        )


if __name__ == "__main__":
    unittest.main()