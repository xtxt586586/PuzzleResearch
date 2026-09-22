import unittest

from fastapi.testclient import TestClient

from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)
import asyncio

from agent.core.event_stream import (
    get_emitter,
    remove_emitter_sync,
)

import main
import json
from agent.core.events import RuntimeEvent
from main import app


class PublicApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_root_returns_healthy_service(self):
        response = self.client.get("/")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(payload["status"], "healthy")
        self.assertEqual(
            payload["service"],
            "PuzzleResearch Research Agent",
        )
        self.assertEqual(payload["version"], "0.1.0")

    def test_health_returns_runtime_snapshot(self):
        response = self.client.get("/health")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(payload["status"], "healthy")
        self.assertIn(
            payload["database"],
            {
                "configured",
                "not configured",
            },
        )
        self.assertIsInstance(
            payload["uptime_seconds"],
            float,
        )
        self.assertTrue(payload["timestamp"])

    def test_agent_health_returns_local_capabilities(self):
        response = self.client.get("/api/health/agent")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(
            payload["agents_count"],
            1,
        )

        self.assertEqual(
            payload["agent_ids"],
            ["default"],
        )
        self.assertGreaterEqual(
            payload["tool_registry_total_tools"],
            1,
        )
        self.assertIn(
            "search_providers_available",
            payload,
        )

    def test_public_config_does_not_expose_api_keys(self):
        response = self.client.get("/api/config/public")

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(
            payload["defaults"]["primary_model"],
            "deepseek-chat",
        )
        self.assertIn(
            "deepseek-chat",
            payload["models"]["options"],
        )
        self.assertFalse(
            payload["features"]["mcp_enabled"],
        )
        self.assertFalse(
            payload["features"]["prometheus_enabled"],
        )

        serialized = str(payload).lower()

        self.assertNotIn(
            "deepseek_api_key",
            serialized,
        )
        self.assertNotIn(
            "tavily_api_key",
            serialized,
        )


    def test_chat_non_stream_returns_response(self):
        with patch.object(
            main.chat_runtime,
            "ainvoke",
            new=AsyncMock(
                return_value={
                    "final_report": "测试回答",
                    "status": "completed",
                }
            ),
        ) as invoke:
            response = self.client.post(
                "/api/chat",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "你好",
                        }
                    ],
                    "stream": False,
                    "search_mode": {
                        "useDeepSearch": True,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["content"],
            "测试回答",
        )

        config = invoke.await_args.kwargs["config"]

        self.assertEqual(
            config["configurable"]["search_mode"]["mode"],
            "deep",
        )
        self.assertTrue(
            config["configurable"]["thread_id"].startswith(
                "thread_"
            )
        )

    def test_chat_rejects_missing_user_message(self):
        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "assistant",
                        "content": "旧回答",
                    }
                ],
                "stream": False,
            },
        )

        self.assertEqual(response.status_code, 400)

    def test_chat_stream_uses_legacy_protocol(self):
        async def fake_events(*args, **kwargs):
            yield RuntimeEvent(
                event="on_chain_end",
                name="finalize",
                data={
                    "output": {
                        "final_report": "流式回答",
                    }
                },
            )

        with patch.object(
            main.chat_runtime,
            "astream_runtime_events",
            new=fake_events,
        ):
            response = self.client.post(
                "/api/chat",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "测试流式回答",
                        }
                    ],
                    "stream": True,
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith(
                "text/event-stream"
            )
        )
        self.assertTrue(
            response.headers["x-thread-id"].startswith(
                "thread_"
            )
        )
        self.assertIn(
            '0:{"type":"completion"',
            response.text,
        )
        self.assertIn(
            '"流式回答"',
            response.text,
        )

    def test_chat_sse_returns_standard_frames(self):
        async def fake_events(*args, **kwargs):
            yield RuntimeEvent(
                event="on_chain_end",
                name="finalize",
                data={
                    "output": {
                        "final_report": "SSE 回答",
                    }
                },
            )

        with patch.object(
            main.chat_runtime,
            "astream_runtime_events",
            new=fake_events,
        ):
            response = self.client.post(
                "/api/chat/sse",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "测试 SSE",
                        }
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "retry: 2000",
            response.text,
        )
        self.assertIn(
            "event: completion",
            response.text,
        )
        self.assertIn(
            "event: done",
            response.text,
        )
        self.assertTrue(
            response.headers["x-thread-id"].startswith(
                "thread_"
            )
        )
        frames = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

        completion_frame = next(
            frame
            for frame in frames
            if frame.get("type") == "completion"
        )

        self.assertEqual(
            completion_frame["data"]["content"],
            "SSE 回答",
        )

        self.assertTrue(
            any(
                frame.get("type") == "done"
                for frame in frames
            )
        )

    def test_chat_sse_uses_error_event_on_runtime_failure(self):
        async def failing_events(*args, **kwargs):
            if False:
                yield RuntimeEvent(
                    event="unused",
                    data={},
                )

            raise RuntimeError("测试流错误")

        with patch.object(
            main.chat_runtime,
            "astream_runtime_events",
            new=failing_events,
        ):
            response = self.client.post(
                "/api/chat/sse",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "触发错误",
                        }
                    ]
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            "event: error",
            response.text,
        )

        frames = [
            json.loads(line.removeprefix("data: "))
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

        error_frame = next(
            frame
            for frame in frames
            if frame.get("type") == "error"
        )

        self.assertEqual(
            error_frame["data"]["message"],
            "测试流错误",
        )


    def test_cancel_chat_cancels_active_task(self):
        task = MagicMock()
        task.done.return_value = False

        main._active_chat_tasks[
            "thread-cancel-001"
        ] = task

        try:
            response = self.client.post(
                "/api/chat/cancel/thread-cancel-001",
                json={
                    "reason": "用户停止任务",
                },
            )
        finally:
            main._active_chat_tasks.clear()

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "cancelled",
        )
        self.assertEqual(
            payload["thread_id"],
            "thread-cancel-001",
        )
        self.assertEqual(
            payload["reason"],
            "用户停止任务",
        )

        task.cancel.assert_called_once()


    def test_cancel_chat_returns_not_found_for_finished_task(self):
        response = self.client.post(
            "/api/chat/cancel/not-found-thread",
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        self.assertEqual(
            response.json()["status"],
            "not_found",
        )


    def test_cancel_all_chats_cancels_active_tasks(self):
        first_task = MagicMock()
        first_task.done.return_value = False

        second_task = MagicMock()
        second_task.done.return_value = False

        main._active_chat_tasks.update(
            {
                "thread-cancel-001": first_task,
                "thread-cancel-002": second_task,
            }
        )

        try:
            response = self.client.post(
                "/api/chat/cancel-all",
            )
        finally:
            main._active_chat_tasks.clear()

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "all_cancelled",
        )
        self.assertEqual(
            payload["cancelled_count"],
            2,
        )

        first_task.cancel.assert_called_once()
        second_task.cancel.assert_called_once()

    def test_research_returns_legacy_stream(self):
        async def fake_events(*args, **kwargs):
            yield RuntimeEvent(
                event="on_chain_end",
                name="finalize",
                data={
                    "output": {
                        "final_report": "Research 回答",
                    }
                },
            )

        with patch.object(
                main.chat_runtime,
                "astream_runtime_events",
                new=fake_events,
        ):
            response = self.client.post(
                "/api/research",
                params={"query": "测试研究"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn(
            '0:{"type":"completion"',
            response.text,
        )
        self.assertTrue(
            response.headers["x-thread-id"].startswith(
                "thread_"
            )
        )

    def test_research_sse_returns_standard_stream(self):
        async def fake_events(*args, **kwargs):
            yield RuntimeEvent(
                event="on_chain_end",
                name="finalize",
                data={
                    "output": {
                        "final_report": "Research SSE 回答",
                    }
                },
            )

        with patch.object(
                main.chat_runtime,
                "astream_runtime_events",
                new=fake_events,
        ):
            response = self.client.post(
                "/api/research/sse",
                json={
                    "query": "测试 SSE 研究",
                    "search_mode": {
                        "useDeepSearch": True,
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("event: completion", response.text)
        self.assertIn("event: done", response.text)
        self.assertTrue(
            response.headers["x-thread-id"].startswith(
                "thread_"
            )
        )

    def test_research_rejects_empty_query(self):
        response = self.client.post(
            "/api/research/sse",
            json={"query": "   "},
        )

        self.assertEqual(response.status_code, 400)

    def test_session_evidence_returns_contract(self):
        with patch.object(
                main.session_manager,
                "get_state",
                return_value={
                    "thread_id": "thread-evidence-001",
                    "state": {
                        "sources": [
                            {
                                "title": "Example",
                                "url": "https://example.com",
                                "provider": "test",
                            }
                        ],
                        "quality_summary": {
                            "score": 0.9,
                        },
                        "deepsearch_artifacts": {
                            "claims": [],
                            "fetched_pages": [],
                            "passages": [],
                        },
                    },
                    "interrupts": [],
                    "checkpoint_id": "checkpoint-001",
                },
        ):
            response = self.client.get(
                "/api/sessions/thread-evidence-001/evidence",
            )

        self.assertEqual(response.status_code, 200)

        payload = response.json()

        self.assertEqual(
            set(payload),
            {
                "sources",
                "claims",
                "quality_summary",
                "fetched_pages",
                "passages",
            },
        )
        self.assertEqual(
            payload["sources"][0]["url"],
            "https://example.com",
        )

    def test_session_evidence_returns_404_for_unknown_thread(self):
        response = self.client.get(
            "/api/sessions/thread-evidence-missing/evidence",
        )

        self.assertEqual(response.status_code, 404)

    def test_events_replay_buffered_events(self):
        thread_id = "thread-events-001"

        async def seed_events():
            emitter = await get_emitter(thread_id)

            await emitter.emit(
                "tool_start",
                {
                    "tool": "echo",
                },
            )

            await emitter.emit(
                "done",
                {},
            )

        asyncio.run(seed_events())

        try:
            response = self.client.get(
                f"/api/events/{thread_id}",
            )
        finally:
            remove_emitter_sync(thread_id)

        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            response.headers["content-type"].startswith(
                "text/event-stream"
            )
        )
        self.assertIn("id: 1", response.text)
        self.assertIn("id: 2", response.text)
        self.assertNotIn(
            "event: tool_start",
            response.text,
        )

        frames = [
            json.loads(
                line.removeprefix("data: ")
            )
            for line in response.text.splitlines()
            if line.startswith("data: ")
        ]

        self.assertEqual(
            frames[0]["type"],
            "tool_start",
        )
        self.assertEqual(
            frames[1]["type"],
            "done",
        )

    def test_events_resume_after_last_event_id(self):
        thread_id = "thread-events-002"

        async def seed_events():
            emitter = await get_emitter(thread_id)

            await emitter.emit(
                "tool_start",
                {
                    "tool": "echo",
                },
            )

            await emitter.emit(
                "done",
                {},
            )

        asyncio.run(seed_events())

        try:
            response = self.client.get(
                f"/api/events/{thread_id}",
                headers={
                    "Last-Event-ID": "1",
                },
            )
        finally:
            remove_emitter_sync(thread_id)

        self.assertEqual(response.status_code, 200)
        self.assertNotIn(
            '"type": "tool_start"',
            response.text,
        )
        self.assertIn(
            '"type": "done"',
            response.text,
        )

    def test_events_are_isolated_by_thread_id(self):
        first_thread_id = "thread-events-isolated-001"
        second_thread_id = "thread-events-isolated-002"

        async def seed_events():
            first_emitter = await get_emitter(
                first_thread_id
            )
            second_emitter = await get_emitter(
                second_thread_id
            )

            await first_emitter.emit(
                "tool_result",
                {
                    "tool": "first",
                },
            )
            await first_emitter.emit(
                "done",
                {},
            )
            await second_emitter.emit(
                "tool_result",
                {
                    "tool": "second",
                },
            )
            await second_emitter.emit(
                "done",
                {},
            )

        asyncio.run(seed_events())

        try:
            response = self.client.get(
                f"/api/events/{first_thread_id}",
            )
        finally:
            remove_emitter_sync(first_thread_id)
            remove_emitter_sync(second_thread_id)

        self.assertEqual(response.status_code, 200)
        self.assertIn('"tool": "first"', response.text)
        self.assertNotIn('"tool": "second"', response.text)

    def test_events_resume_after_terminal_cursor_finishes(self):
        thread_id = "thread-events-terminal-001"

        async def seed_events():
            emitter = await get_emitter(thread_id)
            await emitter.emit(
                "done",
                {},
            )

        asyncio.run(seed_events())

        try:
            response = self.client.get(
                f"/api/events/{thread_id}",
                headers={
                    "Last-Event-ID": "1",
                },
            )
        finally:
            remove_emitter_sync(thread_id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.text, "")

    def test_api_key_middleware_rejects_invalid_key(self):
        with patch.object(
                main,
                "configured_api_key",
                "secret-key",
        ):
            response = self.client.get(
                "/api/config/public",
                headers={
                    "Authorization": "Bearer wrong-key",
                },
            )

        self.assertEqual(
            response.status_code,
            401,
        )

    def test_api_key_middleware_accepts_valid_key(self):
        with patch.object(
                main,
                "configured_api_key",
                "secret-key",
        ):
            response = self.client.get(
                "/api/config/public",
                headers={
                    "Authorization": "Bearer secret-key",
                    "X-Weaver-User": "user-001",
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )

    def test_authenticated_request_cannot_override_owner_with_body_user_id(self):
        with patch.object(
                main,
                "configured_api_key",
                "secret-key",
        ):
            with patch.object(
                    main.chat_runtime,
                    "ainvoke",
                    new=AsyncMock(
                        return_value={
                            "final_report": "测试回答",
                            "status": "completed",
                        },
                    ),
            ) as invoke:
                response = self.client.post(
                    "/api/chat",
                    headers={
                        "Authorization": "Bearer secret-key",
                        "X-Weaver-User": "owner-001",
                    },
                    json={
                        "messages": [
                            {
                                "role": "user",
                                "content": "测试",
                            },
                        ],
                        "stream": False,
                        "user_id": "attacker-001",
                    },
                )

        self.assertEqual(
            response.status_code,
            200,
        )

        runtime_config = invoke.call_args.kwargs["config"]

        self.assertEqual(
            runtime_config["configurable"]["user_id"],
            "owner-001",
        )


if __name__ == "__main__":
    unittest.main()
