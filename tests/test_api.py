import unittest

from fastapi.testclient import TestClient

from unittest.mock import AsyncMock, patch

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

        self.assertEqual(payload["agents_count"], 0)
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


if __name__ == "__main__":
    unittest.main()