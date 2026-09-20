import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main


class FakeSessionManager:
    def __init__(self):
        self.sessions = {
            "thread-001": {
                "thread_id": "thread-001",
                "status": "paused",
                "topic": "测试主题",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:01:00+00:00",
                "route": "agent",
                "has_report": False,
                "revision_count": 0,
                "message_count": 2,
                "agent_id": "default",
                "user_id": "anonymous",
            }
        }

    def list_sessions(
        self,
        *,
        limit=50,
        status=None,
    ):
        values = list(self.sessions.values())

        if status:
            values = [
                item
                for item in values
                if item["status"] == status
            ]

        return values[:limit]

    def get_session(self, thread_id):
        if thread_id not in self.sessions:
            from common.session_manager import (
                SessionNotFoundError,
            )

            raise SessionNotFoundError(
                f"线程不存在：{thread_id}"
            )

        return self.sessions[thread_id]

    def get_state(self, thread_id):
        self.get_session(thread_id)

        return {
            "thread_id": thread_id,
            "status": "paused",
            "state": {
                "status": "paused",
                "route": "agent",
            },
            "interrupts": [
                {
                    "type": "tool_approval",
                    "tool_name": "python_executor",
                }
            ],
            "checkpoint_id": "checkpoint-001",
        }

    def delete_session(self, thread_id):
        self.get_session(thread_id)
        del self.sessions[thread_id]


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.manager = FakeSessionManager()
        self.manager_patch = patch.object(
            main,
            "session_manager",
            self.manager,
        )
        self.manager_patch.start()

    def tearDown(self):
        self.manager_patch.stop()

    def test_list_sessions(self):
        response = self.client.get(
            "/api/sessions"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["count"],
            1,
        )
        self.assertEqual(
            payload["sessions"][0]["thread_id"],
            "thread-001",
        )

    def test_get_session_state(self):
        response = self.client.get(
            "/api/sessions/thread-001/state"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "paused",
        )
        self.assertEqual(
            payload["state"]["route"],
            "agent",
        )
        self.assertEqual(
            payload["checkpoint_id"],
            "checkpoint-001",
        )

    def test_interrupt_status(self):
        response = self.client.get(
            "/api/interrupt/thread-001/status"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertTrue(
            payload["is_interrupted"]
        )
        self.assertIn(
            "approve",
            payload["available_actions"],
        )

    def test_missing_session_returns_404(self):
        response = self.client.get(
            "/api/sessions/missing-thread"
        )

        self.assertEqual(
            response.status_code,
            404,
        )

    def test_delete_session(self):
        response = self.client.delete(
            "/api/sessions/thread-001"
        )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["status"],
            "deleted",
        )

    def test_resume_session(self):
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(
                return_value={
                    "status": "completed",
                    "final_report": "恢复完成",
                }
            ),
        ) as resume:
            response = self.client.post(
                "/api/sessions/thread-001/resume",
                json={
                    "resume_value": True,
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["status"],
            "resumed",
        )
        resume.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()