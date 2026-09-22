import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main


class SupportApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.path_patch = patch.object(
            main.settings,
            "memory_store_path",
            str(
                Path(self.temp_dir.name)
                / "support_memory.json"
            ),
        )
        self.enabled_patch = patch.object(
            main.settings,
            "enable_memory",
            True,
        )

        self.path_patch.start()
        self.enabled_patch.start()

        self.client = TestClient(main.app)

    def tearDown(self):
        self.enabled_patch.stop()
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_memory_status_returns_local_backend(self):
        response = self.client.get(
            "/api/memory/status"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["backend"],
            "memory",
        )
        self.assertTrue(
            payload["local_fallback"],
        )
        self.assertIn(
            "checkpointer",
            payload,
        )

    def test_support_chat_uses_runtime_and_saves_memory(self):
        with patch.object(
            main.chat_runtime,
            "ainvoke",
            new=AsyncMock(
                return_value={
                    "final_report": "这是支持回答。",
                }
            ),
        ) as invoke:
            response = self.client.post(
                "/api/support/chat",
                json={
                    "message": "如何使用 PuzzleResearch？",
                    "user_id": "user-001",
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["content"],
            "这是支持回答。",
        )

        runtime_config = (
            invoke.await_args.kwargs["config"]
        )

        self.assertEqual(
            runtime_config["configurable"]["user_id"],
            "user-001",
        )

        status = self.client.get(
            "/api/memory/status"
        ).json()

        self.assertEqual(
            status["local_store"]["entries"],
            1,
        )

    def test_authenticated_user_cannot_be_overridden(self):
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
                        "final_report": "认证回答",
                    }
                ),
            ) as invoke:
                response = self.client.post(
                    "/api/support/chat",
                    headers={
                        "Authorization": (
                            "Bearer secret-key"
                        ),
                        "X-Weaver-User": "owner-001",
                    },
                    json={
                        "message": "测试身份隔离",
                        "user_id": "attacker-001",
                    },
                )

        self.assertEqual(
            response.status_code,
            200,
        )

        runtime_config = (
            invoke.await_args.kwargs["config"]
        )

        self.assertEqual(
            runtime_config["configurable"]["user_id"],
            "owner-001",
        )

    def test_empty_message_is_rejected(self):
        response = self.client.post(
            "/api/support/chat",
            json={
                "message": "   ",
            },
        )

        self.assertEqual(
            response.status_code,
            422,
        )


if __name__ == "__main__":
    unittest.main()