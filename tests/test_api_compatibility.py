import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main
from agent.core.profile import AgentProfile


class FakeBrowserLiveView:
    async def start(self, url):
        return {
            "url": url,
            "title": "",
        }

    async def capture(
        self,
        *,
        full_page=False,
        quality=70,
    ):
        return {
            "data": "ZmFrZQ==",
            "metadata": {
                "url": "about:blank",
                "title": "",
            },
        }

    async def stop(self):
        return None


class ApiCompatibilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def setUp(self):
        self.old_mcp_enabled = main._mcp_enabled
        self.old_mcp_servers = dict(
            main._mcp_servers_config
        )
        self.old_mcp_loaded = main._mcp_loaded_tools

    def tearDown(self):
        main._mcp_enabled = self.old_mcp_enabled
        main._mcp_servers_config = self.old_mcp_servers
        main._mcp_loaded_tools = self.old_mcp_loaded

    def test_mcp_get_returns_contract(self):
        response = self.client.get(
            "/api/mcp/config"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            set(payload),
            {
                "enabled",
                "servers",
                "loaded_tools",
            },
        )

        self.assertIsInstance(
            payload["servers"],
            dict,
        )

    def test_mcp_post_updates_local_state(self):
        response = self.client.post(
            "/api/mcp/config",
            json={
                "enable": True,
                "servers": {
                    "demo": {
                        "url": "http://localhost:9000"
                    }
                },
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertTrue(payload["enabled"])
        self.assertEqual(
            payload["servers"]["demo"]["url"],
            "http://localhost:9000",
        )
        self.assertEqual(
            payload["loaded_tools"],
            0,
        )

    def test_browser_diagnose_returns_contract(self):
        response = self.client.get(
            "/api/sandbox/browser/diagnose"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            set(payload),
            {
                "ready",
                "missing",
                "sandbox_mode",
                "allow_internet",
                "configured",
                "deep",
            },
        )

        self.assertEqual(
            set(payload["configured"]),
            {
                "e2b_api_key",
                "sandbox_template_browser",
                "e2b_code_interpreter",
                "playwright",
            },
        )

    def test_browser_diagnose_deep_is_best_effort(self):
        with patch.object(
            main,
            "BrowserLiveView",
            FakeBrowserLiveView,
        ):
            response = self.client.get(
                "/api/sandbox/browser/diagnose"
                "?deep=true"
            )

        self.assertEqual(
            response.status_code,
            200,
        )

        deep = response.json()["deep"]

        self.assertTrue(deep["ok"])
        self.assertEqual(
            deep["frame_bytes"],
            4,
        )

    def test_interrupt_resume_success(self):
        runtime_result = {
            "final_report": "恢复完成",
            "status": "completed",
        }

        with patch.object(
            main,
            "_session_manager_with_owner",
            return_value={},
        ), patch.object(
            main,
            "_resolve_agent_runtime",
            return_value=(
                SimpleNamespace(model=""),
                AgentProfile(name="test"),
            ),
        ), patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(
                return_value=runtime_result
            ),
        ) as resume, patch.object(
            main.chat_runtime,
            "aget_state",
            return_value={"values": {"conversation_history": []}},
        ), patch.object(
            main.chat_runtime,
            "aupdate_state",
            new=AsyncMock(),
        ) as update_state:
            response = self.client.post(
                "/api/interrupt/resume",
                json={
                    "thread_id": "thread-resume-001",
                    "payload": {
                        "approved": True,
                    },
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["content"],
            "恢复完成",
        )

        resume.assert_awaited_once()
        self.assertTrue(
            resume.await_args.args[0]
        )
        update_state.assert_awaited_once_with(
            thread_id="thread-resume-001",
            values={
                "conversation_history": [
                    {"role": "assistant", "content": "恢复完成"},
                ],
            },
        )

    def test_interrupt_resume_normalizes_native_edit_decision(self):
        edited_action = {
            "name": "python_executor",
            "args": {"code": "print(6 * 7)"},
        }
        with patch.object(
            main,
            "_session_manager_with_owner",
            return_value={},
        ), patch.object(
            main,
            "_resolve_agent_runtime",
            return_value=(
                SimpleNamespace(model=""),
                AgentProfile(name="test"),
            ),
        ), patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(
                return_value={
                    "final_report": "恢复完成",
                    "status": "completed",
                }
            ),
        ) as resume, patch.object(
            main.chat_runtime,
            "aget_state",
            return_value={"values": {"conversation_history": []}},
        ), patch.object(
            main.chat_runtime,
            "aupdate_state",
            new=AsyncMock(),
        ) as update_state:
            response = self.client.post(
                "/api/interrupt/resume",
                json={
                    "thread_id": "thread-native-edit-001",
                    "payload": {
                        "decisions": [
                            {
                                "type": "edit",
                                "edited_action": edited_action,
                            }
                        ]
                    },
                },
            )

        self.assertEqual(response.status_code, 200)
        resume.assert_awaited_once()
        self.assertEqual(
            resume.await_args.args[0],
            {
                "type": "edit",
                "edited_action": edited_action,
            },
        )
        update_state.assert_awaited_once_with(
            thread_id="thread-native-edit-001",
            values={
                "conversation_history": [
                    {"role": "assistant", "content": "恢复完成"},
                ],
            },
        )

    def test_interrupt_resume_rejects_multiple_native_decisions(self):
        with patch.object(
            main,
            "_session_manager_with_owner",
            return_value={},
        ), patch.object(
            main,
            "_resolve_agent_runtime",
            return_value=(
                SimpleNamespace(model=""),
                AgentProfile(name="test"),
            ),
        ), patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(),
        ) as resume:
            response = self.client.post(
                "/api/interrupt/resume",
                json={
                    "thread_id": "thread-native-edit-batch-001",
                    "payload": {
                        "decisions": [
                            {"type": "approve"},
                            {"type": "reject"},
                        ]
                    },
                },
            )

        self.assertEqual(response.status_code, 400)
        resume.assert_not_awaited()

    def test_interrupt_resume_rejects_blank_thread_id(self):
        response = self.client.post(
            "/api/interrupt/resume",
            json={
                "thread_id": "   ",
                "payload": True,
            },
        )

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_interrupt_resume_returns_not_found(self):
        with patch.object(
            main,
            "_session_manager_with_owner",
            side_effect=main.SessionNotFoundError(
                "线程不存在"
            ),
        ):
            response = self.client.post(
                "/api/interrupt/resume",
                json={
                    "thread_id": "missing-thread",
                    "payload": True,
                },
            )

        self.assertEqual(
            response.status_code,
            404,
        )

    def test_interrupt_resume_returns_forbidden(self):
        with patch.object(
            main,
            "_session_manager_with_owner",
            side_effect=main.SessionOwnerError(
                "无权访问"
            ),
        ):
            response = self.client.post(
                "/api/interrupt/resume",
                json={
                    "thread_id": "other-thread",
                    "payload": True,
                },
            )

        self.assertEqual(
            response.status_code,
            403,
        )

    def test_openapi_exposes_compatibility_paths(self):
        paths = main.app.openapi()["paths"]

        self.assertIn(
            "/api/interrupt/resume",
            paths,
        )
        self.assertIn(
            "/api/mcp/config",
            paths,
        )
        self.assertIn(
            "/api/mcp/diagnose",
            paths,
        )
        self.assertIn(
            "/api/sandbox/browser/diagnose",
            paths,
        )


if __name__ == "__main__":
    unittest.main()
