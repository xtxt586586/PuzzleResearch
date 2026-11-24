import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main
from common import agent_store


class AgentProfileApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.agents_path = (
            Path(self.temp_dir.name)
            / "agents.json"
        )

        self.path_patch = patch.object(
            agent_store,
            "AGENTS_PATH",
            self.agents_path,
        )
        self.path_patch.start()

        self.client = TestClient(main.app)

    def tearDown(self):
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_chat_injects_selected_agent_profile(self):
        create_response = self.client.post(
            "/api/agents",
            json={
                "id": "restricted-agent",
                "name": "Restricted Agent",
                "model": "deepseek-reasoner",
                "system_prompt": "必须使用严谨中文回答。",
                "enabled_tools": {
                    "python_executor": True,
                    "echo": False,
                },
                "metadata": {
                    "runtime_policy": {
                        "tool_call_limit": 1,
                        "require_tool_approval": True,
                        "tool_retry_limit": 2,
                        "browser_context_helper": True,
                    }
                },
            },
        )

        self.assertEqual(
            create_response.status_code,
            201,
        )

        with patch.object(
            main.chat_runtime,
            "ainvoke",
            new=AsyncMock(
                return_value={
                    "final_report": "测试完成",
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
                            "content": "测试 Agent",
                        }
                    ],
                    "stream": False,
                    "agent_id": "restricted-agent",
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )

        runtime_config = (
            invoke.await_args.kwargs["config"]
        )
        profile = (
            runtime_config["configurable"]
            ["agent_profile"]
        )

        self.assertEqual(
            profile.name,
            "Restricted Agent",
        )
        self.assertEqual(
            profile.system_prompt,
            "必须使用严谨中文回答。",
        )
        self.assertEqual(
            profile.enabled_tools,
            frozenset({"python_executor"}),
        )
        self.assertEqual(
            profile.blocked_tools,
            frozenset({"echo"}),
        )
        self.assertEqual(
            profile.tool_call_limit,
            1,
        )
        self.assertTrue(
            profile.require_tool_approval
        )
        self.assertEqual(
            profile.tool_retry_limit,
            2,
        )
        self.assertTrue(profile.browser_context_helper)

        self.assertEqual(
            runtime_config["configurable"]["model"],
            "deepseek-reasoner",
        )
        self.assertEqual(
            runtime_config["configurable"]["model_override"],
            "deepseek-reasoner",
        )

    def test_unknown_agent_is_rejected(self):
        response = self.client.post(
            "/api/chat",
            json={
                "messages": [
                    {
                        "role": "user",
                        "content": "测试未知 Agent",
                    }
                ],
                "stream": False,
                "agent_id": "missing-agent",
            },
        )

        self.assertEqual(
            response.status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
