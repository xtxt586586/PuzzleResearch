import json
import re
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from common import agent_store


class AgentApiTests(unittest.TestCase):
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

    def test_list_returns_default_agent(self):
        response = self.client.get(
            "/api/agents"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            len(payload["agents"]),
            1,
        )
        self.assertEqual(
            payload["agents"][0]["id"],
            "default",
        )
        self.assertFalse(
            self.agents_path.exists()
        )

    def test_create_generates_agent_id(self):
        response = self.client.post(
            "/api/agents",
            json={
                "name": "Research Agent",
                "description": "测试 Agent",
                "system_prompt": "回答研究问题",
                "model": "deepseek-chat",
                "enabled_tools": {
                    "python_executor": True,
                },
                "metadata": {
                    "source": "test",
                },
            },
        )

        self.assertEqual(
            response.status_code,
            201,
        )

        payload = response.json()

        self.assertRegex(
            payload["id"],
            r"^agent_[0-9a-f]{10}$",
        )
        self.assertEqual(
            payload["name"],
            "Research Agent",
        )
        self.assertTrue(
            self.agents_path.exists()
        )

    def test_get_returns_created_agent(self):
        created = self.client.post(
            "/api/agents",
            json={
                "id": "custom-agent",
                "name": "Custom Agent",
            },
        )

        self.assertEqual(
            created.status_code,
            201,
        )

        response = self.client.get(
            "/api/agents/custom-agent"
        )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["name"],
            "Custom Agent",
        )

    def test_update_preserves_created_at(self):
        created = self.client.post(
            "/api/agents",
            json={
                "id": "update-agent",
                "name": "Before",
            },
        ).json()

        response = self.client.put(
            "/api/agents/update-agent",
            json={
                "name": "After",
                "description": "已更新",
                "system_prompt": "新的系统提示",
                "model": "deepseek-reasoner",
                "enabled_tools": {
                    "python_executor": False,
                },
                "metadata": {
                    "updated": True,
                },
            },
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["name"],
            "After",
        )
        self.assertEqual(
            payload["description"],
            "已更新",
        )
        self.assertEqual(
            payload["created_at"],
            created["created_at"],
        )
        self.assertEqual(
            payload["id"],
            "update-agent",
        )

    def test_delete_returns_deleted_status(self):
        self.client.post(
            "/api/agents",
            json={
                "id": "delete-agent",
                "name": "Delete Me",
            },
        )

        response = self.client.delete(
            "/api/agents/delete-agent"
        )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json(),
            {
                "status": "deleted",
                "id": "delete-agent",
            },
        )

        missing = self.client.get(
            "/api/agents/delete-agent"
        )

        self.assertEqual(
            missing.status_code,
            404,
        )

    def test_duplicate_id_is_rejected(self):
        body = {
            "id": "duplicate-agent",
            "name": "Duplicate",
        }

        first = self.client.post(
            "/api/agents",
            json=body,
        )
        second = self.client.post(
            "/api/agents",
            json=body,
        )

        self.assertEqual(
            first.status_code,
            201,
        )
        self.assertEqual(
            second.status_code,
            409,
        )

    def test_default_agent_cannot_be_updated(self):
        response = self.client.put(
            "/api/agents/default",
            json={
                "name": "Changed Default",
            },
        )

        self.assertEqual(
            response.status_code,
            403,
        )

    def test_default_agent_cannot_be_deleted(self):
        response = self.client.delete(
            "/api/agents/default"
        )

        self.assertEqual(
            response.status_code,
            403,
        )

    def test_missing_agent_returns_not_found(self):
        get_response = self.client.get(
            "/api/agents/missing-agent"
        )
        delete_response = self.client.delete(
            "/api/agents/missing-agent"
        )

        self.assertEqual(
            get_response.status_code,
            404,
        )
        self.assertEqual(
            delete_response.status_code,
            404,
        )

    def test_invalid_store_returns_server_error(self):
        self.agents_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self.agents_path.write_text(
            "{invalid-json",
            encoding="utf-8",
        )

        response = self.client.get(
            "/api/agents"
        )

        self.assertEqual(
            response.status_code,
            500,
        )


if __name__ == "__main__":
    unittest.main()