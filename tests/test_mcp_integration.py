from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from common import agent_store
from agent.workflows.nodes import _get_agent_tool_registry
from agent.core.profile import AgentProfile
from tools.builtin import create_default_registry


FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


class MCPIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.agent_path_patch = patch.object(
            agent_store,
            "AGENTS_PATH",
            Path(self.temp_dir.name) / "agents.json",
        )
        self.agent_path_patch.start()
        self.old_enabled = main._mcp_enabled
        self.old_servers = dict(main._mcp_servers_config)
        self.old_loaded = main._mcp_loaded_tools
        self.command_allowlist = patch.object(
            main.settings,
            "mcp_allowed_commands",
            sys.executable,
        )
        self.command_allowlist.start()
        self.host_allowlist = patch.object(
            main.settings,
            "mcp_allowed_sse_hosts",
            "localhost,127.0.0.1,::1",
        )
        self.host_allowlist.start()
        main._mcp_manager.close()
        main._mcp_enabled = False
        main._mcp_servers_config = {}
        main._mcp_loaded_tools = 0

    def tearDown(self):
        main._mcp_manager.close()
        main._mcp_enabled = self.old_enabled
        main._mcp_servers_config = self.old_servers
        main._mcp_loaded_tools = self.old_loaded
        self.host_allowlist.stop()
        self.command_allowlist.stop()
        self.agent_path_patch.stop()
        self.temp_dir.cleanup()

    def mcp_config(self, enable: bool = True) -> dict:
        return {
            "enable": enable,
            "servers": {
                "demo": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(FIXTURE)],
                }
            },
        }

    def test_default_registry_does_not_load_mcp_tools(self):
        registry = create_default_registry()

        self.assertFalse(
            any(tool.name.startswith("mcp_") for tool in registry.list_tools())
        )

    def test_api_loads_tools_and_agent_registry_uses_them(self):
        response = self.client.post(
            "/api/mcp/config",
            json=self.mcp_config(),
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["loaded_tools"], 3)

        catalog = self.client.get("/api/tools/registry")
        self.assertEqual(catalog.status_code, 200)
        catalog_names = {
            item["name"] for item in catalog.json()["tools"]
        }
        self.assertIn("mcp_demo_echo", catalog_names)

        registry = _get_agent_tool_registry(
            {
                "configurable": {
                    "thread_id": "mcp-integration-thread",
                    "mcp_enabled": True,
                    "mcp_tools": list(main._mcp_manager.tools),
                }
            }
        )
        self.assertIn("mcp_demo_add", {
            tool.name for tool in registry.list_tools()
        })
        self.assertEqual(
            registry.invoke("mcp_demo_add", {"left": 4, "right": 5}),
            {"output": {"sum": 9}, "error": ""},
        )

    def test_api_disable_closes_loaded_mcp_tools(self):
        self.assertEqual(
            self.client.post(
                "/api/mcp/config",
                json=self.mcp_config(),
            ).json()["loaded_tools"],
            3,
        )

        response = self.client.post(
            "/api/mcp/config",
            json={"enable": False},
        )

        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["enabled"])
        self.assertEqual(response.json()["loaded_tools"], 0)
        self.assertEqual(main._mcp_manager.connections, {})

    def test_api_rejects_unallowlisted_mcp_command_without_mutating_state(self):
        response = self.client.post(
            "/api/mcp/config",
            json={
                "enable": True,
                "servers": {
                    "bad": {
                        "type": "stdio",
                        "command": "some-untrusted-program",
                    }
                },
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertFalse(main._mcp_enabled)
        self.assertEqual(main._mcp_servers_config, {})

    def test_api_redacts_environment_values_from_mcp_snapshot(self):
        response = self.client.post(
            "/api/mcp/config",
            json={
                "enable": False,
                "servers": {
                    "demo": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": [str(FIXTURE)],
                        "env": {"DEMO_TOKEN": "not-returned"},
                    }
                },
            },
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["servers"]["demo"]["env"]["DEMO_TOKEN"],
            "***",
        )

    def test_mcp_diagnose_reports_connected_and_failed_servers_safely(self):
        secret = "mcp-diagnostic-secret"
        response = self.client.post(
            "/api/mcp/config",
            json={
                "enable": True,
                "servers": {
                    "demo": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": [str(FIXTURE)],
                        "env": {"DEMO_TOKEN": secret},
                    },
                    "broken": {
                        "type": "stdio",
                        "command": sys.executable,
                        "args": ["-c", "pass", "--token", secret],
                    },
                },
            },
        )
        self.assertEqual(response.status_code, 200)

        diagnosis = self.client.get("/api/mcp/diagnose")

        self.assertEqual(diagnosis.status_code, 200)
        payload = diagnosis.json()
        self.assertEqual(payload["status"], "degraded")
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["server_count"], 2)
        self.assertEqual(payload["connected_count"], 1)
        by_id = {server["server_id"]: server for server in payload["servers"]}
        self.assertEqual(by_id["demo"]["status"], "connected")
        self.assertEqual(by_id["demo"]["loaded_tools"], 3)
        self.assertEqual(by_id["broken"]["status"], "failed")
        self.assertNotIn(secret, diagnosis.text)

    def test_mcp_diagnose_reports_disabled_servers(self):
        response = self.client.post(
            "/api/mcp/config",
            json=self.mcp_config(enable=False),
        )
        self.assertEqual(response.status_code, 200)

        diagnosis = self.client.get("/api/mcp/diagnose")

        self.assertEqual(diagnosis.status_code, 200)
        payload = diagnosis.json()
        self.assertEqual(payload["status"], "disabled")
        self.assertFalse(payload["ready"])
        self.assertEqual(payload["servers"][0]["status"], "disabled")
        self.assertIsNone(payload["servers"][0]["message"])

    def test_agent_registry_only_receives_selected_global_mcp_servers(self):
        server = {
            "type": "stdio",
            "command": sys.executable,
            "args": [str(FIXTURE)],
        }
        main._mcp_enabled = True
        main._mcp_servers_config = {"alpha": server, "beta": server}
        main._mcp_manager.load(main._mcp_servers_config)
        main._mcp_loaded_tools = main._mcp_manager.loaded_tool_count

        profile = AgentProfile(
            name="alpha-only",
            mcp_servers=frozenset({"alpha"}),
        )
        config = main._build_chat_config(
            thread_id="agent-mcp-selection",
            model="deepseek-chat",
            search_mode={},
            agent_id="alpha-only",
            user_id="test-user",
            images=None,
            agent_profile=profile,
            model_override=None,
        )
        registry = _get_agent_tool_registry(config)
        names = {tool.name for tool in registry.list_tools()}

        self.assertIn("mcp_alpha_echo", names)
        self.assertNotIn("mcp_beta_echo", names)

    def test_chat_request_only_receives_requested_mcp_server_tools(self):
        server = {
            "type": "stdio",
            "command": sys.executable,
            "args": [str(FIXTURE)],
        }
        main._mcp_enabled = True
        main._mcp_servers_config = {"alpha": server, "beta": server}
        main._mcp_manager.load(main._mcp_servers_config)
        main._mcp_loaded_tools = main._mcp_manager.loaded_tool_count

        captured = {}

        async def fake_chat_stream(**kwargs):
            captured["config"] = kwargs["config"]
            yield "event: done\\ndata: {}\\n\\n"

        with patch.object(
            main,
            "_iter_sse_chat_stream",
            new=fake_chat_stream,
        ):
            response = self.client.post(
                "/api/chat/sse",
                json={
                    "messages": [
                        {"role": "user", "content": "调用 MCP"},
                    ],
                    "search_mode": {"useAgent": True},
                    "mcp_server_id": "alpha",
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            {
                tool.name
                for tool in captured["config"]["configurable"]["mcp_tools"]
            },
            {"mcp_alpha_echo", "mcp_alpha_add", "mcp_alpha_fail"},
        )

    def test_chat_mcp_selection_cannot_exceed_agent_allowlist(self):
        server = {
            "type": "stdio",
            "command": sys.executable,
            "args": [str(FIXTURE)],
        }
        main._mcp_enabled = True
        main._mcp_servers_config = {"alpha": server, "beta": server}
        main._mcp_manager.load(main._mcp_servers_config)
        main._mcp_loaded_tools = main._mcp_manager.loaded_tool_count

        with self.assertRaises(Exception) as context:
            main._build_chat_config(
                thread_id="restricted-mcp-selection",
                model="deepseek-chat",
                search_mode={"use_agent": True},
                agent_id="alpha-only",
                user_id="test-user",
                images=None,
                agent_profile=AgentProfile(
                    name="alpha-only",
                    mcp_servers=frozenset({"alpha"}),
                ),
                model_override=None,
                mcp_server_id="beta",
            )

        self.assertEqual(context.exception.status_code, 403)

    def test_agent_cannot_reference_unconfigured_mcp_server(self):
        main._mcp_servers_config = {"demo": {}}

        response = self.client.post(
            "/api/agents",
            json={
                "id": "agent-bad-mcp",
                "name": "Bad MCP Selection",
                "mcp_servers": {"unknown": True},
            },
        )

        self.assertEqual(response.status_code, 422)

    def test_agent_persists_boolean_server_selection(self):
        main._mcp_servers_config = {
            "alpha": {},
            "beta": {},
        }
        response = self.client.post(
            "/api/agents",
            json={
                "id": "agent-mcp-selection",
                "name": "Selected MCP Agent",
                "mcp_servers": {
                    "alpha": True,
                    "beta": False,
                },
            },
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(
            response.json()["mcp_servers"],
            {"alpha": True, "beta": False},
        )

    def test_environment_mcp_servers_are_parsed_as_json(self):
        with patch.object(
            main.settings,
            "mcp_servers",
            '{"demo":{"type":"sse","url":"http://localhost:9911/sse"}}',
        ):
            self.assertEqual(
                main._initial_mcp_servers(),
                {
                    "demo": {
                        "type": "sse",
                        "url": "http://localhost:9911/sse",
                    }
                },
            )


if __name__ == "__main__":
    unittest.main()
