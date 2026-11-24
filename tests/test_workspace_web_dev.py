from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.workspace_files import SessionWorkspace
from tools.workspace_web_dev import (
    WorkspaceCommandError,
    build_workspace_web_dev_tool_definitions,
    deploy_web_project,
    scaffold_web_project,
)


class WorkspaceWebDevTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = SessionWorkspace(
            "thread-one",
            storage_root=Path(self.temp_dir.name),
        )
        self.workspace_patch = patch(
            "tools.workspace_web_dev.get_session_workspace",
            return_value=self.workspace,
        )
        self.workspace_patch.start()

    def tearDown(self) -> None:
        self.workspace_patch.stop()
        self.temp_dir.cleanup()

    def test_scaffold_generates_vite_command_and_installs_dependencies(self):
        command_results = [
            {"success": True, "stdout": "created", "output": "created"},
            {"success": True, "stdout": "installed", "output": "installed"},
        ]
        with patch(
            "tools.workspace_web_dev.execute_command",
            side_effect=command_results,
        ) as execute:
            result = scaffold_web_project(
                "thread-one",
                "research-ui",
                framework="vite-react",
                package_manager="npm",
                typescript=True,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["project_path"], "research-ui")
        self.assertEqual(execute.call_args_list[0].args[0], "thread-one")
        self.assertIn("npm create vite@latest research-ui", execute.call_args_list[0].args[1])
        self.assertIn("react-ts", execute.call_args_list[0].args[1])
        self.assertEqual(execute.call_args_list[1].kwargs["folder"], "research-ui")
        self.assertEqual(execute.call_args_list[1].args[1], "npm install")

    def test_scaffold_rejects_unsafe_or_existing_project(self):
        with self.assertRaisesRegex(WorkspaceCommandError, "只能包含"):
            scaffold_web_project("thread-one", "../outside")

        self.workspace.root.mkdir(parents=True, exist_ok=True)
        (self.workspace.root / "existing").mkdir()
        with self.assertRaisesRegex(WorkspaceCommandError, "已存在"):
            scaffold_web_project("thread-one", "existing")

    def test_scaffold_returns_install_failure_without_claiming_success(self):
        with patch(
            "tools.workspace_web_dev.execute_command",
            side_effect=[
                {"success": True, "output": "created"},
                {"success": False, "error": "npm unavailable", "output": ""},
            ],
        ):
            result = scaffold_web_project("thread-one", "research-ui")

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "install")
        self.assertIn("npm unavailable", result["error"])

    def test_deploy_builds_starts_and_returns_local_preview(self):
        project = self.workspace.root / "research-ui"
        project.mkdir(parents=True)
        (project / "package.json").write_text(
            json.dumps({"scripts": {"build": "vite build", "dev": "vite"}}),
            encoding="utf-8",
        )
        with patch(
            "tools.workspace_web_dev.execute_command",
            side_effect=[
                {"success": True, "output": "built"},
                {"success": True, "process_id": "proc-001", "output": "started"},
            ],
        ) as execute, patch(
            "tools.workspace_web_dev.expose_port",
            return_value={
                "success": True,
                "url": "http://127.0.0.1:4173",
                "is_public": False,
            },
        ) as expose:
            result = deploy_web_project(
                "thread-one",
                "research-ui",
                port=4173,
            )

        self.assertTrue(result["success"])
        self.assertEqual(result["process_id"], "proc-001")
        self.assertEqual(result["url"], "http://127.0.0.1:4173")
        self.assertEqual(execute.call_args_list[0].kwargs["folder"], "research-ui")
        self.assertEqual(execute.call_args_list[0].args[1], "npm run build")
        self.assertIn("npm run dev", execute.call_args_list[1].args[1])
        self.assertTrue(execute.call_args_list[1].kwargs["background"])
        expose.assert_called_once_with("thread-one", 4173)

    def test_deploy_stops_before_start_when_build_fails(self):
        project = self.workspace.root / "research-ui"
        project.mkdir(parents=True)
        with patch(
            "tools.workspace_web_dev.execute_command",
            return_value={"success": False, "error": "build failed", "output": ""},
        ) as execute:
            result = deploy_web_project("thread-one", "research-ui", expose=False)

        self.assertFalse(result["success"])
        self.assertEqual(result["stage"], "build")
        execute.assert_called_once()

    def test_tool_definitions_describe_web_dev_contract(self):
        tools = build_workspace_web_dev_tool_definitions("thread-one")
        self.assertEqual(
            {tool.name for tool in tools},
            {"scaffold_web_project", "deploy_web_project"},
        )
        schema = tools[0].argument_schema()
        self.assertEqual(schema[0]["name"], "project_name")
        self.assertTrue(schema[0]["required"])


if __name__ == "__main__":
    unittest.main()
