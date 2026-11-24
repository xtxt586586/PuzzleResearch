from __future__ import annotations

import os
import sys
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.builtin import create_default_registry
from tools.workspace_files import SessionWorkspace
from tools.workspace_shell import (
    WorkspaceCommandError,
    _processes,
    build_workspace_shell_tool_definitions,
    check_output,
    execute_command,
    expose_port,
    install_package,
    kill_process,
    list_processes,
)


def python_command(code: str) -> str:
    return f'"{sys.executable}" -c "{code}"'


class WorkspaceShellTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = SessionWorkspace(
            "thread-one",
            storage_root=Path(self.temp_dir.name),
        )
        self.background_ids: list[str] = []
        self.workspace_patch = patch(
            "tools.workspace_shell.get_session_workspace",
            return_value=self.workspace,
        )
        self.workspace_patch.start()

    def tearDown(self) -> None:
        for process_id in self.background_ids:
            try:
                kill_process("thread-one", process_id)
            except WorkspaceCommandError:
                pass
        self.workspace_patch.stop()
        self.temp_dir.cleanup()

    def test_execute_command_uses_workspace_folder_and_returns_streams(self):
        (self.workspace.root / "nested").mkdir(parents=True)

        result = execute_command(
            "thread-one",
            python_command("import sys, os; print(os.path.basename(os.getcwd())); print(2 + 3); print('error', file=sys.stderr)"),
            folder="nested",
        )

        self.assertTrue(result["success"])
        self.assertEqual(result["exit_code"], 0)
        self.assertEqual(result["cwd"], "nested")
        self.assertIn("nested", result["stdout"])
        self.assertIn("5", result["stdout"])
        self.assertIn("error", result["stderr"])

    def test_nonzero_exit_is_reported(self):
        result = execute_command(
            "thread-one",
            python_command("import sys; print('failed'); sys.exit(3)"),
        )

        self.assertFalse(result["success"])
        self.assertEqual(result["exit_code"], 3)
        self.assertIn("failed", result["stdout"])

    def test_command_and_folder_are_validated(self):
        with self.assertRaisesRegex(WorkspaceCommandError, "不能为空"):
            execute_command("thread-one", " ")
        with self.assertRaises(WorkspaceCommandError):
            execute_command("thread-one", "echo blocked", folder="../")
        with self.assertRaisesRegex(WorkspaceCommandError, "目录不存在"):
            execute_command("thread-one", "echo blocked", folder="missing")

    def test_timeout_and_output_limits_are_reported(self):
        timed_out = execute_command(
            "thread-one",
            python_command("import time; time.sleep(2)"),
            timeout=1,
        )
        self.assertFalse(timed_out["success"])
        self.assertTrue(timed_out["timed_out"])

        clipped = execute_command(
            "thread-one",
            python_command("print('x' * 100)"),
            max_output_chars=30,
        )
        self.assertLessEqual(len(clipped["stdout"]), 30)
        self.assertIn("截断", clipped["stdout"])

    def test_background_process_output_and_thread_isolation(self):
        started = execute_command(
            "thread-one",
            python_command("import time; print('ready', flush=True); time.sleep(10)"),
            background=True,
        )
        process_id = started["process_id"]
        self.background_ids.append(process_id)

        self.assertEqual(started["status"], "running")
        self.assertEqual(list_processes("thread-two")["processes"], [])

        observed = None
        for _ in range(20):
            observed = check_output("thread-one", process_id)
            if "ready" in observed["output"]:
                break
            time.sleep(0.05)
        self.assertIsNotNone(observed)
        self.assertIn("ready", observed["output"])
        self.assertIn(
            process_id,
            {item["process_id"] for item in list_processes("thread-one")["processes"]},
        )

        killed = kill_process("thread-one", process_id)
        self.assertTrue(killed["success"])
        final = check_output("thread-one", process_id)
        self.assertNotEqual(final["status"], "running")

    def test_registry_exposes_command_tools_with_argument_contract(self):
        with patch(
            "tools.workspace_files.get_session_workspace",
            return_value=self.workspace,
        ), patch(
            "tools.workspace_shell.get_session_workspace",
            return_value=self.workspace,
        ):
            registry = create_default_registry(thread_id="thread-one")

        names = {tool.name for tool in registry.list_tools()}
        self.assertTrue(
            {
                "execute_command",
                "check_output",
                "kill_process",
                "list_processes",
            }
            <= names
        )
        schema = registry.get("execute_command").argument_schema()
        self.assertEqual(schema[0]["name"], "command")
        self.assertTrue(schema[0]["required"])

        result = registry.invoke(
            "execute_command",
            {"command": python_command("print(6 * 7)")},
        )
        self.assertEqual(result["stdout"].strip(), "42")

    def test_unknown_process_is_rejected(self):
        with self.assertRaisesRegex(WorkspaceCommandError, "不存在"):
            check_output("thread-one", "missing-process")
        with self.assertRaisesRegex(WorkspaceCommandError, "不存在"):
            kill_process("thread-one", "missing-process")

    def test_install_package_uses_workspace_target_without_shell(self):
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="installed",
            stderr="",
        )
        with patch(
            "tools.workspace_shell.subprocess.run",
            return_value=completed,
        ) as run:
            result = install_package(
                "thread-one",
                "requests==2.32.0",
                package_manager="pip",
            )

        command = run.call_args.args[0]
        self.assertEqual(command[:3], [sys.executable, "-m", "pip"])
        self.assertIn("--target", command)
        self.assertEqual(run.call_args.kwargs["cwd"], self.workspace.root)
        self.assertFalse(run.call_args.kwargs.get("shell", False))
        self.assertTrue(result["success"])
        self.assertEqual(result["package_manager"], "pip")
        self.assertEqual(result["target"], ".puzzleresearch/python_packages")

    def test_python_commands_receive_workspace_package_path(self):
        package_path = self.workspace.root / ".puzzleresearch" / "python_packages"
        package_path.mkdir(parents=True)

        result = execute_command(
            "thread-one",
            python_command("import os; print(os.environ['PYTHONPATH'])"),
        )

        self.assertTrue(result["success"])
        self.assertIn(str(package_path), result["stdout"])

    def test_install_package_auto_detects_npm_project(self):
        self.workspace.root.mkdir(parents=True, exist_ok=True)
        (self.workspace.root / "package.json").write_text("{}", encoding="utf-8")
        completed = subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="added 1 package",
            stderr="",
        )
        with patch(
            "tools.workspace_shell.shutil.which",
            return_value="npm.cmd",
        ), patch(
            "tools.workspace_shell.subprocess.run",
            return_value=completed,
        ) as run:
            result = install_package("thread-one", "lodash")

        command = run.call_args.args[0]
        self.assertEqual(command[0], "npm.cmd" if os.name == "nt" else "npm")
        self.assertEqual(command[-1], "lodash")
        self.assertEqual(result["package_manager"], "npm")
        self.assertIsNone(result["target"])

    def test_install_package_rejects_unsafe_or_unsupported_requests(self):
        with self.assertRaisesRegex(WorkspaceCommandError, "命令字符"):
            install_package(
                "thread-one",
                "requests; echo blocked",
                package_manager="pip",
            )
        with self.assertRaisesRegex(WorkspaceCommandError, "不支持 apt"):
            install_package("thread-one", "curl", package_manager="apt")
        with self.assertRaisesRegex(WorkspaceCommandError, "只能是"):
            install_package("thread-one", "curl", package_manager="cargo")

    def test_expose_port_returns_local_url_only_for_listening_port(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        try:
            result = expose_port("thread-one", port)
        finally:
            listener.close()

        self.assertTrue(result["success"])
        self.assertEqual(result["url"], f"http://127.0.0.1:{port}")
        self.assertFalse(result["is_public"])
        self.assertIsNone(result["public_url"])

        missing = expose_port("thread-one", 65_534)
        self.assertFalse(missing["success"])
        self.assertIsNone(missing["url"])

    def test_expose_port_waits_for_delayed_listener_start(self):
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]

        def start_listening():
            time.sleep(0.2)
            listener.listen(1)

        starter = threading.Thread(target=start_listening)
        starter.start()
        try:
            result = expose_port("thread-one", port)
        finally:
            starter.join(timeout=2)
            listener.close()

        self.assertTrue(result["success"])

    def test_expose_port_validates_port(self):
        with self.assertRaisesRegex(WorkspaceCommandError, "1 到 65535"):
            expose_port("thread-one", 0)


if __name__ == "__main__":
    unittest.main()
