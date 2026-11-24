from __future__ import annotations

import os
import signal
import shlex
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.registry import ToolDefinition
from tools.workspace_files import (
    SessionWorkspace,
    WorkspaceFileError,
    WorkspaceFileNotFoundError,
    get_session_workspace,
)


DEFAULT_COMMAND_TIMEOUT_SECONDS = 30
MAX_COMMAND_TIMEOUT_SECONDS = 60
DEFAULT_COMMAND_OUTPUT_CHARS = 4_000
MAX_COMMAND_OUTPUT_CHARS = 12_000
MAX_COMMAND_CHARS = 20_000
MAX_TAIL_LINES = 200
DEFAULT_INSTALL_TIMEOUT_SECONDS = 300
MAX_INSTALL_TIMEOUT_SECONDS = 300
MAX_INSTALL_OUTPUT_CHARS = 4_000
MAX_PACKAGE_COUNT = 20
MAX_PACKAGE_SPEC_CHARS = 2_000
EXPOSE_PORT_WAIT_SECONDS = 5


class WorkspaceCommandError(ValueError):
    pass


@dataclass
class _ProcessRecord:
    process: subprocess.Popen[Any]
    command: str
    cwd: str
    started_at: float
    output_path: Path


_processes: dict[tuple[str, str], _ProcessRecord] = {}
_processes_lock = threading.RLock()


def _clip(value: Any, max_chars: int) -> str:
    text = str(value or "")
    if len(text) <= max_chars:
        return text
    marker = "\n...[命令输出已截断]"
    if max_chars <= len(marker):
        return marker[:max_chars]
    return text[:max_chars - len(marker)] + marker


def _validate_command(command: str, timeout: int, max_output_chars: int) -> None:
    if not isinstance(command, str) or not command.strip():
        raise WorkspaceCommandError("command 不能为空。")
    if len(command) > MAX_COMMAND_CHARS:
        raise WorkspaceCommandError(
            f"command 不能超过 {MAX_COMMAND_CHARS} 个字符。"
        )
    if type(timeout) is not int or not 1 <= timeout <= MAX_COMMAND_TIMEOUT_SECONDS:
        raise WorkspaceCommandError(
            f"timeout 必须是 1 到 {MAX_COMMAND_TIMEOUT_SECONDS} 之间的整数。"
        )
    if (
        type(max_output_chars) is not int
        or not 1 <= max_output_chars <= MAX_COMMAND_OUTPUT_CHARS
    ):
        raise WorkspaceCommandError(
            f"max_output_chars 必须是 1 到 {MAX_COMMAND_OUTPUT_CHARS} 之间的整数。"
        )


def _resolve_cwd(workspace: SessionWorkspace, folder: str | None) -> Path:
    directory = "." if folder is None else folder
    try:
        return workspace.resolve_directory(directory)
    except (WorkspaceFileError, WorkspaceFileNotFoundError) as exc:
        raise WorkspaceCommandError(str(exc)) from exc


def _creationflags() -> int:
    if os.name == "nt":
        return getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    return 0


def _terminate_process(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        if os.name == "nt":
            subprocess.run(
                ["taskkill", "/PID", str(process.pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
        else:
            os.killpg(process.pid, signal.SIGTERM)
    except (OSError, subprocess.SubprocessError):
        try:
            process.kill()
        except OSError:
            pass


def _read_tail(path: Path, tail_lines: int, max_output_chars: int) -> str:
    try:
        content = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        raise WorkspaceCommandError("后台命令输出无法读取。") from exc
    return _clip("".join(content.splitlines(keepends=True)[-tail_lines:]), max_output_chars)


def _split_package_spec(packages: str) -> list[str]:
    if not isinstance(packages, str) or not packages.strip():
        raise WorkspaceCommandError("packages 不能为空。")
    if len(packages) > MAX_PACKAGE_SPEC_CHARS:
        raise WorkspaceCommandError(
            f"packages 不能超过 {MAX_PACKAGE_SPEC_CHARS} 个字符。"
        )
    try:
        values = shlex.split(packages)
    except ValueError as exc:
        raise WorkspaceCommandError("packages 格式无效。") from exc
    if not values or len(values) > MAX_PACKAGE_COUNT:
        raise WorkspaceCommandError(
            f"packages 数量必须在 1 到 {MAX_PACKAGE_COUNT} 个之间。"
        )
    if any(not value or value.startswith("-") for value in values):
        raise WorkspaceCommandError("packages 只能包含包名和版本约束，不能包含命令选项。")
    if any(
        any(character in value for character in "\r\n;|&><`$()")
        for value in values
    ):
        raise WorkspaceCommandError("packages 包含不允许的命令字符。")
    return values


def _validate_package_values(packages: list[str], package_manager: str) -> None:
    for package in packages:
        if package_manager == "npm":
            invalid = ":\\"
            if package.startswith(("/", ".")):
                raise WorkspaceCommandError("npm 包必须使用包名，不能使用工作区外路径。")
        else:
            invalid = ":/\\"
            if package.startswith(("/", ".")):
                raise WorkspaceCommandError("pip 包必须使用包名，不能使用本地路径。")
        if any(character in package for character in invalid):
            raise WorkspaceCommandError("包名不能包含路径或 URL。")


def _choose_package_manager(
    workspace: SessionWorkspace,
    cwd: Path,
    packages: list[str],
    package_manager: str,
) -> str:
    manager = str(package_manager or "").strip().lower()
    if manager not in {"auto", "pip", "npm", "apt"}:
        raise WorkspaceCommandError("package_manager 只能是 auto、pip、npm 或 apt。")
    if manager != "auto":
        return manager
    if (cwd / "package.json").is_file():
        return "npm"
    if any((cwd / name).is_file() for name in ("requirements.txt", "pyproject.toml", "setup.py")):
        return "pip"
    package_text = " ".join(packages).lower()
    if any(token in package_text for token in ("react", "vue", "next", "express", "lodash", "@")):
        return "npm"
    return "pip"


def _normalized_workspace_path(workspace: SessionWorkspace, path: Path) -> str:
    return path.relative_to(workspace.root).as_posix() if path != workspace.root else "."


def _workspace_environment(workspace: SessionWorkspace) -> dict[str, str]:
    environment = os.environ.copy()
    package_path = workspace.root / ".puzzleresearch" / "python_packages"
    if package_path.is_dir():
        existing = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = os.pathsep.join(
            value for value in (str(package_path), existing) if value
        )
    return environment


def execute_command(
    thread_id: str,
    command: str,
    folder: str | None = None,
    timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
    background: bool = False,
    max_output_chars: int = DEFAULT_COMMAND_OUTPUT_CHARS,
) -> dict[str, Any]:
    _validate_command(command, timeout, max_output_chars)
    if type(background) is not bool:
        raise WorkspaceCommandError("background 必须是布尔值。")

    workspace = get_session_workspace(thread_id)
    cwd = _resolve_cwd(workspace, folder)
    normalized_cwd = cwd.relative_to(workspace.root).as_posix() if cwd != workspace.root else "."

    if background:
        process_id = uuid.uuid4().hex[:8]
        output_handle = tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f"puzzleresearch-{process_id}-",
            suffix=".log",
            delete=False,
        )
        output_path = Path(output_handle.name)
        try:
            process = subprocess.Popen(
                command,
                cwd=cwd,
                shell=True,
                stdin=subprocess.DEVNULL,
                stdout=output_handle,
                stderr=subprocess.STDOUT,
                creationflags=_creationflags(),
                start_new_session=os.name != "nt",
                env=_workspace_environment(workspace),
            )
        except OSError as exc:
            output_handle.close()
            output_path.unlink(missing_ok=True)
            raise WorkspaceCommandError(f"命令无法启动：{exc}") from exc
        finally:
            output_handle.close()

        with _processes_lock:
            _processes[(thread_id, process_id)] = _ProcessRecord(
                process=process,
                command=command,
                cwd=normalized_cwd,
                started_at=time.monotonic(),
                output_path=output_path,
            )
        return {
            "success": True,
            "process_id": process_id,
            "cwd": normalized_cwd,
            "status": "running",
            "output": "后台命令已启动。请使用 check_output 查询结果。",
            "error": "",
        }

    process = subprocess.Popen(
        command,
        cwd=cwd,
        shell=True,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=_creationflags(),
        start_new_session=os.name != "nt",
        env=_workspace_environment(workspace),
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        _terminate_process(process)
        stdout, stderr = process.communicate()
        return {
            "success": False,
            "timed_out": True,
            "exit_code": process.returncode,
            "stdout": _clip(stdout or exc.stdout or "", max_output_chars),
            "stderr": _clip(stderr or exc.stderr or "", max_output_chars),
            "cwd": normalized_cwd,
            "output": f"命令执行超过 {timeout} 秒，已终止。",
            "error": f"命令执行超过 {timeout} 秒。",
        }

    exit_code = process.returncode
    return {
        "success": exit_code == 0,
        "timed_out": False,
        "exit_code": exit_code,
        "stdout": _clip(stdout, max_output_chars),
        "stderr": _clip(stderr, max_output_chars),
        "cwd": normalized_cwd,
        "output": _clip(stdout, max_output_chars),
        "error": _clip(stderr, max_output_chars) if exit_code else "",
    }


def check_output(
    thread_id: str,
    process_id: str,
    tail_lines: int = 50,
    max_output_chars: int = DEFAULT_COMMAND_OUTPUT_CHARS,
) -> dict[str, Any]:
    if not isinstance(process_id, str) or not process_id.strip():
        raise WorkspaceCommandError("process_id 不能为空。")
    if type(tail_lines) is not int or not 1 <= tail_lines <= MAX_TAIL_LINES:
        raise WorkspaceCommandError(
            f"tail_lines 必须是 1 到 {MAX_TAIL_LINES} 之间的整数。"
        )
    if (
        type(max_output_chars) is not int
        or not 1 <= max_output_chars <= MAX_COMMAND_OUTPUT_CHARS
    ):
        raise WorkspaceCommandError(
            f"max_output_chars 必须是 1 到 {MAX_COMMAND_OUTPUT_CHARS} 之间的整数。"
        )

    with _processes_lock:
        record = _processes.get((thread_id, process_id))
    if record is None:
        raise WorkspaceCommandError(f"后台进程不存在：{process_id}")

    returncode = record.process.poll()
    status = "running" if returncode is None else (
        "completed" if returncode == 0 else "failed"
    )
    return {
        "success": True,
        "process_id": process_id,
        "command": record.command,
        "cwd": record.cwd,
        "status": status,
        "exit_code": returncode,
        "output": _read_tail(record.output_path, tail_lines, max_output_chars),
        "running_time": round(time.monotonic() - record.started_at, 1),
        "error": "",
    }


def kill_process(thread_id: str, process_id: str) -> dict[str, Any]:
    if not isinstance(process_id, str) or not process_id.strip():
        raise WorkspaceCommandError("process_id 不能为空。")
    with _processes_lock:
        record = _processes.get((thread_id, process_id))
    if record is None:
        raise WorkspaceCommandError(f"后台进程不存在：{process_id}")

    _terminate_process(record.process)
    try:
        record.process.wait(timeout=5)
    except (subprocess.TimeoutExpired, OSError):
        pass
    returncode = record.process.poll()
    return {
        "success": True,
        "process_id": process_id,
        "command": record.command,
        "status": "killed" if returncode is None or returncode != 0 else "completed",
        "output": f"后台进程已终止：{process_id}",
        "error": "",
    }


def list_processes(thread_id: str) -> dict[str, Any]:
    with _processes_lock:
        records = [
            (process_id, record)
            for (owner_thread_id, process_id), record in _processes.items()
            if owner_thread_id == thread_id
        ]

    processes: list[dict[str, Any]] = []
    for process_id, record in records:
        returncode = record.process.poll()
        processes.append({
            "process_id": process_id,
            "command": _clip(record.command, 120),
            "cwd": record.cwd,
            "status": "running" if returncode is None else (
                "completed" if returncode == 0 else "failed"
            ),
            "exit_code": returncode,
            "running_time": round(time.monotonic() - record.started_at, 1),
        })

    processes.sort(key=lambda item: item["process_id"])
    return {
        "success": True,
        "processes": processes,
        "total": len(processes),
        "output": f"当前会话共有 {len(processes)} 个后台进程。",
        "error": "",
    }


def install_package(
    thread_id: str,
    packages: str,
    package_manager: str = "auto",
    folder: str | None = None,
    timeout: int = DEFAULT_INSTALL_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    if type(timeout) is not int or not 1 <= timeout <= MAX_INSTALL_TIMEOUT_SECONDS:
        raise WorkspaceCommandError(
            f"timeout 必须是 1 到 {MAX_INSTALL_TIMEOUT_SECONDS} 之间的整数。"
        )

    workspace = get_session_workspace(thread_id)
    cwd = _resolve_cwd(workspace, folder)
    package_values = _split_package_spec(packages)
    manager = _choose_package_manager(
        workspace,
        cwd,
        package_values,
        package_manager,
    )
    if manager == "apt":
        raise WorkspaceCommandError(
            "当前本机工作区适配器不支持 apt；请使用 pip 或 npm。"
        )
    _validate_package_values(package_values, manager)

    target: Path | None = None
    if manager == "pip":
        target = workspace.root / ".puzzleresearch" / "python_packages"
        target.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--disable-pip-version-check",
            "--no-input",
            "--target",
            str(target),
            *package_values,
        ]
    else:
        npm = "npm.cmd" if os.name == "nt" else "npm"
        if shutil.which(npm) is None:
            raise WorkspaceCommandError("当前环境未找到 npm。")
        command = [npm, "install", "--no-audit", "--no-fund", *package_values]

    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "success": False,
            "timed_out": True,
            "packages": package_values,
            "package_manager": manager,
            "cwd": _normalized_workspace_path(workspace, cwd),
            "output": _clip(exc.stdout or "", MAX_INSTALL_OUTPUT_CHARS),
            "error": f"依赖安装超过 {timeout} 秒，已终止。",
        }
    except OSError as exc:
        raise WorkspaceCommandError(f"依赖安装无法启动：{exc}") from exc

    return {
        "success": completed.returncode == 0,
        "timed_out": False,
        "packages": package_values,
        "package_manager": manager,
        "cwd": _normalized_workspace_path(workspace, cwd),
        "target": (
            _normalized_workspace_path(workspace, target)
            if target is not None
            else None
        ),
        "exit_code": completed.returncode,
        "output": _clip(completed.stdout, MAX_INSTALL_OUTPUT_CHARS),
        "error": (
            _clip(completed.stderr, MAX_INSTALL_OUTPUT_CHARS)
            if completed.returncode
            else ""
        ),
    }


def _wait_for_local_port(port: int, timeout: float = EXPOSE_PORT_WAIT_SECONDS) -> bool:
    deadline = time.monotonic() + timeout
    while True:
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.25):
                return True
        except OSError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


def expose_port(thread_id: str, port: int) -> dict[str, Any]:
    if not isinstance(thread_id, str) or not thread_id.strip():
        raise WorkspaceCommandError("thread_id 不能为空。")
    if type(port) is not int or not 1 <= port <= 65_535:
        raise WorkspaceCommandError("port 必须是 1 到 65535 之间的整数。")

    if not _wait_for_local_port(port):
        return {
            "success": False,
            "port": port,
            "is_public": False,
            "url": None,
            "error": f"本机端口 {port} 当前未监听。",
        }

    url = f"http://127.0.0.1:{port}"
    return {
        "success": True,
        "port": port,
        "is_public": False,
        "url": url,
        "public_url": None,
        "output": (
            f"端口 {port} 已在本机监听，可通过 {url} 访问。"
            "当前运行环境不提供公网隧道。"
        ),
        "error": "",
    }


def build_workspace_shell_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    def execute_command_tool(
        command: str,
        folder: str | None = None,
        timeout: int = DEFAULT_COMMAND_TIMEOUT_SECONDS,
        background: bool = False,
        max_output_chars: int = DEFAULT_COMMAND_OUTPUT_CHARS,
    ) -> dict[str, Any]:
        return execute_command(
            thread_id,
            command,
            folder,
            timeout,
            background,
            max_output_chars,
        )

    def check_output_tool(
        process_id: str,
        tail_lines: int = 50,
        max_output_chars: int = DEFAULT_COMMAND_OUTPUT_CHARS,
    ) -> dict[str, Any]:
        return check_output(
            thread_id,
            process_id,
            tail_lines,
            max_output_chars,
        )

    def kill_process_tool(process_id: str) -> dict[str, Any]:
        return kill_process(thread_id, process_id)

    def list_processes_tool() -> dict[str, Any]:
        return list_processes(thread_id)

    def install_package_tool(
        packages: str,
        package_manager: str = "auto",
        folder: str | None = None,
        timeout: int = DEFAULT_INSTALL_TIMEOUT_SECONDS,
    ) -> dict[str, Any]:
        return install_package(
            thread_id,
            packages,
            package_manager,
            folder,
            timeout,
        )

    def expose_port_tool(port: int) -> dict[str, Any]:
        return expose_port(thread_id, port)

    return [
        ToolDefinition(
            "execute_command",
            "在当前会话工作区执行本机命令；支持超时、输出限制和后台运行。",
            execute_command_tool,
        ),
        ToolDefinition(
            "check_output",
            "读取当前会话后台命令的状态和最近输出。",
            check_output_tool,
        ),
        ToolDefinition(
            "kill_process",
            "终止当前会话启动的后台命令。",
            kill_process_tool,
        ),
        ToolDefinition(
            "list_processes",
            "列出当前会话启动的后台命令。",
            list_processes_tool,
        ),
        ToolDefinition(
            "install_package",
            "在当前会话工作区安装 pip 或 npm 依赖；不支持 apt 或工作区外路径。",
            install_package_tool,
        ),
        ToolDefinition(
            "expose_port",
            "检查并返回当前主机监听端口的本地访问地址；当前适配器不提供公网隧道。",
            expose_port_tool,
        ),
    ]
