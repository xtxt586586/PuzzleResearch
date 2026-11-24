from __future__ import annotations

import json
import re
import shutil
from pathlib import Path
from typing import Any

from tools.registry import ToolDefinition
from tools.workspace_files import (
    SessionWorkspace,
    WorkspaceFileError,
    get_session_workspace,
)
from tools.workspace_shell import (
    WorkspaceCommandError,
    execute_command,
    expose_port,
)


SUPPORTED_FRAMEWORKS = {
    "nextjs",
    "react",
    "vite-react",
    "vite-vue",
    "vite-svelte",
}
SUPPORTED_PACKAGE_MANAGERS = {"npm", "yarn", "pnpm", "bun"}
PROJECT_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
MAX_PROJECT_COMMAND_CHARS = 4_000


def _validate_project_name(project_name: str) -> str:
    if not isinstance(project_name, str) or not PROJECT_NAME_PATTERN.fullmatch(
        project_name.strip()
    ):
        raise WorkspaceCommandError(
            "project_name 只能包含字母、数字、短横线和下划线，长度为 1 到 64。"
        )
    return project_name.strip()


def _validate_framework(framework: str) -> str:
    normalized = str(framework or "").strip().lower()
    if normalized not in SUPPORTED_FRAMEWORKS:
        raise WorkspaceCommandError(
            "framework 必须是 nextjs、react、vite-react、vite-vue 或 vite-svelte。"
        )
    return normalized


def _validate_package_manager(package_manager: str) -> str:
    normalized = str(package_manager or "").strip().lower()
    if normalized not in SUPPORTED_PACKAGE_MANAGERS:
        raise WorkspaceCommandError(
            "package_manager 必须是 npm、yarn、pnpm 或 bun。"
        )
    return normalized


def _validate_boolean(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise WorkspaceCommandError(f"{name} 必须是布尔值。")


def _validate_command_override(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceCommandError(f"{name} 必须是非空命令文本。")
    if len(value) > MAX_PROJECT_COMMAND_CHARS or "\x00" in value:
        raise WorkspaceCommandError(
            f"{name} 不能超过 {MAX_PROJECT_COMMAND_CHARS} 个字符。"
        )
    return value.strip()


def _framework_template(framework: str, typescript: bool) -> str:
    if framework == "vite-react":
        return "react-ts" if typescript else "react"
    if framework == "vite-vue":
        return "vue-ts" if typescript else "vue"
    if framework == "vite-svelte":
        return "svelte-ts" if typescript else "svelte"
    return ""


def _scaffold_command(
    project_name: str,
    framework: str,
    package_manager: str,
    typescript: bool,
) -> str:
    if framework == "nextjs":
        language_flag = "--typescript" if typescript else "--js"
        return (
            f"npx create-next-app@latest {project_name} {language_flag} "
            f"--use-{package_manager} --eslint --no-tailwind --no-git "
            "--src-dir --app --yes"
        )
    if framework == "react":
        template = "typescript" if typescript else "javascript"
        return f"npx create-react-app {project_name} --template {template}"

    template = _framework_template(framework, typescript)
    if package_manager == "npm":
        return f"npm create vite@latest {project_name} -- --template {template}"
    return f"{package_manager} create vite {project_name} --template {template}"


def _install_command(package_manager: str) -> str:
    return f"{package_manager} install"


def _default_start_command(
    package_manager: str,
    port: int,
    project_directory: Path,
) -> str:
    scripts: dict[str, Any] = {}
    package_json = project_directory / "package.json"
    if package_json.is_file():
        try:
            payload = json.loads(package_json.read_text(encoding="utf-8"))
            scripts = payload.get("scripts") or {}
        except (OSError, json.JSONDecodeError):
            scripts = {}

    if "start" in scripts:
        return f"{package_manager} run start -- --hostname 0.0.0.0 --port {port}"
    return f"{package_manager} run dev -- --host 0.0.0.0 --port {port}"


def _workspace_project(workspace: SessionWorkspace, project_name: str) -> Path:
    target = workspace.root / project_name
    workspace.root.mkdir(parents=True, exist_ok=True)
    try:
        target.resolve().relative_to(workspace.root.resolve())
    except ValueError as exc:
        raise WorkspaceCommandError("项目路径不能越出当前会话工作区。") from exc
    return target


def scaffold_web_project(
    thread_id: str,
    project_name: str,
    framework: str = "vite-react",
    package_manager: str = "npm",
    typescript: bool = True,
    overwrite: bool = False,
    install_deps: bool = True,
) -> dict[str, Any]:
    _validate_boolean(typescript, "typescript")
    _validate_boolean(overwrite, "overwrite")
    _validate_boolean(install_deps, "install_deps")
    safe_name = _validate_project_name(project_name)
    selected_framework = _validate_framework(framework)
    selected_manager = _validate_package_manager(package_manager)

    workspace = get_session_workspace(thread_id)
    target = _workspace_project(workspace, safe_name)
    if target.exists():
        if not overwrite:
            raise WorkspaceCommandError(
                f"项目目录已存在：{safe_name}；如需重建请设置 overwrite=true。"
            )
        if target.is_symlink() or not target.is_dir():
            raise WorkspaceCommandError("项目目标必须是普通目录。")
        shutil.rmtree(target)

    scaffold = execute_command(
        thread_id,
        _scaffold_command(
            safe_name,
            selected_framework,
            selected_manager,
            typescript,
        ),
        folder=".",
        timeout=60,
    )
    if not scaffold.get("success"):
        return {
            "success": False,
            "stage": "scaffold",
            "project_path": safe_name,
            "framework": selected_framework,
            "package_manager": selected_manager,
            "scaffold": scaffold,
            "error": scaffold.get("error") or scaffold.get("output", ""),
        }

    install_result: dict[str, Any] | None = None
    if install_deps:
        install_result = execute_command(
            thread_id,
            _install_command(selected_manager),
            folder=safe_name,
            timeout=60,
        )
        if not install_result.get("success"):
            return {
                "success": False,
                "stage": "install",
                "project_path": safe_name,
                "framework": selected_framework,
                "package_manager": selected_manager,
                "scaffold": scaffold,
                "install": install_result,
                "error": install_result.get("error") or install_result.get("output", ""),
            }

    return {
        "success": True,
        "project_path": safe_name,
        "framework": selected_framework,
        "package_manager": selected_manager,
        "typescript": typescript,
        "scaffold": scaffold,
        "install": install_result,
        "output": f"已在当前会话工作区创建 Web 项目：{safe_name}",
        "error": "",
    }


def deploy_web_project(
    thread_id: str,
    project_path: str,
    package_manager: str = "npm",
    build_command: str | None = None,
    start_command: str | None = None,
    port: int = 3000,
    expose: bool = True,
) -> dict[str, Any]:
    selected_manager = _validate_package_manager(package_manager)
    selected_build_command = _validate_command_override(build_command, "build_command")
    selected_start_command = _validate_command_override(start_command, "start_command")
    _validate_boolean(expose, "expose")
    if type(port) is not int or not 1 <= port <= 65_535:
        raise WorkspaceCommandError("port 必须是 1 到 65535 之间的整数。")

    workspace = get_session_workspace(thread_id)
    try:
        project_directory = workspace.resolve_directory(project_path)
    except WorkspaceFileError as exc:
        raise WorkspaceCommandError(str(exc)) from exc
    relative_project_path = project_directory.relative_to(workspace.root).as_posix()

    build = execute_command(
        thread_id,
        selected_build_command or f"{selected_manager} run build",
        folder=relative_project_path,
        timeout=60,
    )
    if not build.get("success"):
        return {
            "success": False,
            "stage": "build",
            "project_path": relative_project_path,
            "build": build,
            "error": build.get("error") or build.get("output", ""),
        }

    start = execute_command(
        thread_id,
        selected_start_command
        or _default_start_command(selected_manager, port, project_directory),
        folder=relative_project_path,
        timeout=60,
        background=True,
    )
    if not start.get("success"):
        return {
            "success": False,
            "stage": "start",
            "project_path": relative_project_path,
            "build": build,
            "start": start,
            "error": start.get("error") or start.get("output", ""),
        }

    port_result = expose_port(thread_id, port) if expose else None
    port_ready = port_result is None or bool(port_result.get("success"))
    return {
        "success": port_ready,
        "project_path": relative_project_path,
        "process_id": start.get("process_id"),
        "port": port,
        "url": port_result.get("url") if port_result else None,
        "port_ready": port_ready,
        "build": build,
        "start": start,
        "expose": port_result,
        "output": (
            f"Web 项目已启动：{relative_project_path}"
            if port_ready
            else "Web 项目后台命令已启动，但端口尚未监听；请使用 check_output 查看输出。"
        ),
        "error": "" if port_ready else (port_result or {}).get("error", ""),
    }


def build_workspace_web_dev_tool_definitions(
    thread_id: str,
) -> list[ToolDefinition]:
    def scaffold_web_project_tool(
        project_name: str,
        framework: str = "vite-react",
        package_manager: str = "npm",
        typescript: bool = True,
        overwrite: bool = False,
        install_deps: bool = True,
    ) -> dict[str, Any]:
        return scaffold_web_project(
            thread_id,
            project_name,
            framework,
            package_manager,
            typescript,
            overwrite,
            install_deps,
        )

    def deploy_web_project_tool(
        project_path: str,
        package_manager: str = "npm",
        build_command: str | None = None,
        start_command: str | None = None,
        port: int = 3000,
        expose: bool = True,
    ) -> dict[str, Any]:
        return deploy_web_project(
            thread_id,
            project_path,
            package_manager,
            build_command,
            start_command,
            port,
            expose,
        )

    return [
        ToolDefinition(
            "scaffold_web_project",
            "在当前会话工作区创建 Next.js、React 或 Vite Web 项目。",
            scaffold_web_project_tool,
        ),
        ToolDefinition(
            "deploy_web_project",
            "构建并启动当前会话工作区中的 Web 项目，可返回本地预览地址。",
            deploy_web_project_tool,
        ),
    ]
