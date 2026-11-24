from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Sequence

from tools.python_executor import PythonExecutor
from tools.registry import ToolDefinition, ToolRegistry
from tools.browser_tools import (
    build_browser_tool_definitions,
)
from tools.ask_human import build_ask_human_tool_definition
from tools.web_crawler import build_crawl_tool_definitions
from tools.web_search_tool import build_web_search_tool_definition
from tools.search.fallback_search import (
    build_fallback_search_tool_definition,
    build_multiple_search_tool_definition,
)
from tools.search.search import build_tavily_search_tool_definition
from tools.task_list import build_task_list_tool_definitions
from tools.planning_tool import build_planning_tool_definition
from tools.workspace_files import build_workspace_tool_definitions
from tools.workspace_shell import build_workspace_shell_tool_definitions
from tools.workspace_web_dev import build_workspace_web_dev_tool_definitions
from tools.workspace_sheets import build_workspace_sheet_tool_definitions
from tools.workspace_presentations import build_workspace_presentation_tool_definitions
from tools.workspace_presentation_outline import build_workspace_presentation_outline_tool_definitions
from tools.workspace_vision import build_workspace_vision_tool_definitions
from tools.workspace_image_edit import build_workspace_image_edit_tool_definitions
from tools.document_search import build_document_search_tool_definition
from common.config import settings
from tools.browser_use import build_browser_use_tool_definitions
from tools.computer_use import build_computer_use_tool_definitions


DEFAULT_PYTHON_TIMEOUT_SECONDS = 5.0
MAX_PYTHON_TIMEOUT_SECONDS = 10.0
DEFAULT_PYTHON_OUTPUT_CHARS = 4_000
MAX_PYTHON_OUTPUT_CHARS = 8_000

_APPROVAL_REQUIRED_TOOL_NAMES = frozenset(
    {
        "python_executor",
        "browser_search",
        "browser_navigate",
        "browser_click",
        "browser_back",
        "browser_extract_text",
        "browser_list_links",
        "browser_reset",
        "browser_screenshot",
        "browser_use",
        "crawl_url",
        "crawl_urls",
        "write_file",
        "update_file",
        "str_replace",
        "delete_file",
        "upload_file",
        "execute_command",
        "kill_process",
        "install_package",
        "scaffold_web_project",
        "deploy_web_project",
        "create_spreadsheet",
        "write_data",
        "format_cells",
        "add_sheet",
        "create_chart",
        "create_presentation",
        "add_slide",
        "add_image_to_slide",
        "add_table_to_slide",
        "add_shape_to_slide",
        "update_slide",
        "delete_slide",
        "outline_to_presentation",
        "apply_image_filter",
        "apply_image_effect",
        "adjust_image",
        "rotate_flip_image",
        "add_watermark",
        "overlay_images",
        "create_thumbnail",
    }
)


_python_executor = PythonExecutor()


def execute_python_tool(
        code: str,
        *,
        timeout_seconds: float = DEFAULT_PYTHON_TIMEOUT_SECONDS,
        max_output_chars: int = DEFAULT_PYTHON_OUTPUT_CHARS,
) -> dict[str,Any]:
    """
    作为 ToolRegistry 工具执行 Python 代码。

    工具层负责限制调用参数，
    PythonExecutor 负责实际执行。
    """

    timeout_seconds = min(
        float(timeout_seconds),
        MAX_PYTHON_TIMEOUT_SECONDS
    )

    max_output_chars = min(
        int(max_output_chars),
        MAX_PYTHON_OUTPUT_CHARS
    )

    result = _python_executor.execute(
        code,
        timeout_seconds=timeout_seconds,
        max_output_chars=max_output_chars,
    )

    return asdict(result)


def create_default_registry(
    *,
    thread_id: str = "default",
    include_browser: bool = False,
    owner_id: str | None = None,
    include_mcp: bool = False,
    mcp_tools: Sequence[ToolDefinition] | None = None,
) -> ToolRegistry:
    definitions = [
        ToolDefinition(
            name="python_executor",
            description=(
                "在受限时间和输出长度下执行 Python 代码"
            ),
            handler=execute_python_tool,
        ),
    ]

    definitions.extend(
        build_task_list_tool_definitions(thread_id)
    )
    definitions.append(build_planning_tool_definition())
    definitions.append(build_ask_human_tool_definition())
    definitions.append(build_web_search_tool_definition())
    definitions.append(build_fallback_search_tool_definition())
    definitions.append(build_multiple_search_tool_definition())
    definitions.append(build_tavily_search_tool_definition())
    definitions.extend(build_crawl_tool_definitions())
    definitions.extend(build_workspace_tool_definitions(thread_id))
    definitions.extend(build_workspace_shell_tool_definitions(thread_id))
    definitions.extend(build_workspace_web_dev_tool_definitions(thread_id))
    definitions.extend(build_workspace_sheet_tool_definitions(thread_id))
    definitions.extend(build_workspace_presentation_tool_definitions(thread_id))
    definitions.extend(build_workspace_presentation_outline_tool_definitions(thread_id))
    definitions.extend(build_workspace_vision_tool_definitions(thread_id))
    definitions.extend(build_workspace_image_edit_tool_definitions(thread_id))

    if bool(getattr(settings, "rag_enabled", False)):
        definitions.append(
            build_document_search_tool_definition(owner_id=owner_id)
        )

    if include_browser:
        definitions.extend(
            build_browser_tool_definitions(
                thread_id
            )
        )
        if bool(getattr(settings, "enable_browser_use", False)):
            definitions.extend(
                build_browser_use_tool_definitions(thread_id)
            )
        if bool(getattr(settings, "enable_computer_use", False)):
            definitions.extend(
                build_computer_use_tool_definitions(thread_id)
            )

    if include_mcp and mcp_tools:
        definitions.extend(
            tool
            for tool in mcp_tools
            if isinstance(tool, ToolDefinition)
        )

    definitions = [
        replace(tool, requires_approval=True)
        if (
            tool.name in _APPROVAL_REQUIRED_TOOL_NAMES
            or tool.name.startswith(("computer_", "mcp_"))
        )
        else tool
        for tool in definitions
    ]

    return ToolRegistry(definitions)
