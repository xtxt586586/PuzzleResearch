from __future__ import annotations

from dataclasses import asdict
from typing import Any

from tools.python_executor import PythonExecutor
from tools.registry import ToolDefinition, ToolRegistry


DEFAULT_PYTHON_TIMEOUT_SECONDS = 5.0
MAX_PYTHON_TIMEOUT_SECONDS = 10.0
DEFAULT_PYTHON_OUTPUT_CHARS = 4_000
MAX_PYTHON_OUTPUT_CHARS = 8_000


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


def create_default_registry() -> ToolRegistry:
    """
    创建默认工具注册表。
    """

    return ToolRegistry(
        [
            ToolDefinition(
                name="python_executor",
                description="在受限时间和输出长度下执行 Python 代码",
                handler=execute_python_tool,
            ),
        ]
    )