from __future__ import annotations

from typing import Any

from tools.registry import ToolDefinition


def ask_human(prompt: str, note: str | None = None) -> str:
    """Return a human-input request for the agent to present to the user."""
    return f"请求用户输入：{prompt}" + (f"（{note}）" if note else "")


def build_ask_human_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="ask_human",
        description=(
            "向用户提出一个需要人工回答的问题。prompt 为问题或需要展示的内容，"
            "note 可补充背景。返回提示文本供 Agent 继续处理。"
        ),
        handler=ask_human,
    )
