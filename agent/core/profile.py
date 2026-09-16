from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AgentProfile:
    """
    Agent 的运行策略。

    Profile 通过 RunnableConfig 注入，不写入 AgentState，
    避免把运行时策略和可持久化状态混在一起。
    """

    name: str = "default"

    enabled_tools: frozenset[str] | None = None

    tool_call_limit: int | None = None

    require_tool_approval: bool | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent Profile 的名称不能为空。")

        if self.tool_call_limit is not None:
            if self.tool_call_limit < 0:
                raise ValueError("Agent Profile 的 tool_call_limit 不能小于 0")


        if self.enabled_tools is not None:
            normalized_tools = frozenset(
                str(tool_name).strip()
                for tool_name in self.enabled_tools
                if str(tool_name).strip()
            )

            object.__setattr__(
                self,
                "enabled_tools",
                normalized_tools,
            )