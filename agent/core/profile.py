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

    system_prompt: str = ""

    model: str | None = None

    enabled_tools: frozenset[str] | None = None

    blocked_tools: frozenset[str] | None = None

    tool_call_limit: int | None = None

    require_tool_approval: bool | None = None

    tool_retry_limit: int | None = None

    deduplicate_tool_calls: bool | None = None

    cache_tool_results: bool | None = None

    max_tool_result_chars: int | None = None

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Agent Profile 的名称不能为空。")

        if self.tool_call_limit is not None:
            if self.tool_call_limit < 0:
                raise ValueError("Agent Profile 的 tool_call_limit 不能小于 0")

        if self.tool_retry_limit is not None:
            if self.tool_retry_limit < 0:
                raise ValueError("Agent Profile 的 tool_retry_limit 不能小于 0")


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

        if self.blocked_tools is not None:
            normalized_blocked_tools = frozenset(
                str(tool_name).strip()
                for tool_name in self.blocked_tools
                if str(tool_name).strip()
            )

            object.__setattr__(
                self,
                "blocked_tools",
                normalized_blocked_tools,
            )

        if self.max_tool_result_chars is not None:
            if self.max_tool_result_chars <= 0:
                raise ValueError(
                    "Agent Profile 的 max_tool_result_chars 必须大于 0"
                )