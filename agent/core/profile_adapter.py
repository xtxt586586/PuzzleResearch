from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from agent.core.profile import AgentProfile
from common.agent_store import AgentRecord


_POLICY_FIELDS = {
    "tool_call_limit",
    "require_tool_approval",
    "tool_retry_limit",
    "deduplicate_tool_calls",
    "cache_tool_results",
    "max_tool_result_chars",
}

def _runtime_policy(
        record: AgentRecord
) -> dict[str,Any]:
    raw_policy = record.metadata.get(
        "runtime_policy",
        {}
    )

    if raw_policy is None:
        return {}

    if not isinstance(raw_policy,Mapping):
        raise ValueError(
            "Agent metadata.runtime_policy 必须是对象。"
        )

    return {
        key: raw_policy[key]
        for key in _POLICY_FIELDS
        if key in raw_policy
    }


def _tool_sets(
        record: AgentRecord
) -> tuple[
   frozenset[str] | None,
   frozenset[str] | None,
]:
    if not record.enabled_tools:
        return None,None

    enabled = frozenset(
        name
        for name ,is_enabled
        in record.enabled_tools.items()
        if is_enabled
    )

    blocked = frozenset(
        name
        for name ,is_enabled
        in record.enabled_tools.items()
        if not is_enabled
    )

    return (
        enabled,
        blocked or None,
    )


def runtime_profile_from_record(
        record: AgentRecord
) -> AgentProfile:
    if not isinstance(record,AgentRecord):
        raise TypeError(
            "record 必须是 AgentRecord 的实例。"
        )

    enabled_tools,blocked_tools = _tool_sets(record)
    policy = _runtime_policy(record)

    return AgentProfile(
        name=record.name,
        system_prompt=record.system_prompt.strip(),
        model=record.model.strip() or None,
        enabled_tools=enabled_tools,
        blocked_tools=blocked_tools,
        tool_call_limit=policy.get(
            "tool_call_limit"
        ),
        require_tool_approval=policy.get(
            "require_tool_approval"
        ),
        tool_retry_limit=policy.get(
            "tool_retry_limit"
        ),
        deduplicate_tool_calls=policy.get(
            "deduplicate_tool_calls"
        ),
        cache_tool_results=policy.get(
            "cache_tool_results"
        ),
        max_tool_result_chars=policy.get(
            "max_tool_result_chars"
        ),
    )