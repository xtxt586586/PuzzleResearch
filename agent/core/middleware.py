from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage

from agent.core.context_manager import get_context_manager
from common.config import settings


def maybe_strip_tool_messages(messages: list[BaseMessage]) -> list[BaseMessage]:
    """Optionally omit tool results from the persisted model context."""
    if not settings.strip_tool_messages:
        return messages
    return [message for message in messages if not isinstance(message, ToolMessage)]


def clear_tool_results_for_context(tool_results: list[Any]) -> list[Any]:
    """Replace older tool outputs with placeholders when their context is large."""
    results = list(tool_results or [])
    if not results:
        return results

    trigger_tokens = max(int(settings.context_edit_trigger_tokens), 0)
    keep_tools = max(int(settings.context_edit_keep_tools), 0)
    serialized = json.dumps(results, ensure_ascii=False, default=str)
    if get_context_manager().count_tokens(serialized) <= trigger_tokens:
        return results

    clear_count = max(len(results) - keep_tools, 0)
    if clear_count == 0:
        return results

    edited = list(results)
    for index in range(clear_count):
        item = results[index]
        if not isinstance(item, dict) or "result" not in item:
            continue
        edited_item = dict(item)
        edited_item["result"] = "[cleared]"
        edited[index] = edited_item
    return edited
