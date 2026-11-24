from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import AIMessage, BaseMessage


STUCK_HINT = "检测到连续重复的助手结果，请尝试不同方案或使用其他工具。"


def detect_stuck(messages: Sequence[BaseMessage], threshold: int = 2) -> bool:
    """Return whether the latest assistant message repeats consecutively."""
    threshold = max(int(threshold), 0)
    if len(messages) < threshold + 1:
        return False

    latest = messages[-1]
    if not isinstance(latest, AIMessage) or not latest.content:
        return False

    repeats = 0
    for message in reversed(messages[:-1]):
        if isinstance(message, AIMessage) and message.content == latest.content:
            repeats += 1
        else:
            break
    return repeats >= threshold


def inject_stuck_hint(messages: Sequence[BaseMessage]) -> list[BaseMessage]:
    """Return a copied message list with a nudge appended."""
    return [*messages, AIMessage(content=STUCK_HINT)]
