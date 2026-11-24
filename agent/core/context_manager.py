from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import tiktoken
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage


logger = logging.getLogger(__name__)

MODEL_TOKEN_LIMITS = {
    "deepseek-chat": 65_536,
    "deepseek-reasoner": 65_536,
    "gpt-4": 8_192,
    "gpt-4-32k": 32_768,
    "gpt-4-turbo": 128_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "gpt-3.5-turbo": 16_385,
    "claude-3-opus": 200_000,
    "claude-3-sonnet": 200_000,
    "claude-3-haiku": 200_000,
    "gemini-pro": 32_000,
    "gemini-1.5-pro": 1_000_000,
    "gemini-1.5-flash": 1_000_000,
    "default": 8_192,
}


@dataclass
class TokenStats:
    total_tokens: int = 0
    system_tokens: int = 0
    user_tokens: int = 0
    assistant_tokens: int = 0
    tool_tokens: int = 0
    message_count: int = 0
    truncated_count: int = 0


@dataclass
class TruncationConfig:
    max_tokens: int = 8_000
    reserve_tokens: int = 1_000
    keep_system_messages: int = 2
    keep_recent_messages: int = 4
    strategy: str = "smart"
    min_message_tokens: int = 50


def _normalize_model_name(model: str) -> str:
    name = str(model or "").strip().lower()
    if "deepseek" in name:
        return "deepseek-reasoner" if any(key in name for key in ("reasoner", "r1")) else "deepseek-chat"
    if "gpt-4o" in name:
        return "gpt-4o-mini" if "mini" in name else "gpt-4o"
    if "gpt-4-turbo" in name or "gpt-4-1106" in name:
        return "gpt-4-turbo"
    if "gpt-4-32k" in name:
        return "gpt-4-32k"
    if "gpt-4" in name:
        return "gpt-4"
    if "gpt-3.5" in name:
        return "gpt-3.5-turbo"
    if "claude-3" in name:
        if "opus" in name:
            return "claude-3-opus"
        if "haiku" in name:
            return "claude-3-haiku"
        return "claude-3-sonnet"
    if "gemini-1.5" in name:
        return "gemini-1.5-flash" if "flash" in name else "gemini-1.5-pro"
    if "gemini" in name:
        return "gemini-pro"
    return name or "default"


class ContextManager:
    """Estimate message tokens and trim history while prioritizing useful context."""

    def __init__(self, model: str = "deepseek-chat", config: TruncationConfig | None = None):
        self.model = _normalize_model_name(model)
        self.config = config or TruncationConfig()
        self.max_context_tokens = MODEL_TOKEN_LIMITS.get(self.model, MODEL_TOKEN_LIMITS["default"])
        try:
            self._encoder = tiktoken.get_encoding("cl100k_base")
        except Exception as exc:
            logger.warning("无法加载 tiktoken 编码器，使用字符数估算：%s", exc)
            self._encoder = None

    @staticmethod
    def _content_text(content: Any) -> tuple[str, int]:
        if isinstance(content, str):
            return content, 0
        if not isinstance(content, list):
            return (str(content) if content is not None else ""), 0

        text_parts: list[str] = []
        multimodal_blocks = 0
        for part in content:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str):
                    text_parts.append(text)
                kind = str(part.get("type", "")).lower()
                if any(token in kind for token in ("image", "audio", "video", "file")):
                    multimodal_blocks += 1
        return "\n".join(text_parts), multimodal_blocks

    def count_tokens(self, text: str) -> int:
        if not text:
            return 0
        if self._encoder is not None:
            try:
                return len(self._encoder.encode(text))
            except Exception:
                pass

        ascii_chars = sum(ord(char) < 128 for char in text)
        other_chars = len(text) - ascii_chars
        return math.ceil(ascii_chars / 4 + other_chars / 1.5)

    def count_message_tokens(self, message: BaseMessage) -> int:
        content, multimodal_blocks = self._content_text(message.content)
        tokens = self.count_tokens(content) + 4 + multimodal_blocks * 1_024
        name = getattr(message, "name", None)
        if name:
            tokens += self.count_tokens(str(name)) + 1

        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls:
            tokens += self.count_tokens(str(tool_calls))
        return tokens

    def count_messages_tokens(self, messages: list[BaseMessage]) -> TokenStats:
        stats = TokenStats(message_count=len(messages))
        for message in messages:
            tokens = self.count_message_tokens(message)
            stats.total_tokens += tokens
            if isinstance(message, SystemMessage):
                stats.system_tokens += tokens
            elif isinstance(message, HumanMessage):
                stats.user_tokens += tokens
            elif isinstance(message, AIMessage):
                stats.assistant_tokens += tokens
            elif isinstance(message, ToolMessage):
                stats.tool_tokens += tokens
        stats.total_tokens += 3
        return stats

    def truncate_messages(
        self,
        messages: list[BaseMessage],
        max_tokens: int | None = None,
        strategy: str | None = None,
    ) -> tuple[list[BaseMessage], TokenStats]:
        limit = self.config.max_tokens if max_tokens is None else max_tokens
        available = max(int(limit) - max(int(self.config.reserve_tokens), 0), 0)
        selected_strategy = strategy or self.config.strategy
        stats = self.count_messages_tokens(messages)
        if stats.total_tokens <= available:
            return list(messages), stats

        if selected_strategy == "fifo":
            result = self._truncate_fifo(messages, available)
        elif selected_strategy == "middle":
            result = self._truncate_middle(messages, available)
        else:
            result = self._truncate_smart(messages, available)

        result_stats = self.count_messages_tokens(result)
        result_stats.truncated_count = len(messages) - len(result)
        logger.info(
            "上下文裁剪：%s 条消息降至 %s 条，估算 token %s/%s",
            len(messages),
            len(result),
            result_stats.total_tokens,
            available,
        )
        return result, result_stats

    def _select_by_priority(
        self,
        messages: list[BaseMessage],
        max_tokens: int,
        protected: list[int],
        priority: list[int],
    ) -> list[BaseMessage]:
        remaining = max(max_tokens - 3, 0)
        selected: set[int] = set()
        for index in protected:
            cost = self.count_message_tokens(messages[index])
            if cost <= remaining:
                selected.add(index)
                remaining -= cost
        for index in priority:
            if index in selected:
                continue
            cost = self.count_message_tokens(messages[index])
            if cost <= remaining:
                selected.add(index)
                remaining -= cost
        return [message for index, message in enumerate(messages) if index in selected]

    def _protected_system_indices(self, messages: list[BaseMessage]) -> list[int]:
        keep = max(int(self.config.keep_system_messages), 0)
        return [
            index
            for index, message in enumerate(messages)
            if isinstance(message, SystemMessage)
        ][:keep]

    def _other_indices(self, messages: list[BaseMessage], protected: list[int]) -> list[int]:
        protected_set = set(protected)
        return [index for index in range(len(messages)) if index not in protected_set]

    def _truncate_smart(self, messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
        protected = self._protected_system_indices(messages)
        other = self._other_indices(messages, protected)
        recent_count = max(int(self.config.keep_recent_messages), 0)
        recent = other[-recent_count:] if recent_count else []
        middle = other[:-recent_count] if recent_count else other
        priority = list(reversed(recent)) + list(reversed(middle))
        return self._select_by_priority(messages, max_tokens, protected, priority)

    def _truncate_fifo(self, messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
        protected = self._protected_system_indices(messages)
        other = self._other_indices(messages, protected)
        return self._select_by_priority(messages, max_tokens, protected, list(reversed(other)))

    def _truncate_middle(self, messages: list[BaseMessage], max_tokens: int) -> list[BaseMessage]:
        protected = self._protected_system_indices(messages)
        other = self._other_indices(messages, protected)
        priority: list[int] = []
        left, right = 0, len(other) - 1
        while left <= right:
            priority.append(other[left])
            left += 1
            if left <= right:
                priority.append(other[right])
                right -= 1
        return self._select_by_priority(messages, max_tokens, protected, priority)

    def get_available_tokens(
        self,
        messages: list[BaseMessage],
        reserve_for_response: int = 1_000,
    ) -> int:
        used = self.count_messages_tokens(messages).total_tokens
        return max(self.max_context_tokens - used - max(reserve_for_response, 0), 0)

    def should_truncate(self, messages: list[BaseMessage], threshold: float = 0.9) -> bool:
        if not 0 < threshold <= 1:
            raise ValueError("threshold 必须大于 0 且不超过 1")
        used = self.count_messages_tokens(messages).total_tokens
        return used / self.max_context_tokens >= threshold

    def summarize_long_message(
        self,
        message: BaseMessage,
        max_tokens: int = 500,
    ) -> BaseMessage:
        if not isinstance(message.content, str) or self.count_tokens(message.content) <= max_tokens:
            return message

        suffix = "\n... [truncated]"
        suffix_tokens = self.count_tokens(suffix)
        content_budget = max(int(max_tokens) - suffix_tokens, 0)
        if self._encoder is not None:
            token_ids = self._encoder.encode(message.content)[:content_budget]
            shortened = self._encoder.decode(token_ids)
        else:
            shortened = message.content[: content_budget * 4]
        return message.model_copy(update={"content": shortened + suffix})


_context_managers: dict[tuple[Any, ...], ContextManager] = {}


def get_context_manager(
    model: str = "deepseek-chat",
    config: TruncationConfig | None = None,
) -> ContextManager:
    key = (
        model,
        config.max_tokens,
        config.reserve_tokens,
        config.keep_system_messages,
        config.keep_recent_messages,
        config.strategy,
        config.min_message_tokens,
    ) if config else (model, "default")
    if key not in _context_managers:
        _context_managers[key] = ContextManager(model=model, config=config)
    return _context_managers[key]


def truncate_for_model(
    messages: list[BaseMessage],
    model: str = "deepseek-chat",
    max_tokens: int | None = None,
) -> list[BaseMessage]:
    manager = get_context_manager(model)
    limit = max_tokens if max_tokens is not None else int(manager.max_context_tokens * 0.8)
    truncated, stats = manager.truncate_messages(messages, max_tokens=limit)
    if stats.truncated_count:
        logger.info("已为模型 %s 移除 %s 条上下文消息", model, stats.truncated_count)
    return truncated
