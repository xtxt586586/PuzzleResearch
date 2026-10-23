from __future__ import annotations

import time
import unicodedata
from collections.abc import Mapping
from typing import Any


DEFAULT_DEEPSEARCH_MAX_SECONDS = 300.0
DEFAULT_DEEPSEARCH_MAX_TOKENS = 50_000


def estimate_tokens(value: Any) -> int:
    """Estimate text tokens while excluding image payloads and non-text data."""
    if value is None:
        return 0
    if isinstance(value, str):
        if not value:
            return 0
        wide_characters = sum(
            unicodedata.east_asian_width(character) in {"W", "F"}
            for character in value
        )
        narrow_characters = len(value) - wide_characters
        return wide_characters + (narrow_characters + 3) // 4
    if isinstance(value, bytes):
        return 0
    if isinstance(value, Mapping):
        content_type = str(value.get("type", "")).lower()
        if content_type in {"image_url", "input_image"}:
            return 0
        if content_type in {"text", "input_text"}:
            return estimate_tokens(value.get("text", ""))
        return sum(
            estimate_tokens(item)
            for key, item in value.items()
            if str(key).lower() not in {"url", "data", "image_url"}
        )
    if isinstance(value, (list, tuple, set)):
        return sum(estimate_tokens(item) for item in value)
    content = getattr(value, "content", None)
    if content is not None:
        return estimate_tokens(content)
    return estimate_tokens(str(value))


def reported_token_usage(response: Any) -> int | None:
    """Read common LangChain/OpenAI usage metadata shapes."""
    candidates = [
        getattr(response, "usage_metadata", None),
        getattr(response, "response_metadata", {}).get("token_usage")
        if isinstance(getattr(response, "response_metadata", None), Mapping)
        else None,
        getattr(response, "response_metadata", {}).get("usage")
        if isinstance(getattr(response, "response_metadata", None), Mapping)
        else None,
    ]
    for usage in candidates:
        if not isinstance(usage, Mapping):
            continue
        for key in ("total_tokens", "total_token_count"):
            try:
                total = int(usage.get(key))
            except (TypeError, ValueError):
                continue
            if total >= 0:
                return total
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
        try:
            total = int(input_tokens or 0) + int(output_tokens or 0)
        except (TypeError, ValueError):
            continue
        if total > 0:
            return total
    return None


def model_call_tokens(messages: Any, response: Any) -> int:
    reported = reported_token_usage(response)
    if reported is not None:
        return reported
    return estimate_tokens(messages) + estimate_tokens(
        getattr(response, "content", "")
    )


def new_research_budget(
    artifacts: Mapping[str, Any] | None,
    *,
    max_seconds: float,
    max_tokens: int,
    initial_tokens: int = 0,
    now: float | None = None,
) -> dict[str, Any]:
    """Create or resume the shared budget from workflow artifacts."""
    current_time = time.time() if now is None else float(now)
    source: Mapping[str, Any] = {}
    if isinstance(artifacts, Mapping):
        existing = artifacts.get("research_budget")
        if isinstance(existing, Mapping):
            source = existing
        else:
            for key in ("linear_budget", "tree_budget"):
                legacy = artifacts.get(key)
                if isinstance(legacy, Mapping):
                    source = legacy
                    break

    def number(name: str, default: float, cast: Any) -> Any:
        try:
            return cast(source.get(name, default))
        except (TypeError, ValueError):
            return cast(default)

    prior_elapsed = max(number("elapsed_seconds", 0.0, float), 0.0)
    prior_tokens = max(number("tokens_used", 0, int), 0)
    prior_searches = max(number("searches_used", 0, int), 0)
    previous_update = number("last_updated_epoch", current_time, float)
    started_at = number("started_at_epoch", 0.0, float)
    if started_at <= 0:
        started_at = current_time
        previous_update = current_time

    # Zero means "use the bounded application default", not unlimited work.
    configured_seconds = max(float(max_seconds or 0.0), 0.0)
    configured_tokens = max(int(max_tokens or 0), 0)
    budget = {
        "max_seconds": configured_seconds or DEFAULT_DEEPSEARCH_MAX_SECONDS,
        "max_tokens": configured_tokens or DEFAULT_DEEPSEARCH_MAX_TOKENS,
        "elapsed_seconds": prior_elapsed,
        "tokens_used": max(prior_tokens, int(initial_tokens or 0)),
        "searches_used": prior_searches,
        "stop_reason": str(source.get("stop_reason", "") or ""),
        "started_at_epoch": started_at,
        "last_updated_epoch": previous_update,
    }
    return refresh_research_budget(budget, now=current_time)


def refresh_research_budget(
    budget: Mapping[str, Any],
    *,
    now: float | None = None,
) -> dict[str, Any]:
    current_time = time.time() if now is None else float(now)
    result = dict(budget)
    try:
        elapsed = max(float(result.get("elapsed_seconds", 0.0)), 0.0)
        last_updated = float(result.get("last_updated_epoch", current_time))
    except (TypeError, ValueError):
        elapsed = 0.0
        last_updated = current_time
    elapsed += max(current_time - last_updated, 0.0)
    result["elapsed_seconds"] = round(elapsed, 3)
    result["last_updated_epoch"] = current_time
    result["stop_reason"] = str(result.get("stop_reason", "") or "")

    if not result["stop_reason"]:
        try:
            max_seconds = max(float(result.get("max_seconds", 0.0)), 0.0)
            max_tokens = max(int(result.get("max_tokens", 0)), 0)
            tokens_used = max(int(result.get("tokens_used", 0)), 0)
        except (TypeError, ValueError):
            max_seconds = max_tokens = tokens_used = 0
        if max_seconds and elapsed >= max_seconds:
            result["stop_reason"] = "time_budget_exceeded"
        elif max_tokens and tokens_used >= max_tokens:
            result["stop_reason"] = "token_budget_exceeded"
    return result


def add_budget_tokens(
    budget: Mapping[str, Any],
    token_count: int,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    result = dict(budget)
    try:
        current = max(int(result.get("tokens_used", 0)), 0)
    except (TypeError, ValueError):
        current = 0
    result["tokens_used"] = current + max(int(token_count or 0), 0)
    return refresh_research_budget(result, now=now)


def set_budget_elapsed(
    budget: Mapping[str, Any],
    elapsed_seconds: float,
    *,
    now: float | None = None,
) -> dict[str, Any]:
    current_time = time.time() if now is None else float(now)
    result = dict(budget)
    result["elapsed_seconds"] = max(
        float(elapsed_seconds or 0.0),
        float(result.get("elapsed_seconds", 0.0) or 0.0),
    )
    result["last_updated_epoch"] = current_time
    return refresh_research_budget(result, now=current_time)


def budget_state_fields(budget: Mapping[str, Any]) -> dict[str, Any]:
    """Expose the shared budget through the stable AgentState fields."""
    refreshed = refresh_research_budget(budget)
    return {
        "budget_stop_reason": refreshed["stop_reason"],
        "deepsearch_tokens_used": int(refreshed.get("tokens_used", 0) or 0),
        "deepsearch_elapsed_seconds": float(
            refreshed.get("elapsed_seconds", 0.0) or 0.0
        ),
    }
