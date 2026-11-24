"""Retry and circuit-breaker helpers for search providers."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ReliabilityPolicy:
    max_retries: int = 2
    retry_backoff_seconds: float = 0.5
    circuit_breaker_failures: int = 3
    circuit_breaker_reset_seconds: float = 60.0


@dataclass
class _ProviderState:
    consecutive_failures: int = 0
    opened_at: float | None = None


class ProviderReliabilityManager:
    """Run provider calls with retry and per-provider circuit protection."""

    def __init__(self, policy: ReliabilityPolicy | None = None) -> None:
        self.policy = policy or ReliabilityPolicy()
        self._states: dict[str, _ProviderState] = {}
        self._lock = threading.RLock()

    def _state(self, provider: str) -> _ProviderState:
        with self._lock:
            return self._states.setdefault(provider, _ProviderState())

    def _reset_expired(self, provider: str) -> None:
        state = self._state(provider)
        if state.opened_at is None:
            return
        reset_after = max(0.0, float(self.policy.circuit_breaker_reset_seconds))
        if reset_after == 0.0 or time.monotonic() - state.opened_at >= reset_after:
            with self._lock:
                state.consecutive_failures = 0
                state.opened_at = None

    def is_open(self, provider: str) -> bool:
        self._reset_expired(provider)
        return self._state(provider).opened_at is not None

    def _record_success(self, provider: str) -> None:
        state = self._state(provider)
        with self._lock:
            state.consecutive_failures = 0
            state.opened_at = None

    def _record_failure(self, provider: str) -> None:
        state = self._state(provider)
        with self._lock:
            state.consecutive_failures += 1
            threshold = max(1, int(self.policy.circuit_breaker_failures))
            if state.consecutive_failures >= threshold:
                state.opened_at = time.monotonic()

    def call(self, provider: str, function: Callable[[], Any]) -> Any:
        if self.is_open(provider):
            return []

        attempts = max(1, int(self.policy.max_retries) + 1)
        for attempt in range(attempts):
            try:
                result = function()
                self._record_success(provider)
                return result
            except Exception:
                self._record_failure(provider)
                if self.is_open(provider) or attempt == attempts - 1:
                    return []
                delay = max(0.0, float(self.policy.retry_backoff_seconds)) * (2**attempt)
                if delay:
                    time.sleep(delay)
        return []

    def snapshot(self, provider: str) -> dict[str, Any]:
        self._reset_expired(provider)
        state = self._state(provider)
        opened_for = None
        resets_in = None
        if state.opened_at is not None:
            opened_for = max(0.0, time.monotonic() - state.opened_at)
            reset_after = max(0.0, float(self.policy.circuit_breaker_reset_seconds))
            resets_in = 0.0 if reset_after == 0.0 else max(0.0, reset_after - opened_for)
        return {
            "provider": provider,
            "is_open": state.opened_at is not None,
            "consecutive_failures": state.consecutive_failures,
            "opened_for_seconds": opened_for,
            "resets_in_seconds": resets_in,
            "policy": {
                "max_retries": int(self.policy.max_retries),
                "retry_backoff_seconds": float(self.policy.retry_backoff_seconds),
                "circuit_breaker_failures": int(self.policy.circuit_breaker_failures),
                "circuit_breaker_reset_seconds": float(self.policy.circuit_breaker_reset_seconds),
            },
        }
