from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Awaitable, Callable

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


@dataclass
class CancellationToken:
    task_id: str
    status: TaskStatus = TaskStatus.PENDING
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    cancelled_at: datetime | None = None
    current_checkpoint: str = ""
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    _cancelled: bool = field(default=False, repr=False)
    _event: asyncio.Event = field(
        default_factory=asyncio.Event,
        repr=False,
    )
    _cleanup_callbacks: list[
        Callable[[], Awaitable[None]]
    ] = field(default_factory=list, repr=False)

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled

    def cancel(self, reason: str = "User requested cancellation") -> None:
        if self._cancelled:
            return

        self._cancelled = True
        self.cancelled_at = datetime.now(timezone.utc)
        self.status = TaskStatus.CANCELLED
        self.metadata["cancel_reason"] = reason
        self._event.set()

    def check(self, checkpoint: str = "") -> None:
        if checkpoint:
            self.current_checkpoint = checkpoint
            self.checkpoints.append(
                {
                    "checkpoint": checkpoint,
                    "timestamp": datetime.now(
                        timezone.utc
                    ).isoformat(),
                }
            )

        if self._cancelled:
            raise asyncio.CancelledError(
                self.metadata.get(
                    "cancel_reason",
                    "Task cancelled",
                )
            )

    def mark_running(self) -> None:
        if not self._cancelled:
            self.status = TaskStatus.RUNNING

    def mark_paused(self) -> None:
        if not self._cancelled:
            self.status = TaskStatus.PAUSED

    def mark_completed(self) -> None:
        if not self._cancelled:
            self.status = TaskStatus.COMPLETED

    def mark_failed(self, error: str) -> None:
        if not self._cancelled:
            self.status = TaskStatus.FAILED
            self.metadata["error"] = error

    def register_cleanup(
        self,
        callback: Callable[[], Awaitable[None]],
    ) -> None:
        self._cleanup_callbacks.append(callback)

    async def run_cleanup(self) -> None:
        for callback in reversed(self._cleanup_callbacks):
            try:
                await callback()
            except Exception:
                logger.exception(
                    "Cancellation cleanup callback failed"
                )

        self._cleanup_callbacks.clear()

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "status": self.status.value,
            "is_cancelled": self.is_cancelled,
            "created_at": self.created_at.isoformat(),
            "cancelled_at": (
                self.cancelled_at.isoformat()
                if self.cancelled_at
                else None
            ),
            "current_checkpoint": self.current_checkpoint,
            "checkpoint_count": len(self.checkpoints),
            "metadata": dict(self.metadata),
        }


class CancellationManager:
    def __init__(self) -> None:
        self._tokens: dict[str, CancellationToken] = {}
        self._lock = asyncio.Lock()

    async def create_token(
        self,
        task_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CancellationToken:
        async with self._lock:
            old_token = self._tokens.get(task_id)

            if old_token is not None and not old_token.is_cancelled:
                old_token.cancel("Replaced by new task")
                await old_token.run_cleanup()

            token = CancellationToken(
                task_id=task_id,
                metadata=dict(metadata or {}),
            )
            self._tokens[task_id] = token
            return token

    def create_token_sync(
        self,
        task_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> CancellationToken:
        old_token = self._tokens.get(task_id)

        if old_token is not None and not old_token.is_cancelled:
            old_token.cancel("Replaced by new task")

        token = CancellationToken(
            task_id=task_id,
            metadata=dict(metadata or {}),
        )
        self._tokens[task_id] = token
        return token

    def get_token(
        self,
        task_id: str,
    ) -> CancellationToken | None:
        return self._tokens.get(task_id)

    async def cancel(
        self,
        task_id: str,
        reason: str = "User requested cancellation",
    ) -> bool:
        async with self._lock:
            token = self._tokens.get(task_id)

            if token is None:
                return False

            token.cancel(reason)
            await token.run_cleanup()
            return True

    async def cancel_all(
        self,
        reason: str = "Batch cancellation",
    ) -> int:
        cancelled_count = 0

        async with self._lock:
            for token in self._tokens.values():
                if token.status not in {
                    TaskStatus.PENDING,
                    TaskStatus.RUNNING,
                    TaskStatus.PAUSED,
                }:
                    continue

                token.cancel(reason)
                await token.run_cleanup()
                cancelled_count += 1

        return cancelled_count

    def get_active_tasks(self) -> dict[str, dict[str, Any]]:
        active_statuses = {
            TaskStatus.PENDING,
            TaskStatus.RUNNING,
            TaskStatus.PAUSED,
        }

        return {
            task_id: token.to_dict()
            for task_id, token in self._tokens.items()
            if token.status in active_statuses
        }

    def get_stats(self) -> dict[str, int]:
        stats = {
            status.value: 0
            for status in TaskStatus
        }

        for token in self._tokens.values():
            stats[token.status.value] += 1

        stats["total"] = len(self._tokens)
        return stats


cancellation_manager = CancellationManager()


def check_cancellation(
    task_id: str,
    checkpoint: str = "",
) -> None:
    token = cancellation_manager.get_token(task_id)

    if token is not None:
        token.check(checkpoint)


def check_state_cancellation(
    state: dict[str, Any],
    checkpoint: str = "",
) -> None:
    if state.get("is_cancelled"):
        raise asyncio.CancelledError(
            "任务已取消"
        )

    token_id = state.get("cancel_token_id")

    if isinstance(token_id, str) and token_id:
        check_cancellation(token_id, checkpoint)