from __future__ import annotations

import asyncio
import json
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Any


_DEFAULT_EMITTER_TTL_SECONDS = 60.0 * 60.0
_MAX_EMITTERS = 5000
_TERMINAL_EVENT_TYPES = frozenset(
    {
        "done",
        "cancelled",
        "error",
    }
)


@dataclass(frozen=True)
class ThreadEvent:
    event_type: str
    data: dict[str,Any]
    thread_id: str
    seq: int
    event_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:8]
    )
    timestamp: float = field(
        default_factory=time.time
    )

    def to_dict(self) -> dict[str,Any]:
        return {
            "type": self.event_type,
            "data": self.data,
            "event_id": self.event_id,
            "seq": self.seq,
            "timestamp": self.timestamp,
            "thread_id": self.thread_id,
        }

    def to_sse(self) -> str:
        return (
            f"id: {self.seq}\n"
            f"event: {self.event_type}\n"
            "data: "
            + json.dumps(
                self.to_dict(),
                ensure_ascii=False
            )
            + "\n\n"
        )


class ThreadEventEmitter:
    def __init__(
            self,
            thread_id: str,
            buffer_size: int = 100,
    ):
        self.thread_id = thread_id
        self.buffer_size = max(int(buffer_size),1)
        self._sequence = 0
        self._events: list[ThreadEvent] = []
        self._queues: set[asyncio.Queue[ThreadEvent]] = set()
        self._lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._last_activity = time.monotonic()
        self._terminal = False

    async def emit(
            self,
            event_type: str,
            data: dict[str,Any] | None = None,
    ) -> ThreadEvent:
        with self._lock:
            self._sequence += 1

            event = ThreadEvent(
                event_type=str(event_type),
                data=dict(data or {}),
                thread_id=self.thread_id,
                seq=self._sequence,
            )

            self._events.append(event)

            if len(self._events) > self.buffer_size:
                self._events.pop(0)

            self._last_activity = time.monotonic()

            if event.event_type in _TERMINAL_EVENT_TYPES:
                self._terminal = True

            queues = list(self._queues)

        for queue in queues:
            await queue.put(event)

        return event


    def buffered_events(self) -> list[ThreadEvent]:
        with self._lock:
            return list(self._events)

    def subscribe(
            self,
    ) -> asyncio.Queue[ThreadEvent]:

        queue: asyncio.Queue[ThreadEvent] = asyncio.Queue()

        with self._lock:
            self._queues.add(queue)
            self._last_activity = time.monotonic()

        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            pass


        return queue

    def unsubscribe(
            self,
            queue: asyncio.Queue[ThreadEvent],
    ) -> None:
        with self._lock:
            self._queues.discard(queue)
            self._last_activity = time.monotonic()


    @property
    def is_terminal(self) -> bool:
        with self._lock:
            return self._terminal

    @property
    def has_subscribers(self) -> bool:
        with self._lock:
            return bool(self._queues)

    @property
    def last_activity(self) -> float:
        with self._lock:
            return self._last_activity

    def emit_sync(
            self,
            event_type: str,
            data: dict[str,Any] | None = None,
    ) -> None:
        coroutine = self.emit(
            event_type,
            data,
        )

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if (
            running_loop is not None
            and (
                self._loop is None
                or self._loop is running_loop
            )
        ):
            running_loop.create_task(coroutine)
            return

        if (
            self._loop is not None
            and self._loop.is_running()
        ):
            asyncio.run_coroutine_threadsafe(
                coroutine,
                self._loop,
            )
            return

        if running_loop is not None:
            running_loop.create_task(coroutine)
            return

        asyncio.run(coroutine)

    # Backward-compatible alias for callers from the first event-stream pass.
    def _emit_sync(
            self,
            event_type: str,
            data: dict[str,Any] | None = None,
    ) -> None:
        self.emit_sync(event_type, data)



_emitters: dict[str,ThreadEventEmitter] = {}
_emitters_lock = threading.Lock()


def _prune_emitters_locked(
        now: float,
) -> None:
    expired_ids = [
        thread_id
        for thread_id, emitter in _emitters.items()
        if (
            not emitter.has_subscribers
            and now - emitter.last_activity
            > _DEFAULT_EMITTER_TTL_SECONDS
        )
    ]

    for thread_id in expired_ids:
        _emitters.pop(thread_id, None)

    if len(_emitters) <= _MAX_EMITTERS:
        return

    inactive_emitters = sorted(
        (
            emitter.last_activity,
            thread_id,
        )
        for thread_id, emitter in _emitters.items()
        if not emitter.has_subscribers
    )

    for _, thread_id in inactive_emitters[
        : max(len(_emitters) - _MAX_EMITTERS, 0)
    ]:
        _emitters.pop(thread_id, None)


async def get_emitter(
        thread_id: str,
) -> ThreadEventEmitter:
    return get_emitter_sync(thread_id)


def get_emitter_sync(
        thread_id: str,
) -> ThreadEventEmitter:
    normalized_id = str(thread_id).strip()

    if not normalized_id:
        raise ValueError("thread_id 不能为空。")

    with _emitters_lock:
        _prune_emitters_locked(time.monotonic())
        emitter = _emitters.get(normalized_id)

        if emitter is None:
            emitter = ThreadEventEmitter(normalized_id)
            _emitters[normalized_id] = emitter

        return emitter


def remove_emitter_sync(
        thread_id: str,
) -> None:
    with _emitters_lock:
        _emitters.pop(str(thread_id).strip(),None)

def _parser_last_event_id(
        value: str | None,
) -> int | None:
    if value is None:
        return None

    try:
        return int(str(value).strip())
    except (TypeError,ValueError):
        return None


async def event_stream_generator(
        thread_id: str,
        *,
        timeout: float = 300.0,
        last_event_id: str | None = None,
):
    emitter = await get_emitter(thread_id)
    queue = emitter.subscribe()
    cursor = _parser_last_event_id(last_event_id)

    try:
        for event in emitter.buffered_events():
            if cursor is not None and event.seq <= cursor:
                if event.event_type in _TERMINAL_EVENT_TYPES:
                    return
                continue

            cursor = event.seq
            yield event.to_sse()

            if event.event_type == "done":
                return

            if event.event_type in _TERMINAL_EVENT_TYPES:
                return

        started_at = time.monotonic()

        while True:
            remaining = timeout - (
                time.monotonic() - started_at
            )

            if remaining <= 0:
                return

            try:
                event = await asyncio.wait_for(
                    queue.get(),
                    timeout=min(10.0,remaining),
                )

            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if cursor is not None and event.seq <= cursor:
                continue

            cursor = event.seq
            yield event.to_sse()

            if event.event_type == "done":
                return

            if event.event_type in _TERMINAL_EVENT_TYPES:
                return

    finally:
        emitter.unsubscribe(queue)
