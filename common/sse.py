from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable


async def _close_iterator(iterator: AsyncIterator[str]) -> None:
    close = getattr(iterator, "aclose", None)
    if close is not None:
        try:
            await close()
        except (RuntimeError, GeneratorExit):
            pass


async def _stop_pending(task: asyncio.Task[str] | None) -> None:
    if task is None:
        return
    if not task.done():
        task.cancel()
    try:
        await task
    except BaseException:
        pass


async def iter_with_sse_keepalive(
    source: AsyncIterable[str],
    *,
    interval_s: float = 15.0,
) -> AsyncIterator[str]:
    iterator = aiter(source)
    pending: asyncio.Task[str] | None = None
    interval = max(float(interval_s), 0.0)

    try:
        while True:
            if pending is None:
                pending = asyncio.create_task(anext(iterator))

            done, _ = await asyncio.wait(
                {pending},
                timeout=interval,
            )
            if not done:
                yield ": keepalive\n\n"
                continue

            task = pending
            pending = None
            try:
                yield task.result()
            except StopAsyncIteration:
                return
    finally:
        await _stop_pending(pending)
        await _close_iterator(iterator)


async def iter_abort_on_disconnect(
    source: AsyncIterable[str],
    *,
    is_disconnected: Callable[[], Awaitable[bool]],
    check_interval_s: float = 0.25,
) -> AsyncIterator[str]:
    iterator = aiter(source)
    pending: asyncio.Task[str] | None = None
    interval = max(float(check_interval_s), 0.0)

    try:
        while True:
            try:
                if await is_disconnected():
                    return
            except Exception:
                pass

            if pending is None:
                pending = asyncio.create_task(anext(iterator))

            done, _ = await asyncio.wait(
                {pending},
                timeout=interval,
            )
            if not done:
                continue

            task = pending
            pending = None
            try:
                yield task.result()
            except StopAsyncIteration:
                return
    finally:
        await _stop_pending(pending)
        await _close_iterator(iterator)
