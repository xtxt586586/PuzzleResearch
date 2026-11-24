from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import main
from common.sse import (
    iter_abort_on_disconnect,
    iter_with_sse_keepalive,
)


class SSELifecycleTests(unittest.IsolatedAsyncioTestCase):
    async def test_keepalive_is_emitted_while_source_is_idle(self):
        gate = asyncio.Event()
        started = asyncio.Event()
        closed = asyncio.Event()

        async def source():
            started.set()
            try:
                await gate.wait()
                yield "data: ready\n\n"
            finally:
                closed.set()

        stream = iter_with_sse_keepalive(
            source(),
            interval_s=0.01,
        )
        self.assertEqual(
            await anext(stream),
            ": keepalive\n\n",
        )
        self.assertTrue(started.is_set())
        await stream.aclose()
        self.assertTrue(closed.is_set())

    async def test_keepalive_preserves_source_items(self):
        async def source():
            yield "event: completion\ndata: done\n\n"

        stream = iter_with_sse_keepalive(
            source(),
            interval_s=30,
        )
        self.assertEqual(
            await anext(stream),
            "event: completion\ndata: done\n\n",
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

    async def test_disconnect_cancels_pending_source_iteration(self):
        started = asyncio.Event()
        closed = asyncio.Event()
        checks = 0

        async def source():
            started.set()
            try:
                await asyncio.Event().wait()
                yield "unreachable"
            finally:
                closed.set()

        async def is_disconnected():
            nonlocal checks
            checks += 1
            return checks > 1

        stream = iter_abort_on_disconnect(
            source(),
            is_disconnected=is_disconnected,
            check_interval_s=0.01,
        )
        with self.assertRaises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), timeout=1)

        self.assertTrue(started.is_set())
        self.assertTrue(closed.is_set())

    async def test_disconnect_before_first_event_skips_source(self):
        started = False

        async def source():
            nonlocal started
            started = True
            yield "data: should-not-be-sent\n\n"

        async def is_disconnected():
            return True

        stream = iter_abort_on_disconnect(
            source(),
            is_disconnected=is_disconnected,
        )
        with self.assertRaises(StopAsyncIteration):
            await anext(stream)

        self.assertFalse(started)

    async def _assert_chat_route_skips_disconnected_client(
        self,
        route,
        payload,
    ):
        class DisconnectedRequest:
            headers = {}

            async def is_disconnected(self):
                return True

        source_started = False

        async def blocked_stream(**_kwargs):
            nonlocal source_started
            source_started = True
            await asyncio.Event().wait()
            yield "unreachable"

        with patch.object(
            main,
            "_resolve_request_principal",
            return_value=SimpleNamespace(user_id="test-user"),
        ), patch.object(
            main,
            "_resolve_agent_runtime",
            return_value=(SimpleNamespace(model=""), None),
        ), patch.object(
            main,
            "_build_chat_config",
            return_value={},
        ), patch.object(
            main,
            "_register_thread_owner",
        ), patch.object(
            main,
            "_iter_sse_chat_stream",
            new=blocked_stream,
        ):
            response = await route(payload, DisconnectedRequest())
            with self.assertRaises(StopAsyncIteration):
                await asyncio.wait_for(
                    anext(response.body_iterator),
                    timeout=1,
                )

        self.assertFalse(source_started)

    async def test_chat_sse_does_not_start_work_for_disconnected_client(self):
        payload = main.ChatRequest(
            messages=[main.Message(role="user", content="test question")]
        )
        await self._assert_chat_route_skips_disconnected_client(
            main.chat_sse,
            payload,
        )

    async def test_research_sse_does_not_start_work_for_disconnected_client(
        self,
    ):
        payload = main.ResearchRequest(query="test question")
        await self._assert_chat_route_skips_disconnected_client(
            main.research_sse_api,
            payload,
        )


if __name__ == "__main__":
    unittest.main()
