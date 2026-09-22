import asyncio
import unittest
from unittest.mock import MagicMock, patch

from agent.core.event_stream import (
    event_stream_generator,
    remove_emitter_sync,
)
from tools.browser_tools import (
    _emit_browser_event,
)


class BrowserEventTests(unittest.TestCase):
    def test_browser_event_is_buffered(self):
        thread_id = "browser-events-001"
        emitter = MagicMock()

        try:
            with patch(
                "tools.browser_tools.get_emitter_sync",
                return_value=emitter,
            ):
                _emit_browser_event(
                    thread_id,
                    "tool_progress",
                    {
                        "tool": "browser_navigate",
                        "action": "navigate_completed",
                        "url": "https://example.com",
                    },
                )

            emitter.emit_sync.assert_called_once_with(
                "tool_progress",
                {
                    "thread_id": thread_id,
                    "tool": "browser_navigate",
                    "action": "navigate_completed",
                    "url": "https://example.com",
                },
            )

        finally:
            remove_emitter_sync(thread_id)

    def test_browser_screenshot_event_is_replayed(self):
        thread_id = "browser-events-002"

        async def collect():
            from agent.core.event_stream import (
                get_emitter,
            )

            emitter = await get_emitter(
                thread_id
            )

            await emitter.emit(
                "tool_screenshot",
                {
                    "tool": "browser_screenshot",
                    "filename": "browser-test.jpg",
                    "page_url": "https://example.com",
                },
            )

            await emitter.emit(
                "done",
                {},
            )

            return [
                frame
                async for frame in event_stream_generator(
                    thread_id,
                    timeout=1.0,
                )
            ]

        try:
            frames = asyncio.run(
                collect()
            )

            self.assertIn(
                "event: tool_screenshot",
                frames[0],
            )
            self.assertIn(
                "browser-test.jpg",
                frames[0],
            )

        finally:
            remove_emitter_sync(thread_id)


if __name__ == "__main__":
    unittest.main()