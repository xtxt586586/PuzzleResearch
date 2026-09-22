import json
import unittest

import httpx

from sdk.python.puzzleresearch_sdk import (
    PuzzleResearchApiError,
    PuzzleResearchClient,
)


class PuzzleResearchClientTests(unittest.TestCase):
    def test_research_sse_parses_envelopes_and_thread_header(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(request.url.path, "/api/research/sse")
            self.assertEqual(
                json.loads(request.content)["query"],
                "测试研究",
            )
            return httpx.Response(
                200,
                headers={
                    "Content-Type": "text/event-stream",
                    "X-Thread-ID": "thread-sdk-001",
                },
                content=(
                    "id: 1\n"
                    "event: completion\n"
                    "data: {\"type\": \"completion\", "
                    "\"data\": {\"content\": \"完成\"}}\n\n"
                    "id: 2\n"
                    "event: done\n"
                    "data: {\"type\": \"done\", \"data\": {}}\n\n"
                ).encode("utf-8"),
                request=request,
            )

        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = PuzzleResearchClient(
            base_url="http://example.test/",
            http=http,
        )

        events = list(
            client.research_sse(
                {"query": "测试研究"},
            )
        )

        self.assertEqual(client.last_thread_id, "thread-sdk-001")
        self.assertEqual(
            [event["type"] for event in events],
            ["completion", "done"],
        )
        self.assertEqual(
            events[0]["data"]["content"],
            "完成",
        )
        client.close()

    def test_stream_events_sends_last_event_id(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.headers["last-event-id"],
                "3",
            )
            return httpx.Response(
                200,
                headers={"Content-Type": "text/event-stream"},
                content=(
                    "id: 4\n"
                    "data: {\"type\": \"done\", \"data\": {}}\n\n"
                ).encode("utf-8"),
                request=request,
            )

        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = PuzzleResearchClient(http=http)

        events = list(
            client.stream_events(
                "thread/sdk",
                last_event_id=3,
            )
        )

        self.assertEqual(events, [{"type": "done", "data": {}}])
        client.close()

    def test_request_json_raises_typed_error(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                404,
                text="missing",
                request=request,
            )

        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = PuzzleResearchClient(http=http)

        with self.assertRaises(PuzzleResearchApiError) as raised:
            client.get_session("missing")

        self.assertEqual(raised.exception.status, 404)
        self.assertEqual(raised.exception.path, "/api/sessions/missing")
        client.close()


    def test_get_research_tree_uses_encoded_thread_id(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn(
                "/api/sessions/thread%2Ftree/research-tree",
                str(request.url),
            )

            return httpx.Response(
                200,
                json={
                    "thread_id": "thread/tree",
                    "tree": {
                        "topic": "LangGraph",
                        "nodes": [],
                    },
                    "quality": {
                        "overall_score": 0.9,
                    },
                },
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(
            http=http,
        )

        result = client.get_research_tree(
            "thread/tree"
        )

        self.assertEqual(
            result["thread_id"],
            "thread/tree",
        )
        self.assertEqual(
            result["tree"]["topic"],
            "LangGraph",
        )
        self.assertEqual(
            result["quality"]["overall_score"],
            0.9,
        )

        client.close()

    def test_stream_research_events_keeps_quality_events(self):
        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={
                    "Content-Type": (
                        "text/event-stream"
                    ),
                },
                content=(
                    "id: 1\n"
                    "event: research_tree_update\n"
                    "data: "
                    "{\"type\": \"research_tree_update\", "
                    "\"data\": {\"tree\": {}}}\n\n"
                    "id: 2\n"
                    "event: quality_update\n"
                    "data: "
                    "{\"type\": \"quality_update\", "
                    "\"data\": {\"stage\": \"final\"}}\n\n"
                ).encode("utf-8"),
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(
            http=http,
        )

        events = list(
            client.stream_research_events(
                "thread-observe-001"
            )
        )

        self.assertEqual(
            [
                event["type"]
                for event in events
            ],
            [
                "research_tree_update",
                "quality_update",
            ],
        )

        self.assertEqual(
            events[0]["data"]["tree"],
            {},
        )
        self.assertEqual(
            events[1]["data"]["stage"],
            "final",
        )

        client.close()


if __name__ == "__main__":
    unittest.main()
