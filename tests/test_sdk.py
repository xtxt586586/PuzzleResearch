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


    def test_get_session_state_preserves_checkpoint_fields(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertIn(
                "/api/sessions/thread%2Fstate/state",
                str(request.url),
            )

            return httpx.Response(
                200,
                json={
                    "thread_id": "thread/state",
                    "status": "paused",
                    "state": {
                        "research_plan": [
                            "查询一",
                        ],
                    },
                    "interrupts": [],
                    "checkpoint_id": "cp-001",
                    "checkpoint_ts": "2026-09-23T00:00:00Z",
                    "parent_checkpoint_id": None,
                    "deepsearch_artifacts": {
                        "phase": "paused",
                    },
                },
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(http=http)

        result = client.get_session_state(
            "thread/state"
        )

        self.assertEqual(
            result["checkpoint_id"],
            "cp-001",
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["phase"],
            "paused",
        )

        client.close()


    def test_resume_interrupt_sends_weaver_payload(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                request.method,
                "POST",
            )
            self.assertEqual(
                request.url.path,
                "/api/interrupt/resume",
            )

            body = json.loads(
                request.content
            )

            self.assertEqual(
                body["thread_id"],
                "thread-resume",
            )
            self.assertEqual(
                body["payload"]["approved"],
                True,
            )
            self.assertEqual(
                body["model"],
                "deepseek-chat",
            )

            return httpx.Response(
                200,
                json={
                    "status": "resumed",
                },
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(http=http)

        result = client.resume_interrupt(
            "thread-resume",
            {
                "approved": True,
            },
            model="deepseek-chat",
        )

        self.assertEqual(
            result["status"],
            "resumed",
        )

        client.close()


    def test_mcp_and_browser_compatibility_methods(self):
        requests: list[tuple[str, str]] = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(
                (
                    request.method,
                    request.url.path,
                )
            )

            if request.url.path == "/api/mcp/config":
                if request.method == "POST":
                    body = json.loads(
                        request.content
                    )
                    self.assertTrue(
                        body["enable"]
                    )
                    self.assertEqual(
                        body["servers"]["demo"]["url"],
                        "http://localhost:9000",
                    )

                return httpx.Response(
                    200,
                    json={
                        "enabled": True,
                        "servers": {
                            "demo": {
                                "url": "http://localhost:9000",
                            }
                        },
                        "loaded_tools": 0,
                    },
                    request=request,
                )

            return httpx.Response(
                200,
                json={
                    "ready": True,
                    "missing": [],
                    "sandbox_mode": "local",
                    "allow_internet": True,
                    "configured": {
                        "e2b_api_key": False,
                        "sandbox_template_browser": False,
                        "e2b_code_interpreter": False,
                        "playwright": True,
                    },
                    "deep": None,
                },
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(http=http)

        config = client.get_mcp_config()

        self.assertTrue(
            config["enabled"]
        )

        updated = client.update_mcp_config(
            enable=True,
            servers={
                "demo": {
                    "url": "http://localhost:9000",
                }
            },
        )

        self.assertEqual(
            updated["loaded_tools"],
            0,
        )

        diagnosis = client.diagnose_browser()

        self.assertTrue(
            diagnosis["ready"]
        )

        self.assertEqual(
            requests,
            [
                (
                    "GET",
                    "/api/mcp/config",
                ),
                (
                    "POST",
                    "/api/mcp/config",
                ),
                (
                    "GET",
                    "/api/sandbox/browser/diagnose",
                ),
            ],
        )

        client.close()


if __name__ == "__main__":
    unittest.main()
