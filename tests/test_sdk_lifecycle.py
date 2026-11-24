import json
import unittest
from urllib.parse import urlsplit

import httpx

from sdk.python.puzzleresearch_sdk import (
    PuzzleResearchClient,
)


class PuzzleResearchLifecycleSdkTests(
    unittest.TestCase
):
    def test_json_requests_capture_thread_id(self):
        def handler(request):
            return httpx.Response(
                200,
                json={
                    "status": "cancelled",
                },
                headers={
                    "X-Thread-ID": "thread-json-001",
                },
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(
            base_url="http://example.test",
            http=http,
        )

        result = client.cancel_all_chats()

        self.assertEqual(
            result["status"],
            "cancelled",
        )
        self.assertEqual(
            client.last_thread_id,
            "thread-json-001",
        )

        client.close()

    def test_cancel_reason_is_sent_as_json(self):
        captured = {}

        def handler(request):
            parsed = urlsplit(str(request.url))

            captured["method"] = request.method
            captured["path"] = parsed.path
            captured["body"] = json.loads(
                request.content
            )

            return httpx.Response(
                200,
                json={
                    "status": "cancelled",
                },
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(
            base_url="http://example.test",
            http=http,
        )

        client.cancel_chat(
            "thread/one",
            reason="user_cancelled",
        )

        self.assertEqual(
            captured,
            {
                "method": "POST",
                "path": (
                    "/api/chat/cancel/"
                    "thread%2Fone"
                ),
                "body": {
                    "reason": "user_cancelled",
                },
            },
        )

        client.close()

    def test_context_manager_closes_client(self):
        http = httpx.Client(
            transport=httpx.MockTransport(
                lambda request: httpx.Response(
                    200,
                    json={"ok": True},
                    request=request,
                )
            )
        )

        with PuzzleResearchClient(
            base_url="http://example.test",
            http=http,
        ) as client:
            self.assertFalse(
                client._http.is_closed
            )

        self.assertTrue(
            client._http.is_closed
        )


if __name__ == "__main__":
    unittest.main()