import json
import unittest
from urllib.parse import urlsplit

import httpx

from sdk.python.puzzleresearch_sdk import (
    PuzzleResearchClient,
)


class PuzzleResearchCollaborationSdkTests(
    unittest.TestCase
):
    def test_collaboration_and_version_routes(self):
        calls = []

        def handler(request):
            parsed = urlsplit(str(request.url))

            calls.append(
                (
                    request.method,
                    parsed.path,
                    request.url.query.decode(),
                    request.content,
                )
            )

            return httpx.Response(
                200,
                json={"ok": True},
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(
            base_url="http://example.test",
            http=http,
        )

        client.create_share(
            "thread/one",
            permissions="view",
            expires_hours=72,
        )

        client.get_share("share/one")
        client.delete_share("share/one")

        client.add_session_comment(
            "thread/one",
            "Important finding",
            message_id="message/one",
        )

        client.get_session_comments(
            "thread/one",
            message_id="message/one",
        )

        client.get_session_versions(
            "thread/one"
        )

        client.create_session_version(
            "thread/one",
            label="Before revision",
        )

        client.restore_session_version(
            "thread/one",
            "version/one",
        )

        self.assertEqual(
            [
                (
                    method,
                    path,
                    query,
                )
                for method, path, query, _ in calls
            ],
            [
                (
                    "POST",
                    "/api/sessions/thread%2Fone/share",
                    "",
                ),
                (
                    "GET",
                    "/api/share/share%2Fone",
                    "",
                ),
                (
                    "DELETE",
                    "/api/share/share%2Fone",
                    "",
                ),
                (
                    "POST",
                    "/api/sessions/thread%2Fone/comments",
                    "",
                ),
                (
                    "GET",
                    "/api/sessions/thread%2Fone/comments",
                    "message_id=message%2Fone",
                ),
                (
                    "GET",
                    "/api/sessions/thread%2Fone/versions",
                    "",
                ),
                (
                    "POST",
                    "/api/sessions/thread%2Fone/versions",
                    "label=Before+revision",
                ),
                (
                    "POST",
                    "/api/sessions/thread%2Fone/restore/version%2Fone",
                    "",
                ),
            ],
        )

        self.assertEqual(
            json.loads(calls[0][3]),
            {
                "permissions": "view",
                "expires_hours": 72,
            },
        )

        self.assertEqual(
            json.loads(calls[3][3]),
            {
                "content": "Important finding",
                "message_id": "message/one",
            },
        )

        client.close()


if __name__ == "__main__":
    unittest.main()