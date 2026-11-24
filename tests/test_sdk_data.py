import unittest
from urllib.parse import urlsplit

import httpx

from sdk.python.puzzleresearch_sdk import (
    PuzzleResearchClient,
)


class PuzzleResearchDataSdkTests(
    unittest.TestCase
):
    def test_document_export_and_screenshot_routes(self):
        calls = []

        def handler(request):
            parsed = urlsplit(str(request.url))

            calls.append(
                (
                    request.method,
                    parsed.path,
                    request.url.query.decode(),
                )
            )

            if parsed.path.startswith(
                "/api/export/"
            ):
                return httpx.Response(
                    200,
                    content=b"export-data",
                    headers={
                        "content-type": "text/html",
                    },
                    request=request,
                )

            if parsed.path.startswith(
                "/api/screenshots/"
            ):
                return httpx.Response(
                    200,
                    content=b"image-data",
                    headers={
                        "content-type": "image/png",
                    },
                    request=request,
                )

            if parsed.path == "/api/documents/upload":
                self.assertIn(
                    "multipart/form-data",
                    request.headers["content-type"],
                )
                self.assertIn(
                    b"notes.txt",
                    request.content,
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

        client.upload_document(
            "notes.txt",
            b"research notes",
            content_type="text/plain",
        )

        client.list_documents(limit=20)

        client.delete_document(
            "folder/notes.txt"
        )

        client.search_documents(
            "research",
            n_results=8,
        )

        client.list_export_templates()

        exported, export_type = (
            client.export_report(
                "thread/one",
                format="html",
                template="default",
                title="Research",
            )
        )

        self.assertEqual(
            exported,
            b"export-data",
        )
        self.assertEqual(
            export_type,
            "text/html",
        )

        image, image_type = (
            client.get_screenshot(
                "folder/example.png"
            )
        )

        self.assertEqual(
            image,
            b"image-data",
        )
        self.assertEqual(
            image_type,
            "image/png",
        )

        client.list_screenshots(
            thread_id="thread/one",
            limit=10,
        )

        client.cleanup_screenshots()

        client.get_browser_session_info(
            "thread/one"
        )

        client.trigger_browser_screenshot(
            "thread/one"
        )

        self.assertEqual(
            calls,
            [
                (
                    "POST",
                    "/api/documents/upload",
                    "",
                ),
                (
                    "GET",
                    "/api/documents/list",
                    "limit=20",
                ),
                (
                    "DELETE",
                    "/api/documents/folder%2Fnotes.txt",
                    "",
                ),
                (
                    "POST",
                    "/api/documents/search",
                    "query=research&n_results=8",
                ),
                (
                    "GET",
                    "/api/export/templates",
                    "",
                ),
                (
                    "GET",
                    "/api/export/thread%2Fone",
                    "format=html&template=default&title=Research",
                ),
                (
                    "GET",
                    "/api/screenshots/folder%2Fexample.png",
                    "",
                ),
                (
                    "GET",
                    "/api/screenshots",
                    "limit=10&thread_id=thread%2Fone",
                ),
                (
                    "POST",
                    "/api/screenshots/cleanup",
                    "",
                ),
                (
                    "GET",
                    "/api/browser/thread%2Fone/info",
                    "",
                ),
                (
                    "POST",
                    "/api/browser/thread%2Fone/screenshot",
                    "",
                ),
            ],
        )

        client.close()


if __name__ == "__main__":
    unittest.main()