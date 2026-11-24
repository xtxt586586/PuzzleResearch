import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main


PNG = b"\x89PNG\r\n\x1a\npuzzleresearch"


class ScreenshotApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.path_patch = patch.object(
            main.settings,
            "screenshot_store_path",
            self.temp_dir.name,
        )
        self.path_patch.start()

        self.client = TestClient(main.app)

    def tearDown(self):
        self.path_patch.stop()
        self.temp_dir.cleanup()

    def test_save_list_and_get_screenshot(self):
        saved = main._screenshot_store().save_screenshot(
            PNG,
            action="browser_manual",
            thread_id="thread-001",
            page_url="https://example.com/research",
            metadata={"title": "Research page"},
        )

        listed = self.client.get(
            "/api/screenshots",
            params={
                "thread_id": "thread-001",
            },
        )

        self.assertEqual(
            listed.status_code,
            200,
        )
        self.assertEqual(
            listed.json()["count"],
            1,
        )
        self.assertEqual(
            listed.json()["screenshots"][0]["action"],
            "browser_manual",
        )
        self.assertEqual(
            listed.json()["screenshots"][0]["page_url"],
            "https://example.com/research",
        )
        self.assertEqual(
            listed.json()["screenshots"][0]["title"],
            "Research page",
        )

        response = self.client.get(
            saved["url"]
        )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.headers["content-type"],
            "image/png",
        )
        self.assertEqual(
            response.content,
            PNG,
        )

    def test_list_screenshots_does_not_match_thread_id_substrings(self):
        store = main._screenshot_store()
        matching = store.save_screenshot(
            PNG,
            action="browser_screenshot",
            thread_id="thread-001",
        )
        store.save_screenshot(
            PNG,
            action="browser_screenshot",
            thread_id="other-thread-001",
        )

        listed = store.list_screenshots(
            thread_id="thread-001",
        )

        self.assertEqual(
            [item["filename"] for item in listed],
            [matching["filename"]],
        )

    def test_list_screenshots_keeps_sanitized_thread_ids_distinct(self):
        store = main._screenshot_store()
        matching = store.save_screenshot(
            PNG,
            action="browser_screenshot",
            thread_id="thread/a",
        )
        store.save_screenshot(
            PNG,
            action="browser_screenshot",
            thread_id="thread?a",
        )

        listed = store.list_screenshots(thread_id="thread/a")

        self.assertEqual(
            [item["filename"] for item in listed],
            [matching["filename"]],
        )

    def test_missing_screenshot_returns_404(self):
        response = self.client.get(
            "/api/screenshots/missing.png"
        )

        self.assertEqual(
            response.status_code,
            404,
        )

    def test_path_traversal_is_rejected(self):
        response = self.client.get(
            "/api/screenshots/../secret.png"
        )

        self.assertIn(
            response.status_code,
            {404, 307},
        )

    def test_cleanup_removes_expired_files(self):
        saved = main._screenshot_store().save_screenshot(
            PNG,
            action="old",
        )

        old_timestamp = 1
        os.utime(
            saved["path"],
            (old_timestamp, old_timestamp),
        )

        with patch.object(
            main.settings,
            "screenshot_retention_hours",
            1,
        ):
            response = self.client.post(
                "/api/screenshots/cleanup"
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["deleted_count"],
            1,
        )
        self.assertFalse(
            Path(f"{saved['path']}.json").exists(),
        )


if __name__ == "__main__":
    unittest.main()
