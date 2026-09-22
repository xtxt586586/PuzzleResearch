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
            action="test",
            thread_id="thread-001",
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


if __name__ == "__main__":
    unittest.main()