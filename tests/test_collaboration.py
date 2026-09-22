import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from common import collaboration


class CollaborationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

        self.env_patch = patch.dict(
            os.environ,
            {
                "PUZZLERESEARCH_DATA_DIR": (
                    self.temp_dir.name
                )
            },
            clear=False,
        )
        self.env_patch.start()

    def tearDown(self):
        self.env_patch.stop()
        self.temp_dir.cleanup()

    def test_share_link_can_be_created_and_viewed(self):
        link = collaboration.create_share_link(
            "thread-share-001",
            "owner-001",
        )

        self.assertTrue(link["id"])
        self.assertEqual(
            link["thread_id"],
            "thread-share-001",
        )
        self.assertEqual(
            link["owner_id"],
            "owner-001",
        )
        self.assertEqual(
            link["permissions"],
            "view",
        )
        self.assertEqual(
            link["view_count"],
            0,
        )

        first_view = collaboration.get_share_link(
            link["id"]
        )
        second_view = collaboration.get_share_link(
            link["id"]
        )

        self.assertEqual(
            first_view["view_count"],
            1,
        )
        self.assertEqual(
            second_view["view_count"],
            2,
        )

    def test_expired_share_link_is_unavailable(self):
        created_at = datetime(
            2026,
            1,
            1,
            tzinfo=timezone.utc,
        )

        with patch.object(
            collaboration,
            "_now",
            return_value=created_at,
        ):
            link = collaboration.create_share_link(
                "thread-share-002",
                "owner-002",
                expires_hours=1,
            )

        with patch.object(
            collaboration,
            "_now",
            return_value=created_at + timedelta(hours=2),
        ):
            result = collaboration.get_share_link(
                link["id"]
            )

        self.assertIsNone(result)

    def test_share_link_requires_owner_to_delete(self):
        link = collaboration.create_share_link(
            "thread-share-003",
            "owner-003",
        )

        self.assertFalse(
            collaboration.delete_share_link(
                link["id"],
                "wrong-owner",
            )
        )

        self.assertTrue(
            collaboration.delete_share_link(
                link["id"],
                "owner-003",
            )
        )

        self.assertIsNone(
            collaboration.get_share_link(
                link["id"]
            )
        )

    def test_comments_can_be_filtered_by_message(self):
        first = collaboration.add_comment(
            "thread-comment-001",
            "第一条评论",
            "user-001",
            "message-001",
        )
        collaboration.add_comment(
            "thread-comment-001",
            "第二条评论",
            "user-002",
            "message-002",
        )

        self.assertEqual(
            first["author"],
            "user-001",
        )

        all_comments = collaboration.get_comments(
            "thread-comment-001"
        )
        self.assertEqual(
            len(all_comments),
            2,
        )

        filtered = collaboration.get_comments(
            "thread-comment-001",
            "message-001",
        )

        self.assertEqual(
            len(filtered),
            1,
        )
        self.assertEqual(
            filtered[0]["content"],
            "第一条评论",
        )

    def test_versions_are_saved_and_owner_scoped(self):
        snapshot = {
            "thread_id": "thread-version-001",
            "state": {
                "status": "completed",
                "final_report": "测试报告",
            },
        }

        version = collaboration.save_version(
            "thread-version-001",
            "owner-001",
            snapshot,
            "初始版本",
        )

        self.assertEqual(
            version["version_number"],
            1,
        )
        self.assertEqual(
            version["label"],
            "初始版本",
        )

        versions = collaboration.list_versions(
            "thread-version-001",
            "owner-001",
        )

        self.assertEqual(
            len(versions),
            1,
        )
        self.assertEqual(
            versions[0]["id"],
            version["id"],
        )

        self.assertEqual(
            collaboration.list_versions(
                "thread-version-001",
                "wrong-owner",
            ),
            [],
        )

        restored = collaboration.get_version_snapshot(
            "thread-version-001",
            version["id"],
            "owner-001",
        )

        self.assertEqual(
            restored,
            snapshot,
        )

        self.assertIsNone(
            collaboration.get_version_snapshot(
                "thread-version-001",
                version["id"],
                "wrong-owner",
            )
        )

    def test_share_parameters_are_validated(self):
        with self.assertRaises(ValueError):
            collaboration.create_share_link(
                "thread-invalid-001",
                "owner-001",
                permissions="edit",
            )

        with self.assertRaises(ValueError):
            collaboration.create_share_link(
                "thread-invalid-002",
                "owner-001",
                expires_hours=0,
            )


if __name__ == "__main__":
    unittest.main()