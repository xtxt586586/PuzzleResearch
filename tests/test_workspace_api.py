from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from common.session_manager import SessionNotFoundError, SessionOwnerError
from tools.workspace_files import SessionWorkspace
from tools.workspace_files import MAX_FILE_BYTES


class FakeWorkspaceSessionManager:
    def __init__(self):
        self.owners = {"workspace-thread": "owner-001"}

    def get_session(self, thread_id: str, *, owner_id: str = "anonymous"):
        if thread_id not in self.owners:
            raise SessionNotFoundError(f"线程不存在：{thread_id}")
        if self.owners[thread_id] != owner_id:
            raise SessionOwnerError("当前用户无权访问该会话。")
        return {"thread_id": thread_id, "owner_id": owner_id}


class WorkspaceApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.workspace = SessionWorkspace(
            "workspace-thread",
            storage_root=Path(self.temp_dir.name),
        )
        self.manager_patch = patch.object(
            main,
            "session_manager",
            FakeWorkspaceSessionManager(),
        )
        self.workspace_patch = patch.object(
            main,
            "get_session_workspace",
            return_value=self.workspace,
        )
        self.api_key_patch = patch.object(main, "configured_api_key", "")
        self.manager_patch.start()
        self.workspace_patch.start()
        self.api_key_patch.start()
        self.client = TestClient(main.app)

    def tearDown(self):
        self.api_key_patch.stop()
        self.workspace_patch.stop()
        self.manager_patch.stop()
        self.temp_dir.cleanup()

    def test_list_files_requires_owner_and_returns_session_files(self):
        self.workspace.write_file("notes/research.md", "研究结论")

        response = self.client.get(
            "/api/sessions/workspace-thread/workspace/files",
            headers={"X-Weaver-User": "owner-001"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["count"], 1)
        self.assertEqual(response.json()["files"], ["notes/research.md"])

    def test_download_returns_file_bytes_and_attachment_name(self):
        self.workspace.write_file("reports/final.md", "最终报告内容")

        response = self.client.get(
            "/api/sessions/workspace-thread/workspace/files/reports/final.md",
            headers={"X-Weaver-User": "owner-001"},
        )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content.decode("utf-8"), "最终报告内容")
        self.assertIn(
            "filename*=UTF-8''final.md",
            response.headers["content-disposition"],
        )
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_workspace_files_are_not_visible_to_another_owner(self):
        response = self.client.get(
            "/api/sessions/workspace-thread/workspace/files",
            headers={"X-Weaver-User": "intruder-001"},
        )

        self.assertEqual(response.status_code, 403)

    def test_missing_file_returns_not_found(self):
        response = self.client.get(
            "/api/sessions/workspace-thread/workspace/files/missing.txt",
            headers={"X-Weaver-User": "owner-001"},
        )

        self.assertEqual(response.status_code, 404)

    def test_upload_creates_utf8_text_file_and_returns_it_in_list(self):
        response = self.client.post(
            "/api/sessions/workspace-thread/workspace/files/upload",
            headers={"X-Weaver-User": "owner-001"},
            files={"file": ("研究记录.md", "阶段结论".encode("utf-8"), "text/plain")},
        )

        self.assertEqual(response.status_code, 201)
        self.assertEqual(response.json()["path"], "研究记录.md")
        self.assertEqual(
            self.workspace.read_file("研究记录.md")["content"],
            "阶段结论",
        )
        listing = self.client.get(
            "/api/sessions/workspace-thread/workspace/files",
            headers={"X-Weaver-User": "owner-001"},
        )
        self.assertEqual(listing.json()["files"], ["研究记录.md"])

    def test_upload_rejects_duplicate_invalid_name_and_oversized_files(self):
        endpoint = "/api/sessions/workspace-thread/workspace/files/upload"
        headers = {"X-Weaver-User": "owner-001"}
        self.workspace.write_file("existing.txt", "original")

        duplicate = self.client.post(
            endpoint,
            headers=headers,
            files={"file": ("existing.txt", b"replacement", "text/plain")},
        )
        binary = self.client.post(
            endpoint,
            headers=headers,
            files={"file": ("image.png", b"\x89PNG\x00\xff", "image/png")},
        )
        invalid_name = self.client.post(
            endpoint,
            headers=headers,
            files={"file": ("../invalid.bin", b"binary", "application/octet-stream")},
        )
        oversized = self.client.post(
            endpoint,
            headers=headers,
            files={"file": ("large.txt", b"x" * (MAX_FILE_BYTES + 1), "text/plain")},
        )

        self.assertEqual(duplicate.status_code, 409)
        self.assertEqual(binary.status_code, 201)
        self.assertEqual(invalid_name.status_code, 400)
        self.assertEqual(oversized.status_code, 413)
        self.assertEqual(self.workspace.read_file("existing.txt")["content"], "original")
        self.assertEqual(self.workspace._read_bytes(self.workspace._target("image.png")), b"\x89PNG\x00\xff")

    def test_upload_requires_session_owner(self):
        response = self.client.post(
            "/api/sessions/workspace-thread/workspace/files/upload",
            headers={"X-Weaver-User": "intruder-001"},
            files={"file": ("private.txt", b"text", "text/plain")},
        )

        self.assertEqual(response.status_code, 403)

    def test_delete_file_requires_owner_and_removes_file(self):
        self.workspace.write_file("reports/final.md", "研究报告")

        forbidden = self.client.delete(
            "/api/sessions/workspace-thread/workspace/files/reports/final.md",
            headers={"X-Weaver-User": "intruder-001"},
        )
        deleted = self.client.delete(
            "/api/sessions/workspace-thread/workspace/files/reports/final.md",
            headers={"X-Weaver-User": "owner-001"},
        )

        self.assertEqual(forbidden.status_code, 403)
        self.assertEqual(deleted.status_code, 200)
        self.assertEqual(deleted.json()["path"], "reports/final.md")
        self.assertEqual(self.workspace.list_files()["files"], [])

    def test_delete_file_returns_not_found_for_missing_file(self):
        response = self.client.delete(
            "/api/sessions/workspace-thread/workspace/files/missing.txt",
            headers={"X-Weaver-User": "owner-001"},
        )

        self.assertEqual(response.status_code, 404)

    def test_missing_session_returns_not_found(self):
        response = self.client.get(
            "/api/sessions/missing-thread/workspace/files",
            headers={"X-Weaver-User": "owner-001"},
        )

        self.assertEqual(response.status_code, 404)


if __name__ == "__main__":
    unittest.main()
