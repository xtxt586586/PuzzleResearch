from __future__ import annotations

import base64
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.workflows.nodes import agent_tool_node
from tools.builtin import create_default_registry
from tools.registry import ToolRegistry
from tools.workspace_files import (
    MAX_BINARY_DOWNLOAD_CHUNK_BYTES,
    MAX_FILE_BYTES,
    SessionWorkspace,
    WorkspaceFileConflictError,
    WorkspaceFileError,
    WorkspaceFileNotFoundError,
    build_workspace_tool_definitions,
)


class WorkspaceFileTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_write_read_replace_and_list_files(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        created = workspace.write_file("notes/research.txt", "旧结论\n")
        self.assertTrue(created["success"])

        with self.assertRaisesRegex(WorkspaceFileError, "已存在"):
            workspace.write_file("notes/research.txt", "覆盖")

        replaced = workspace.str_replace(
            "notes/research.txt",
            "旧结论",
            "新结论",
        )
        self.assertEqual(replaced["replacements"], 1)
        self.assertEqual(
            workspace.read_file("notes/research.txt")["content"],
            "新结论\n",
        )
        self.assertEqual(
            workspace.list_files()["files"],
            ["notes/research.txt"],
        )

    def test_update_rewrites_existing_file_but_does_not_create_missing_file(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        workspace.write_file("notes/research.txt", "初稿")

        result = workspace.update_file("notes/research.txt", "完整替换后的报告")

        self.assertTrue(result["success"])
        self.assertEqual(result["bytes"], len("完整替换后的报告".encode("utf-8")))
        self.assertEqual(
            workspace.read_file("notes/research.txt")["content"],
            "完整替换后的报告",
        )
        with self.assertRaises(WorkspaceFileNotFoundError):
            workspace.update_file("notes/missing.txt", "不会创建")

    def test_workspaces_are_isolated_per_thread(self):
        first = SessionWorkspace("thread-one", storage_root=self.root)
        second = SessionWorkspace("thread-two", storage_root=self.root)
        first.write_file("private.txt", "first session")

        with self.assertRaisesRegex(WorkspaceFileError, "不存在"):
            second.read_file("private.txt")

    def test_list_files_supports_nonrecursive_and_depth_limited_modes(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        workspace.write_file("root.txt", "root")
        workspace.write_file("notes/direct.txt", "direct")
        workspace.write_file("notes/archive/deep.txt", "deep")
        workspace.write_file("notes/archive/old/older.txt", "older")

        top_level = workspace.list_files(recursive=False)
        self.assertEqual(top_level["files"], ["root.txt"])
        self.assertEqual(
            top_level["directories"],
            [{"path": "notes", "type": "directory"}],
        )

        one_level = workspace.list_files(recursive=True, max_depth=1)
        self.assertEqual(
            one_level["files"],
            ["notes/direct.txt", "root.txt"],
        )
        self.assertEqual(
            [entry["path"] for entry in one_level["directories"]],
            ["notes", "notes/archive"],
        )
        self.assertEqual(one_level["total_files"], 2)
        self.assertEqual(one_level["total_dirs"], 2)
        self.assertEqual(
            one_level["file_details"][0],
            {"path": "notes/direct.txt", "type": "file", "size": 6},
        )

    def test_list_files_validates_recursion_options(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        with self.assertRaisesRegex(WorkspaceFileError, "recursive"):
            workspace.list_files(recursive=1)
        with self.assertRaisesRegex(WorkspaceFileError, "max_depth"):
            workspace.list_files(max_depth=21)

    def test_paths_cannot_escape_or_target_absolute_files(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        outside = self.root / "outside.txt"
        outside.write_text("protected", encoding="utf-8")

        for path in (
            "../outside.txt",
            "nested/../../outside.txt",
            "C:/outside.txt",
            str(outside),
            "nested/./file.txt",
        ):
            with self.subTest(path=path):
                with self.assertRaises(WorkspaceFileError):
                    workspace.write_file(path, "not allowed")

        self.assertEqual(outside.read_text(encoding="utf-8"), "protected")

    def test_file_size_and_read_limits_are_enforced(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        with self.assertRaisesRegex(WorkspaceFileError, "不能超过"):
            workspace.write_file("large.txt", "x" * (MAX_FILE_BYTES + 1))

        workspace.write_file("small.txt", "abcdef")
        result = workspace.read_file("small.txt", max_chars=3)
        self.assertEqual(result["content"], "abc")
        self.assertTrue(result["truncated"])

    def test_upload_accepts_utf8_text_without_overwriting(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        result = workspace.upload_text_file("研究记录.md", "结论".encode("utf-8"))

        self.assertEqual(result["path"], "研究记录.md")
        self.assertEqual(
            workspace.read_file("研究记录.md")["content"],
            "结论",
        )
        with self.assertRaises(WorkspaceFileConflictError):
            workspace.upload_text_file("研究记录.md", b"replacement")
        self.assertEqual(
            workspace.read_file("研究记录.md")["content"],
            "结论",
        )

    def test_upload_rejects_paths_binary_and_oversized_files(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        invalid_uploads = (
            ("../outside.txt", b"text"),
            ("folder/nested.txt", b"text"),
            ("binary.txt", b"\xff\xfe"),
            ("nul.txt", b"a\x00b"),
            ("large.txt", b"x" * (MAX_FILE_BYTES + 1)),
        )

        for filename, content in invalid_uploads:
            with self.subTest(filename=filename):
                with self.assertRaises(WorkspaceFileError):
                    workspace.upload_text_file(filename, content)

    def test_binary_upload_and_download_support_exact_base64_chunks(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        content = bytes(range(256)) * 60
        uploaded = workspace.upload_binary_file(
            "assets/sample.bin",
            base64.b64encode(content).decode("ascii"),
        )

        self.assertEqual(uploaded["size"], len(content))
        offset = 0
        chunks = []
        while True:
            result = workspace.download_file_chunk(
                "assets/sample.bin",
                offset=offset,
                max_bytes=MAX_BINARY_DOWNLOAD_CHUNK_BYTES,
            )
            chunk = base64.b64decode(result["content_base64"], validate=True)
            self.assertLessEqual(len(chunk), MAX_BINARY_DOWNLOAD_CHUNK_BYTES)
            self.assertEqual(result["offset"], offset)
            chunks.append(chunk)
            offset = result["next_offset"]
            if result["eof"]:
                break

        self.assertEqual(b"".join(chunks), content)

    def test_binary_upload_rejects_invalid_base64_oversize_and_overwrite(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        with self.assertRaisesRegex(WorkspaceFileError, "Base64"):
            workspace.upload_binary_file("broken.bin", "not base64!")
        with self.assertRaisesRegex(WorkspaceFileError, "不能超过"):
            workspace.upload_binary_file(
                "large.bin",
                base64.b64encode(b"x" * (MAX_FILE_BYTES + 1)).decode("ascii"),
            )

        workspace.upload_binary_file("same.bin", "AAE=")
        with self.assertRaises(WorkspaceFileConflictError):
            workspace.upload_binary_file("same.bin", "AgM=")
        workspace.upload_binary_file("same.bin", "AgM=", overwrite=True)
        self.assertEqual(workspace.download_file("same.bin")["content"], b"\x02\x03")

    def test_binary_download_validates_ranges_and_missing_files(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        workspace.upload_binary_file("small.bin", "AAE=")

        with self.assertRaisesRegex(WorkspaceFileError, "offset"):
            workspace.download_file_chunk("small.bin", offset=3)
        with self.assertRaisesRegex(WorkspaceFileError, "max_bytes"):
            workspace.download_file_chunk(
                "small.bin",
                max_bytes=MAX_BINARY_DOWNLOAD_CHUNK_BYTES + 1,
            )
        with self.assertRaises(WorkspaceFileNotFoundError):
            workspace.download_file_chunk("missing.bin")

    def test_delete_removes_only_files_inside_the_workspace(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        workspace.write_file("notes/research.txt", "content")

        result = workspace.delete_file("notes/research.txt")

        self.assertTrue(result["success"])
        self.assertEqual(result["path"], "notes/research.txt")
        self.assertFalse((workspace.root / "notes" / "research.txt").exists())
        with self.assertRaises(WorkspaceFileNotFoundError):
            workspace.delete_file("notes/research.txt")
        with self.assertRaises(WorkspaceFileError):
            workspace.delete_file("notes")
        with self.assertRaises(WorkspaceFileError):
            workspace.delete_file("../outside.txt")

    def test_empty_search_is_rejected(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        workspace.write_file("notes.txt", "content")
        with self.assertRaisesRegex(WorkspaceFileError, "非空"):
            workspace.str_replace("notes.txt", "", "replacement")

    def test_str_replace_requires_exactly_one_match(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        original = "first match\nsecond match\n"
        workspace.write_file("notes.txt", original)

        with self.assertRaisesRegex(WorkspaceFileError, "不存在"):
            workspace.str_replace("notes.txt", "missing", "replacement")
        with self.assertRaisesRegex(WorkspaceFileError, "出现 2 次"):
            workspace.str_replace("notes.txt", "match", "replacement")

        self.assertEqual(workspace.read_file("notes.txt")["content"], original)

    def test_str_replace_preserves_requested_encoding(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        workspace.root.mkdir(parents=True, exist_ok=True)
        target = workspace.root / "legacy.txt"
        target.write_bytes("café".encode("cp1252"))

        workspace.str_replace("legacy.txt", "café", "résumé", "cp1252")

        self.assertEqual(target.read_bytes(), "résumé".encode("cp1252"))

    def test_symlink_cannot_read_outside_workspace(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        workspace.root.mkdir(parents=True, exist_ok=True)
        outside = self.root / "secret.txt"
        outside.write_text("secret", encoding="utf-8")
        link = workspace.root / "outside-link.txt"
        try:
            link.symlink_to(outside)
        except (OSError, NotImplementedError):
            self.skipTest("当前 Windows 环境不允许创建符号链接。")

        with self.assertRaisesRegex(WorkspaceFileError, "符号链接"):
            workspace.read_file("outside-link.txt")

    def test_default_registry_exposes_workspace_tools(self):
        names = {tool.name for tool in create_default_registry().list_tools()}
        self.assertTrue(
            {
                "write_file",
                "update_file",
                "read_file",
                "list_files",
                "str_replace",
                "delete_file",
                "upload_file",
                "download_file",
            }
            <= names
        )

    def test_workspace_tool_definitions_are_invokable(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        with patch(
            "tools.workspace_files.get_session_workspace",
            return_value=workspace,
        ) as workspace_factory:
            registry = ToolRegistry(
                build_workspace_tool_definitions("thread-one")
            )
        workspace_factory.assert_called_once_with("thread-one")

        result = registry.invoke(
            "write_file",
            {"path": "answer.md", "content": "draft"},
        )
        self.assertIn("answer.md", result["output"])
        updated = registry.invoke(
            "update_file",
            {"path": "answer.md", "content": "revised draft"},
        )
        self.assertTrue(updated["success"])
        self.assertEqual(
            registry.invoke("read_file", {"path": "answer.md"})["content"],
            "revised draft",
        )
        deleted = registry.invoke("delete_file", {"path": "answer.md"})
        self.assertTrue(deleted["success"])
        with self.assertRaises(WorkspaceFileNotFoundError):
            workspace.read_file("answer.md")

        uploaded_binary = registry.invoke(
            "upload_file",
            {"path": "data/sample.bin", "content_base64": "AAEC/xA="},
        )
        downloaded_binary = registry.invoke(
            "download_file",
            {"path": "data/sample.bin", "offset": 0, "max_bytes": 5},
        )
        self.assertEqual(uploaded_binary["size"], 5)
        self.assertEqual(downloaded_binary["content_base64"], "AAEC/xA=")
        self.assertTrue(downloaded_binary["eof"])

    def test_agent_tool_node_can_write_and_replace_workspace_files(self):
        workspace = SessionWorkspace("thread-one", storage_root=self.root)
        with patch(
            "tools.workspace_files.get_session_workspace",
            return_value=workspace,
        ):
            registry = create_default_registry(thread_id="thread-one")

        written = agent_tool_node(
            {
                "pending_tool_calls": [{
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "report.md",
                        "content": "Draft result",
                    },
                }],
                "tool_approved": True,
            },
            config={"configurable": {"tool_registry": registry}},
        )
        self.assertIn("report.md", written["final_report"])

        replaced = agent_tool_node(
            {
                "pending_tool_calls": [{
                    "tool_name": "str_replace",
                    "arguments": {
                        "path": "report.md",
                        "search": "Draft",
                        "replace": "Final",
                    },
                }],
                "tool_approved": True,
            },
            config={"configurable": {"tool_registry": registry}},
        )
        self.assertIn("替换 1 处", replaced["final_report"])
        self.assertEqual(
            workspace.read_file("report.md")["content"],
            "Final result",
        )


if __name__ == "__main__":
    unittest.main()
