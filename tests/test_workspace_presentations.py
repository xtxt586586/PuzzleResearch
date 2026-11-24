from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from pptx import Presentation

from tools.builtin import create_default_registry
from tools.registry import ToolRegistry
from tools.workspace_files import SessionWorkspace, WorkspaceFileError
from tools.workspace_presentations import (
    WorkspacePresentationError,
    add_image_to_slide,
    add_shape_to_slide,
    add_slide,
    add_table_to_slide,
    build_workspace_presentation_tool_definitions,
    create_presentation,
    delete_slide,
    get_presentation_info,
    update_slide,
)


class WorkspacePresentationTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.workspace = SessionWorkspace("presentation-thread", storage_root=self.root)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _patch_workspace(self):
        return patch(
            "tools.workspace_presentations.get_session_workspace",
            return_value=self.workspace,
        )

    def test_create_and_edit_presentation(self):
        with self._patch_workspace():
            created = create_presentation(
                "presentation-thread",
                "reports/research.pptx",
                "研究项目汇报",
                "PuzzleResearch",
            )
            self.assertEqual(created["slide_count"], 1)
            added = add_slide(
                "presentation-thread",
                "reports/research.pptx",
                title="研究结论",
                content="第一条结论\n第二条结论",
                notes="说明研究结论和证据来源。",
            )
            self.assertEqual(added["slide_number"], 2)
            updated = update_slide(
                "presentation-thread",
                "reports/research.pptx",
                2,
                title="更新后的研究结论",
                content="更新后的内容",
            )
            self.assertTrue(updated["success"])
            info = get_presentation_info(
                "presentation-thread",
                "reports/research.pptx",
            )
            self.assertEqual(info["slide_count"], 2)
            self.assertEqual(info["slides"][1]["title"], "更新后的研究结论")

        target = next(self.root.rglob("research.pptx"))
        presentation = Presentation(str(target))
        self.assertEqual(len(presentation.slides), 2)
        self.assertEqual(presentation.slides[0].shapes.title.text, "研究项目汇报")

    def test_table_shape_image_and_delete(self):
        with self._patch_workspace():
            create_presentation(
                "presentation-thread",
                "visuals.pptx",
                "研究可视化",
            )
            image_path = self.workspace._target("source.png")
            Image.new("RGB", (40, 30), (20, 120, 200)).save(image_path)
            table = add_table_to_slide(
                "presentation-thread",
                "visuals.pptx",
                1,
                [["指标", "数值"], ["来源", "12"]],
            )
            shape = add_shape_to_slide(
                "presentation-thread",
                "visuals.pptx",
                1,
                "rounded_rectangle",
                1,
                1,
                2,
                1,
                text="可信度 0.92",
                fill_color="#D9EAF7",
            )
            image = add_image_to_slide(
                "presentation-thread",
                "visuals.pptx",
                1,
                "source.png",
                left=5,
                top=1,
                width=2,
            )
            self.assertTrue(table["success"])
            self.assertTrue(shape["success"])
            self.assertTrue(image["success"])
            deleted = delete_slide("presentation-thread", "visuals.pptx", 1)
            self.assertEqual(deleted["slide_count"], 0)

    def test_registry_exposes_presentation_tools(self):
        names = {tool.name for tool in create_default_registry().list_tools()}
        self.assertTrue(
            {
                "create_presentation",
                "add_slide",
                "add_image_to_slide",
                "add_table_to_slide",
                "add_shape_to_slide",
                "get_presentation_info",
                "update_slide",
                "delete_slide",
            }
            <= names
        )
        with self._patch_workspace():
            registry = ToolRegistry(
                build_workspace_presentation_tool_definitions("presentation-thread")
            )
            result = registry.invoke(
                "create_presentation",
                {"file_path": "registry.pptx", "title": "工具测试"},
            )
        self.assertTrue(result["success"])

    def test_invalid_paths_and_slide_numbers_are_rejected(self):
        with self._patch_workspace():
            with self.assertRaises(WorkspaceFileError):
                create_presentation("presentation-thread", "../outside.pptx")
            with self.assertRaises(WorkspacePresentationError):
                create_presentation("presentation-thread", "notes.pdf")
            create_presentation("presentation-thread", "empty.pptx")
            with self.assertRaises(WorkspacePresentationError):
                add_slide("presentation-thread", "empty.pptx", layout="unsupported")
            with self.assertRaises(WorkspacePresentationError):
                update_slide("presentation-thread", "empty.pptx", 1, title="不会存在")


if __name__ == "__main__":
    unittest.main()
