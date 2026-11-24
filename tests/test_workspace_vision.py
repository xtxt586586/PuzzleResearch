from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageChops

from tools.builtin import create_default_registry
from tools.workspace_files import SessionWorkspace, WorkspaceFileError
from tools.workspace_vision import (
    WorkspaceVisionError,
    compare_images,
    convert_image,
    crop_image,
    get_image_info,
    resize_image,
)


class WorkspaceVisionTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.workspace = SessionWorkspace("vision-thread", storage_root=self.root)
        self.source = self.workspace._target("source.png")
        image = Image.new("RGB", (100, 60), (20, 120, 200))
        image.save(self.source)
        Image.new("RGB", (100, 60), (200, 40, 60)).save(
            self.workspace._target("different.png")
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_info_resize_convert_crop_and_compare(self):
        import unittest.mock

        with unittest.mock.patch(
            "tools.workspace_vision.get_session_workspace",
            return_value=self.workspace,
        ):
            info = get_image_info("vision-thread", "source.png")
            resized = resize_image(
                "vision-thread", "source.png", "derived/resized.png", width=50
            )
            converted = convert_image(
                "vision-thread", "source.png", "derived/source.jpg", quality=90
            )
            cropped = crop_image(
                "vision-thread", "source.png", "derived/cropped.png", 10, 10, 50, 30
            )
            same = compare_images("vision-thread", "source.png", "source.png")
            different = compare_images(
                "vision-thread", "source.png", "different.png"
            )

        self.assertEqual((info["width"], info["height"], info["format"]), (100, 60, "PNG"))
        self.assertEqual(resized["new_size"], [50, 30])
        self.assertEqual(converted["output_format"], "JPG")
        self.assertEqual(cropped["new_size"], [40, 20])
        self.assertTrue(same["identical"])
        self.assertEqual(same["similarity_percent"], 100.0)
        self.assertFalse(different["identical"])

        with Image.open(self.workspace._target("derived/resized.png")) as resized_image:
            self.assertEqual(resized_image.size, (50, 30))
        with Image.open(self.workspace._target("derived/source.jpg")) as converted_image:
            self.assertEqual(converted_image.format, "JPEG")

    def test_invalid_image_requests_are_rejected(self):
        import unittest.mock

        with unittest.mock.patch(
            "tools.workspace_vision.get_session_workspace",
            return_value=self.workspace,
        ):
            with self.assertRaises(WorkspaceVisionError):
                resize_image("vision-thread", "source.png", "bad.txt", width=20)
            with self.assertRaises(WorkspaceVisionError):
                resize_image("vision-thread", "source.png", "small.png")
            with self.assertRaises(WorkspaceVisionError):
                crop_image("vision-thread", "source.png", "bad.png", 0, 0, 101, 20)
            with self.assertRaises(WorkspaceFileError):
                get_image_info("vision-thread", "../outside.png")

    def test_default_registry_exposes_vision_tools(self):
        names = {tool.name for tool in create_default_registry().list_tools()}
        self.assertTrue(
            {
                "get_image_info",
                "resize_image",
                "convert_image",
                "crop_image",
                "compare_images",
            }
            <= names
        )


if __name__ == "__main__":
    unittest.main()
