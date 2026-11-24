import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from tools.builtin import create_default_registry
from tools.workspace_files import SessionWorkspace, WorkspaceFileError
from tools.workspace_image_edit import (
    WorkspaceImageEditError,
    add_watermark,
    adjust_image,
    apply_image_effect,
    apply_image_filter,
    create_thumbnail,
    overlay_images,
    rotate_flip_image,
)


class WorkspaceImageEditTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.workspace = SessionWorkspace("image-edit-thread", storage_root=self.root)

        source = Image.new("RGB", (100, 60), (20, 120, 200))
        for x in range(50, 100):
            for y in range(30, 60):
                source.putpixel((x, y), (220, 40, 70))
        source.save(self.workspace._target("source.png"))

        overlay = Image.new("RGBA", (20, 20), (255, 255, 255, 180))
        overlay.save(self.workspace._target("overlay.png"))

    def tearDown(self):
        self.temp_dir.cleanup()

    def _run(self, operation, *args, **kwargs):
        with patch(
            "tools.workspace_image_edit.get_session_workspace",
            return_value=self.workspace,
        ):
            return operation("image-edit-thread", *args, **kwargs)

    def test_filters_effects_adjustments_rotate_and_thumbnail(self):
        filtered = self._run(
            apply_image_filter,
            "source.png",
            "output/sharpened.png",
            "sharpen",
        )
        effected = self._run(
            apply_image_effect,
            "source.png",
            "output/grayscale.png",
            "grayscale",
        )
        adjusted = self._run(
            adjust_image,
            "source.png",
            "output/adjusted.png",
            brightness=1.2,
            contrast=1.1,
        )
        rotated = self._run(
            rotate_flip_image,
            "source.png",
            "output/rotated.png",
            rotate_degrees=90,
        )
        thumbnail = self._run(
            create_thumbnail,
            "source.png",
            "output/thumbnail.png",
            max_size=25,
        )

        self.assertEqual(filtered["filter"], "sharpen")
        self.assertEqual(effected["effect"], "grayscale")
        self.assertEqual(adjusted["adjustments"]["brightness"], 1.2)
        self.assertEqual(rotated["new_size"], [60, 100])
        self.assertEqual(thumbnail["thumbnail_size"], [25, 15])

        with Image.open(self.workspace._target("output/grayscale.png")) as image:
            self.assertEqual(image.mode, "L")
        with Image.open(self.workspace._target("output/thumbnail.png")) as image:
            self.assertEqual(image.size, (25, 15))

    def test_watermark_and_overlay_create_expected_outputs(self):
        watermarked = self._run(
            add_watermark,
            "source.png",
            "output/watermarked.png",
            "PuzzleResearch",
            position="bottom-right",
            opacity=0.75,
            font_size=18,
        )
        overlaid = self._run(
            overlay_images,
            "source.png",
            "overlay.png",
            "output/overlaid.png",
            x=10,
            y=8,
            opacity=0.5,
            resize_overlay=[30, 20],
        )

        self.assertEqual(watermarked["watermark_position"], "bottom-right")
        self.assertEqual(overlaid["new_size"], [100, 60])
        self.assertTrue(
            self.workspace._target("output/watermarked.png").is_file()
        )
        self.assertTrue(self.workspace._target("output/overlaid.png").is_file())

    def test_invalid_parameters_and_paths_are_rejected(self):
        with self.assertRaises(WorkspaceImageEditError):
            self._run(
                apply_image_filter,
                "source.png",
                "output/bad.png",
                "unknown",
            )
        with self.assertRaises(WorkspaceImageEditError):
            self._run(
                add_watermark,
                "source.png",
                "output/bad.png",
                "text",
                opacity=2.0,
            )
        with self.assertRaises(WorkspaceImageEditError):
            self._run(
                rotate_flip_image,
                "source.png",
                "output/bad.png",
                rotate_degrees=90.0,
            )
        with self.assertRaises(WorkspaceFileError):
            self._run(
                create_thumbnail,
                "source.png",
                "output/bad.txt",
            )
        with self.assertRaises(WorkspaceFileError):
            self._run(
                apply_image_effect,
                "../source.png",
                "output/bad.png",
                "grayscale",
            )

    def test_default_registry_exposes_image_edit_tools(self):
        names = {tool.name for tool in create_default_registry().list_tools()}
        self.assertTrue(
            {
                "apply_image_filter",
                "apply_image_effect",
                "adjust_image",
                "rotate_flip_image",
                "add_watermark",
                "overlay_images",
                "create_thumbnail",
            }
            <= names
        )


if __name__ == "__main__":
    unittest.main()
