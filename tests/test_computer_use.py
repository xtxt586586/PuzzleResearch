import io
import unittest
from unittest.mock import MagicMock, patch

from PIL import Image

from tools.builtin import create_default_registry


class ComputerUseTests(unittest.TestCase):
    def setUp(self):
        self.gui = MagicMock()
        self.gui.size.return_value = (800, 600)
        self.gui.position.return_value = (100, 120)
        screenshot = Image.new("RGB", (16, 12), (20, 40, 60))
        self.gui.screenshot.return_value = screenshot
        self.saved = {
            "url": "/api/screenshots/computer.png",
            "filename": "computer.png",
            "mime_type": "image/png",
        }

    def _registry(self):
        return create_default_registry(
            thread_id="computer-use-test",
            include_browser=True,
        )

    def test_desktop_actions_use_pyautogui_and_save_screenshots(self):
        with (
            patch("tools.builtin.settings.enable_computer_use", True),
            patch("tools.computer_use._pyautogui", return_value=self.gui),
            patch("tools.computer_use.get_screenshot_store") as store,
        ):
            store.return_value.save_screenshot.return_value = self.saved
            registry = self._registry()
            moved = registry.invoke("computer_move_mouse", {"x": 900, "y": -10})
            clicked = registry.invoke(
                "computer_click",
                {"x": 10, "y": 20, "button": "left", "clicks": 2},
            )
            typed = registry.invoke(
                "computer_type",
                {"text": "PuzzleResearch", "interval": 0.01},
            )
            pressed = registry.invoke("computer_press", {"keys": "ctrl+c"})
            scrolled = registry.invoke(
                "computer_scroll",
                {"direction": "down", "amount": 3},
            )
            screen = registry.invoke("computer_screen_info", {})
            dragged = registry.invoke(
                "computer_drag",
                {
                    "start_x": 20,
                    "start_y": 30,
                    "end_x": 100,
                    "end_y": 120,
                    "duration": 0.1,
                },
            )
            captured = registry.invoke("computer_screenshot", {})

        self.assertEqual(moved["position"], {"x": 799, "y": 0})
        self.assertTrue(clicked["success"])
        self.assertEqual(typed["text_length"], len("PuzzleResearch"))
        self.assertEqual(pressed["keys"], "ctrl+c")
        self.assertEqual(scrolled["direction"], "down")
        self.assertEqual(screen["screen"], {"width": 800, "height": 600})
        self.assertEqual(dragged["end"], {"x": 100, "y": 120})
        self.assertEqual(captured["screenshot_url"], self.saved["url"])
        self.gui.moveTo.assert_any_call(799, 0, duration=0.2)
        self.gui.click.assert_called_once_with(10, 20, clicks=2, button="left")
        self.gui.write.assert_called_once_with("PuzzleResearch", interval=0.01)
        self.gui.hotkey.assert_called_once_with("ctrl", "c")
        self.gui.scroll.assert_called_once_with(-3)
        self.gui.dragTo.assert_called_once_with(100, 120, duration=0.1, button="left")

    def test_invalid_coordinates_and_actions_return_structured_errors(self):
        with (
            patch("tools.builtin.settings.enable_computer_use", True),
            patch("tools.computer_use._pyautogui", return_value=self.gui),
        ):
            registry = self._registry()
            invalid_button = registry.invoke(
                "computer_click",
                {"x": 10, "y": 20, "button": "invalid"},
            )
            invalid_direction = registry.invoke(
                "computer_scroll",
                {"direction": "sideways"},
            )

        self.assertFalse(invalid_button["success"])
        self.assertIn("button", invalid_button["error"])
        self.assertFalse(invalid_direction["success"])
        self.assertIn("direction", invalid_direction["error"])

    def test_tools_are_disabled_by_default(self):
        with patch("tools.builtin.settings.enable_computer_use", False):
            names = {
                tool.name
                for tool in create_default_registry(
                    thread_id="computer-use-test",
                    include_browser=True,
                ).list_tools()
            }

        self.assertNotIn("computer_click", names)


if __name__ == "__main__":
    unittest.main()
