import unittest
from unittest.mock import patch

from tools.builtin import (
    MAX_PYTHON_OUTPUT_CHARS,
    MAX_PYTHON_TIMEOUT_SECONDS,
    create_default_registry,
)


class BuiltinToolTests(unittest.TestCase):
    def test_default_registry_contains_python_executor(self):
        registry = create_default_registry()

        tools = registry.list_tools()

        self.assertEqual(
            {tool.name for tool in tools},
            {
                "python_executor",
                "create_tasks",
                "view_tasks",
                "update_task",
                "get_next_task",
                "plan_steps",
                "ask_human",
                "web_search",
                "fallback_search",
                "search_multiple_queries",
                "tavily_search",
                "crawl_url",
                "crawl_urls",
                "write_file",
                "update_file",
                "read_file",
                "list_files",
                "str_replace",
                "delete_file",
                "upload_file",
                "download_file",
                "execute_command",
                "check_output",
                "kill_process",
                "list_processes",
                "install_package",
                "expose_port",
                "scaffold_web_project",
                "deploy_web_project",
                "create_spreadsheet",
                "write_data",
                "read_data",
                "format_cells",
                "add_sheet",
                "create_chart",
                "create_presentation",
                "add_slide",
                "add_image_to_slide",
                "add_table_to_slide",
                "add_shape_to_slide",
                "get_presentation_info",
                "update_slide",
                "delete_slide",
                "generate_presentation_outline",
                "outline_to_presentation",
                "refine_presentation_outline",
                "expand_slide_content",
                "get_image_info",
                "resize_image",
                "convert_image",
                "crop_image",
                "compare_images",
                "apply_image_filter",
                "apply_image_effect",
                "adjust_image",
                "rotate_flip_image",
                "add_watermark",
                "overlay_images",
                "create_thumbnail",
            },
        )

    def test_ask_human_returns_prompt_and_optional_note(self):
        registry = create_default_registry()

        result = registry.invoke(
            "ask_human",
            {
                "prompt": "你希望报告侧重成本还是效率？",
                "note": "这会影响后续的比较维度。",
            },
        )

        self.assertEqual(
            result,
            "请求用户输入：你希望报告侧重成本还是效率？"
            "（这会影响后续的比较维度。）",
        )

    def test_ask_human_note_is_optional(self):
        registry = create_default_registry()

        result = registry.invoke(
            "ask_human",
            {"prompt": "请确认研究范围。"},
        )

        self.assertEqual(
            result,
            "请求用户输入：请确认研究范围。",
        )

    def test_python_executor_can_be_invoked_from_registry(self):
        registry = create_default_registry()

        result = registry.invoke(
            "python_executor",
            {
                "code": "print(6 * 7)",
                "timeout_seconds": 2,
            },
        )

        self.assertEqual(result["output"].strip(), "42")
        self.assertEqual(result["error"], "")
        self.assertEqual(result["return_code"], 0)
        self.assertFalse(result["timed_out"])

    def test_tool_parameters_are_capped(self):
        registry = create_default_registry()

        result = registry.invoke(
            "python_executor",
            {
                "code": "print('x' * 100)",
                "timeout_seconds": 100,
                "max_output_chars": 100_000,
            },
        )

        self.assertLessEqual(
            len(result["output"]),
            MAX_PYTHON_OUTPUT_CHARS + 20,
        )

        self.assertLessEqual(
            MAX_PYTHON_TIMEOUT_SECONDS,
            10.0,
        )

    def test_browser_tools_are_available_when_enabled(self):
        registry = create_default_registry(
            thread_id="thread-001",
            include_browser=True,
        )

        names = {
            tool.name
            for tool in registry.list_tools()
        }

        self.assertIn(
            "browser_search",
            names,
        )
        self.assertIn(
            "browser_navigate",
            names,
        )
        self.assertIn(
            "browser_reset",
            names,
        )

    def test_rag_search_is_available_only_when_enabled(self):
        with patch("tools.builtin.settings.rag_enabled", False):
            disabled_names = {
                tool.name for tool in create_default_registry().list_tools()
            }
        with patch("tools.builtin.settings.rag_enabled", True):
            enabled_names = {
                tool.name for tool in create_default_registry().list_tools()
            }

        self.assertNotIn("rag_search", disabled_names)
        self.assertIn("rag_search", enabled_names)


if __name__ == "__main__":
    unittest.main()
