import unittest

from tools.builtin import (
    MAX_PYTHON_OUTPUT_CHARS,
    MAX_PYTHON_TIMEOUT_SECONDS,
    create_default_registry,
)


class BuiltinToolTests(unittest.TestCase):
    def test_default_registry_contains_python_executor(self):
        registry = create_default_registry()

        tools = registry.list_tools()

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "python_executor")

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


if __name__ == "__main__":
    unittest.main()