import unittest

from tools.registry import ToolDefinition, ToolRegistry


def add_numbers(
    left: int,
    right: int,
) -> int:
    return left + right


class ToolRegistryTests(unittest.TestCase):
    def test_register_and_invoke_tool(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="add_numbers",
                    description="计算两个整数的和",
                    handler=add_numbers,
                )
            ]
        )

        result = registry.invoke(
            "add_numbers",
            {
                "left": 2,
                "right": 3,
            },
        )

        self.assertEqual(result, 5)

    def test_list_tools(self):
        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="add_numbers",
                    description="计算两个整数的和",
                    handler=add_numbers,
                )
            ]
        )

        tools = registry.list_tools()

        self.assertEqual(len(tools), 1)
        self.assertEqual(tools[0].name, "add_numbers")

    def test_duplicate_tool_name_is_rejected(self):
        tool = ToolDefinition(
            name="add_numbers",
            description="计算两个整数的和",
            handler=add_numbers,
        )

        registry = ToolRegistry([tool])

        with self.assertRaises(ValueError):
            registry.register(tool)

    def test_unknown_tool_is_rejected(self):
        registry = ToolRegistry()

        with self.assertRaises(KeyError):
            registry.invoke("missing_tool", {})


if __name__ == "__main__":
    unittest.main()