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


    def test_missing_required_argument_is_rejected(self):
        attempts = {
            "count": 0,
        }

        def add_numbers(left: int, right: int) -> int:
            attempts["count"] += 1
            return left + right

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="add",
                    description="计算两个数的和",
                    handler=add_numbers,
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "参数无效"):
            registry.invoke(
                "add",
                {
                    "left": 1,
                },
            )

        self.assertEqual(attempts["count"], 0)


    def test_unexpected_argument_is_rejected(self):
        attempts = {
            "count": 0,
        }

        def echo(text: str) -> str:
            attempts["count"] += 1
            return text

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="echo",
                    description="返回文本",
                    handler=echo,
                ),
            ]
        )

        with self.assertRaisesRegex(ValueError, "参数无效"):
            registry.invoke(
                "echo",
                {
                    "text": "hello",
                    "unused": True,
                },
            )

        self.assertEqual(attempts["count"], 0)


if __name__ == "__main__":
    unittest.main()