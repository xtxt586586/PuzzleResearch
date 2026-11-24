import unittest

from tools.registry import (
    ToolDefinition,
    ToolRegistry,
    get_tool_usage_stats,
    reset_tool_usage_stats,
)


def add_numbers(
    left: int,
    right: int,
) -> int:
    return left + right


class ToolRegistryTests(unittest.TestCase):
    def test_invocations_record_usage_statistics(self):
        reset_tool_usage_stats()
        self.addCleanup(reset_tool_usage_stats)

        def fail() -> None:
            raise RuntimeError("tool failure")

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="add_numbers",
                    description="计算两个整数的和",
                    handler=add_numbers,
                ),
                ToolDefinition(
                    name="failing_tool",
                    description="用于验证失败统计",
                    handler=fail,
                ),
            ]
        )

        self.assertEqual(
            registry.invoke(
                "add_numbers",
                {"left": 2, "right": 3},
            ),
            5,
        )
        with self.assertRaisesRegex(RuntimeError, "tool failure"):
            registry.invoke("failing_tool")

        stats = registry.get_statistics()
        self.assertEqual(stats["total_calls"], 2)
        self.assertEqual(stats["total_successes"], 1)
        self.assertEqual(stats["overall_success_rate"], 0.5)
        self.assertEqual(
            get_tool_usage_stats("failing_tool")["failure_count"],
            1,
        )
        metadata = registry.get("add_numbers").metadata()
        self.assertEqual(metadata["call_count"], 1)
        self.assertEqual(metadata["success_count"], 1)
        self.assertEqual(metadata["failure_count"], 0)
        self.assertGreaterEqual(metadata["average_duration_ms"], 0.0)
        self.assertTrue(metadata["last_called"])
        self.assertEqual(metadata["return_type"], "int")

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


    def test_wrong_argument_type_is_rejected(self):
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

        with self.assertRaisesRegex(
            ValueError,
            "类型无效",
        ):
            registry.invoke(
                "add",
                {
                    "left": "1",
                    "right": 2,
                },
            )

        self.assertEqual(
            attempts["count"],
            0,
        )



    def test_tool_argument_contract_is_described(self):
        def search(
                query: str,
                limit: int = 5,
        ) -> str:
            return f"{query}:{limit}"

        tool = ToolDefinition(
            name="search",
            description="搜索内容",
            handler=search,
        )

        self.assertEqual(
            tool.format_argument_contract(),
            "query (必填), limit (可选，默认=5)",
        )


    def test_no_argument_tool_has_empty_contract(self):
        def ping() -> str:
            return "pong"

        tool = ToolDefinition(
            name="ping",
            description="健康检查",
            handler=ping,
        )

        self.assertEqual(
            tool.format_argument_contract(),
            "无参数",
        )

    def test_argument_schema_describes_required_and_optional(self):
        def search(
                query: str,
                limit: int = 5,
        ) -> str:
            return f"{query}:{limit}"

        tool = ToolDefinition(
            name="search",
            description="搜索内容",
            handler=search,
        )

        self.assertEqual(
            tool.argument_schema(),
            [
                {
                    "name": "query",
                    "required": True,
                    "annotation": "str",
                },
                {
                    "name": "limit",
                    "required": False,
                    "annotation": "int",
                    "default": 5,
                },
            ],
        )


    def test_registry_describes_tools(self):
        def ping() -> str:
            return "pong"

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="ping",
                    description="健康检查",
                    handler=ping,
                ),
            ]
        )

        result = registry.describe_tools()

        self.assertEqual(
            result,
            [
                {
                    "name": "ping",
                    "description": "健康检查",
                    "parameters": [],
                },
            ],
        )


    def test_argument_schema_ignores_variadic_parameters(self):
        def variadic(
                required: str,
                *values,
                **options,
        ) -> str:
            return required

        tool = ToolDefinition(
            name="variadic",
            description="可变参数工具",
            handler=variadic,
        )

        self.assertEqual(
            tool.argument_schema(),
            [
                {
                    "name": "required",
                    "required": True,
                    "annotation": "str",
                },
            ],
        )

    def test_argument_schema_preserves_missing_annotation(self):
        def unannotated(value):
            return value

        tool = ToolDefinition(
            name="unannotated",
            description="无类型标注工具",
            handler=unannotated,
        )

        self.assertEqual(
            tool.argument_schema(),
            [
                {
                    "name": "value",
                    "required": True,
                    "annotation": None,
                },
            ],
        )

    def test_argument_schema_preserves_none_default(self):
        def optional(value: str | None = None):
            return value

        tool = ToolDefinition(
            name="optional",
            description="可选参数工具",
            handler=optional,
        )

        self.assertEqual(
            tool.argument_schema(),
            [
                {
                    "name": "value",
                    "required": False,
                    "annotation": "str | None",
                    "default": None,
                },
            ],
        )


    def test_integer_is_accepted_for_float_argument(self):
        def measure(
                timeout_seconds: float,
        ) -> float:
            return timeout_seconds

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="measure",
                    description="测试浮点参数",
                    handler=measure,
                ),
            ]
        )

        result = registry.invoke(
            "measure",
            {
                "timeout_seconds": 2,
            },
        )

        self.assertEqual(
            result,
            2,
        )


    def test_union_argument_accepts_allowed_types(self):
        def normalize(
                value: str | None,
        ) -> str:
            return value or "empty"

        registry = ToolRegistry(
            [
                ToolDefinition(
                    name="normalize",
                    description="规范化文本",
                    handler=normalize,
                ),
            ]
        )

        self.assertEqual(
            registry.invoke(
                "normalize",
                {
                    "value": "hello",
                },
            ),
            "hello",
        )

        self.assertEqual(
            registry.invoke(
                "normalize",
                {
                    "value": None,
                },
            ),
            "empty",
        )

        with self.assertRaisesRegex(
            ValueError,
            "类型无效",
        ):
            registry.invoke(
                "normalize",
                {
                    "value": 42,
                },
            )

        def test_bool_is_not_accepted_as_int(self):
            def repeat(
                    count: int,
            ) -> int:
                return count

            registry = ToolRegistry(
                [
                    ToolDefinition(
                        name="repeat",
                        description="测试整数参数",
                        handler=repeat,
                    ),
                ]
            )

            with self.assertRaisesRegex(
                    ValueError,
                    "类型无效",
            ):
                registry.invoke(
                    "repeat",
                    {
                        "count": True,
                    },
                )


if __name__ == "__main__":
    unittest.main()
