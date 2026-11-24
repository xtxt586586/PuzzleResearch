from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable,Mapping,Union,get_args,get_origin,get_type_hints
from datetime import datetime, timezone
import inspect
import threading
import time
import types


ToolHandler = Callable[..., Any]

_TOOL_USAGE: dict[str, dict[str, Any]] = {}
_TOOL_USAGE_LOCK = threading.Lock()


def get_tool_usage_stats(name: str) -> dict[str, Any]:
    with _TOOL_USAGE_LOCK:
        stats = dict(_TOOL_USAGE.get(name, {}))

    calls = int(stats.get("call_count", 0))
    stats.setdefault("call_count", 0)
    stats.setdefault("success_count", 0)
    stats.setdefault("failure_count", 0)
    stats.setdefault("last_called", None)
    stats.setdefault("average_duration_ms", 0.0)
    stats["success_rate"] = (
        stats["success_count"] / calls
        if calls
        else 0.0
    )
    return stats


def reset_tool_usage_stats() -> None:
    with _TOOL_USAGE_LOCK:
        _TOOL_USAGE.clear()


def _record_tool_usage(
    name: str,
    *,
    succeeded: bool,
    duration_ms: float,
) -> None:
    with _TOOL_USAGE_LOCK:
        stats = _TOOL_USAGE.setdefault(
            name,
            {
                "call_count": 0,
                "success_count": 0,
                "failure_count": 0,
                "last_called": None,
                "average_duration_ms": 0.0,
            },
        )
        calls = stats["call_count"] + 1
        stats["call_count"] = calls
        if succeeded:
            stats["success_count"] += 1
        else:
            stats["failure_count"] += 1
        previous_average = stats["average_duration_ms"]
        stats["average_duration_ms"] = (
            previous_average * (calls - 1) + duration_ms
        ) / calls
        stats["last_called"] = datetime.now(
            timezone.utc
        ).isoformat()


@dataclass(frozen=True)
class ToolDefinition:
    """
    单个工具的定义。
    """

    name: str
    description: str
    handler: ToolHandler
    parameters_schema: Mapping[str, Any] | None = None
    requires_approval: bool = False


    @staticmethod
    def _matches_annotation(
            value: Any,
            annotation: Any,
    ) -> bool:
        """
        判断运行时参数是否符合类型标注。

        对无法解析的复杂类型只做结构校验，
        不阻塞工具调用。
        """

        if (
            annotation is inspect.Parameter.empty
            or annotation is Any
        ):
            return True

        if annotation is None:
            return value is None

        if annotation is bool:
            return type(value) is bool

        if annotation is int:
            return type(value) is int

        if annotation is float:
            return type(value) in {
                int,
                float,
            }

        origin = get_origin(annotation)

        if origin in {
            Union,
            types.UnionType,
        }:
            return any(
                ToolDefinition._matches_annotation(
                    value,
                    item,
                )
                for item in get_args(annotation)
            )

        if origin is not None:
            try:
                return isinstance(value,origin)

            except TypeError:
                return True

        try:
            return isinstance(value,annotation)

        except TypeError:
            return True




    def argument_schema(self) -> list[dict[str,Any]]:
        """
        返回工具参数的机器可读描述。
        """
        if self.parameters_schema is not None:
            return self._json_schema_parameters(self.parameters_schema)

        try:
            signature = inspect.signature(self.handler)
        except (TypeError, ValueError):
            return []

        parameters: list[dict[str,Any]] = []

        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue

            annotation = parameter.annotation

            if annotation is inspect.Parameter.empty:
                annotation_name = None

            elif isinstance(annotation,str):
                annotation_name = annotation

            else:
                annotation_name = getattr(
                    annotation,
                    "__name__",
                    str(annotation),
                )

            item: dict[str,Any] = {
                "name": parameter.name,
                "required": (
                    parameter.default
                    is inspect.Parameter.empty
                ),
                "annotation": annotation_name,
            }

            if parameter.default is not inspect.Parameter.empty:
                item["default"] = parameter.default

            parameters.append(item)

        return parameters

    @staticmethod
    def _json_schema_parameters(
            schema: Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            return []
        required = schema.get("required", [])
        required_names = {
            str(name)
            for name in required
            if isinstance(name, str)
        } if isinstance(required, list) else set()

        parameters: list[dict[str, Any]] = []
        for name, value in properties.items():
            if not isinstance(value, Mapping):
                value = {}
            item: dict[str, Any] = {
                "name": str(name),
                "required": str(name) in required_names,
                "annotation": str(value.get("type", "Any")),
            }
            if "default" in value:
                item["default"] = value["default"]
            if "description" in value:
                item["description"] = str(value["description"])
            parameters.append(item)
        return parameters


    def to_schema(self) -> dict[str,Any]:
        """
        返回单个工具的机器可读定义。
        """

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.argument_schema(),
        }

    def metadata(self) -> dict[str, Any]:
        try:
            signature = inspect.signature(self.handler)
            annotations = get_type_hints(self.handler)
            return_annotation = annotations.get(
                "return",
                signature.return_annotation,
            )
        except (TypeError, ValueError, NameError, AttributeError):
            return_annotation = inspect.Signature.empty

        if return_annotation is inspect.Signature.empty:
            return_type = None
        elif isinstance(return_annotation, str):
            return_type = return_annotation
        else:
            return_type = getattr(
                return_annotation,
                "__name__",
                str(return_annotation),
            )

        handler = self.handler
        module_name = str(getattr(handler, "__module__", "") or "")
        function_name = str(getattr(handler, "__name__", "") or "")
        bound_instance = getattr(handler, "__self__", None)
        class_name = (
            type(bound_instance).__name__
            if bound_instance is not None
            else ""
        )
        return {
            "return_type": return_type,
            "module_name": module_name,
            "class_name": class_name,
            "function_name": function_name,
            "version": "1.0.0",
            "tags": [],
            "enabled": True,
            "deprecated": False,
            "deprecation_message": None,
            **get_tool_usage_stats(self.name),
        }




    def format_argument_contract(self) -> str:
        """
        生成人类和模型都容易理解的参数契约。
        """

        if self.parameters_schema is not None:
            parameters = self.argument_schema()
            if not parameters:
                return "无参数"
            parts = []
            for parameter in parameters:
                if parameter["required"]:
                    parts.append(f"{parameter['name']} (必填)")
                else:
                    default = parameter.get("default", "无")
                    parts.append(
                        f"{parameter['name']} (可选，默认={default!r})"
                    )
            return ", ".join(parts)

        try:
            signature = inspect.signature(self.handler)
        except (TypeError, ValueError):
            return "参数信息不可用"

        parts: list[str] = []

        for parameter in signature.parameters.values():
            if parameter.kind in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }:
                continue

            if parameter.default is inspect.Parameter.empty:
                parts.append(
                    f"{parameter.name} (必填)"
                )
            else:
                parts.append(
                    f"{parameter.name} "
                    f"(可选，默认={parameter.default!r})"
                )

        return ", ".join(parts) or "无参数"



    def validate_arguments(
            self,
            arguments: Mapping[str,Any] | None = None,
    ) -> dict[str,Any]:
        """
        校验工具参数。

        校验内容包括：

        - 必填参数是否缺失；
        - 是否传入工具不接受的参数；
        - 已提供参数是否符合类型标注。
        """

        payload = dict(arguments or {})

        if self.parameters_schema is not None:
            return self._validate_json_schema_arguments(payload)

        try:

            signature = inspect.signature(self.handler,)

            bound = signature.bind(**payload)

            try:
                type_hints = get_type_hints(self.handler,)

            except (TypeError, NameError,AttributeError,):
                type_hints = {}

            for parameter_name,value in bound.arguments.items():
                parameter = signature.parameters[parameter_name]

                if parameter.kind in {
                    inspect.Parameter.VAR_POSITIONAL,
                    inspect.Parameter.VAR_KEYWORD,
                }:
                    continue

                annotation = type_hints.get(
                    parameter_name,
                    parameter.annotation,
                )

                if annotation is inspect.Parameter.empty:
                    continue

                if not self._matches_annotation(
                    value,
                    annotation,
                ):
                    annotation_name = getattr(
                        annotation,
                        "__name__",
                        str(annotation),
                    )

                    raise TypeError(
                        f"参数「{parameter_name}」类型无效："
                        f"期望 {annotation_name}，"
                        f"实际为 {type(value).__name__}"
                    )

        except Exception as exc:
            raise ValueError(
                f"工具「{self.name}」参数无效：{exc}"
            ) from exc

        return payload

    def _validate_json_schema_arguments(
            self,
            payload: dict[str, Any],
    ) -> dict[str, Any]:
        schema = self.parameters_schema or {}
        properties = schema.get("properties", {})
        if not isinstance(properties, Mapping):
            properties = {}
        required = schema.get("required", [])
        required_names = {
            str(name)
            for name in required
            if isinstance(name, str)
        } if isinstance(required, list) else set()

        missing = sorted(required_names - set(payload))
        if missing:
            raise ValueError(
                f"工具「{self.name}」参数无效：缺少必填参数 {', '.join(missing)}"
            )

        unexpected = sorted(set(payload) - {str(name) for name in properties})
        if unexpected:
            raise ValueError(
                f"工具「{self.name}」参数无效：不支持参数 {', '.join(unexpected)}"
            )

        for name, value in payload.items():
            definition = properties.get(name, {})
            if not isinstance(definition, Mapping):
                continue
            expected = definition.get("type")
            valid = {
                "string": isinstance(value, str),
                "integer": type(value) is int,
                "number": type(value) in {int, float},
                "boolean": type(value) is bool,
                "array": isinstance(value, list),
                "object": isinstance(value, Mapping),
            }.get(str(expected), True)
            if not valid:
                raise ValueError(
                    f"工具「{self.name}」参数无效：参数「{name}」类型无效，"
                    f"期望 {expected}，实际为 {type(value).__name__}"
                )
            enum = definition.get("enum")
            if isinstance(enum, list) and value not in enum:
                raise ValueError(
                    f"工具「{self.name}」参数无效：参数「{name}」取值不受支持。"
                )
        return payload



    def invoke(
            self,
            arguments: Mapping[str,Any] | None = None,
    ) -> Any:
        """
        使用字典参数调用工具。
        """

        payload = self.validate_arguments(arguments)

        return self.handler(**payload)


class ToolRegistry:
    """
    工具注册表。

    负责：

    - 注册工具；
    - 查询工具；
    - 列出工具；
    - 按名称调用工具。
    """

    def __init__(
            self,
            tools: list[ToolDefinition] | None = None,
    ) -> None:
        self._tools: dict[str,ToolDefinition] = {}

        for tool in tools or []:
            self.register(tool)

    def register(
            self,
            tool: ToolDefinition,
    ) -> None:
        """
        注册一个工具。
        :param tool:
        :return:
        """

        name = tool.name.strip()

        if not name:
            raise ValueError("工具名称不能为空。")

        if name in self._tools:
            raise ValueError("工具已注册。")

        self._tools[name] = tool


    def get(self,name: str) -> ToolDefinition:
        """
        根据名称获取工具。
        :param name:
        :return:
        """

        normalized_name = name.strip()

        try:
            return self._tools[normalized_name]
        except KeyError as exc:
            raise KeyError(
                f"未找到工具：{normalized_name}"
            ) from exc


    def list_tools(self) -> list[ToolDefinition]:
        """
        返回已经注册的工具。
        """

        return list(self._tools.values())


    def describe_tools(self) -> list[dict[str,Any]]:
        """
        返回所有工具的机器可读目录。
        """

        return [
            tool.to_schema()
            for tool in self.list_tools()
        ]


    def invoke(
            self,
            name: str,
            arguments: Mapping[str,Any] | None = None,
    ) -> Any:
        """
        根据名称调用工具。
        :param name:
        :param arguments:
        :return:
        """

        tool = self.get(name)
        started_at = time.perf_counter()
        try:
            result = tool.invoke(arguments)
        except Exception:
            _record_tool_usage(
                tool.name,
                succeeded=False,
                duration_ms=(time.perf_counter() - started_at) * 1000,
            )
            raise

        _record_tool_usage(
            tool.name,
            succeeded=True,
            duration_ms=(time.perf_counter() - started_at) * 1000,
        )
        return result

    def get_statistics(self) -> dict[str, Any]:
        tools = self.list_tools()
        usage = {
            tool.name: get_tool_usage_stats(tool.name)
            for tool in tools
        }
        total_calls = sum(item["call_count"] for item in usage.values())
        total_successes = sum(
            item["success_count"] for item in usage.values()
        )
        most_used = sorted(
            (
                (name, item["call_count"])
                for name, item in usage.items()
                if item["call_count"]
            ),
            key=lambda item: (-item[1], item[0]),
        )[:10]
        return {
            "total_tools": len(tools),
            "enabled_tools": len(tools),
            "deprecated_tools": 0,
            "total_calls": total_calls,
            "total_successes": total_successes,
            "overall_success_rate": (
                total_successes / total_calls
                if total_calls
                else 0.0
            ),
            "most_used": most_used,
            "by_type": {"builtin": len(tools)},
            "tags": [],
        }
