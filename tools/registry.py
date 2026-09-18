from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable,Mapping,Union,get_args,get_origin,get_type_hints
import inspect
import types


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    """
    单个工具的定义。
    """

    name: str
    description: str
    handler: ToolHandler


    @staticmethod
    def _matches_annotation(
            value: str,
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

        if annotation is float:
            return type(value) in {
                int,
                float,
            }

        origin = get_origin(annotation)

        if origin is {
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


    def to_schema(self) -> dict[str,Any]:
        """
        返回单个工具的机器可读定义。
        """

        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.argument_schema(),
        }




    def format_argument_contract(self) -> str:
        """
        生成人类和模型都容易理解的参数契约。
        """

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
        return tool.invoke(arguments)