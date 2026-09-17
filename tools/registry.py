from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable,Mapping
import inspect


ToolHandler = Callable[..., Any]


@dataclass(frozen=True)
class ToolDefinition:
    """
    单个工具的定义。
    """

    name: str
    description: str
    handler: ToolHandler


    def validate_arguments(
            self,
            arguments: Mapping[str,Any] | None = None,
    ) -> dict[str,Any]:
        """
        校验工具参数。

        bind 会检查：
        - 必填参数是否缺失；
        - 是否传入工具不接受的参数。
        """

        payload = dict(arguments or {})

        try:
            inspect.signature(
                self.handler,
            ).bind(**payload)

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