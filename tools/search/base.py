from __future__ import annotations

from typing import Protocol,TypedDict


class SearchItem(TypedDict):
    """
    单条搜索结果的统一结构。

    所有搜索 Provider 都必须把原始结果转换成这个结构。
    """

    title: str
    url: str
    snippet: str
    content: str
    provider: str


class SearchProvider(Protocol):
    """
    搜索 Provider 的统一接口。

    Tavily、DuckDuckGo、Brave 等服务虽然 API 不同，
    但在业务层都只需要暴露以下三个内容。
    """

    name: str

    def is_available(self) -> bool:
        """
        判断当前 Provider 是否可以使用。
        """

        ...

    def search(
            self,
            query: str,
            max_results: int = 5,
    ) -> list[SearchItem]:
        """
        执行搜索并返回统一格式的结果。
        """
        ...