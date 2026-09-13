from __future__ import annotations

import importlib.util

from tools.search.base import SearchItem


class DuckDuckGoProvider:
    """
    DuckDuckGo 搜索适配器。

    DuckDuckGo 不需要 API Key，适合作为本地开发阶段的免费搜索 Provider。
    """

    name = "duckduckgo"

    def is_available(self) -> bool:
        """
        判断 duckduckgo-search 是否已经安装。
        """

        return importlib.util.find_spec("duckduckgo_search") is not None

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchItem]:
        """
        调用 DuckDuckGo，并把结果转换为 SearchItem。
        """

        query = query.strip()

        if not query:
            raise ValueError("搜索关键词不能为空")

        if not self.is_available():
            raise RuntimeError(
                "未安装 duckduckgo-search，请先执行："
                "python -m pip install duckduckgo-search"
            )

        from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            raw_results = list(
                ddgs.text(
                    query,
                    max_results=max_results,
                )
            )

        normalized_results: list[SearchItem] = []

        for item in raw_results:
            normalized_results.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(
                        item.get("href")
                        or item.get("url")
                        or ""
                    ),
                    "snippet": str(
                        item.get("body")
                        or item.get("snippet")
                        or ""
                    ),
                    "content": "",
                    "provider": self.name,
                }
            )

        return normalized_results