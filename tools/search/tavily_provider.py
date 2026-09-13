from __future__ import annotations

from common.config import settings
from tools.search.base import SearchItem


class TavilyProvider:
    """
    Tavily 搜索适配器。

    Tavily 是当前项目的主要搜索 Provider。
    """

    name = "tavily"

    def is_available(self) -> bool:
        """
        判断 Tavily API Key 是否已经配置。
        """

        return bool(
            settings.tavily_api_key
            and settings.tavily_api_key.strip()
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchItem]:
        """
        调用 Tavily 并转换为统一搜索结果结构。
        """

        query = query.strip()

        if not query:
            raise ValueError("搜索关键词不能为空")

        if not self.is_available():
            raise RuntimeError(
                "未配置 TAVILY_API_KEY"
            )

        from tavily import TavilyClient

        client = TavilyClient(
            api_key=settings.tavily_api_key.strip()
        )

        response = client.search(
            query=query,
            max_results=max_results,
            search_depth="basic",
        )

        raw_results = response.get("results", [])
        normalized_results: list[SearchItem] = []

        for item in raw_results:
            normalized_results.append(
                {
                    "title": str(item.get("title", "")),
                    "url": str(item.get("url", "")),
                    "snippet": str(item.get("content", "")),
                    "content": str(
                        item.get("raw_content")
                        or item.get("content")
                        or ""
                    ),
                    "provider": self.name,
                }
            )

        return normalized_results