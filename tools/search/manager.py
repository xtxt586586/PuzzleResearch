from __future__ import annotations

from typing import Dict

from common.config import settings
from tools.search.base import SearchItem,SearchProvider
from tools.search.duckduckgo_provider import DuckDuckGoProvider
from tools.search.tavily_provider import TavilyProvider


class SearchManager:
    """
    搜索 Provider 管理器。

    职责：

    - 根据配置选择搜索服务；
    - 跳过未安装或未配置的 Provider；
    - 统一调用 Provider；
    - 在当前 Provider 失败时尝试下一个 Provider；
    - 向上层返回统一结构的搜索结果。
    """

    def __init__(
            self,
            providers: Dict[str, SearchProvider] = None,
    ) -> None:
        if providers is None:
            providers = {
                "tavily": TavilyProvider(),
                "duckduckgo": DuckDuckGoProvider(),
            }

        self.providers = providers

    def available_providers(self) -> list[str]:
        """
        返回当前配置下可用的 Provider 名称。
        """

        available: list[str] = []

        for provider_name in settings.search_engines_list:
            provider = self.providers.get(provider_name)

            # 配置中可能写了尚未实现的 Provider，
            # 例如 tavily、brave、serper。
            if provider is None:
                continue

            if provider.is_available():
                available.append(provider_name)

        return available

    def search(
            self,
            query: str,
            max_results: int = 5,
    ) -> list[SearchItem]:
        """
        按配置顺序执行搜索，并提供 fallback。
        """

        query = query.strip()

        if not query:
            raise ValueError("搜索关键词不能为空")

        errors: list[str] = []

        for provider_name in settings.search_engines_list:
            provider = self.providers.get(provider_name)

            if provider is None:
                continue

            if not provider.is_available():
                continue

            try:
                results = provider.search(
                    query=query,
                    max_results=max_results,
                )

                if results:
                    return results

                errors.append(
                    f"{provider_name} 返回了空结果"
                )

            except Exception as exc:
                errors.append(
                    f"{provider_name} 调用失败：{exc}"
                )

        available = self.available_providers()

        if not available:
            raise RuntimeError(
                "没有可用的搜索 Provider。"
                f"当前配置：{settings.search_engines_list}"
            )

        raise RuntimeError(
            "所有搜索 Provider 都没有返回结果："
            + "；".join(errors)
        )