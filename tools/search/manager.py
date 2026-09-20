from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Dict

from common.config import settings
from tools.search.base import SearchItem,SearchProvider
from tools.search.duckduckgo_provider import DuckDuckGoProvider
from tools.search.tavily_provider import TavilyProvider


@dataclass
class ProviderCallStats:
    calls: int = 0
    successes: int = 0
    failures: int = 0
    empty_results: int = 0
    last_error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "call": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "empty_results": self.empty_results,
            "last_error": self.last_error,
        }


@dataclass
class SearchRuntimeState:
    cache: dict[
        tuple[str,int,tuple[str,...]],
        list[SearchItem],
    ] = field(default_factory=dict)
    cache_hits: int = 0
    cache_misses: int = 0
    provider_stats: dict[str,ProviderCallStats] = field(
        default_factory=dict
    )


_DEFAULT_SEARCH_RUNTIME_STATE = SearchRuntimeState()

def get_default_search_runtime_state() -> SearchRuntimeState:
    return _DEFAULT_SEARCH_RUNTIME_STATE



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
            providers: Dict[str, SearchProvider] | None = None,
            runtime_state: SearchRuntimeState | None = None
    ) -> None:
        if providers is None:
            providers = {
                "tavily": TavilyProvider(),
                "duckduckgo": DuckDuckGoProvider(),
            }

        self.providers = providers
        self.runtime_state = (
            runtime_state
            or SearchRuntimeState()
        )

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

        max_results = max(int(max_results), 1)
        cache_key = self._cache_key(
            query,
            max_results,
        )

        if cache_key in self.runtime_state.cache:
            self.runtime_state.cache_hits += 1
            return deepcopy(
                self.runtime_state.cache[cache_key]
            )

        self.runtime_state.cache_misses += 1

        errors: list[str] = []

        for provider_name in settings.search_engines_list:
            provider = self.providers.get(provider_name)

            if provider is None:
                continue

            if not provider.is_available():
                continue


            stats = self._provider_call_stats(provider_name)

            stats.calls += 1


            try:
                results = provider.search(
                    query=query,
                    max_results=max_results,
                )

                if results:
                    stats.successes += 1
                    self.runtime_state.cache[cache_key] = deepcopy(results)
                    return deepcopy(results)

                stats.empty_results += 1

                errors.append(
                    f"{provider_name} 返回了空结果"
                )

            except Exception as exc:
                stats.failures += 1
                stats.last_error = str(exc)
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

    def _cache_key(
            self,
            query: str,
            max_results: int,
    ) -> tuple[str,int,tuple[str,...]]:
        return (
            query,
            max_results,
            tuple(settings.search_engines_list),
        )


    def _provider_call_stats(
            self,
            provider_name: str,
    ) -> ProviderCallStats:
        return self.runtime_state.provider_stats.setdefault(
            provider_name,
            ProviderCallStats(),
        )

    def cache_stats(self) -> dict[str, Any]:
        total = (
            self.runtime_state.cache_hits
            + self.runtime_state.cache_misses
        )

        return {
            "entries": len(self.runtime_state.cache),
            "hits": self.runtime_state.cache_hits,
            "misses": self.runtime_state.cache_misses,
            "hit_rate": (
                self.runtime_state.cache_hits / total
                if total
                else 0.0
            ),
        }

    def clear_cache(self) -> int:
        cleared = len(self.runtime_state.cache)
        self.runtime_state.cache.clear()
        return cleared

    def provider_call_stats(self) -> dict[str,dict[str,Any]]:
        names = set(self.providers)
        names.update(settings.search_engines_list)

        return {
            name: self._provider_call_stats(name).to_dict()
            for name in names
        }