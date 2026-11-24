from __future__ import annotations

import threading
import time
from copy import deepcopy
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from collections import OrderedDict
from typing import Any, Dict

from common.config import settings
from tools.search.base import SearchItem,SearchProvider
from tools.search.academic.arxiv_provider import ArxivProvider
from tools.search.academic.pubmed_provider import PubMedProvider
from tools.search.academic.semantic_scholar_provider import SemanticScholarProvider
from tools.search.brave_provider import BraveProvider
from tools.search.duckduckgo_provider import DuckDuckGoProvider
from tools.search.feeds.hackernews_provider import HackerNewsProvider
from tools.search.feeds.reddit_provider import RedditProvider
from tools.search.feeds.twitter_provider import TwitterProvider
from tools.search.serper_provider import SerperProvider
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
            "calls": self.calls,
            "successes": self.successes,
            "failures": self.failures,
            "empty_results": self.empty_results,
            "last_error": self.last_error,
        }


@dataclass
class SearchCacheEntry:
    query: str
    results: list[SearchItem]
    timestamp: float


@dataclass
class SearchRuntimeState:
    cache: OrderedDict[
        tuple[str,int,tuple[str,...]],
        SearchCacheEntry,
    ] = field(default_factory=OrderedDict)
    cache_hits: int = 0
    cache_misses: int = 0
    cache_similar_hits: int = 0
    provider_stats: dict[str,ProviderCallStats] = field(
        default_factory=dict
    )
    cache_lock: threading.RLock = field(
        default_factory=threading.RLock,
        repr=False,
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
        self._uses_default_providers = providers is None
        if providers is None:
            providers = {
                "tavily": TavilyProvider(),
                "duckduckgo": DuckDuckGoProvider(),
                "brave": BraveProvider(),
                "serper": SerperProvider(),
                "arxiv": ArxivProvider(),
                "semantic_scholar": SemanticScholarProvider(),
                "pubmed": PubMedProvider(),
                "hackernews": HackerNewsProvider(),
                "reddit": RedditProvider(),
                "twitter": TwitterProvider(),
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
            strategy: str | None = None,
            provider_profile: list[str] | None = None,
    ) -> list[SearchItem]:
        """
        按配置顺序执行搜索，并提供 fallback。
        """

        query = query.strip()

        if not query:
            raise ValueError("搜索关键词不能为空")

        max_results = max(int(max_results), 1)

        # Deep Research may opt into the multi-provider orchestrator while the
        # legacy path keeps its existing cache and fallback semantics.
        if strategy:
            from tools.search.multi_search import (
                MultiSearchOrchestrator,
                SearchStrategy,
            )

            try:
                selected_strategy = SearchStrategy(str(strategy).strip().lower())
            except ValueError:
                selected_strategy = SearchStrategy.FALLBACK

            if self._uses_default_providers:
                from tools.search.multi_search import get_search_orchestrator

                orchestrator = get_search_orchestrator()
            else:
                orchestrator = MultiSearchOrchestrator(
                    list(self.providers.values()),
                    strategy=selected_strategy,
                )
            return [
                item.to_dict()
                for item in orchestrator.search(
                    query,
                    max_results=max_results,
                    strategy=selected_strategy,
                    provider_profile=provider_profile,
                )
            ]

        cache_key = self._cache_key(
            query,
            max_results,
        )

        cached = self._get_cached_results(cache_key, query)
        if cached is not None:
            return cached

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
                    self._store_cached_results(cache_key, query, results)
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
            self._normalize_query(query),
            max_results,
            tuple(settings.search_engines_list),
        )

    @staticmethod
    def _normalize_query(query: str) -> str:
        return " ".join(query.lower().split())

    def _cache_ttl(self) -> float:
        return max(
            1.0,
            float(getattr(settings, "search_cache_ttl_seconds", 1800.0)),
        )

    def _cache_max_size(self) -> int:
        return max(
            1,
            int(getattr(settings, "search_cache_max_size", 200)),
        )

    def _purge_expired_cache(self, now: float) -> None:
        ttl = self._cache_ttl()
        cache = self.runtime_state.cache
        expired = [
            key
            for key, entry in cache.items()
            if now - entry.timestamp > ttl
        ]
        for key in expired:
            del cache[key]

    def _get_cached_results(
        self,
        cache_key: tuple[str, int, tuple[str, ...]],
        query: str,
    ) -> list[SearchItem] | None:
        state = self.runtime_state
        now = time.monotonic()
        normalized_query = self._normalize_query(query)
        with state.cache_lock:
            self._purge_expired_cache(now)
            entry = state.cache.get(cache_key)
            if entry is not None:
                state.cache.move_to_end(cache_key)
                state.cache_hits += 1
                return deepcopy(entry.results)

            threshold = min(
                1.0,
                max(
                    0.0,
                    float(
                        getattr(
                            settings,
                            "search_cache_similarity_threshold",
                            0.9,
                        )
                    ),
                ),
            )
            context = cache_key[1:]
            for candidate_key, candidate in reversed(state.cache.items()):
                if candidate_key[1:] != context:
                    continue
                similarity = SequenceMatcher(
                    None,
                    normalized_query,
                    candidate.query,
                ).ratio()
                if similarity >= threshold:
                    state.cache.move_to_end(candidate_key)
                    state.cache_similar_hits += 1
                    return deepcopy(candidate.results)

            state.cache_misses += 1
            return None

    def _store_cached_results(
        self,
        cache_key: tuple[str, int, tuple[str, ...]],
        query: str,
        results: list[SearchItem],
    ) -> None:
        state = self.runtime_state
        with state.cache_lock:
            now = time.monotonic()
            self._purge_expired_cache(now)
            state.cache.pop(cache_key, None)
            state.cache[cache_key] = SearchCacheEntry(
                query=self._normalize_query(query),
                results=deepcopy(results),
                timestamp=now,
            )
            while len(state.cache) > self._cache_max_size():
                state.cache.popitem(last=False)


    def _provider_call_stats(
            self,
            provider_name: str,
    ) -> ProviderCallStats:
        return self.runtime_state.provider_stats.setdefault(
            provider_name,
            ProviderCallStats(),
        )

    def cache_stats(self) -> dict[str, Any]:
        state = self.runtime_state
        with state.cache_lock:
            self._purge_expired_cache(time.monotonic())
            total = state.cache_hits + state.cache_similar_hits + state.cache_misses

            return {
                "entries": len(state.cache),
                "hits": state.cache_hits,
                "misses": state.cache_misses,
                "hit_rate": (
                    (state.cache_hits + state.cache_similar_hits) / total
                    if total
                    else 0.0
                ),
            }

    def cache_api_stats(self) -> dict[str, Any]:
        state = self.runtime_state
        with state.cache_lock:
            self._purge_expired_cache(time.monotonic())
            total = state.cache_hits + state.cache_similar_hits + state.cache_misses
            return {
                "size": len(state.cache),
                "max_size": self._cache_max_size(),
                "hits": state.cache_hits,
                "similar_hits": state.cache_similar_hits,
                "misses": state.cache_misses,
                "hit_rate": round(
                    (state.cache_hits + state.cache_similar_hits) / total
                    if total
                    else 0.0,
                    3,
                ),
            }

    def clear_cache(self) -> int:
        state = self.runtime_state
        with state.cache_lock:
            cleared = len(state.cache)
            state.cache.clear()
            state.cache_hits = 0
            state.cache_similar_hits = 0
            state.cache_misses = 0
            return cleared

    def provider_call_stats(self) -> dict[str,dict[str,Any]]:
        names = set(self.providers)
        names.update(settings.search_engines_list)

        return {
            name: self._provider_call_stats(name).to_dict()
            for name in names
        }
