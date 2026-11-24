"""Multi-provider search orchestration used by Deep Research."""

from __future__ import annotations

import copy
import hashlib
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime, timezone
from difflib import SequenceMatcher
from enum import Enum
from typing import Any, Iterable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from common.config import settings
from tools.search.academic.arxiv_provider import ArxivProvider
from tools.search.academic.pubmed_provider import PubMedProvider
from tools.search.academic.semantic_scholar_provider import SemanticScholarProvider
from tools.search.brave_provider import BraveProvider
from tools.search.duckduckgo_provider import DuckDuckGoProvider
from tools.search.feeds.hackernews_provider import HackerNewsProvider
from tools.search.feeds.reddit_provider import RedditProvider
from tools.search.feeds.twitter_provider import TwitterProvider
from tools.search.reliability import ProviderReliabilityManager, ReliabilityPolicy
from tools.search.serper_provider import SerperProvider
from tools.search.tavily_provider import TavilyProvider


_TRACKING_KEYS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "source"}


def canonicalize_result_url(value: str) -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "://" not in raw:
        raw = f"https://{raw}"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        return raw
    if not parsed.netloc:
        return raw
    query = []
    for key, item in parse_qsl(parsed.query, keep_blank_values=True):
        key_lower = key.lower()
        if key_lower.startswith("utm_") or key_lower in _TRACKING_KEYS:
            continue
        query.append((key, item))
    path = parsed.path or "/"
    if path != "/":
        path = path.rstrip("/") or "/"
    return urlunsplit(
        ((parsed.scheme or "https").lower(), parsed.netloc.lower(), path, urlencode(sorted(query), doseq=True), "")
    )


@dataclass
class ProviderStats:
    name: str
    total_calls: int = 0
    success_count: int = 0
    error_count: int = 0
    total_latency_ms: float = 0.0
    avg_result_quality: float = 0.5
    last_error: str | None = None
    last_error_time: str | None = None
    is_healthy: bool = True
    consecutive_failures: int = 0

    @property
    def success_rate(self) -> float:
        return self.success_count / self.total_calls if self.total_calls else 1.0

    @property
    def avg_latency_ms(self) -> float:
        return self.total_latency_ms / self.success_count if self.success_count else 0.0

    def record_success(self, latency_ms: float, quality: float = 0.5) -> None:
        self.total_calls += 1
        self.success_count += 1
        self.total_latency_ms += max(0.0, float(latency_ms))
        self.avg_result_quality = self.avg_result_quality * 0.9 + float(quality) * 0.1
        self.consecutive_failures = 0
        self.is_healthy = True

    def record_failure(self, error: str) -> None:
        self.total_calls += 1
        self.error_count += 1
        self.last_error = str(error)
        self.last_error_time = datetime.now(timezone.utc).isoformat()
        self.consecutive_failures += 1
        if self.consecutive_failures >= 3:
            self.is_healthy = False


class SearchProvider:
    """Base class for providers accepted by MultiSearchOrchestrator."""

    def __init__(self, name: str, api_key: str | None = None) -> None:
        self.name = name
        self.api_key = api_key
        self.stats = ProviderStats(name=name)

    def is_available(self) -> bool:
        raise NotImplementedError

    def search(self, query: str, max_results: int = 10) -> list["SearchResult"]:
        raise NotImplementedError

    def get_stats(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "total_calls": self.stats.total_calls,
            "success_count": self.stats.success_count,
            "error_count": self.stats.error_count,
            "success_rate": self.stats.success_rate,
            "avg_latency_ms": self.stats.avg_latency_ms,
            "avg_result_quality": self.stats.avg_result_quality,
            "healthy": self.stats.is_healthy,
            "last_error": self.stats.last_error,
        }


class SearchStrategy(str, Enum):
    FALLBACK = "fallback"
    PARALLEL = "parallel"
    ROUND_ROBIN = "round_robin"
    BEST_FIRST = "best_first"


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    content: str = ""
    score: float = 0.0
    published_date: str | None = None
    provider: str = ""
    raw_data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.url = canonicalize_result_url(self.url)

    @property
    def url_hash(self) -> str:
        return hashlib.md5(self.url.encode("utf-8")).hexdigest()[:12]

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "content": self.content,
            "score": self.score,
            "published_date": self.published_date,
            "provider": self.provider,
        }


class _LegacyProvider(SearchProvider):
    def __init__(self, provider: Any):
        super().__init__(provider.name)
        self.provider = provider

    def is_available(self) -> bool:
        return bool(self.provider.is_available())

    def search(self, query: str, max_results: int = 10) -> list[SearchResult]:
        raw = self.provider.search(query=query, max_results=max_results)
        return [_coerce_result(item, self.name) for item in raw or []]


def _coerce_result(item: Any, provider: str = "") -> SearchResult:
    if isinstance(item, SearchResult):
        return item
    data = item if isinstance(item, dict) else {}
    return SearchResult(
        title=str(data.get("title") or ""),
        url=str(data.get("url") or ""),
        snippet=str(data.get("snippet") or data.get("summary") or ""),
        content=str(data.get("content") or data.get("raw_excerpt") or ""),
        score=float(data.get("score") or 0.0),
        published_date=data.get("published_date"),
        provider=str(data.get("provider") or provider),
        raw_data=copy.deepcopy(data),
    )


class MultiSearchOrchestrator:
    def __init__(
        self,
        providers: list[SearchProvider] | None = None,
        strategy: SearchStrategy = SearchStrategy.FALLBACK,
        similarity_threshold: float = 0.7,
        reliability_manager: ProviderReliabilityManager | None = None,
    ) -> None:
        raw_providers = providers if providers is not None else self._default_providers()
        self.providers = [
            provider
            if isinstance(provider, SearchProvider)
            else _LegacyProvider(provider)
            for provider in raw_providers
        ]
        self.strategy = strategy
        self.similarity_threshold = max(0.0, min(1.0, float(similarity_threshold)))
        self._round_robin_index = 0
        policy = ReliabilityPolicy(
            max_retries=max(0, int(getattr(settings, "search_reliability_max_retries", 2))),
            retry_backoff_seconds=max(0.0, float(getattr(settings, "search_reliability_retry_backoff_seconds", 0.5))),
            circuit_breaker_failures=max(1, int(getattr(settings, "search_reliability_circuit_breaker_failures", 3))),
            circuit_breaker_reset_seconds=max(0.0, float(getattr(settings, "search_reliability_circuit_breaker_reset_seconds", 60.0))),
        )
        self.reliability_manager = reliability_manager or ProviderReliabilityManager(policy)
        self._cache: dict[str, tuple[float, list[SearchResult]]] = {}
        self._cache_lock = threading.RLock()

    @staticmethod
    def _default_providers() -> list[SearchProvider]:
        providers: list[SearchProvider] = []
        for provider in (
            TavilyProvider(),
            DuckDuckGoProvider(),
            BraveProvider(),
            SerperProvider(),
            ArxivProvider(),
            SemanticScholarProvider(),
            PubMedProvider(),
            HackerNewsProvider(),
            RedditProvider(),
            TwitterProvider(),
        ):
            if provider.is_available():
                providers.append(_LegacyProvider(provider))
        return providers

    def get_available_providers(self) -> list[SearchProvider]:
        return [
            provider
            for provider in self.providers
            if provider.is_available() and provider.stats.is_healthy
        ]

    def search(
        self,
        query: str,
        max_results: int = 10,
        strategy: SearchStrategy | None = None,
        provider_profile: list[str] | None = None,
    ) -> list[SearchResult]:
        query = " ".join(str(query or "").split())
        if not query:
            raise ValueError("搜索关键词不能为空")
        max_results = max(1, int(max_results))
        selected_strategy = strategy or self.strategy
        providers = self._apply_profile(self.get_available_providers(), provider_profile)
        if not providers:
            return []

        cache_key = f"{selected_strategy.value}:{max_results}:{','.join(provider_profile or [])}:{query.lower()}"
        now = time.monotonic()
        with self._cache_lock:
            cached = self._cache.get(cache_key)
            ttl = max(1.0, float(getattr(settings, "search_cache_ttl_seconds", 1800.0)))
            if cached and now - cached[0] <= ttl:
                return copy.deepcopy(cached[1])
            self._cache.pop(cache_key, None)

        if selected_strategy == SearchStrategy.PARALLEL:
            results = self._parallel(query, max_results, providers)
        elif selected_strategy == SearchStrategy.ROUND_ROBIN:
            provider = providers[self._round_robin_index % len(providers)]
            self._round_robin_index += 1
            results = self._call(provider, query, max_results)
            if not results:
                results = self._fallback(query, max_results, providers)
        elif selected_strategy == SearchStrategy.BEST_FIRST:
            ordered = sorted(providers, key=self._provider_score, reverse=True)
            results = self._fallback(query, max_results, ordered)
        else:
            results = self._fallback(query, max_results, providers)

        ranked = self._deduplicate_and_rank(results, max_results, query)
        if ranked:
            with self._cache_lock:
                self._cache[cache_key] = (time.monotonic(), copy.deepcopy(ranked))
        return ranked

    def _apply_profile(self, providers: list[SearchProvider], profile: list[str] | None) -> list[SearchProvider]:
        preferred = [str(item).strip().lower() for item in profile or [] if str(item).strip()]
        if not preferred:
            return providers
        by_name = {provider.name.lower(): provider for provider in providers}
        selected = [by_name[name] for name in preferred if name in by_name]
        selected_ids = {id(item) for item in selected}
        return selected + [item for item in providers if id(item) not in selected_ids]

    def _call(self, provider: SearchProvider, query: str, max_results: int) -> list[SearchResult]:
        def invoke() -> list[SearchResult]:
            started = time.monotonic()
            raw = provider.search(query, max_results)
            results = [_coerce_result(item, provider.name) for item in raw or []]
            elapsed_ms = (time.monotonic() - started) * 1000
            if results:
                provider.stats.record_success(elapsed_ms, max(float(item.score or 0.0) for item in results))
            else:
                raise RuntimeError(f"{provider.name} 返回空结果")
            return results

        result = self.reliability_manager.call(provider.name, invoke)
        if not result:
            provider.stats.record_failure("搜索失败或返回空结果")
        return list(result or [])

    def _fallback(self, query: str, max_results: int, providers: list[SearchProvider]) -> list[SearchResult]:
        for provider in providers:
            results = self._call(provider, query, max_results)
            if results:
                return results
        return []

    def _parallel(self, query: str, max_results: int, providers: list[SearchProvider]) -> list[SearchResult]:
        results: list[SearchResult] = []
        workers = max(1, min(len(providers), int(getattr(settings, "search_parallel_max_workers", 8))))
        timeout = max(0.0, float(getattr(settings, "search_parallel_timeout_seconds", 30.0)))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(self._call, provider, query, max_results) for provider in providers]
            try:
                for future in as_completed(futures, timeout=timeout or None):
                    results.extend(future.result())
            except TimeoutError:
                for future in futures:
                    future.cancel()
        return results

    def _provider_score(self, provider: SearchProvider) -> float:
        return provider.stats.success_rate * provider.stats.avg_result_quality / max(provider.stats.avg_latency_ms, 100.0) * 1000

    def _deduplicate_and_rank(self, results: Iterable[SearchResult], max_results: int, query: str) -> list[SearchResult]:
        unique: list[SearchResult] = []
        by_url: dict[str, SearchResult] = {}
        for result in results:
            item = _coerce_result(result)
            if not item.url:
                continue
            existing = by_url.get(item.url)
            if existing is None or self._ranking_score(item, query) > self._ranking_score(existing, query):
                by_url[item.url] = item
        for item in by_url.values():
            if any(SequenceMatcher(None, item.snippet[:300].lower(), other.snippet[:300].lower()).ratio() >= self.similarity_threshold for other in unique):
                continue
            unique.append(item)
        unique.sort(key=lambda item: self._ranking_score(item, query), reverse=True)
        return unique[:max_results]

    def _ranking_score(self, result: SearchResult, query: str) -> float:
        score = max(0.0, min(1.0, float(result.score or 0.0)))
        if not getattr(settings, "search_enable_freshness_ranking", True) or not self._is_time_sensitive(query):
            return score
        fresh = self._freshness_score(result.published_date)
        weight = max(0.0, min(1.0, float(getattr(settings, "search_freshness_weight", 0.35))))
        return score * (1.0 - weight) + fresh * weight

    @staticmethod
    def _is_time_sensitive(query: str) -> bool:
        lowered = query.lower()
        return any(marker in lowered for marker in ("latest", "today", "recent", "current", "news", "最新", "近期", "今年")) or bool(re.search(r"\b20\d{2}\b", lowered))

    @staticmethod
    def _freshness_score(value: str | None) -> float:
        if not value:
            return 0.5
        text = str(value).strip().replace("Z", "+00:00")
        try:
            published = datetime.fromisoformat(text)
        except ValueError:
            try:
                published = datetime.strptime(text, "%Y-%m-%d").replace(tzinfo=timezone.utc)
            except ValueError:
                return 0.5
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_days = max(0.0, (datetime.now(timezone.utc) - published.astimezone(timezone.utc)).total_seconds() / 86400.0)
        half_life = max(1.0, float(getattr(settings, "search_freshness_half_life_days", 30.0)))
        return 2 ** (-age_days / half_life)

    def get_provider_stats(self) -> list[dict[str, Any]]:
        return [provider.get_stats() for provider in self.providers]

    def provider_snapshots(self) -> list[dict[str, Any]]:
        return [
            {
                **provider.get_stats(),
                "available": bool(provider.is_available()),
                "circuit": self.reliability_manager.snapshot(provider.name),
            }
            for provider in self.providers
        ]

    def reset_provider_health(self) -> None:
        for provider in self.providers:
            provider.stats.is_healthy = True
            provider.stats.consecutive_failures = 0


_global_orchestrator: MultiSearchOrchestrator | None = None


def get_search_orchestrator() -> MultiSearchOrchestrator:
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = MultiSearchOrchestrator()
    return _global_orchestrator


def reset_search_orchestrator() -> None:
    global _global_orchestrator
    _global_orchestrator = None


def multi_search(
    query: str,
    max_results: int = 10,
    strategy: SearchStrategy = SearchStrategy.FALLBACK,
    provider_profile: list[str] | None = None,
) -> list[dict[str, Any]]:
    return [
        item.to_dict()
        for item in get_search_orchestrator().search(
            query,
            max_results=max_results,
            strategy=strategy,
            provider_profile=provider_profile,
        )
    ]
