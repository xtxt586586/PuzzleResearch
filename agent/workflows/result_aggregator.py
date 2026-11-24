from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher
import re
from typing import Any, Iterable
from urllib.parse import urlsplit

from tools.research.source_registry import SourceRegistry


@dataclass
class RankedSearchResult:
    data: dict[str, Any]
    score: float = 0.0
    canonical_url: str = ""
    source_id: str = ""

    @property
    def title(self) -> str:
        return str(self.data.get("title") or "")

    @property
    def content(self) -> str:
        return str(
            self.data.get("fetched_content")
            or self.data.get("content")
            or self.data.get("snippet")
            or ""
        )

    @property
    def domain(self) -> str:
        return str(urlsplit(self.canonical_url or self.data.get("url", "")).netloc)

    def to_dict(self) -> dict[str, Any]:
        result = dict(self.data)
        result["score"] = round(float(self.score), 4)
        result["canonical_url"] = self.canonical_url
        result["source_id"] = self.source_id
        return result


@dataclass
class AggregatedResults:
    tier_1: list[RankedSearchResult] = field(default_factory=list)
    tier_2: list[RankedSearchResult] = field(default_factory=list)
    tier_3: list[RankedSearchResult] = field(default_factory=list)
    total_before: int = 0
    total_after: int = 0
    duplicates_removed: int = 0

    def all_results(self) -> list[RankedSearchResult]:
        return [*self.tier_1, *self.tier_2, *self.tier_3]

    def to_dicts(self, limit: int | None = None) -> list[dict[str, Any]]:
        results = self.all_results()
        if limit is not None:
            results = results[: max(0, int(limit))]
        return [item.to_dict() for item in results]

    def to_context(
        self,
        max_tier_1: int = 5,
        max_tier_2: int = 3,
        max_tier_3: int = 2,
        max_content_length: int = 500,
    ) -> tuple[str, str]:
        blocks: list[str] = []
        sources: list[str] = []
        tiers = (
            ("主要来源", self.tier_1, max_tier_1),
            ("辅助来源", self.tier_2, max_tier_2),
            ("补充来源", self.tier_3, max_tier_3),
        )
        for label, results, limit in tiers:
            if not results or limit <= 0:
                continue
            blocks.append(f"## {label}")
            for index, result in enumerate(results[:limit], start=1):
                excerpt = result.content[:max_content_length]
                if len(result.content) > max_content_length:
                    excerpt += "..."
                blocks.extend(
                    [
                        f"【来源 {index}】{result.title}",
                        f"链接：{result.canonical_url or result.data.get('url', '')}",
                        f"内容：{excerpt}",
                        "",
                    ]
                )
                sources.append(
                    f"{result.title}：{result.canonical_url or result.data.get('url', '')}"
                )
        return "\n".join(blocks), "\n".join(sources)


class ResultAggregator:
    """对多查询搜索结果执行统一去重、排序和分层。"""

    def __init__(
        self,
        similarity_threshold: float = 0.7,
        max_results_per_query: int = 10,
        tier_1_threshold: float = 0.6,
        tier_2_threshold: float = 0.3,
    ) -> None:
        self.similarity_threshold = max(0.0, min(1.0, float(similarity_threshold)))
        self.max_results_per_query = max(1, int(max_results_per_query))
        self.tier_1_threshold = float(tier_1_threshold)
        self.tier_2_threshold = float(tier_2_threshold)

    def aggregate(
        self,
        search_results: Iterable[dict[str, Any]],
        original_query: str = "",
    ) -> AggregatedResults:
        normalized = self._normalize_results(search_results)
        total_before = len(normalized)
        by_url: dict[str, RankedSearchResult] = {}
        for result in normalized:
            previous = by_url.get(result.canonical_url)
            if previous is None or len(result.content) > len(previous.content):
                by_url[result.canonical_url] = result

        url_results = list(by_url.values())
        unique: list[RankedSearchResult] = []
        for result in sorted(url_results, key=lambda item: len(item.content), reverse=True):
            if any(
                self._similarity(result.content, existing.content)
                >= self.similarity_threshold
                for existing in unique
            ):
                continue
            unique.append(result)

        for result in unique:
            result.score = self._relevance_score(result, original_query)
        unique.sort(key=lambda item: item.score, reverse=True)

        tier_1 = [item for item in unique if item.score >= self.tier_1_threshold]
        tier_2 = [
            item for item in unique
            if self.tier_2_threshold <= item.score < self.tier_1_threshold
        ]
        tier_3 = [item for item in unique if item.score < self.tier_2_threshold]
        return AggregatedResults(
            tier_1=tier_1,
            tier_2=tier_2,
            tier_3=tier_3,
            total_before=total_before,
            total_after=len(unique),
            duplicates_removed=total_before - len(unique),
        )

    def _normalize_results(
        self,
        search_results: Iterable[dict[str, Any]],
    ) -> list[RankedSearchResult]:
        registry = SourceRegistry()
        normalized: list[RankedSearchResult] = []
        query_counts: dict[str, int] = {}
        for item in search_results:
            if not isinstance(item, dict):
                continue
            query = str(item.get("query") or "")
            count = query_counts.get(query, 0)
            if count >= self.max_results_per_query:
                continue
            query_counts[query] = count + 1
            url = str(item.get("url") or "").strip()
            source = registry.register(url=url, title=str(item.get("title") or ""))
            if source is None:
                continue
            data = dict(item)
            data["content"] = str(
                data.get("fetched_content")
                or data.get("content")
                or data.get("snippet")
                or ""
            )
            normalized.append(
                RankedSearchResult(
                    data=data,
                    canonical_url=source.canonical_url,
                    source_id=source.source_id,
                )
            )
        return normalized

    def _relevance_score(self, result: RankedSearchResult, query: str) -> float:
        base = max(0.0, min(1.0, float(result.data.get("score") or 0.0)))
        terms = self._tokenize(query)
        if not terms:
            relevance = 0.5
        else:
            title_terms = self._tokenize(result.title)
            content_terms = self._tokenize(result.content[:1500])
            query_terms = self._tokenize(str(result.data.get("query") or ""))
            title_overlap = len(terms & title_terms) / len(terms)
            content_overlap = len(terms & content_terms) / len(terms)
            query_overlap = len(terms & query_terms) / len(terms)
            relevance = title_overlap * 0.4 + content_overlap * 0.3 + query_overlap * 0.2
        authority = 0.1 if any(
            marker in result.domain.lower()
            for marker in ("arxiv.org", "pubmed", ".gov", ".edu", "nature.com", "github.com")
        ) else 0.0
        return min(1.0, max(base * 0.5 + relevance + authority, base))

    @staticmethod
    def _tokenize(value: str) -> set[str]:
        return {
            item
            for item in re.findall(r"[\w]+", str(value or "").lower())
            if len(item) > 1
        }

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        return SequenceMatcher(None, left.lower()[:1000], right.lower()[:1000]).ratio()
