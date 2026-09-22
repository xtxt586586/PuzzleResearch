from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable


_CITATION_PATTERN = re.compile(
    r"\[(?:来源|source)?\s*(\d+)\]",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class ResearchQuality:
    overall_score: float
    citation_coverage: float
    source_count: int
    cited_source_count: int
    missing_topics: list[str]
    gap_count: int
    sufficient: bool
    summary: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall_score": self.overall_score,
            "citation_coverage": (
                self.citation_coverage
            ),
            "source_count": self.source_count,
            "cited_source_count": (
                self.cited_source_count
            ),
            "missing_topics": list(
                self.missing_topics
            ),
            "gap_count": self.gap_count,
            "sufficient": self.sufficient,
            "summary": self.summary,
        }


def _clean_topics(
    topics: Iterable[Any],
) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()

    for item in topics:
        value = " ".join(
            str(item or "").split()
        ).strip()

        if not value:
            continue

        key = value.casefold()

        if key in seen:
            continue

        seen.add(key)
        result.append(value)

    return result


def _unique_sources(
    sources: Iterable[Any],
) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()

    for item in sources:
        if not isinstance(item, dict):
            continue

        url = str(
            item.get("url", "")
        ).strip()

        key = url or str(
            item.get("title", "")
        ).strip()

        if not key or key in seen:
            continue

        seen.add(key)
        result.append(item)

    return result


def _citation_indexes(
    report: str,
) -> set[int]:
    body = str(report or "")

    if "参考来源：" in body:
        body = body.split(
            "参考来源：",
            1,
        )[0]

    if "References:" in body:
        body = body.split(
            "References:",
            1,
        )[0]

    return {
        int(value)
        for value in _CITATION_PATTERN.findall(
            body
        )
    }


def analyze_research_quality(
    *,
    report: str,
    sources: list[dict[str, Any]],
    missing_topics: Iterable[Any] = (),
    model_dimensions: dict[str, Any] | None = None,
    citation_threshold: float = 0.6,
) -> ResearchQuality:
    unique_sources = _unique_sources(
        sources
    )

    source_count = len(unique_sources)
    citation_indexes = _citation_indexes(
        report
    )

    valid_citations = {
        index
        for index in citation_indexes
        if 1 <= index <= source_count
    }

    cited_source_count = len(
        valid_citations
    )

    citation_coverage = (
        cited_source_count / source_count
        if source_count
        else 0.0
    )

    gaps = _clean_topics(
        missing_topics
    )

    if source_count == 0:
        gaps.append("没有可验证的研究来源")

    if (
        source_count > 0
        and citation_coverage
        < float(citation_threshold)
    ):
        gaps.append("报告引用覆盖率不足")

    gaps = _clean_topics(gaps)

    dimensions = model_dimensions or {}
    numeric_scores: list[float] = []

    for key in (
        "accuracy",
        "completeness",
        "citation",
        "citation_coverage",
    ):
        value = dimensions.get(key)

        if isinstance(value, (int, float)):
            numeric_scores.append(
                max(
                    0.0,
                    min(float(value), 1.0),
                )
            )

    model_score = (
        sum(numeric_scores)
        / len(numeric_scores)
        if numeric_scores
        else 1.0
    )

    overall_score = (
        model_score * 0.6
        + citation_coverage * 0.4
    )

    sufficient = (
        source_count > 0
        and citation_coverage
        >= float(citation_threshold)
        and not gaps
    )

    if sufficient:
        summary = "研究来源和引用覆盖率达到质量门槛。"
    else:
        summary = (
            "研究仍存在知识缺口或引用覆盖不足。"
        )

    return ResearchQuality(
        overall_score=round(
            overall_score,
            4,
        ),
        citation_coverage=round(
            citation_coverage,
            4,
        ),
        source_count=source_count,
        cited_source_count=cited_source_count,
        missing_topics=gaps,
        gap_count=len(gaps),
        sufficient=sufficient,
        summary=summary,
    )