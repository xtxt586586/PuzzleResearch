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


def normalize_knowledge_gap_analysis(
    value: Any,
    *,
    coverage_threshold: float = 0.8,
) -> dict[str, Any] | None:
    """Validate model-produced semantic coverage and targeted query data."""
    if not isinstance(value, dict):
        return None

    def score(raw: Any, default: float) -> float:
        try:
            return max(0.0, min(float(raw), 1.0))
        except (TypeError, ValueError):
            return default

    gaps: list[dict[str, str]] = []
    seen_aspects: set[str] = set()
    raw_gaps = value.get("gaps", [])
    if isinstance(raw_gaps, list):
        for item in raw_gaps:
            if isinstance(item, str):
                raw_gap = {"aspect": item}
            elif isinstance(item, dict):
                raw_gap = item
            else:
                continue

            aspect = " ".join(
                str(raw_gap.get("aspect", "") or "").split()
            ).strip()
            key = aspect.casefold()
            if not aspect or key in seen_aspects:
                continue
            seen_aspects.add(key)

            importance = str(
                raw_gap.get("importance", "medium") or "medium"
            ).strip().lower()
            if importance not in {"high", "medium", "low"}:
                importance = "medium"
            gaps.append({
                "aspect": aspect,
                "importance": importance,
                "reason": " ".join(
                    str(raw_gap.get("reason", "") or "").split()
                ).strip(),
            })

    suggested_queries = _clean_topics(
        value.get("suggested_queries", [])
        if isinstance(value.get("suggested_queries", []), list)
        else []
    )[:10]
    covered_aspects = _clean_topics(
        value.get("covered_aspects", [])
        if isinstance(value.get("covered_aspects", []), list)
        else []
    )[:20]
    coverage = score(value.get("overall_coverage"), 0.5)
    confidence = score(value.get("confidence"), 0.5)
    threshold = score(coverage_threshold, 0.8)
    sufficient = coverage >= threshold and not any(
        gap["importance"] == "high" for gap in gaps
    )

    return {
        "overall_coverage": coverage,
        "confidence": confidence,
        "gaps": gaps,
        "suggested_queries": suggested_queries,
        "covered_aspects": covered_aspects,
        "analysis": str(value.get("analysis", "") or "").strip(),
        "is_sufficient": sufficient,
    }


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
