"""Deterministic query coverage helpers for Deep Research."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any


_QUERY_DIMENSIONS = ("freshness", "official", "evidence", "risk", "implementation")
_EN_TIME_MARKERS = (
    "latest", "recent", "today", "current", "update", "updates",
    "new", "this week", "this month", "news",
)
_ZH_TIME_MARKERS = ("最新", "近期", "今天", "当下", "更新", "本周", "本月", "动态", "新闻")
_OFFICIAL_MARKERS = (
    "official", "documentation", "docs", "release notes", "changelog", "roadmap",
    "官方", "文档", "发布说明", "路线图",
)
_EVIDENCE_MARKERS = (
    "benchmark", "evaluation", "metrics", "data", "report", "study", "paper",
    "评测", "评估", "指标", "数据", "报告", "论文",
)
_RISK_MARKERS = (
    "risk", "risks", "limitation", "limitations", "criticism", "criticisms",
    "tradeoff", "trade-offs", "争议", "风险", "局限", "缺点", "问题",
)
_IMPLEMENTATION_MARKERS = (
    "implementation", "how to", "best practices", "case study", "architecture",
    "playbook", "实践", "案例", "最佳实践", "架构", "落地",
)
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_YEAR_RE = re.compile(r"\b20\d{2}\b")


def _contains_any(text: str, markers: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in markers)


def is_time_sensitive_topic(topic: str) -> bool:
    text = str(topic or "").strip()
    return bool(
        text
        and (
            _contains_any(text, _EN_TIME_MARKERS)
            or any(marker in text for marker in _ZH_TIME_MARKERS)
            or _YEAR_RE.search(text)
        )
    )


def query_dimensions(query: str) -> set[str]:
    text = str(query or "").strip()
    if not text:
        return set()
    dimensions: set[str] = set()
    if is_time_sensitive_topic(text):
        dimensions.add("freshness")
    for name, markers in (
        ("official", _OFFICIAL_MARKERS),
        ("evidence", _EVIDENCE_MARKERS),
        ("risk", _RISK_MARKERS),
        ("implementation", _IMPLEMENTATION_MARKERS),
    ):
        if _contains_any(text, markers) or any(marker in text for marker in markers):
            dimensions.add(name)
    return dimensions


def analyze_query_coverage(queries: list[str] | None) -> dict[str, Any]:
    hits = {name: 0 for name in _QUERY_DIMENSIONS}
    for query in queries or []:
        for dimension in query_dimensions(query):
            hits[dimension] += 1
    covered = sorted(name for name, count in hits.items() if count)
    missing = sorted(name for name in _QUERY_DIMENSIONS if not hits[name])
    return {
        "score": round(len(covered) / len(_QUERY_DIMENSIONS), 3),
        "covered_dimensions": covered,
        "missing_dimensions": missing,
        "dimension_hits": hits,
        "total_queries": len(queries or []),
    }


def _seed_templates(topic: str, year: int) -> list[dict[str, str]]:
    if _CJK_RE.search(topic):
        return [
            {"dimension": "freshness", "query": f"{topic} 最新进展 {year}"},
            {"dimension": "official", "query": f"{topic} 官方文档 发布说明"},
            {"dimension": "evidence", "query": f"{topic} 数据 报告 评测"},
            {"dimension": "risk", "query": f"{topic} 局限 风险 争议"},
            {"dimension": "implementation", "query": f"{topic} 实践 案例 最佳实践"},
        ]
    return [
        {"dimension": "freshness", "query": f"{topic} latest updates {year}"},
        {"dimension": "official", "query": f"{topic} official documentation release notes"},
        {"dimension": "evidence", "query": f"{topic} benchmark evaluation metrics"},
        {"dimension": "risk", "query": f"{topic} limitations risks tradeoffs"},
        {"dimension": "implementation", "query": f"{topic} implementation best practices case study"},
    ]


def backfill_diverse_queries(
    topic: str,
    existing_queries: list[str] | None,
    historical_queries: list[str] | None,
    query_num: int,
) -> list[str]:
    target = max(1, int(query_num or 1))
    seen = {
        str(query).strip().casefold()
        for query in historical_queries or []
        if str(query).strip()
    }
    final: list[str] = []
    for raw_query in existing_queries or []:
        query = str(raw_query or "").strip()
        key = query.casefold()
        if not query or key in seen:
            continue
        seen.add(key)
        final.append(query)
        if len(final) >= target:
            return final[:target]

    missing = set(analyze_query_coverage(final)["missing_dimensions"])
    seeds = _seed_templates(str(topic or "").strip() or "topic", datetime.now().year)
    ordered = [item for item in seeds if item["dimension"] in missing]
    ordered.extend(item for item in seeds if item["dimension"] not in missing)
    for seed in ordered:
        query = seed["query"].strip()
        if query.casefold() in seen:
            continue
        seen.add(query.casefold())
        final.append(query)
        if len(final) >= target:
            break
    return final[:target]


def merge_query_history(
    existing_queries: list[str] | None,
    new_queries: list[str] | None,
) -> list[str]:
    """Append queries in order while deduplicating case-insensitively."""
    merged: list[str] = []
    seen: set[str] = set()
    for raw_query in [*(existing_queries or []), *(new_queries or [])]:
        query = str(raw_query or "").strip()
        key = query.casefold()
        if query and key not in seen:
            seen.add(key)
            merged.append(query)
    return merged


def new_query_history_entries(
    existing_queries: list[str] | None,
    candidate_queries: list[str] | None,
) -> list[str]:
    """Return only unseen additions for append-based state reducers."""
    seen = {
        query.casefold()
        for query in merge_query_history([], existing_queries)
    }
    additions: list[str] = []
    for query in merge_query_history([], candidate_queries):
        key = query.casefold()
        if key not in seen:
            seen.add(key)
            additions.append(query)
    return additions


def _parse_datetime(value: Any) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        text = str(value).strip()
        if not text:
            return None
        normalized = text[:-1] + "+00:00" if text.endswith("Z") else text
        parsed = None
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%d %H:%M:%S"):
                try:
                    parsed = datetime.strptime(text, fmt)
                    break
                except ValueError:
                    continue
        if parsed is None:
            return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def summarize_freshness(search_runs: list[dict[str, Any]] | None) -> dict[str, Any]:
    total = known = unknown = fresh_7 = fresh_30 = stale_180 = 0
    now = datetime.now(timezone.utc)
    for run in search_runs or []:
        results = run.get("results") if isinstance(run, dict) else []
        if not isinstance(results, list):
            continue
        for result in results:
            total += 1
            value = result.get("published_date") if isinstance(result, dict) else None
            published = _parse_datetime(value)
            if published is None:
                unknown += 1
                continue
            known += 1
            age_days = max(0.0, (now - published).total_seconds() / 86400.0)
            fresh_7 += age_days <= 7
            fresh_30 += age_days <= 30
            stale_180 += age_days > 180
    return {
        "total_results": total,
        "known_count": known,
        "unknown_count": unknown,
        "fresh_7_count": fresh_7,
        "fresh_30_count": fresh_30,
        "stale_180_count": stale_180,
        "fresh_30_ratio": round(fresh_30 / known, 3) if known else 0.0,
        "stale_180_ratio": round(stale_180 / known, 3) if known else 0.0,
    }


def build_quality_diagnostics(
    topic: str,
    queries: list[str] | None,
    search_runs: list[dict[str, Any]] | None,
    *,
    min_known_results: int = 3,
    min_fresh_ratio: float = 0.4,
) -> dict[str, Any]:
    """Build Weaver-compatible query coverage and source freshness diagnostics."""
    query_coverage = analyze_query_coverage(queries)
    freshness_summary = summarize_freshness(search_runs)
    time_sensitive_query = is_time_sensitive_topic(topic)
    min_known = max(1, int(min_known_results or 3))
    fresh_ratio = max(0.0, min(1.0, float(min_fresh_ratio or 0.4)))

    freshness_warning = ""
    if (
        time_sensitive_query
        and freshness_summary["known_count"] >= min_known
        and freshness_summary["fresh_30_ratio"] < fresh_ratio
    ):
        freshness_warning = "low_freshness_for_time_sensitive_query"

    return {
        "query_coverage": query_coverage,
        "query_coverage_score": query_coverage["score"],
        "query_dimensions_covered": query_coverage["covered_dimensions"],
        "query_dimensions_missing": query_coverage["missing_dimensions"],
        "query_dimension_hits": query_coverage["dimension_hits"],
        "freshness_summary": freshness_summary,
        "time_sensitive_query": time_sensitive_query,
        "freshness_warning": freshness_warning,
    }
