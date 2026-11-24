from __future__ import annotations

import logging
from typing import Any, Iterable

from common.config import settings
from tools.search.base import SearchItem
from tools.search.manager import SearchManager, get_default_search_runtime_state
from tools.search.multi_search import canonicalize_result_url
from tools.registry import ToolDefinition


logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
MAX_SEARCH_RESULTS = 10
MAX_QUERY_CHARS = 1_000
MAX_ENGINE_COUNT = 10
MAX_MULTI_QUERIES = 10

_ENGINE_ALIASES = {
    "google": "google_cse",
    "googlecse": "google_cse",
    "google_custom_search": "google_cse",
    "ddg": "duckduckgo",
    "semantic-scholar": "semantic_scholar",
    "pub_med": "pubmed",
    "hacker_news": "hackernews",
}


def _normalize_query(query: str) -> str:
    normalized = " ".join(str(query or "").split())
    if not normalized:
        raise ValueError("搜索关键词不能为空。")
    if len(normalized) > MAX_QUERY_CHARS:
        raise ValueError(f"搜索关键词不能超过 {MAX_QUERY_CHARS} 个字符。")
    return normalized


def _normalize_max_results(max_results: int) -> int:
    if type(max_results) is not int or not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise ValueError(
            f"max_results 必须是 1 到 {MAX_SEARCH_RESULTS} 之间的整数。"
        )
    return max_results


def _normalize_engines(engines: list[str] | None) -> list[str] | None:
    if engines is None:
        return None
    if not isinstance(engines, list):
        raise ValueError("engines 必须是搜索 Provider 名称列表。")
    if len(engines) > MAX_ENGINE_COUNT:
        raise ValueError(f"engines 不能超过 {MAX_ENGINE_COUNT} 个 Provider。")

    normalized: list[str] = []
    for engine in engines:
        value = str(engine or "").strip().lower()
        if not value:
            continue
        normalized.append(_ENGINE_ALIASES.get(value, value))
    return normalized


def _result_dict(item: Any, provider: str) -> dict[str, Any]:
    data = item if isinstance(item, dict) else {}
    return {
        "title": str(data.get("title", "")).strip() or "无标题",
        "url": str(data.get("url", "")).strip(),
        "snippet": str(data.get("snippet", "")).strip(),
        "content": str(data.get("content", "")).strip(),
        "provider": str(data.get("provider", "")).strip() or provider,
    }


def run_fallback_search(
    *,
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    engines: list[str] | None = None,
) -> tuple[str | None, list[dict[str, Any]]]:
    """按指定顺序执行 Provider，返回首个成功 Provider 及其结果。"""

    normalized_query = _normalize_query(query)
    normalized_max_results = _normalize_max_results(max_results)
    selected_engines = _normalize_engines(engines)
    manager = SearchManager(runtime_state=get_default_search_runtime_state())
    engine_list = selected_engines or settings.search_engines_list or ["tavily"]

    for engine in engine_list:
        provider = getattr(manager, "providers", {}).get(engine)
        if provider is None:
            logger.info("跳过未知搜索 Provider：%s", engine)
            continue
        try:
            if not provider.is_available():
                continue
            raw_results = provider.search(
                query=normalized_query,
                max_results=normalized_max_results,
            )
            results = [
                _result_dict(item, engine)
                for item in (raw_results or [])
            ]
            results = [item for item in results if item["url"] or item["snippet"]]
            if results:
                return engine, results[:normalized_max_results]
        except Exception as exc:
            # Provider 异常只影响当前候选，具体错误不返回给调用方，避免泄露配置或密钥。
            logger.warning("搜索 Provider %s 执行失败：%s", engine, type(exc).__name__)
            continue

    return None, []


def _format_results(title: str, results: Iterable[dict[str, Any]]) -> str:
    lines = [title]
    for index, item in enumerate(results, start=1):
        title_value = str(item.get("title", "")).strip() or "无标题"
        url = str(item.get("url", "")).strip()
        snippet = str(item.get("snippet", "")).strip()
        provider = str(item.get("provider", "")).strip()
        lines.extend(
            [
                f"{index}. {title_value}",
                f"   链接：{url}" if url else "   链接：无",
                (
                    f"   摘要：{snippet}"
                    if snippet
                    else "   摘要：无"
                ),
                f"   来源：{provider}" if provider else "",
            ]
        )
    return "\n".join(line for line in lines if line)


def fallback_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    """执行单查询 Provider 回退搜索。"""

    normalized_query = _normalize_query(query)
    engine, results = run_fallback_search(
        query=normalized_query,
        max_results=max_results,
        engines=engines,
    )
    return {
        "output": _format_results(
            f"搜索关键词：{normalized_query}",
            results,
        ),
        "error": "" if engine else "没有可用的搜索 Provider。",
        "query": normalized_query,
        "engine": engine or "",
        "results": results,
    }


def search_multiple_queries(
    queries: list[str],
    max_results: int = DEFAULT_MAX_RESULTS,
    engines: list[str] | None = None,
) -> dict[str, Any]:
    """逐个执行多个查询；单个查询失败不会中断其余查询。"""

    if not isinstance(queries, list) or not queries:
        raise ValueError("queries 必须是非空的查询列表。")
    if len(queries) > MAX_MULTI_QUERIES:
        raise ValueError(f"queries 不能超过 {MAX_MULTI_QUERIES} 条。")
    normalized_max_results = _normalize_max_results(max_results)
    normalized_engines = _normalize_engines(engines)

    all_results: list[dict[str, Any]] = []
    query_reports: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    seen: set[str] = set()

    for raw_query in queries:
        try:
            normalized_query = _normalize_query(raw_query)
            engine, results = run_fallback_search(
                query=normalized_query,
                max_results=normalized_max_results,
                engines=normalized_engines,
            )
        except ValueError as exc:
            errors.append({"query": str(raw_query), "error": str(exc)})
            continue
        except Exception:
            errors.append({"query": str(raw_query), "error": "查询执行失败。"})
            continue

        query_reports.append(
            {
                "query": normalized_query,
                "engine": engine or "",
                "result_count": len(results),
            }
        )
        if not engine:
            errors.append(
                {"query": normalized_query, "error": "没有可用的搜索 Provider。"}
            )
            continue

        for result in results:
            canonical_url = canonicalize_result_url(result["url"])
            dedupe_key = canonical_url or (
                f"text:{result['title'].lower()}:{result['snippet'].lower()}"
            )
            if dedupe_key in seen:
                continue
            seen.add(dedupe_key)
            all_results.append({**result, "query": normalized_query})

    output = _format_results("多查询搜索结果", all_results)
    if errors:
        output += "\n\n部分查询未完成：" + "；".join(
            f"{item['query']}（{item['error']}）" for item in errors
        )
    return {
        "output": output,
        "error": "" if not errors else "部分查询未完成。",
        "queries": [item["query"] for item in query_reports],
        "query_reports": query_reports,
        "errors": errors,
        "results": all_results,
    }


def build_fallback_search_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="fallback_search",
        description="按指定 Provider 顺序搜索，并在 Provider 无结果或失败时自动回退。",
        handler=fallback_search,
    )


def build_multiple_search_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="search_multiple_queries",
        description="逐个执行多个搜索查询，隔离单个查询失败并去重合并结果。",
        handler=search_multiple_queries,
    )
