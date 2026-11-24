from __future__ import annotations

from typing import Any

from tools.registry import ToolDefinition
from tools.search.manager import (
    SearchManager,
    get_default_search_runtime_state,
)


DEFAULT_MAX_RESULTS = 5
MAX_SEARCH_RESULTS = 10
MAX_QUERY_CHARS = 1_000
MAX_SNIPPET_CHARS = 800


def web_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
) -> dict[str, Any]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise ValueError("搜索关键词不能为空。")
    if len(normalized_query) > MAX_QUERY_CHARS:
        raise ValueError(f"搜索关键词不能超过 {MAX_QUERY_CHARS} 个字符。")
    if type(max_results) is not int or not 1 <= max_results <= MAX_SEARCH_RESULTS:
        raise ValueError(
            f"max_results 必须是 1 到 {MAX_SEARCH_RESULTS} 之间的整数。"
        )

    manager = SearchManager(
        runtime_state=get_default_search_runtime_state()
    )
    raw_results = manager.search(
        normalized_query,
        max_results=max_results,
    )

    results = [
        {
            "title": str(item.get("title", "")).strip() or "无标题",
            "url": str(item.get("url", "")).strip(),
            "snippet": str(item.get("snippet", "")).strip()[:MAX_SNIPPET_CHARS],
            "provider": str(item.get("provider", "")).strip(),
        }
        for item in raw_results[:max_results]
    ]

    lines = [f"搜索关键词：{normalized_query}"]
    for index, item in enumerate(results, start=1):
        title = item["title"]
        url = item["url"]
        snippet = item["snippet"]
        provider = item["provider"]
        lines.extend(
            [
                f"{index}. {title}",
                f"   链接：{url}" if url else "   链接：无",
                f"   摘要：{snippet}" if snippet else "   摘要：无",
                f"   来源：{provider}" if provider else "",
            ]
        )

    return {
        "output": "\n".join(line for line in lines if line),
        "error": "",
        "query": normalized_query,
        "results": results,
    }


def build_web_search_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="web_search",
        description=(
            "使用已配置的搜索 Provider 查找公开网页资料，按配置顺序自动回退。"
            f"max_results 默认为 {DEFAULT_MAX_RESULTS}，范围为 1 到 {MAX_SEARCH_RESULTS}。"
        ),
        handler=web_search,
    )
