from __future__ import annotations

import logging
from typing import Any

from agent.prompts import get_prompt_manager
from common.config import settings
from tools.registry import ToolDefinition
from tools.search.multi_search import canonicalize_result_url


logger = logging.getLogger(__name__)

DEFAULT_MAX_RESULTS = 5
MAX_RESULTS = 20
MAX_QUERY_CHARS = 1_000
MAX_RAW_EXCERPT_CHARS = 1_200
MAX_SNIPPET_CHARS = 600


def _trim_text(text: str, max_len: int) -> str:
    return str(text or "")[:max_len].strip()


def _validate_search_args(
    query: str,
    max_results: int,
    search_depth: str,
    include_answer: bool,
) -> tuple[str, int, str, bool]:
    normalized_query = " ".join(str(query or "").split())
    if not normalized_query:
        raise ValueError("搜索关键词不能为空。")
    if len(normalized_query) > MAX_QUERY_CHARS:
        raise ValueError(f"搜索关键词不能超过 {MAX_QUERY_CHARS} 个字符。")
    if type(max_results) is not int or not 1 <= max_results <= MAX_RESULTS:
        raise ValueError(f"max_results 必须是 1 到 {MAX_RESULTS} 之间的整数。")
    normalized_depth = str(search_depth or "").strip().lower()
    if normalized_depth not in {"basic", "advanced"}:
        raise ValueError("search_depth 只能是 basic 或 advanced。")
    if type(include_answer) is not bool:
        raise ValueError("include_answer 必须是布尔值。")
    return normalized_query, max_results, normalized_depth, include_answer


def _summarize_content(raw_content: str) -> str | None:
    """使用已配置的 DeepSeek 对首条正文做可选摘要，失败时返回 None。"""

    if not raw_content or not str(getattr(settings, "deepseek_api_key", "")).strip():
        return None
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_openai import ChatOpenAI

        model = ChatOpenAI(
            model=settings.primary_model,
            api_key=settings.deepseek_api_key.strip(),
            base_url=settings.deepseek_base_url,
            timeout=float(settings.model_timeout),
        )
        response = model.invoke(
            [
                SystemMessage(
                    content=get_prompt_manager().get_prompt("search_result_summarizer")
                ),
                HumanMessage(content=_trim_text(raw_content, 3_500)),
            ]
        )
        value = getattr(response, "content", "")
        return str(value or "").strip() or None
    except Exception as exc:
        logger.warning("网页正文摘要失败：%s", type(exc).__name__)
        return None


def _tavily_client(api_key: str) -> Any:
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


def enhanced_tavily_search(
    query: str,
    max_results: int = DEFAULT_MAX_RESULTS,
    search_depth: str = "advanced",
    include_answer: bool = True,
) -> dict[str, Any]:
    """调用 Tavily 的增强搜索接口并返回结构化结果。"""

    normalized_query, normalized_max, depth, with_answer = _validate_search_args(
        query,
        max_results,
        search_depth,
        include_answer,
    )
    api_key = str(getattr(settings, "tavily_api_key", "") or "").strip()
    if not api_key:
        return {
            "output": "",
            "error": "未配置 Tavily API Key。",
            "query": normalized_query,
            "results": [],
            "count": 0,
            "search_depth": depth,
        }

    try:
        response = _tavily_client(api_key).search(
            query=normalized_query,
            search_depth=depth,
            max_results=normalized_max,
            include_answer=with_answer,
            include_raw_content=True,
        )
    except Exception as exc:
        logger.warning("Tavily 搜索失败：%s", type(exc).__name__)
        return {
            "output": "",
            "error": "Tavily 搜索暂时不可用。",
            "query": normalized_query,
            "results": [],
            "count": 0,
            "search_depth": depth,
        }

    normalized_results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    raw_results = response.get("results", []) if isinstance(response, dict) else []
    ranked_results = sorted(
        raw_results if isinstance(raw_results, list) else [],
        key=lambda item: float(item.get("score", 0) or 0)
        if isinstance(item, dict)
        else 0.0,
        reverse=True,
    )

    for item in ranked_results:
        if not isinstance(item, dict):
            continue
        raw_url = str(item.get("url", "")).strip()
        url = canonicalize_result_url(raw_url)
        if not url or url in seen_urls:
            continue
        seen_urls.add(url)
        snippet = _trim_text(item.get("content", ""), MAX_SNIPPET_CHARS)
        raw_content = str(item.get("raw_content") or item.get("content") or "")
        summary = snippet
        if not normalized_results:
            summary = _summarize_content(raw_content) or snippet
        normalized_results.append(
            {
                "title": str(item.get("title", "")).strip() or "无标题",
                "url": url,
                "summary": summary,
                "snippet": snippet,
                "raw_excerpt": _trim_text(raw_content, MAX_RAW_EXCERPT_CHARS),
                "score": float(item.get("score", 0) or 0),
                "provider": "tavily",
            }
        )
        if len(normalized_results) >= normalized_max:
            break

    result: dict[str, Any] = {
        "output": "\n".join(
            f"{index}. {item['title']}\n   链接：{item['url']}\n   摘要：{item['summary']}"
            for index, item in enumerate(normalized_results, start=1)
        ),
        "error": "",
        "query": normalized_query,
        "results": normalized_results,
        "count": len(normalized_results),
        "search_depth": depth,
    }
    if with_answer and isinstance(response, dict) and response.get("answer"):
        result["ai_answer"] = str(response["answer"])
    return result


def tavily_search(query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[dict[str, Any]]:
    """兼容 PuzzleResearch 旧调用方式，返回 Tavily 结果列表。"""

    return enhanced_tavily_search(query, max_results=max_results)["results"]


def build_tavily_search_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="tavily_search",
        description="使用 Tavily 进行深度网页搜索，支持正文截取和可选 AI 答案。",
        handler=enhanced_tavily_search,
    )
