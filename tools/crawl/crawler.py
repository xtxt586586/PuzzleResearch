from __future__ import annotations

from typing import Any


def crawl_urls(urls: list[str], timeout: int = 10) -> list[dict[str, Any]]:
    """Adapt the existing crawler to the research fetcher contract."""

    from tools.web_crawler import crawl_urls as legacy_crawl_urls

    results = legacy_crawl_urls(urls, timeout=timeout)
    normalized: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        normalized.append(
            {
                "url": item.get("url", ""),
                "content": item.get("content", item.get("output", "")),
                "error": item.get("error", ""),
            }
        )
    return normalized
