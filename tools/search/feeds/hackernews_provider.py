from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import requests

from common.config import settings
from tools.search.base import SearchItem


class HackerNewsProvider:
    """Hacker News Algolia/Firebase API 适配器。"""

    name = "hackernews"
    search_endpoint = "https://hn.algolia.com/api/v1/search"
    date_search_endpoint = "https://hn.algolia.com/api/v1/search_by_date"
    top_stories_endpoint = "https://hacker-news.firebaseio.com/v0/topstories.json"
    item_endpoint = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"

    def is_available(self) -> bool:
        return bool(getattr(settings, "hackernews_enabled", True))

    def search(
        self,
        query: str,
        max_results: int = 5,
        tags: str | None = None,
        sort_by_date: bool = False,
        min_points: int = 0,
    ) -> list[SearchItem]:
        query = query.strip()
        if not query and not tags:
            raise ValueError("搜索关键词不能为空")
        if not self.is_available():
            raise RuntimeError("Hacker News Provider 已禁用")

        params: dict[str, Any] = {
            "query": query,
            "hitsPerPage": max(1, min(int(max_results), 50)),
        }
        if tags:
            params["tags"] = tags
        if min_points > 0:
            params["numericFilters"] = f"points>={int(min_points)}"
        response = requests.get(
            self.date_search_endpoint if sort_by_date else self.search_endpoint,
            params=params,
            headers={"User-Agent": "PuzzleResearch/1.0"},
            timeout=float(getattr(settings, "search_request_timeout_seconds", 10.0)),
        )
        self._ensure_ok(response)
        payload = response.json()
        hits = payload.get("hits", []) if isinstance(payload, dict) else []
        return [self._normalize_hit(item) for item in hits]

    def get_top_stories(self, max_results: int = 30) -> list[SearchItem]:
        if not self.is_available():
            raise RuntimeError("Hacker News Provider 已禁用")
        response = requests.get(
            self.top_stories_endpoint,
            headers={"User-Agent": "PuzzleResearch/1.0"},
            timeout=float(getattr(settings, "search_request_timeout_seconds", 10.0)),
        )
        self._ensure_ok(response)
        story_ids = response.json()[: max(1, min(int(max_results), 30))]
        results: list[SearchItem] = []
        for story_id in story_ids:
            item_response = requests.get(
                self.item_endpoint.format(item_id=story_id),
                headers={"User-Agent": "PuzzleResearch/1.0"},
                timeout=5.0,
            )
            if not item_response.ok:
                continue
            item = item_response.json()
            if isinstance(item, dict) and item.get("type") == "story":
                results.append(self._normalize_item(item))
        return results

    def get_ask_hn(self, max_results: int = 20) -> list[SearchItem]:
        return self.search("", max_results=max_results, tags="ask_hn", sort_by_date=True)

    def get_show_hn(self, max_results: int = 20) -> list[SearchItem]:
        return self.search("", max_results=max_results, tags="show_hn", sort_by_date=True)

    def _normalize_hit(self, item: Any) -> SearchItem:
        data = item if isinstance(item, dict) else {}
        is_story = "story" in (data.get("_tags") or [])
        object_id = str(data.get("objectID") or "")
        title = str(data.get("title") or "")
        if is_story:
            url = str(data.get("url") or f"https://news.ycombinator.com/item?id={object_id}")
            snippet = title
        else:
            author = str(data.get("author") or "unknown")
            title = f"评论：{author}"
            url = f"https://news.ycombinator.com/item?id={object_id}"
            snippet = str(data.get("comment_text") or "")[:500]
        content = str(data.get("comment_text") or data.get("story_text") or "")
        metadata = [
            f"作者：{data.get('author')}" if data.get("author") else "",
            f"积分：{data.get('points')}" if data.get("points") is not None else "",
            f"评论数：{data.get('num_comments')}" if data.get("num_comments") is not None else "",
        ]
        return {
            "title": title,
            "url": url,
            "snippet": snippet,
            "content": "\n".join([content, *[item for item in metadata if item]]).strip(),
            "provider": self.name,
        }

    def _normalize_item(self, item: dict[str, Any]) -> SearchItem:
        story_id = str(item.get("id") or "")
        return {
            "title": str(item.get("title") or ""),
            "url": str(item.get("url") or f"https://news.ycombinator.com/item?id={story_id}"),
            "snippet": str(item.get("title") or ""),
            "content": str(item.get("text") or ""),
            "provider": self.name,
        }

    @staticmethod
    def _ensure_ok(response: Any) -> None:
        if not response.ok:
            raise RuntimeError(f"Hacker News 请求失败（HTTP {response.status_code}）")
