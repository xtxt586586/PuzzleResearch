from __future__ import annotations

from typing import Any

import requests

from common.config import settings
from tools.search.base import SearchItem


class RedditProvider:
    """Reddit OAuth API 适配器，不依赖 PRAW。"""

    name = "reddit"
    token_endpoint = "https://www.reddit.com/api/v1/access_token"
    api_endpoint = "https://oauth.reddit.com"

    def __init__(self) -> None:
        self._access_token = ""

    def is_available(self) -> bool:
        return bool(
            str(getattr(settings, "reddit_client_id", "") or "").strip()
            and str(getattr(settings, "reddit_client_secret", "") or "").strip()
        )

    def search(
        self,
        query: str,
        max_results: int = 5,
        subreddit: str | None = None,
        sort: str = "relevance",
        time_filter: str = "all",
    ) -> list[SearchItem]:
        query = query.strip()
        if not query:
            raise ValueError("搜索关键词不能为空")
        if not self.is_available():
            raise RuntimeError("未配置 REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET")

        path = f"/r/{subreddit.strip()}/search" if subreddit else "/r/all/search"
        payload = self._request_json(
            path,
            params={
                "q": query,
                "sort": sort,
                "t": time_filter,
                "limit": max(1, min(int(max_results), 100)),
                "restrict_sr": "on" if subreddit else "off",
            },
        )
        children = payload.get("data", {}).get("children", []) if isinstance(payload, dict) else []
        return [self._normalize(item.get("data")) for item in children if isinstance(item, dict)]

    def get_hot_posts(self, subreddit: str = "all", max_results: int = 10) -> list[SearchItem]:
        if not self.is_available():
            raise RuntimeError("未配置 REDDIT_CLIENT_ID/REDDIT_CLIENT_SECRET")
        payload = self._request_json(
            f"/r/{subreddit.strip() or 'all'}/hot",
            params={"limit": max(1, min(int(max_results), 100))},
        )
        children = payload.get("data", {}).get("children", []) if isinstance(payload, dict) else []
        return [self._normalize(item.get("data")) for item in children if isinstance(item, dict)]

    def _request_json(self, path: str, params: dict[str, Any]) -> Any:
        response = requests.get(
            f"{self.api_endpoint}{path}",
            params=params,
            headers={
                "Authorization": f"bearer {self._get_access_token()}",
                "User-Agent": str(getattr(settings, "reddit_user_agent", "PuzzleResearch/1.0")),
            },
            timeout=float(getattr(settings, "search_request_timeout_seconds", 15.0)),
        )
        if not response.ok:
            raise RuntimeError(f"Reddit 请求失败（HTTP {response.status_code}）")
        return response.json()

    def _get_access_token(self) -> str:
        if self._access_token:
            return self._access_token
        response = requests.post(
            self.token_endpoint,
            auth=(
                str(settings.reddit_client_id).strip(),
                str(settings.reddit_client_secret).strip(),
            ),
            data={"grant_type": "client_credentials"},
            headers={
                "User-Agent": str(getattr(settings, "reddit_user_agent", "PuzzleResearch/1.0")),
            },
            timeout=float(getattr(settings, "search_request_timeout_seconds", 15.0)),
        )
        if not response.ok:
            raise RuntimeError(f"Reddit OAuth 请求失败（HTTP {response.status_code}）")
        payload = response.json()
        token = str(payload.get("access_token") or "") if isinstance(payload, dict) else ""
        if not token:
            raise RuntimeError("Reddit OAuth 未返回访问令牌")
        self._access_token = token
        return token

    def _normalize(self, item: Any) -> SearchItem:
        data = item if isinstance(item, dict) else {}
        permalink = str(data.get("permalink") or "")
        body = str(data.get("selftext") or "")
        title = str(data.get("title") or "")
        content = body or str(data.get("url") or "")
        metadata = [
            f"版块：{data.get('subreddit')}" if data.get("subreddit") else "",
            f"作者：{data.get('author')}" if data.get("author") else "",
            f"赞数：{data.get('score')}" if data.get("score") is not None else "",
            f"评论数：{data.get('num_comments')}" if data.get("num_comments") is not None else "",
        ]
        return {
            "title": title,
            "url": f"https://www.reddit.com{permalink}" if permalink.startswith("/") else permalink,
            "snippet": body[:500] if body else title,
            "content": "\n".join([content, *[item for item in metadata if item]]).strip(),
            "provider": self.name,
        }
