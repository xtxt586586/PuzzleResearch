from __future__ import annotations

from typing import Any

import requests

from common.config import settings
from tools.search.base import SearchItem


class TwitterProvider:
    """Twitter/X API v2 最近搜索适配器。"""

    name = "twitter"
    endpoint = "https://api.twitter.com/2/tweets/search/recent"

    def is_available(self) -> bool:
        return bool(str(getattr(settings, "twitter_bearer_token", "") or "").strip())

    def search(self, query: str, max_results: int = 10) -> list[SearchItem]:
        query = query.strip()
        if not query:
            raise ValueError("搜索关键词不能为空")
        if not self.is_available():
            raise RuntimeError("未配置 TWITTER_BEARER_TOKEN")

        response = requests.get(
            self.endpoint,
            params={
                "query": query,
                "max_results": max(10, min(int(max_results), 100)),
                "tweet.fields": "author_id,created_at,public_metrics,text",
                "expansions": "author_id",
                "user.fields": "name,username",
            },
            headers={
                "Authorization": f"Bearer {str(settings.twitter_bearer_token).strip()}"
            },
            timeout=float(getattr(settings, "search_request_timeout_seconds", 15.0)),
        )
        if not response.ok:
            raise RuntimeError(f"Twitter 请求失败（HTTP {response.status_code}）")
        payload = response.json()
        users = {
            str(user.get("id")): user
            for user in (payload.get("includes", {}).get("users", []) if isinstance(payload, dict) else [])
            if isinstance(user, dict) and user.get("id")
        }
        return [
            self._normalize(tweet, users.get(str(tweet.get("author_id"))))
            for tweet in (payload.get("data", []) if isinstance(payload, dict) else [])
            if isinstance(tweet, dict)
        ]

    def _normalize(self, tweet: dict[str, Any], user: dict[str, Any] | None) -> SearchItem:
        tweet_id = str(tweet.get("id") or "")
        text = str(tweet.get("text") or "")
        author = ""
        if user:
            author = str(user.get("username") or user.get("name") or "")
        return {
            "title": f"@{author}：{text[:100]}" if author else text[:100],
            "url": f"https://x.com/i/web/status/{tweet_id}" if tweet_id else "",
            "snippet": text[:500],
            "content": f"作者：@{author}\n{text}" if author else text,
            "provider": self.name,
        }
