from __future__ import annotations

from typing import Any

import requests

from common.config import settings
from tools.search.base import SearchItem


class BraveProvider:
    """Brave Search API 适配器。"""

    name = "brave"
    endpoint = "https://api.search.brave.com/res/v1/web/search"

    def is_available(self) -> bool:
        return bool(str(getattr(settings, "brave_api_key", "") or "").strip())

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchItem]:
        query = query.strip()
        if not query:
            raise ValueError("搜索关键词不能为空")
        if not self.is_available():
            raise RuntimeError("未配置 BRAVE_API_KEY")

        limit = max(1, min(int(max_results), 20))
        response = requests.get(
            self.endpoint,
            headers={
                "Accept": "application/json",
                "X-Subscription-Token": str(settings.brave_api_key).strip(),
            },
            params={"q": query, "count": limit},
            timeout=float(getattr(settings, "search_request_timeout_seconds", 15.0)),
        )
        if not response.ok:
            raise RuntimeError(f"Brave Search 请求失败（HTTP {response.status_code}）")

        payload = response.json()
        web = payload.get("web", {}) if isinstance(payload, dict) else {}
        raw_results = web.get("results", []) if isinstance(web, dict) else []
        return [self._normalize_result(item) for item in raw_results]

    def _normalize_result(self, item: Any) -> SearchItem:
        data = item if isinstance(item, dict) else {}
        extra = data.get("extra_snippets")
        content = str(data.get("description") or "")
        if isinstance(extra, list):
            content = "\n".join(
                [content, *[str(value) for value in extra if value]]
            ).strip()
        return {
            "title": str(data.get("title") or ""),
            "url": str(data.get("url") or ""),
            "snippet": str(data.get("description") or ""),
            "content": content,
            "provider": self.name,
        }
