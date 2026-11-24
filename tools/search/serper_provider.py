from __future__ import annotations

from typing import Any

import requests

from common.config import settings
from tools.search.base import SearchItem


class SerperProvider:
    """Serper.dev Google Search API 适配器。"""

    name = "serper"
    endpoint = "https://google.serper.dev/search"

    def is_available(self) -> bool:
        return bool(str(getattr(settings, "serper_api_key", "") or "").strip())

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[SearchItem]:
        query = query.strip()
        if not query:
            raise ValueError("搜索关键词不能为空")
        if not self.is_available():
            raise RuntimeError("未配置 SERPER_API_KEY")

        limit = max(1, min(int(max_results), 100))
        response = requests.post(
            self.endpoint,
            headers={
                "X-API-KEY": str(settings.serper_api_key).strip(),
                "Content-Type": "application/json",
            },
            json={"q": query, "num": limit},
            timeout=float(getattr(settings, "search_request_timeout_seconds", 15.0)),
        )
        if not response.ok:
            raise RuntimeError(f"Serper 请求失败（HTTP {response.status_code}）")

        payload = response.json()
        raw_results = payload.get("organic", []) if isinstance(payload, dict) else []
        return [self._normalize_result(item) for item in raw_results]

    def _normalize_result(self, item: Any) -> SearchItem:
        data = item if isinstance(item, dict) else {}
        snippet = str(data.get("snippet") or "")
        return {
            "title": str(data.get("title") or ""),
            "url": str(data.get("link") or data.get("url") or ""),
            "snippet": snippet,
            "content": snippet,
            "provider": self.name,
        }
