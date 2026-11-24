from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse

import httpx


DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
)


def _title(html: str) -> str:
    match = re.search(
        r"<title[^>]*>(.*?)</title>",
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return ""

    return re.sub(
        r"\s+",
        " ",
        match.group(1),
    ).strip()[:200]


def _text(html: str) -> str:
    value = re.sub(
        r"<(script|style|noscript)[^>]*>.*?</\1>",
        " ",
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    )
    value = re.sub(r"<[^>]+>", " ", value)

    return re.sub(
        r"\s+",
        " ",
        value,
    ).strip()[:20000]


def _links(
    html: str,
    base_url: str,
) -> list[dict[str, str]]:
    result = []
    seen: set[str] = set()

    for match in re.finditer(
        r"<a\s+[^>]*href=[\"']([^\"']+)[\"'][^>]*>(.*?)</a>",
        html or "",
        flags=re.IGNORECASE | re.DOTALL,
    ):
        href = match.group(1).strip()

        if not href or href.startswith("#"):
            continue

        absolute_url = urljoin(
            base_url,
            href,
        )

        if urlparse(absolute_url).scheme not in {
            "http",
            "https",
        }:
            continue

        if absolute_url in seen:
            continue

        seen.add(absolute_url)

        result.append(
            {
                "text": re.sub(
                    r"\s+",
                    " ",
                    re.sub(
                        r"<[^>]+>",
                        " ",
                        match.group(2),
                    ),
                ).strip()[:200],
                "url": absolute_url,
            }
        )

        if len(result) >= 30:
            break

    return result


@dataclass
class BrowserPage:
    url: str
    title: str
    text: str
    links: list[dict[str, str]]
    fetched_at: str


class BrowserSession:
    def __init__(
        self,
        *,
        timeout_seconds: float = 20.0,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.current: BrowserPage | None = None
        self.history: list[BrowserPage] = []
        self._client: httpx.Client | None = None

    def _get_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=True,
                headers={
                    "User-Agent": DEFAULT_USER_AGENT,
                },
            )

        return self._client

    def navigate(self, url: str) -> BrowserPage:
        normalized_url = str(url or "").strip()

        if not normalized_url:
            raise ValueError("url 不能为空。")

        if urlparse(normalized_url).scheme not in {
            "http",
            "https",
        }:
            raise ValueError(
                "只支持 http 和 https URL。"
            )

        response = self._get_client().get(
            normalized_url
        )
        response.raise_for_status()

        page = BrowserPage(
            url=str(response.url),
            title=_title(response.text),
            text=_text(response.text),
            links=_links(
                response.text,
                str(response.url),
            ),
            fetched_at=datetime.now(
                timezone.utc
            ).isoformat(),
        )

        if self.current is not None:
            self.history.append(self.current)

        self.current = page
        return page

    def search(
        self,
        query: str,
        *,
        engine: str = "duckduckgo",
    ) -> BrowserPage:
        normalized_query = str(query or "").strip()

        if not normalized_query:
            raise ValueError("query 不能为空。")

        normalized_engine = (
            engine or "duckduckgo"
        ).strip().lower()

        if normalized_engine in {
            "duckduckgo",
            "duck",
            "ddg",
        }:
            url = (
                "https://html.duckduckgo.com/html/"
                f"?q={quote_plus(normalized_query)}"
            )
        elif normalized_engine == "bing":
            url = (
                "https://www.bing.com/search"
                f"?q={quote_plus(normalized_query)}"
            )
        else:
            raise ValueError(
                f"不支持的搜索引擎：{engine}"
            )

        return self.navigate(url)

    def back(self) -> BrowserPage:
        if not self.history:
            raise ValueError("没有可返回的历史页面。")

        self.current = self.history.pop()
        return self.current

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None


class BrowserSessionManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, BrowserSession] = {}

    @staticmethod
    def _normalize(thread_id: str) -> str:
        value = str(thread_id or "").strip()
        return value or "default"

    def get(self, thread_id: str) -> BrowserSession:
        normalized = self._normalize(thread_id)

        with self._lock:
            if normalized not in self._sessions:
                self._sessions[normalized] = (
                    BrowserSession()
                )

            return self._sessions[normalized]

    def peek(
        self,
        thread_id: str,
    ) -> BrowserSession | None:
        normalized = self._normalize(thread_id)

        with self._lock:
            return self._sessions.get(normalized)

    def reset(self, thread_id: str) -> None:
        normalized = self._normalize(thread_id)

        with self._lock:
            session = self._sessions.pop(
                normalized,
                None,
            )

        if session is not None:
            session.close()

    def reset_all(self) -> int:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            try:
                session.close()
            except Exception:
                pass

        return len(sessions)


browser_sessions = BrowserSessionManager()
