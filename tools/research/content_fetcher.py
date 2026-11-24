from __future__ import annotations

import html
import ipaddress
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from html.parser import HTMLParser
from urllib.parse import urlsplit

import requests

from common.config import settings
from tools.research.models import FetchedPage, truncate_bytes
from tools.research.page_cache import get_fetched_page_cache
from tools.research.reader_client import ReaderClient
from tools.research.source_registry import SourceRegistry


DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0 Safari/537.36"
)


class _VisibleTextParser(HTMLParser):
    _ignored = {"script", "style", "noscript", "svg", "template", "title"}
    _blocks = {"article", "br", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "p", "pre", "section", "td", "th", "tr", "ul", "ol"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self.ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.ignored_depth:
            self.ignored_depth += 1
        elif tag in self._ignored:
            self.ignored_depth = 1
        elif tag in self._blocks:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self.ignored_depth:
            self.ignored_depth -= 1
        elif tag in self._blocks:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self.ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[^\S\n]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        return re.sub(r"\n{3,}", "\n\n", value).strip()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _content_type(headers: object) -> str:
    getter = getattr(headers, "get", None)
    if not callable(getter):
        return ""
    return str(
        getter("content-type")
        or getter("Content-Type")
        or ""
    )


def _title(raw_html: str) -> str | None:
    match = re.search(r"<title[^>]*>(.*?)</title>", raw_html, re.I | re.S)
    if not match:
        match = re.search(r"<h1[^>]*>(.*?)</h1>", raw_html, re.I | re.S)
    if not match:
        return None
    value = re.sub(r"<[^>]+>", " ", match.group(1))
    value = html.unescape(re.sub(r"\s+", " ", value)).strip()
    return value or None


def _html_to_markdown(raw_html: str) -> str:
    """Convert common HTML structure to Markdown without a hard dependency."""

    if not raw_html:
        return ""
    try:
        from markdownify import markdownify

        return str(markdownify(raw_html) or "").strip()
    except Exception:
        value = raw_html
        value = re.sub(
            r"<h([1-6])[^>]*>(.*?)</h\1>",
            lambda match: "\n\n"
            + ("#" * int(match.group(1)))
            + " "
            + match.group(2)
            + "\n\n",
            value,
            flags=re.I | re.S,
        )
        value = re.sub(
            r"<li[^>]*>(.*?)</li>",
            lambda match: "\n- " + match.group(1),
            value,
            flags=re.I | re.S,
        )
        value = re.sub(
            r"<(strong|b)[^>]*>(.*?)</\1>",
            r"**\2**",
            value,
            flags=re.I | re.S,
        )
        value = re.sub(
            r"<(em|i)[^>]*>(.*?)</\1>",
            r"*\2*",
            value,
            flags=re.I | re.S,
        )
        return _strip_html(value)


def _looks_like_javascript_interstitial(text: str) -> bool:
    lowered = str(text or "").lower()
    if not lowered:
        return False
    return any(
        phrase in lowered
        for phrase in (
            "please enable javascript",
            "enable javascript and cookies",
            "checking your browser",
            "just a moment",
            "verify you are human",
        )
    )


def _strip_html(raw_html: str) -> str:
    parser = _VisibleTextParser()
    parser.feed(raw_html)
    parser.close()
    return parser.text()


def _blocked_target(url: str) -> bool:
    parsed = urlsplit(url)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.hostname:
        return True
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost" or host.endswith((".localhost", ".local", ".internal")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return not address.is_global


def _read_bytes(response: object) -> bytes:
    limit = int(getattr(settings, "research_fetch_max_bytes", 0) or 0)
    iterator = getattr(response, "iter_content", None)
    if callable(iterator):
        chunks: list[bytes] = []
        total = 0
        for chunk in iterator(chunk_size=65536):
            if not chunk:
                continue
            data = chunk if isinstance(chunk, bytes) else str(chunk).encode()
            if limit > 0:
                data = data[: max(0, limit - total)]
            chunks.append(data)
            total += len(data)
            if limit > 0 and total >= limit:
                break
        return b"".join(chunks)
    data = getattr(response, "content", b"") or b""
    if isinstance(data, str):
        data = data.encode()
    return truncate_bytes(bytes(data), max_bytes=limit)


def _response_text(
    response: object,
) -> tuple[str, str | None, str | None, str | None]:
    content_type = _content_type(getattr(response, "headers", None)).lower()
    raw = _read_bytes(response)
    text = raw.decode("utf-8", errors="replace")
    if not raw:
        text = str(getattr(response, "text", "") or "")
    if "html" in content_type:
        return (
            _strip_html(text),
            _title(text),
            _html_to_markdown(text)
            if getattr(settings, "research_fetch_extract_markdown", True)
            else None,
            content_type,
        )
    return re.sub(r"\s+", " ", text).strip(), None, None, content_type


def _close(response: object) -> None:
    close = getattr(response, "close", None)
    if callable(close):
        close()


class ContentFetcher:
    def __init__(
        self,
        *,
        reader_mode: str | None = None,
        reader_public_base: str | None = None,
        reader_self_hosted_base: str | None = None,
    ) -> None:
        self.registry = SourceRegistry()
        self.reader_mode = reader_mode if reader_mode is not None else settings.reader_fallback_mode
        self.reader_public_base = reader_public_base if reader_public_base is not None else settings.reader_public_base
        self.reader_self_hosted_base = reader_self_hosted_base if reader_self_hosted_base is not None else settings.reader_self_hosted_base

    def _reader_url(self, canonical_url: str) -> tuple[str, str] | None:
        mode = str(self.reader_mode or "").strip().lower()
        try:
            reader_url = ReaderClient(
                mode=mode,
                public_base=str(self.reader_public_base or ""),
                self_hosted_base=str(self.reader_self_hosted_base or ""),
            ).build_reader_url(canonical_url)
        except ValueError:
            return None
        if mode == "both":
            method = (
                "reader_self_hosted"
                if self.reader_self_hosted_base
                else "reader_public"
            )
        else:
            method = f"reader_{mode}"
        return reader_url, method

    def _request(self, url: str) -> object:
        return requests.get(
            url,
            timeout=float(getattr(settings, "research_fetch_timeout_s", 25.0)),
            headers={"User-Agent": DEFAULT_UA},
            stream=True,
        )

    def _reader_fallback(self, canonical_url: str, raw_url: str, attempts: int) -> FetchedPage | None:
        target = self._reader_url(canonical_url)
        if target is None:
            return None
        reader_url, method = target
        try:
            response = self._request(reader_url)
            try:
                text, title, markdown, _ = _response_text(response)
                status = int(getattr(response, "status_code", 0) or 0)
            finally:
                _close(response)
        except Exception:
            return None
        if status != 200 or not text:
            return None
        return FetchedPage(
            url=canonical_url,
            raw_url=raw_url,
            method=method,
            text=text,
            title=title,
            markdown=markdown,
            http_status=status,
            retrieved_at=_now(),
            attempts=attempts,
        )

    def _render_fallback(self, canonical_url: str, raw_url: str, attempts: int) -> FetchedPage | None:
        try:
            from tools.crawl.crawler import crawl_urls

            results = crawl_urls(
                [canonical_url],
                timeout=max(
                    1,
                    int(float(getattr(settings, "research_fetch_timeout_s", 25.0))),
                ),
            )
            result = results[0] if results else {}
            text = str(
                result.get("content", result.get("output", ""))
                if isinstance(result, dict)
                else ""
            ).strip()
        except Exception:
            return None
        if not text or text.startswith(("抓取失败", "Crawl failed", "Exception:")):
            return None
        return FetchedPage(
            url=canonical_url,
            raw_url=raw_url,
            method="render_crawler",
            text=text,
            http_status=200,
            retrieved_at=_now(),
            attempts=attempts,
        )

    def fetch(self, url: str) -> FetchedPage:
        raw_url = str(url or "").strip()
        canonical_url = self.registry.canonicalize_url(raw_url)
        if not canonical_url:
            return FetchedPage(url="", raw_url=raw_url, method="direct_http", error="url is required")
        if _blocked_target(canonical_url):
            return FetchedPage(url=canonical_url, raw_url=raw_url, method="direct_http", error="blocked fetch target url")

        cache = get_fetched_page_cache()
        render_mode = str(
            getattr(settings, "research_fetch_render_mode", "off") or "off"
        ).lower()
        cache_key = f"{canonical_url}::render={render_mode}::reader={str(self.reader_mode).lower()}"
        if cache is not None:
            cached = cache.get(cache_key)
            if cached is not None:
                cached.raw_url = raw_url
                return cached

        direct = FetchedPage(
            url=canonical_url,
            raw_url=raw_url,
            method="direct_http",
            attempts=1,
        )
        try:
            response = self._request(canonical_url)
            try:
                direct.http_status = int(getattr(response, "status_code", 0) or 0)
                (
                    direct.text,
                    direct.title,
                    direct.markdown,
                    content_type,
                ) = _response_text(response)
            finally:
                _close(response)
            if direct.http_status != 200:
                direct.error = f"HTTP {direct.http_status}"
        except Exception as exc:
            content_type = ""
            direct.error = str(exc)
        direct.retrieved_at = _now()

        min_chars = int(
            getattr(settings, "research_fetch_render_min_chars", 200) or 200
        )
        needs_render = render_mode == "always" or (
            render_mode == "auto"
            and "html" in content_type
            and (
                len((direct.text or "").strip()) < max(1, min_chars)
                or _looks_like_javascript_interstitial(direct.text or "")
            )
        )
        if render_mode != "off" and (direct.http_status != 200 or not direct.text):
            needs_render = True
        final = self._render_fallback(canonical_url, raw_url, 2) if needs_render else None
        if final is None and (direct.http_status != 200 or not (direct.text or "").strip()):
            final = self._reader_fallback(canonical_url, raw_url, 2)
        final = final or direct

        if cache is not None and (final.text or final.markdown):
            cache.set(cache_key, final)
        return final

    def fetch_many(self, urls: list[str]) -> list[FetchedPage]:
        candidates: list[str] = []
        seen: set[str] = set()
        for raw_url in urls or []:
            canonical = self.registry.canonicalize_url(str(raw_url or ""))
            if canonical and canonical not in seen:
                seen.add(canonical)
                candidates.append(canonical)

        workers = max(1, int(getattr(settings, "research_fetch_concurrency", 6) or 6))
        per_domain = int(getattr(settings, "research_fetch_concurrency_per_domain", 2) or 2)
        if per_domain <= 0:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                return list(executor.map(self.fetch, candidates))

        import threading
        semaphores: dict[str, threading.Semaphore] = {}
        lock = threading.RLock()

        def fetch_limited(target: str) -> FetchedPage:
            domain = urlsplit(target).netloc.lower()
            with lock:
                semaphore = semaphores.setdefault(domain, threading.Semaphore(per_domain))
            with semaphore:
                return self.fetch(target)

        with ThreadPoolExecutor(max_workers=workers) as executor:
            return list(executor.map(fetch_limited, candidates))
