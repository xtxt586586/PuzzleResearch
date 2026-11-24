from __future__ import annotations

import ipaddress
import re
import socket
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit, urlunsplit

import httpx

from tools.registry import ToolDefinition


DEFAULT_TIMEOUT_SECONDS = 10
MAX_TIMEOUT_SECONDS = 60
DEFAULT_MAX_CHARS = 4_000
MAX_MAX_CHARS = 20_000
DEFAULT_MAX_CHARS_PER_URL = 2_000
MAX_URLS_PER_CALL = 10
MAX_REDIRECTS = 5
MAX_RESPONSE_BYTES = 2_000_000
_USER_AGENT = "PuzzleResearch/1.0 (+web page text extraction)"
_REDIRECT_STATUSES = {301, 302, 303, 307, 308}


class _VisibleTextParser(HTMLParser):
    _ignored_tags = {"script", "style", "noscript", "svg", "template"}
    _block_tags = {
        "address", "article", "blockquote", "br", "dd", "div", "dl",
        "dt", "figcaption", "h1", "h2", "h3", "h4", "h5", "h6",
        "hr", "li", "main", "ol", "p", "pre", "section", "table",
        "td", "th", "tr", "ul",
    }

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._ignored_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self._ignored_depth:
            self._ignored_depth += 1
        elif tag in self._ignored_tags:
            self._ignored_depth = 1
        elif tag in self._block_tags:
            self.parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if self._ignored_depth:
            self._ignored_depth -= 1
        elif tag in self._block_tags:
            self.parts.append("\n")

    def handle_data(self, data: str) -> None:
        if not self._ignored_depth:
            self.parts.append(data)

    def text(self) -> str:
        value = "".join(self.parts)
        value = re.sub(r"[^\S\n]+", " ", value)
        value = re.sub(r" *\n *", "\n", value)
        return re.sub(r"\n{3,}", "\n\n", value).strip()


def _resolve_host_addresses(host: str, port: int) -> set[str]:
    try:
        records = socket.getaddrinfo(
            host,
            port,
            type=socket.SOCK_STREAM,
        )
    except OSError as exc:
        raise ValueError(f"无法解析目标主机：{host}") from exc
    return {record[4][0].split("%", 1)[0] for record in records}


def _validate_public_url(url: str) -> str:
    candidate = str(url or "").strip()
    try:
        parsed = urlsplit(candidate)
        scheme = parsed.scheme.lower()
        host = parsed.hostname
        port = parsed.port
    except ValueError as exc:
        raise ValueError("URL 格式无效。") from exc

    if scheme not in {"http", "https"} or not host:
        raise ValueError("只支持包含主机名的 HTTP 或 HTTPS URL。")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("URL 不允许包含用户名或密码。")

    normalized_host = host.rstrip(".").lower()
    if normalized_host == "localhost" or normalized_host.endswith(
        (".localhost", ".local", ".internal")
    ):
        raise ValueError("不允许抓取本机或内部网络地址。")

    try:
        addresses = {str(ipaddress.ip_address(normalized_host))}
    except ValueError:
        addresses = _resolve_host_addresses(
            normalized_host,
            port or (443 if scheme == "https" else 80),
        )

    if not addresses or any(
        not ipaddress.ip_address(address).is_global
        for address in addresses
    ):
        raise ValueError("不允许抓取本机或内部网络地址。")

    return urlunsplit(
        (scheme, parsed.netloc, parsed.path or "/", parsed.query, "")
    )


def _validate_limits(timeout: int, max_chars: int) -> None:
    if type(timeout) is not int or not 1 <= timeout <= MAX_TIMEOUT_SECONDS:
        raise ValueError(f"timeout 必须是 1 到 {MAX_TIMEOUT_SECONDS} 之间的整数。")
    if type(max_chars) is not int or not 200 <= max_chars <= MAX_MAX_CHARS:
        raise ValueError(f"max_chars 必须是 200 到 {MAX_MAX_CHARS} 之间的整数。")


def _create_client(timeout: int) -> httpx.Client:
    return httpx.Client(
        timeout=timeout,
        follow_redirects=False,
        trust_env=False,
        headers={"User-Agent": _USER_AGENT},
    )


def _read_limited_body(response: httpx.Response) -> bytes:
    body = bytearray()
    for chunk in response.iter_bytes():
        remaining = MAX_RESPONSE_BYTES - len(body)
        if remaining <= 0:
            break
        body.extend(chunk[:remaining])
        if len(chunk) > remaining:
            break
    return bytes(body)


def _decode_body(response: httpx.Response) -> str:
    encoding = response.encoding or "utf-8"
    body = _read_limited_body(response)
    try:
        return body.decode(encoding, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _extract_text(response: httpx.Response) -> str:
    content_type = response.headers.get("content-type", "").lower()
    raw_text = _decode_body(response)

    if "text/plain" in content_type:
        return re.sub(r"\s+", " ", raw_text).strip()
    if any(
        kind in content_type
        for kind in ("application/json", "application/xml", "text/xml")
    ):
        return re.sub(r"\s+", " ", raw_text).strip()
    if content_type and not any(
        kind in content_type
        for kind in ("text/html", "application/xhtml+xml")
    ):
        return f"无法提取非文本页面（{content_type.split(';', 1)[0]}）。"

    parser = _VisibleTextParser()
    parser.feed(raw_text)
    parser.close()
    return parser.text()


def _truncate(text: str, max_chars: int) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "..."


def _fetch_page(url: str, timeout: int) -> dict[str, str]:
    current_url = _validate_public_url(url)

    try:
        with _create_client(timeout) as client:
            for _ in range(MAX_REDIRECTS + 1):
                with client.stream("GET", current_url) as response:
                    if response.status_code in _REDIRECT_STATUSES:
                        location = response.headers.get("location")
                        if not location:
                            return {
                                "url": current_url,
                                "output": "抓取失败：重定向缺少目标地址。",
                                "error": "",
                            }
                        current_url = _validate_public_url(
                            urljoin(current_url, location)
                        )
                        continue

                    if not 200 <= response.status_code < 300:
                        return {
                            "url": current_url,
                            "output": f"抓取失败：HTTP {response.status_code}。",
                            "error": "",
                        }

                    return {
                        "url": current_url,
                        "output": _extract_text(response),
                        "error": "",
                    }
    except httpx.HTTPError as exc:
        return {
            "url": current_url,
            "output": f"抓取失败：{exc}",
            "error": "",
        }

    return {
        "url": current_url,
        "output": "抓取失败：重定向次数过多。",
        "error": "",
    }


def crawl_url(
    url: str,
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_chars: int = DEFAULT_MAX_CHARS,
) -> dict[str, str]:
    _validate_limits(timeout, max_chars)
    result = _fetch_page(url, timeout)
    result["output"] = _truncate(result["output"], max_chars)
    return result


def crawl_urls(
    urls: list[str],
    timeout: int = DEFAULT_TIMEOUT_SECONDS,
    max_chars_per_url: int = DEFAULT_MAX_CHARS_PER_URL,
) -> list[dict[str, str]]:
    _validate_limits(timeout, max_chars_per_url)
    if not isinstance(urls, list) or any(not isinstance(url, str) for url in urls):
        raise ValueError("urls 必须是字符串列表。")
    if len(urls) > MAX_URLS_PER_CALL:
        raise ValueError(f"每次最多抓取 {MAX_URLS_PER_CALL} 个 URL。")

    results: list[dict[str, str]] = []
    for url in urls:
        try:
            result = _fetch_page(url, timeout)
        except ValueError as exc:
            result = {
                "url": url,
                "output": f"抓取失败：{exc}",
                "error": "",
            }
        result["output"] = _truncate(result["output"], max_chars_per_url)
        results.append(result)
    return results


def build_crawl_tool_definitions() -> list[ToolDefinition]:
    return [
        ToolDefinition(
            name="crawl_url",
            description=(
                "抓取一个公开 HTTP/HTTPS 网页并返回正文文本。"
                "timeout 范围为 1 到 60 秒，max_chars 范围为 200 到 20000。"
            ),
            handler=crawl_url,
        ),
        ToolDefinition(
            name="crawl_urls",
            description=(
                f"依次抓取最多 {MAX_URLS_PER_CALL} 个公开 HTTP/HTTPS 网页并返回正文文本。"
                "timeout 范围为 1 到 60 秒，max_chars_per_url 范围为 200 到 20000。"
            ),
            handler=crawl_urls,
        ),
    ]
