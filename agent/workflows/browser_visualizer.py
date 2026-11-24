from __future__ import annotations

import ipaddress
import logging
import threading
import time
from collections.abc import Iterable, Mapping
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from common.config import settings
from tools.browser_use import browser_use


logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_ACTIVE_THREADS: set[str] = set()
_LAST_STARTED: dict[str, float] = {}
_VISITED_URLS: dict[str, set[str]] = {}
_WORKER_SLOTS = threading.BoundedSemaphore(3)
_MIN_INTERVAL_SECONDS = 1.5
_MAX_URLS_PER_THREAD = 500
_BLOCKED_HOSTS = {
    "youtube.com",
    "youtu.be",
    "bilibili.com",
    "tiktok.com",
    "douyin.com",
}


def _thread_id(state: Mapping[str, Any], config: Any) -> str:
    configurable = config.get("configurable", {}) if isinstance(config, Mapping) else {}
    if not isinstance(configurable, Mapping):
        configurable = {}
    return str(
        configurable.get("thread_id")
        or state.get("thread_id")
        or state.get("cancel_token_id")
        or ""
    ).strip()


def _enabled(config: Any) -> bool:
    configurable = config.get("configurable", {}) if isinstance(config, Mapping) else {}
    if not isinstance(configurable, Mapping):
        configurable = {}
    configured = configurable.get(
        "deepsearch_visualize_browser",
        getattr(settings, "deepsearch_visualize_browser", True),
    )
    return bool(configured) and bool(getattr(settings, "enable_browser_use", False))


def _normalize_url(value: Any) -> str:
    raw = str(value or "").strip()
    try:
        parts = urlsplit(raw)
        scheme = parts.scheme.lower()
        host = (parts.hostname or "").lower()
        if scheme not in {"http", "https"} or not host:
            return ""
        if (
            parts.username
            or parts.password
            or host == "localhost"
            or host.endswith((".localhost", ".local", ".internal", ".lan", ".intranet"))
        ):
            return ""
        try:
            address = ipaddress.ip_address(host)
            if not address.is_global:
                return ""
        except ValueError:
            address = None
        if any(host == blocked or host.endswith(f".{blocked}") for blocked in _BLOCKED_HOSTS):
            return ""
        host_for_netloc = f"[{host}]" if address and address.version == 6 else host
        netloc = host_for_netloc
        if parts.port is not None and not (
            (scheme == "http" and parts.port == 80)
            or (scheme == "https" and parts.port == 443)
        ):
            netloc = f"{host_for_netloc}:{parts.port}"
        return urlunsplit((scheme, netloc, parts.path or "/", parts.query, ""))
    except (TypeError, ValueError):
        return ""


def _spawn(callback: Any) -> None:
    threading.Thread(target=callback, name="research-browser-preview", daemon=True).start()


def _visualize(
    thread_id: str,
    urls: list[str],
    reason: str,
) -> None:
    try:
        for url in urls:
            try:
                logger.debug("Research browser preview (%s): %s", reason, url)
                result = browser_use(thread_id, "go_to_url", url=url)
                if not isinstance(result, Mapping) or result.get("error"):
                    continue
                browser_use(thread_id, "scroll_down", scroll_amount=520)
            except Exception:
                logger.debug("Research browser preview failed for %s", url, exc_info=True)
    finally:
        with _LOCK:
            _ACTIVE_THREADS.discard(thread_id)
        _WORKER_SLOTS.release()


def visualize_urls_from_results(
    *,
    state: Mapping[str, Any],
    config: Any,
    results: Iterable[Any],
    max_urls: int = 1,
    reason: str = "deepsearch",
) -> None:
    """Preview a few research sources without making browser errors research errors."""
    if state.get("is_cancelled") or not _enabled(config):
        return
    thread_id = _thread_id(state, config)
    if not thread_id:
        return
    try:
        limit = max(0, min(int(max_urls), 3))
    except (TypeError, ValueError):
        return
    if limit == 0:
        return

    candidates: list[str] = []
    for item in results:
        if not isinstance(item, Mapping):
            continue
        url = _normalize_url(item.get("url"))
        if url and url not in candidates:
            candidates.append(url)
        if len(candidates) >= limit:
            break
    if not candidates:
        return

    now = time.monotonic()
    with _LOCK:
        if thread_id in _ACTIVE_THREADS:
            return
        if now - _LAST_STARTED.get(thread_id, 0.0) < _MIN_INTERVAL_SECONDS:
            return
        if not _WORKER_SLOTS.acquire(blocking=False):
            return
        visited = _VISITED_URLS.setdefault(thread_id, set())
        selected = [url for url in candidates if url not in visited]
        if not selected:
            _WORKER_SLOTS.release()
            return
        visited.update(selected)
        if len(visited) > _MAX_URLS_PER_THREAD:
            _VISITED_URLS[thread_id] = set(list(visited)[-_MAX_URLS_PER_THREAD:])
        _ACTIVE_THREADS.add(thread_id)
        _LAST_STARTED[thread_id] = now

    try:
        _spawn(lambda: _visualize(thread_id, selected, str(reason or "deepsearch")))
    except Exception:
        with _LOCK:
            _ACTIVE_THREADS.discard(thread_id)
        _WORKER_SLOTS.release()
        logger.debug("Could not start research browser preview", exc_info=True)


__all__ = ["visualize_urls_from_results"]
