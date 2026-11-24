from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass

from common.config import settings
from tools.research.models import FetchedPage


@dataclass(frozen=True)
class _CacheEntry:
    expires_at: float
    payload: dict


class FetchedPageCache:
    def __init__(self, *, max_entries: int, ttl_s: float) -> None:
        self.max_entries = max(1, int(max_entries))
        self.ttl_s = max(0.0, float(ttl_s))
        self._data: OrderedDict[str, _CacheEntry] = OrderedDict()
        self._lock = threading.RLock()

    def get(self, key: str) -> FetchedPage | None:
        if not key:
            return None
        with self._lock:
            entry = self._data.get(key)
            if entry is None:
                return None
            if entry.expires_at <= time.time():
                self._data.pop(key, None)
                return None
            self._data.move_to_end(key)
            try:
                return FetchedPage(**dict(entry.payload))
            except Exception:
                self._data.pop(key, None)
                return None

    def set(self, key: str, page: FetchedPage) -> None:
        if not key:
            return
        with self._lock:
            self._data[key] = _CacheEntry(
                expires_at=time.time() + self.ttl_s,
                payload=page.to_dict(),
            )
            self._data.move_to_end(key)
            while len(self._data) > self.max_entries:
                self._data.popitem(last=False)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()


_cache: FetchedPageCache | None = None
_cache_lock = threading.RLock()


def get_fetched_page_cache() -> FetchedPageCache | None:
    ttl_s = float(getattr(settings, "research_fetch_cache_ttl_s", 0.0) or 0.0)
    max_entries = int(getattr(settings, "research_fetch_cache_max_entries", 0) or 0)
    if ttl_s <= 0 or max_entries <= 0:
        return None

    global _cache
    with _cache_lock:
        if _cache is None or _cache.ttl_s != ttl_s or _cache.max_entries != max_entries:
            _cache = FetchedPageCache(max_entries=max_entries, ttl_s=ttl_s)
        return _cache


def clear_fetched_page_cache() -> None:
    global _cache
    with _cache_lock:
        if _cache is not None:
            _cache.clear()
        _cache = None
