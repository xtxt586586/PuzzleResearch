from __future__ import annotations

import json
import os
import re
import tempfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any


class SupportMemoryStore:
    def __init__(
        self,
        path: str | Path,
        *,
        max_entries: int = 20,
    ) -> None:
        self.path = Path(path).expanduser()
        self.max_entries = max(int(max_entries), 1)
        self._lock = RLock()

    @staticmethod
    def _normalize_user_id(user_id: str | None) -> str:
        value = str(user_id or "").strip()
        return value or "default_user"

    def _read_unlocked(self) -> dict[str, Any]:
        if not self.path.exists():
            return {
                "version": 1,
                "users": {},
            }

        try:
            payload = json.loads(
                self.path.read_text(
                    encoding="utf-8",
                )
            )
        except (OSError, ValueError, TypeError):
            return {
                "version": 1,
                "users": {},
            }

        if not isinstance(payload, dict):
            return {
                "version": 1,
                "users": {},
            }

        users = payload.get("users")

        if not isinstance(users, dict):
            payload["users"] = {}

        return payload

    def _write_unlocked(self, payload: dict[str, Any]) -> None:
        self.path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            dir=str(self.path.parent),
            text=True,
        )

        temporary_path = Path(temporary_name)

        try:
            with os.fdopen(
                descriptor,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    payload,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")

            os.replace(
                temporary_path,
                self.path,
            )
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    def add(
        self,
        user_id: str,
        query: str,
        response: str,
    ) -> dict[str, Any]:
        normalized_user_id = self._normalize_user_id(user_id)
        normalized_query = str(query or "").strip()
        normalized_response = str(response or "").strip()

        if not normalized_response:
            return {}

        entry = {
            "query": normalized_query,
            "response": normalized_response,
            "created_at": datetime.now(
                timezone.utc
            ).isoformat(),
        }

        with self._lock:
            payload = self._read_unlocked()
            users = payload.setdefault("users", {})
            entries = users.setdefault(
                normalized_user_id,
                [],
            )

            if not isinstance(entries, list):
                entries = []

            entries.append(entry)
            users[normalized_user_id] = entries[
                -self.max_entries:
            ]
            self._write_unlocked(payload)

        return dict(entry)

    @staticmethod
    def _tokens(value: str) -> set[str]:
        tokens: set[str] = set()
        parts = re.findall(
            r"[A-Za-z0-9_]+|[\u3400-\u4dbf\u4e00-\u9fff]+",
            value,
            flags=re.UNICODE,
        )

        for part in parts:
            normalized = part.casefold()
            if not re.fullmatch(
                r"[\u3400-\u4dbf\u4e00-\u9fff]+",
                part,
            ):
                tokens.add(normalized)
                continue

            if len(part) == 1:
                tokens.add(normalized)
                continue

            tokens.update(
                part[index:index + 2]
                for index in range(len(part) - 1)
            )

        return tokens

    def search(
        self,
        user_id: str,
        query: str = "",
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        normalized_user_id = self._normalize_user_id(user_id)
        normalized_query = str(query or "").strip()
        query_tokens = self._tokens(normalized_query)
        normalized_limit = max(int(limit), 0)

        if normalized_limit == 0:
            return []

        with self._lock:
            payload = self._read_unlocked()
            entries = (
                payload.get("users", {})
                .get(normalized_user_id, [])
            )

            if not isinstance(entries, list):
                return []

            copied = [
                dict(item)
                for item in entries
                if isinstance(item, dict)
            ]

        if not query_tokens:
            return list(
                reversed(copied[-normalized_limit:])
            )

        ranked: list[tuple[int, int, dict[str, Any]]] = []

        for index, entry in enumerate(copied):
            searchable = " ".join(
                [
                    str(entry.get("query", "")),
                    str(entry.get("response", "")),
                ]
            )
            score = len(
                query_tokens
                & self._tokens(searchable)
            )

            if score > 0:
                ranked.append(
                    (
                        score,
                        index,
                        entry,
                    )
                )

        ranked.sort(
            key=lambda item: (
                item[0],
                item[1],
            ),
            reverse=True,
        )

        return [
            dict(item[2])
            for item in ranked[:normalized_limit]
        ]

    def clear(self, user_id: str) -> int:
        normalized_user_id = self._normalize_user_id(user_id)

        with self._lock:
            payload = self._read_unlocked()
            users = payload.setdefault("users", {})
            entries = users.pop(
                normalized_user_id,
                [],
            )

            if entries:
                self._write_unlocked(payload)

            return (
                len(entries)
                if isinstance(entries, list)
                else 0
            )

    def status(self) -> dict[str, Any]:
        with self._lock:
            payload = self._read_unlocked()
            users = payload.get("users", {})

            if not isinstance(users, dict):
                users = {}

            entries = sum(
                len(value)
                for value in users.values()
                if isinstance(value, list)
            )

        return {
            "path": str(self.path),
            "users": len(users),
            "entries": entries,
        }


@lru_cache(maxsize=32)
def get_support_memory_store(
    path: str = "./data/support_memory.json",
    max_entries: int = 20,
) -> SupportMemoryStore:
    return SupportMemoryStore(
        path,
        max_entries=max_entries,
    )
