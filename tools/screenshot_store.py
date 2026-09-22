from __future__ import annotations

import base64
import re
import uuid
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any


class ScreenshotStore:
    def __init__(
        self,
        root: str | Path,
        *,
        base_url: str = "/api/screenshots",
        retention_hours: int = 24,
        max_bytes: int = 10 * 1024 * 1024,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.base_url = base_url.rstrip("/")
        self.retention_hours = max(
            int(retention_hours),
            1,
        )
        self.max_bytes = max(
            int(max_bytes),
            1,
        )
        self._lock = RLock()
        self.root.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _image_type(
        data: bytes,
    ) -> tuple[str, str]:
        if data.startswith(b"\x89PNG"):
            return "png", "image/png"

        if data.startswith(b"\xff\xd8\xff"):
            return "jpg", "image/jpeg"

        raise ValueError(
            "仅支持 PNG 或 JPEG 图片。"
        )

    @staticmethod
    def _safe_part(
        value: str,
        fallback: str,
    ) -> str:
        normalized = re.sub(
            r"[^a-zA-Z0-9_-]",
            "_",
            str(value or "").strip(),
        )

        return normalized[:40] or fallback

    def _filename(
        self,
        action: str,
        thread_id: str | None,
        extension: str,
    ) -> str:
        parts = []

        if thread_id:
            parts.append(
                self._safe_part(
                    thread_id,
                    "thread",
                )
            )

        parts.append(
            self._safe_part(
                action,
                "screenshot",
            )
        )
        parts.append(
            datetime.now(timezone.utc).strftime(
                "%Y%m%d_%H%M%S_%f"
            )
        )
        parts.append(uuid.uuid4().hex[:8])

        return "_".join(parts) + f".{extension}"

    def save_screenshot(
        self,
        image_data: bytes,
        *,
        action: str = "screenshot",
        thread_id: str | None = None,
        page_url: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not image_data:
            raise ValueError(
                "图片内容不能为空。"
            )

        if len(image_data) > self.max_bytes:
            raise ValueError(
                "图片超过大小限制。"
            )

        extension, mime_type = self._image_type(
            image_data
        )
        filename = self._filename(
            action,
            thread_id,
            extension,
        )
        path = self.root / filename

        with self._lock:
            path.write_bytes(image_data)

        return {
            "url": f"{self.base_url}/{filename}",
            "filename": filename,
            "path": str(path),
            "action": action,
            "thread_id": thread_id,
            "page_url": page_url,
            "mime_type": mime_type,
            "timestamp": datetime.now(
                timezone.utc
            ).isoformat(),
            "size_bytes": len(image_data),
            "metadata": dict(metadata or {}),
        }

    def save_base64(
        self,
        value: str,
        **kwargs: Any,
    ) -> dict[str, Any]:
        normalized = str(value or "").strip()

        if normalized.startswith("data:") and "," in normalized:
            normalized = normalized.split(
                ",",
                1,
            )[1]

        try:
            image_data = base64.b64decode(
                normalized,
                validate=True,
            )
        except Exception as exc:
            raise ValueError(
                "Base64 图片数据无效。"
            ) from exc

        return self.save_screenshot(
            image_data,
            **kwargs,
        )

    def get_path(
        self,
        filename: str,
    ) -> Path | None:
        requested = Path(str(filename or ""))

        if (
            not requested.name
            or requested.name != str(filename)
        ):
            return None

        candidate = (
            self.root / requested.name
        ).resolve()

        if candidate.parent != self.root:
            return None

        if not candidate.is_file():
            return None

        if candidate.suffix.lower() not in {
            ".png",
            ".jpg",
            ".jpeg",
        }:
            return None

        return candidate

    def list_screenshots(
        self,
        *,
        thread_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        normalized_limit = max(
            min(int(limit), 200),
            1,
        )
        result = []

        files = sorted(
            (
                path
                for path in self.root.iterdir()
                if path.is_file()
                and path.suffix.lower()
                in {".png", ".jpg", ".jpeg"}
            ),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )

        for path in files:
            if (
                thread_id
                and thread_id not in path.name
            ):
                continue

            suffix = path.suffix.lower()
            result.append(
                {
                    "url": (
                        f"{self.base_url}/"
                        f"{path.name}"
                    ),
                    "filename": path.name,
                    "mime_type": (
                        "image/png"
                        if suffix == ".png"
                        else "image/jpeg"
                    ),
                    "size_bytes": path.stat().st_size,
                    "created_at": datetime.fromtimestamp(
                        path.stat().st_mtime,
                        timezone.utc,
                    ).isoformat(),
                }
            )

            if len(result) >= normalized_limit:
                break

        return result

    def cleanup_old(self) -> int:
        cutoff = datetime.now(
            timezone.utc
        ) - timedelta(
            hours=self.retention_hours
        )
        deleted = 0

        for path in self.root.iterdir():
            if (
                not path.is_file()
                or path.suffix.lower()
                not in {".png", ".jpg", ".jpeg"}
            ):
                continue

            modified = datetime.fromtimestamp(
                path.stat().st_mtime,
                timezone.utc,
            )

            if modified < cutoff:
                path.unlink()
                deleted += 1

        return deleted


@lru_cache(maxsize=16)
def get_screenshot_store(
    root: str = "./data/screenshots",
    retention_hours: int = 24,
    max_bytes: int = 10 * 1024 * 1024,
) -> ScreenshotStore:
    return ScreenshotStore(
        root,
        retention_hours=retention_hours,
        max_bytes=max_bytes,
    )