from __future__ import annotations

import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any
import secrets


_LOCK = threading.RLock()


def _data_dir() -> Path:
    root = os.getenv("PUZZLERESEARCH_DATA_DIR", "").strip()

    if root:
        return Path(root).expanduser().resolve() / "collaboration"

    return Path(__file__).resolve().parents[1] / "data" / "collaboration"


def _load(name: str) -> dict[str, Any]:
    path = _data_dir() / name

    if not path.exists():
        return {}

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"无法读取协作数据：{name}") from exc

    if not isinstance(value, dict):
        raise RuntimeError(f"协作数据格式无效：{name}")

    return value


def _save(name: str, value: dict[str, Any]) -> None:
    directory = _data_dir()
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".collaboration-",
        suffix=".tmp",
        dir=directory,
    )
    temporary_path = Path(temporary_name)

    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

        os.replace(temporary_path, directory / name)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _timestamp() -> str:
    return _now().isoformat()


def create_share_link(
    thread_id: str,
    owner_id: str,
    *,
    permissions: str = "view",
    expires_hours: int | None = 72,
) -> dict[str, Any]:
    if permissions != "view":
        raise ValueError("当前只支持 view 分享权限。")

    if expires_hours is not None and not 1 <= expires_hours <= 8760:
        raise ValueError("expires_hours 必须在 1 到 8760 之间，或设为 null。")

    created_at = _now()
    link = {
        "id": secrets.token_urlsafe(12),
        "thread_id": thread_id,
        "owner_id": owner_id,
        "permissions": permissions,
        "created_at": created_at.isoformat(),
        "expires_at": (
            (created_at + timedelta(hours=expires_hours)).isoformat()
            if expires_hours is not None
            else None
        ),
        "view_count": 0,
    }

    with _LOCK:
        shares = _load("shares.json")
        shares[link["id"]] = link
        _save("shares.json", shares)

    return link


def get_share_link(share_id: str) -> dict[str, Any] | None:
    with _LOCK:
        shares = _load("shares.json")
        link = shares.get(share_id)

        if not isinstance(link, dict):
            return None

        expires_at = link.get("expires_at")
        if expires_at:
            expiry = datetime.fromisoformat(
                str(expires_at).replace("Z", "+00:00")
            )
            if _now() >= expiry:
                return None

        link["view_count"] = int(link.get("view_count", 0)) + 1
        shares[share_id] = link
        _save("shares.json", shares)
        return dict(link)


def delete_share_link(share_id: str, owner_id: str) -> bool:
    with _LOCK:
        shares = _load("shares.json")
        link = shares.get(share_id)

        if not isinstance(link, dict) or link.get("owner_id") != owner_id:
            return False

        del shares[share_id]
        _save("shares.json", shares)
        return True


def add_comment(
    thread_id: str,
    content: str,
    author: str,
    message_id: str | None = None,
) -> dict[str, Any]:
    now = _timestamp()
    comment = {
        "id": uuid.uuid4().hex[:12],
        "thread_id": thread_id,
        "message_id": message_id,
        "author": author,
        "content": content,
        "created_at": now,
        "updated_at": now,
    }

    with _LOCK:
        comments = _load("comments.json")
        comments.setdefault(thread_id, []).append(comment)
        _save("comments.json", comments)

    return comment


def get_comments(
    thread_id: str,
    message_id: str | None = None,
) -> list[dict[str, Any]]:
    comments = _load("comments.json").get(thread_id, [])
    if not isinstance(comments, list):
        return []

    return [
        dict(item)
        for item in comments
        if isinstance(item, dict)
        and (message_id is None or item.get("message_id") == message_id)
    ]


def save_version(
    thread_id: str,
    owner_id: str,
    snapshot: dict[str, Any],
    label: str | None = None,
) -> dict[str, Any]:
    version_id = uuid.uuid4().hex[:12]
    snapshot_text = json.dumps(
        snapshot,
        ensure_ascii=False,
        default=str,
    )
    snapshot_path = f"snapshot_{version_id}.json"
    record = {
        "id": version_id,
        "thread_id": thread_id,
        "owner_id": owner_id,
        "version_number": 1,
        "label": label or "Version 1",
        "created_at": _timestamp(),
        "snapshot_size": len(snapshot_text),
        "snapshot_path": snapshot_path,
    }

    with _LOCK:
        versions = _load("versions.json")
        history = versions.setdefault(thread_id, [])
        record["version_number"] = len(history) + 1

        directory = _data_dir()
        directory.mkdir(parents=True, exist_ok=True)
        (directory / snapshot_path).write_text(
            snapshot_text,
            encoding="utf-8",
        )

        history.append(record)
        _save("versions.json", versions)

    return {key: value for key, value in record.items() if key != "owner_id"}


def list_versions(
    thread_id: str,
    owner_id: str,
) -> list[dict[str, Any]]:
    versions = _load("versions.json").get(thread_id, [])
    if not isinstance(versions, list):
        return []

    return [
        {key: value for key, value in item.items() if key != "owner_id"}
        for item in versions
        if isinstance(item, dict) and item.get("owner_id") == owner_id
    ]


def get_version_snapshot(
    thread_id: str,
    version_id: str,
    owner_id: str,
) -> dict[str, Any] | None:
    versions = _load("versions.json").get(thread_id, [])

    record = next(
        (
            item
            for item in versions
            if isinstance(item, dict)
            and item.get("id") == version_id
            and item.get("owner_id") == owner_id
        ),
        None,
    )
    if record is None:
        return None

    try:
        value = json.loads(
            (_data_dir() / record["snapshot_path"]).read_text(
                encoding="utf-8"
            )
        )
    except (OSError, json.JSONDecodeError, KeyError):
        return None

    return value if isinstance(value, dict) else None