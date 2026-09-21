from __future__ import annotations

import contextlib
import json
from typing import Any

from .types import SseEvent, StreamEvent


def parse_sse_frame(frame: str) -> SseEvent | None:
    event_name = ""
    event_id = ""
    data_lines: list[str] = []

    for raw_line in str(frame or "").replace("\r\n", "\n").split("\n"):
        if raw_line.startswith(":"):
            continue
        if raw_line.startswith("event:"):
            event_name = raw_line[6:].strip()
            continue
        if raw_line.startswith("id:"):
            event_id = raw_line[3:].strip()
            continue
        if raw_line.startswith("data:"):
            data_lines.append(raw_line[5:].lstrip())

    if not data_lines:
        return None

    try:
        data: Any = json.loads("\n".join(data_lines))
    except (TypeError, ValueError):
        return None

    result: SseEvent = {"data": data}
    if event_name:
        result["event"] = event_name
    if event_id:
        with contextlib.suppress(TypeError, ValueError):
            result["id"] = int(event_id)
    return result


def normalize_sse_event(parsed: SseEvent) -> StreamEvent | None:
    data = parsed.get("data")

    if isinstance(data, dict) and "type" in data and "data" in data:
        return {
            "type": str(data["type"]),
            "data": data["data"],
        }

    event_name = parsed.get("event")
    if isinstance(event_name, str) and event_name:
        return {
            "type": event_name,
            "data": data,
        }

    return None

