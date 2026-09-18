from __future__ import annotations

from dataclasses import dataclass,field
from typing import Any,Mapping


@dataclass(frozen=True)
class RuntimeEvent:
    """
    PuzzleResearch 对 LangGraph 原始事件的统一封装。
    """

    event: str
    name: str = ""
    run_id: str | None = None
    parent_ids: tuple[str,...] = ()
    tags: tuple[str,...] = ()
    metadata: dict[str,Any] = field(
        default_factory=dict,
    )
    data: Any = None


    @classmethod
    def from_raw(
            cls,
            raw_event: Any,
    ) -> "RuntimeEvent":
        if not isinstance(raw_event,Mapping):
            return cls(
                event="unknown",
                data=raw_event,
            )

        raw_parent_ids = raw_event.get(
            "parent_ids"
        ) or []

        raw_tags = raw_event.get(
            "tags"
        )

        return cls(
            event=str(
                raw_event.get(
                    "event",
                    "unknown",
                )
            ),
            name=str(
                raw_event.get(
                    "name",
                    ""
                )
            ),
            run_id=(
                str(raw_event["run_id"])
                if raw_event.get("run_id") is not None
                else None
            ),
            parent_ids=tuple(
                str(item)
                for item in raw_parent_ids
            ),
            tags=tuple(
                str(item)
                for item in (
                    raw_event.get("tags")
                    or []
                )
            ),
            metadata=dict(
                raw_event.get("metadata") or {}
            ),
            data=raw_event.get("data")
        )


    @property
    def tool_events(self) -> list[dict[str,Any]]:
        """
        返回当前运行事件中携带的工具事件。

        兼容直接放在 data.tool_events，
        以及 LangGraph 常见的 data.output.tool_events。
        """

        candidates: list[Any] = [self.data]

        if isinstance(self.data,Mapping):
            for key in ("output","data"):
                nested = self.data.get(key)

                if isinstance(nested,Mapping):
                    candidates.append(nested)

        for candidate in candidates:
            if not isinstance(candidate,Mapping):
                continue

            raw_events = candidate.get("tool_events")

            if not isinstance(raw_events,list):
                continue

            return [
                dict(item)
                for item in raw_events
                if isinstance(
                    item,
                    Mapping,
                )
            ]

        return []



    def to_dict(self) -> dict[str,Any]:
        return {
            "event": self.event,
            "name": self.name,
            "run_id": self.run_id,
            "parent_ids": list(self.parent_ids),
            "tags": list(self.tags),
            "metadata": dict(self.metadata),
            "data": self.data,
        }