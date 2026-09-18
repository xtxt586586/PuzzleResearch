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