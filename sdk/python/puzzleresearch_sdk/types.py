from __future__ import annotations

from typing import Any, TypedDict


class StreamEvent(TypedDict):
    type: str
    data: Any


class SseEvent(TypedDict, total=False):
    id: int
    event: str
    data: Any


class ResearchTreeResponse(TypedDict):
    thread_id: str
    tree: dict[str, Any]
    quality: dict[str, Any]


class ResearchTreeUpdateData(TypedDict, total=False):
    thread_id: str
    node: str
    tree: dict[str, Any]
    quality: dict[str, Any]


class QualityUpdateData(TypedDict, total=False):
    thread_id: str
    stage: str
    overall_score: float
    citation_coverage: float
    gap_count: int
    sufficient: bool