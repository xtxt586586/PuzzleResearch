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


class SessionStateResponse(TypedDict, total=False):
    thread_id: str
    status: str
    state: dict[str, Any]
    interrupts: list[Any]
    checkpoint_id: str | None
    checkpoint_ts: str
    parent_checkpoint_id: str | None
    deepsearch_artifacts: dict[str, Any]


class McpConfigResponse(TypedDict, total=False):
    enabled: bool
    servers: dict[str, Any]
    loaded_tools: int


class SandboxBrowserDiagnoseResponse(TypedDict, total=False):
    ready: bool
    missing: list[str]
    sandbox_mode: str
    allow_internet: bool
    configured: dict[str, bool]
    deep: dict[str, Any] | None