from __future__ import annotations

from typing import Any, TypedDict


class StreamEvent(TypedDict):
    type: str
    data: Any


class ChatResponse(TypedDict, total=False):
    id: str
    content: str
    timestamp: str


class RagEmbeddingStatus(TypedDict, total=False):
    provider: str
    available: bool
    dimensions: int | None
    reason: str


class AgentHealthResponse(TypedDict, total=False):
    agents_count: int
    agent_ids: list[str]
    tool_registry_total_tools: int
    enhanced_tool_discovery_enabled: bool
    enhanced_tool_discovery_recursive: bool
    rag_enabled: bool
    rag_embedding: RagEmbeddingStatus
    search_strategy: str
    search_engines: list[str]
    search_providers_available: list[str]


class KnowledgeDocument(TypedDict, total=False):
    source: str
    filename: str
    size: int
    chunks: int
    uploaded_at: str


class DocumentUploadResponse(TypedDict, total=False):
    success: bool
    filename: str
    chunks: int
    message: str


class DocumentListResponse(TypedDict, total=False):
    total_chunks: int
    documents: list[KnowledgeDocument]


class DocumentDeleteResponse(TypedDict, total=False):
    success: bool
    source: str


class DocumentSearchResult(TypedDict, total=False):
    source: str
    chunk_index: int
    text: str
    score: float


class DocumentSearchResponse(TypedDict, total=False):
    query: str
    results: list[DocumentSearchResult]


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


class EvidenceSource(TypedDict, total=False):
    title: str
    url: str
    rawUrl: str | None
    domain: str | None
    provider: str | None
    publishedDate: str | None


class EvidenceClaimEvidence(TypedDict, total=False):
    url: str
    snippet_hash: str | None
    quote: str | None
    heading_path: list[str] | None


class EvidenceClaim(TypedDict, total=False):
    claim: str
    status: str
    evidence_urls: list[str]
    evidence_passages: list[EvidenceClaimEvidence]
    score: float
    notes: str


class FetchPageItem(TypedDict, total=False):
    url: str
    raw_url: str
    method: str
    text: str | None
    title: str | None
    published_date: str | None
    retrieved_date: str | None
    retrieved_at: str | None
    markdown: str | None
    http_status: int | None
    error: str | None
    attempts: int


class EvidencePassageItem(TypedDict, total=False):
    url: str
    text: str
    start_char: int
    end_char: int
    heading: str | None
    heading_path: list[str] | None
    page_title: str | None
    retrieved_at: str | None
    method: str | None
    quote: str | None
    snippet_hash: str | None


class EvidenceResponse(TypedDict, total=False):
    sources: list[EvidenceSource]
    claims: list[EvidenceClaim]
    quality_summary: dict[str, Any]
    fetched_pages: list[FetchPageItem]
    passages: list[EvidencePassageItem]


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


class WorkspaceFilesResponse(TypedDict, total=False):
    thread_id: str
    count: int
    files: list[str]


class WorkspaceUploadResponse(TypedDict, total=False):
    thread_id: str
    path: str
    bytes: int


class WorkspaceDeleteResponse(TypedDict, total=False):
    thread_id: str
    path: str


class McpConfigResponse(TypedDict, total=False):
    enabled: bool
    servers: dict[str, Any]
    loaded_tools: int


class McpServerDiagnostic(TypedDict, total=False):
    server_id: str
    transport: str
    status: str
    loaded_tools: int
    message: str | None


class McpDiagnoseResponse(TypedDict, total=False):
    enabled: bool
    status: str
    ready: bool
    server_count: int
    connected_count: int
    servers: list[McpServerDiagnostic]


class SandboxBrowserDiagnoseResponse(TypedDict, total=False):
    ready: bool
    missing: list[str]
    sandbox_mode: str
    allow_internet: bool
    configured: dict[str, bool]
    deep: dict[str, Any] | None
