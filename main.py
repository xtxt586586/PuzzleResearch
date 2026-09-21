from __future__ import annotations

from time import monotonic

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Mapping
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
)
from pydantic import (
    BaseModel,
    Field,
    field_validator,
)

from agent.core.events import RuntimeEvent
from agent.core.runtime import AgentRuntime
from common.config import settings
from tools.builtin import create_default_registry
from tools.search import (
    SearchManager,
    get_default_search_runtime_state,
)
from common import agent_store
from agent.core.profile import AgentProfile
from agent.core.profile_adapter import (
    runtime_profile_from_record,
)
from langgraph.checkpoint.memory import (
    InMemorySaver,
)

from common.session_manager import (
    SessionManager,
    SessionManagerError,
    SessionNotFoundError,
)
from agent.core.event_stream import (
    event_stream_generator,
    get_emitter,
)


APP_STARTED_AT = monotonic()

app = FastAPI(
    title="PuzzleResearch",
    description="PuzzleResearch AI research platform",
    version="0.1.0",
)



chat_checkpointer = InMemorySaver()

chat_runtime = AgentRuntime(
    checkpointer=chat_checkpointer,
)

session_manager = SessionManager(
    chat_runtime,
    chat_checkpointer,
)

_active_chat_tasks: dict[str,asyncio.Task] = {}



class SessionSummaryResponse(BaseModel):
    thread_id: str
    status: str
    topic: str
    created_at: str
    updated_at: str
    route: str
    has_report: bool
    revision_count: int
    message_count: int
    agent_id: str
    user_id: str


class SessionListResponse(BaseModel):
    count: int
    sessions: list[SessionSummaryResponse]



class SessionStateResponse(BaseModel):
    thread_id: str
    status: str
    state: dict[str,Any]
    interrupts: list[Any]
    checkpoint_id: str | None = None


class SessionResumeRequest(BaseModel):
    resume_value: Any = True

class InterruptResumeRequest(BaseModel):
    approved: bool | None = None
    resume_value: Any = None
    action: str | None = None



class AgentHealthResponse(BaseModel):
    agents_count: int
    agent_ids: list[str]
    tool_registry_total_tools: int
    enhanced_tool_discovery_enabled: bool
    enhanced_tool_discovery_recursive: bool
    rag_enabled: bool
    search_strategy: str
    search_engines: list[str]
    search_providers_available: list[str]


class ToolRegistryState(BaseModel):
    total_tools: int
    enabled_tools: int
    deprecated_tools: int
    total_calls: int
    total_successes: int
    overall_success_rate: float
    most_used: list[dict[str,Any]]
    by_type: dict[str,int]
    tags: list[str]


class ToolRegistryTool(BaseModel):
    name: str
    description: str
    parameters: list[dict[str,Any]]
    enabled: bool
    source: str
    tool_type: str

class ToolRegistryResponse(BaseModel):
    stats: ToolRegistryState
    tools: list[ToolRegistryTool]


class ToolRegistryRefreshResponse(BaseModel):
    discovered: int
    total_tools: int


class SearchProviderSnapshot(BaseModel):
    name: str
    configured: bool
    available: bool
    healthy: bool
    reason: str
    calls: int = 0
    successes: int = 0
    failures: int = 0
    empty_results: int = 0
    last_error: str = ""


class SearchProviderResponse(BaseModel):
    providers: list[SearchProviderSnapshot]


class SearchProviderResetResponse(BaseModel):
    reset: bool
    message: str



class PublicConfigDefaults(BaseModel):
    port: int
    primary_model: str
    reasoning_model: str


class PublicConfigFeatures(BaseModel):
    mcp_enabled: bool
    sandbox_mode: str
    prometheus_enabled: bool
    tracing_enabled: bool


class PublicConfigStreamEndpoint(BaseModel):
    protocol: str
    endpoint: str


class PublicConfigStreaming(BaseModel):
    chat: PublicConfigStreamEndpoint
    research: PublicConfigStreamEndpoint
    events: PublicConfigStreamEndpoint
    browser: PublicConfigStreamEndpoint


class PublicConfigModels(BaseModel):
    default: str
    options: list[str]


class PublicConfigResponse(BaseModel):
    version: str
    defaults: PublicConfigDefaults
    features: PublicConfigFeatures
    streaming: PublicConfigStreaming
    models: PublicConfigModels


class Message(BaseModel):
    role: str
    content: str


class SearchMode(BaseModel):
    useWebSearch: bool = False
    useAgent: bool = False
    useDeepSearch: bool = False


class ImagePayload(BaseModel):
    name: str | None = None
    data: str
    mime: str | None = None


class CancelRequest(BaseModel):
    reason: str = "用户取消"



class AgentUpsertRequest(BaseModel):
    id: str | None = None
    name: str
    description: str = ""
    system_prompt: str = ""
    model: str = ""
    enabled_tools: dict[str,bool] = Field(
        default_factory=dict
    )
    mcp_servers: dict[str,Any] | None = None
    metadata: dict[str,Any] = Field(
        default_factory=dict
    )

    @field_validator("id")
    @classmethod
    def _normalize_id(
            cls,
            value: str | None,
    ) -> str | None:
        if value is None:
            return None

        normalized = value.strip()
        return normalized or None

    @field_validator("name")
    @classmethod
    def _normalize_name(
            cls,
            value: str,
    ) -> str:
        normalized = value.strip()

        if not normalized:
            raise ValueError("名称不能为空。")

        return normalized


class AgentRecordResponse(BaseModel):
    id: str
    name: str
    description: str
    system_prompt: str
    model: str
    enabled_tools: dict[str,bool]
    mcp_servers: dict[str,Any] | None
    metadata: dict[str,Any]
    created_at: str
    updated_at: str


class AgentListsResponse(BaseModel):
    agents: list[AgentRecordResponse]



class SearchCacheStateResponse(BaseModel):
    entries: int
    hits: int
    misses: int
    hit_rate: float


class SearchCacheClearResponse(BaseModel):
    cleared: int


def _normalize_search_mode(
        search_mode: SearchMode | Mapping[str,Any] | str | None,
) -> dict[str,Any]:
    if isinstance(search_mode, SearchMode):
        use_web = search_mode.useWebSearch
        use_agent = search_mode.useAgent
        use_deep = search_mode.useDeepSearch

    elif isinstance(search_mode, Mapping):
        use_web = bool(
            search_mode.get(
                "useWebSearch",
                search_mode.get("use_web",False)
            )
        )
        use_agent = bool(
            search_mode.get(
                "useAgent",
                search_mode.get("use_agent",False)
            )
        )
        use_deep = bool(
            search_mode.get(
                "useDeepSearch",
                search_mode.get("use_deep",False)
            )
        )

        if not (use_web or use_agent or use_deep):
            raw_mode = str(
                search_mode.get("mode","")
            ).strip().lower()

            if raw_mode == "web":
                use_web = True
            elif raw_mode == "agent":
                use_agent = True
            elif raw_mode == "deep":
                use_agent = True
                use_deep = True

    elif isinstance(search_mode,str):
        mode = search_mode.strip().lower()

        use_web = mode in {
            "web",
            "search",
            "tavily",
        }

        use_agent = mode in {
            "agent",
            "mcp",
            "deep",
            "deep_agent",
            "deep-agent",
            "ultra",
        }
        use_deep = mode in {
            "deep",
            "deep_agent",
            "deep-agent",
            "ultra",
        }

    else:
        use_web = False
        use_agent = False
        use_deep = False

    if use_deep:
        use_agent = True

    if use_agent:
        mode = "deep" if use_deep else "agent"
    elif use_web:
        mode = "web"
    else:
        mode = "direct"

    return {
        "use_web": use_web,
        "use_agent": use_agent,
        "use_deep": use_deep,
        "use_deep_prompt": use_deep,
        "mode": mode,
    }


def _coerce_search_mode_input(
        value: Any,
) -> SearchMode | None:
    if value is None:
        return None

    if isinstance(value, SearchMode):
        return value

    normalized = _normalize_search_mode(value)

    return SearchMode(
        useWebSearch=normalized["use_web"],
        useAgent=normalized["use_agent"],
        useDeepSearch=normalized["use_deep"],
    )



class ChatRequest(BaseModel):
    messages: list[Message]
    stream: bool = True
    model: str | None = None
    search_mode: SearchMode | None = None
    agent_id: str | None = None
    user_id: str | None = None
    images: list[ImagePayload] | None = None

    @field_validator("search_mode", mode="before")
    @classmethod
    def _coerce_search_mode(
            cls,
            value: Any,
    ) -> SearchMode | None:
        return _coerce_search_mode_input(value)



class ResearchRequest(BaseModel):
    query: str
    model: str | None = None
    search_mode: SearchMode | None = None
    agent_id: str | None = None
    user_id: str | None = None
    images: list[ImagePayload] | None = None

    @field_validator("search_mode", mode="before")
    @classmethod
    def _coerce_search_mode(
            cls,
            value: Any,
    ) -> SearchMode | None:
        return _coerce_search_mode_input(value)


class ChatResponse(BaseModel):
    id: str
    content: str
    role: str = "assistant"
    timestamp: str


class EvidenceSource(BaseModel):
    title: str = ""
    url: str
    rawUrl: str | None = None
    domain: str | None = None
    provider: str | None = None
    publishedDate: str | None = None


class EvidenceClaimEvidence(BaseModel):
    url: str
    snippet_hash: str | None = None
    quote: str | None = None
    heading_path: list[str] | None = None


class EvidenceClaim(BaseModel):
    claim: str
    status: str
    evidence_urls: list[str] = Field(default_factory=list)
    evidence_passages: list[EvidenceClaimEvidence] = Field(
        default_factory=list
    )
    score: float = 0.0
    notes: str = ""


class FetchPageItem(BaseModel):
    url: str
    raw_url: str
    method: str
    text: str | None = None
    title: str | None = None
    published_date: str | None = None
    retrieved_date: str | None = None
    markdown: str | None = None
    http_status: int | None = None
    error: str | None = None
    attempts: int = 1


class EvidencePassageItem(BaseModel):
    url: str
    text: str
    start_char: int
    end_char: int
    heading: str | None = None
    heading_path: list[str] | None = None
    page_title: str | None = None
    retrieved_at: str | None = None
    method: str | None = None
    quote: str | None = None
    snippet_hash: str | None = None


class EvidenceResponse(BaseModel):
    sources: list[EvidenceSource] = Field(default_factory=list)
    claims: list[EvidenceClaim] = Field(default_factory=list)
    quality_summary: dict[str,Any] = Field(default_factory=dict)
    fetched_pages: list[FetchPageItem] = Field(default_factory=list)
    passages: list[EvidencePassageItem] = Field(default_factory=list)



def _get_last_user_message(
        messages: list[Message],
) -> tuple[int,str]:
    for index in range(len(messages) - 1,-1,-1):
        message = messages[index]

        if message.role.strip().lower() == "user":
            content = message.content.strip()

            if not content:
                raise HTTPException(
                    status_code=400,
                    detail="用户消息不能为空。"
                )

            return index,content

    raise HTTPException(
        status_code=400,
        detail="没有发现用户消息。"
    )


def _to_runtime_message(
        messages: list[Message]
) -> list[Any]:
    result: list[Any] = []

    for message in messages:
        role = message.role.strip().lower()

        if role == "system":
            result.append(
                SystemMessage(content=message.content)
            )
        elif role == "assistant":
            result.append(
                AIMessage(content=message.content)
            )
        else:
            result.append(
                HumanMessage(content=message.content)
            )

    return result


def _build_chat_config(
        *,
        thread_id: str,
        model: str,
        search_mode: dict[str,Any],
        agent_id: str,
        user_id: str,
        images: list[ImagePayload] | None,
        agent_profile: AgentProfile,
        model_override: str | None,
) -> dict[str,Any]:
    return {
        "configurable": {
            "thread_id": thread_id,
            "model": model,
            "search_mode": search_mode,
            "agent_id": agent_id,
            "user_id": user_id,
            "images": [
                image.model_dump(mode="json")
                for image in images or []
            ],
            "agent_profile": agent_profile,
            "model_override": model_override,
        },
        "recursion_limit": 50,
    }


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _model_options() -> list[str]:
    options: list[str] = []

    for value in (
        settings.primary_model,
        settings.reasoning_model,
        settings.planner_model,
        settings.researcher_model,
        settings.writer_model,
        settings.evaluator_model
    ):
        model_name = str(value or "").strip()

        if model_name and model_name not in options:
            options.append(model_name)

    return options


def _tool_registry_snapshot() -> dict[str,Any]:
    registry = create_default_registry()
    schemas = registry.describe_tools()

    tools: list[dict[str,Any]] = []

    for schema in schemas:
        tools.append(
            {
                **schema,
                "enabled": True,
                "source": "builtin",
                "tool_type": "builtin",
            }
        )

    return {
        "stats": {
            "total_tools": len(tools),
            "enabled_tools": len(tools),
            "deprecated_tools": 0,
            "total_calls": 0,
            "total_successes": 0,
            "overall_success_rate": 0.0,
            "most_used": [],
            "by_type": {
                "builtin": len(tools),
            },
            "tags": [],
        },
        "tools": tools,
    }



def _resolve_agent_runtime(
        agent_id: str,
) -> tuple[
    agent_store.AgentRecord,
    AgentProfile,
]:
    normalized_id = agent_id.strip()

    if not normalized_id:
        normalized_id = "default"

    try:
        record = agent_store.get_agent(normalized_id)

    except agent_store.AgentStoreError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )

    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Agent 不存在：{normalized_id}",
        )

    try:
        profile = runtime_profile_from_record(
            record
        )

    except (
        TypeError,
        ValueError,
    ) as exc:
        raise HTTPException(
            status_code=442,
            detail=str(exc),
        ) from exc

    return record,profile



def _search_provider_snapshots() -> list[
    dict[str,Any]
]:
    manager = SearchManager(
        runtime_state=get_default_search_runtime_state()
    )
    provider_stats_method = getattr(manager, "provider_call_stats", None)
    provider_stats = (
        provider_stats_method()
        if callable(provider_stats_method)
        else {}
    )
    configured_name = set(
        settings.search_engines_list
    )

    snapshots: list[dict[str,Any]] = []

    for name,provider in manager.providers.items():
        provider_name = str(name).strip()

        try:
            available = bool(
                provider.is_available()
            )

        except Exception as exc:
            available = False
            reason = str(exc)
        else:
            if available:
                reason = ""
            elif provider_name not in configured_name:
                reason = "未在当前搜索配置中启用。"
            else:
                reason = "Provider 不可用。"


        stats = provider_stats.get(
            provider_name,
            {},
        )

        snapshots.append(
            {
                "name": provider_name,
                "configured": provider_name in configured_name,
                "available": available,
                "healthy": available,
                "reason": reason,
                "calls": stats.get("calls", 0),
                "successes": stats.get("successes", 0),
                "failures": stats.get("failures", 0),
                "empty_results": stats.get("empty_results", 0),
                "last_error": stats.get("last_error", ""),
            }
        )

    return snapshots



def _agent_response(
        record: agent_store.AgentRecord,
) -> AgentRecordResponse:
    return AgentRecordResponse.model_validate(
        record.model_dump(mode="json")
    )


def _record_from_request(
        payload: AgentUpsertRequest,
        *,
        agent_id: str
) -> agent_store.AgentRecord:
    timestamp = _timestamp()

    return agent_store.AgentRecord(
        id=agent_id,
        name=payload.name,
        description=payload.description,
        system_prompt=payload.system_prompt,
        model=payload.model,
        enabled_tools=dict(
            payload.enabled_tools
        ),
        mcp_servers=(
            dict(payload.mcp_servers)
            if payload.mcp_servers is not None
            else None
        ),
        metadata=dict(payload.metadata),
        created_at=timestamp,
        updated_at=timestamp,
    )



def _raise_agent_store_error(
        error: agent_store.AgentStoreError,
) -> None:
    raise HTTPException(
        status_code=500,
        detail=str(error),
    ) from error


@app.get("/")
async def root() -> dict[str,Any]:
    return {
        "status": "healthy",
        "service": "PuzzleResearch Research Agent",
        "version": app.version,
    }



@app.get("/health")
async def health() -> dict[str,Any]:
    return {
        "status": "healthy",
        "database": (
            "configured"
            if settings.database_url
            else "not configured"
        ),
        "version": app.version,
        "uptime_seconds": monotonic() - APP_STARTED_AT,
        "timestamp": _timestamp(),
    }


@app.get(
    "/api/health/agent",
    response_model=AgentHealthResponse,
)
async def agent_health() -> AgentHealthResponse:
    registry = create_default_registry()

    try:
        available_providers = SearchManager().available_providers()

    except Exception:
        available_providers = []

    try:
        records = agent_store.list_agents()
    except agent_store.AgentStoreError as exc:
        _raise_agent_store_error(exc)


    return AgentHealthResponse(
        agents_count=len(records),
        agent_ids=[
            record.id
            for record in records
        ],
        tool_registry_total_tools=len(
            registry.list_tools()
        ),
        enhanced_tool_discovery_enabled=False,
        enhanced_tool_discovery_recursive=False,
        rag_enabled=bool(
            getattr(settings,"rag_enabled",False)
        ),
        search_strategy="fallback",
        search_engines=list(
            settings.search_engines_list
        ),
        search_providers_available=sorted(
            str(name)
            for name in available_providers
        ),
    )


@app.get(
    "/api/tools/registry",
    response_model=ToolRegistryResponse,
)
async def get_tool_registry_api() -> ToolRegistryResponse:
    return ToolRegistryResponse.model_validate(
        _tool_registry_snapshot()
    )

@app.post(
    "/api/tools/registry/refresh",
    response_model=ToolRegistryRefreshResponse,
)
async def refresh_tool_registry_api(
        reset: bool = False,
) -> ToolRegistryRefreshResponse:
    snapshot = _tool_registry_snapshot()

    return ToolRegistryRefreshResponse(
        discovered=0,
        total_tools=snapshot["stats"]["total_tools"],
    )


@app.get(
    "/api/search/providers",
    response_model=SearchProviderResponse,
)
async def get_search_provider_api() -> SearchProviderResponse:
    return SearchProviderResponse(
        providers=[
            SearchProviderSnapshot.model_validate(
                item
            )
            for item in _search_provider_snapshots()
        ]
    )

@app.post(
    "/api/search/providers/reset",
    response_model=SearchProviderResetResponse,
)
async def reset_search_provider_api() -> SearchProviderResetResponse:
    return SearchProviderResetResponse(
        reset=True,
        message=(
            "当前搜索管理器没有持久化 Provider 状态，"
            "已完成逻辑重置。"
        ),
    )


@app.get(
    "/api/config/public",
    response_model=PublicConfigResponse,
)
async def public_config() -> PublicConfigResponse:
    return PublicConfigResponse(
        version=app.version,
        defaults=PublicConfigDefaults(
            port=settings.port,
            primary_model=settings.primary_model,
            reasoning_model=settings.reasoning_model,
        ),
        features=PublicConfigFeatures(
            mcp_enabled=False,
            sandbox_mode="unavailable",
            prometheus_enabled=False,
            tracing_enabled=False,
        ),
        streaming=PublicConfigStreaming(
            chat=PublicConfigStreamEndpoint(
                protocol="sse",
                endpoint="/api/chat/sse",
            ),
            research=PublicConfigStreamEndpoint(
                protocol="sse",
                endpoint="/api/research/sse",
            ),
            events=PublicConfigStreamEndpoint(
                protocol="sse",
                endpoint="/api/events/{thread_id}",
            ),
            browser=PublicConfigStreamEndpoint(
                protocol="websocket",
                endpoint="/api/browser/{thread_id}/stream",
            ),
        ),
        models=PublicConfigModels(
            default=settings.primary_model,
            options=_model_options(),
        ),
    )


@app.get(
    "/api/agents",
    response_model=AgentListsResponse,
)
async def list_agents_api() -> AgentListsResponse:
    try:
        records = agent_store.list_agents()

    except agent_store.AgentStoreError as exc:
        _raise_agent_store_error(exc)

    return AgentListsResponse(
        agents=[
            _agent_response(record)
            for record in records
        ]
    )


@app.get(
    "/api/agents/{agent_id}",
    response_model=AgentRecordResponse,
)
async def get_agent_api(
        agent_id: str,
) -> AgentRecordResponse:
    try:
        record = agent_store.get_agent(agent_id)

    except agent_store.AgentStoreError as exc:
        _raise_agent_store_error(exc)

    if record is None:
        raise HTTPException(
            status_code=404,
            detail="Agent 不存在。"
        )

    return _agent_response(record)

@app.post(
    "/api/agents",
    response_model=AgentRecordResponse,
    status_code=201,
)
async def create_agent_api(
        payload: AgentUpsertRequest,
) -> AgentRecordResponse:
    agent_id = payload.id or (
        f"agent_{uuid.uuid4().hex[:10]}"
    )



    record = _record_from_request(
        payload,
        agent_id=agent_id,
    )

    try:
        created = agent_store.create_agent(
            record
        )

    except agent_store.AgentAlreadyExistError as exc:
        raise HTTPException(
            status_code=409,
            detail=str(exc),
        ) from exc

    except agent_store.AgentStoreError as exc:
        _raise_agent_store_error(exc)

    return _agent_response(created)


@app.put(
    "/api/agents/{agent_id}",
    response_model=AgentRecordResponse,
)
async def update_agent_api(
        agent_id: str,
        payload: AgentUpsertRequest,
) -> AgentRecordResponse:
    normalized_id = agent_id.strip()

    if normalized_id == agent_store.DEFAULT_AGENT_ID:
        raise HTTPException(
            status_code=403,
            detail="默认 Agent 不允许更新。"
        )

    if (
        payload.id is not None
        and payload.id != normalized_id
    ):
        raise HTTPException(
            status_code=400,
            detail="请求体中的 id 必须与路径参数一致。"
        )

    try:
        current = agent_store.get_agent(
            normalized_id
        )

    except agent_store.AgentStoreError as exc:
        _raise_agent_store_error(exc)

    if current is None:
        raise HTTPException(
            status_code=404,
            detail="Agent 不存在。"
        )

    updated_record = current.model_copy(
        update={
            "name": payload.name,
            "description": payload.description,
            "system_prompt": payload.system_prompt,
            "model": payload.model,
            "enabled_tools": dict(
                payload.enabled_tools
            ),
            "mcp_servers": (
                dict(payload.mcp_servers)
                if payload.mcp_servers is not None
                else None
            ),
            "metadata": dict(payload.metadata),
            "updated_at": _timestamp(),
        }
    )

    try:
        updated = agent_store.update_agent(
            normalized_id,
            updated_record,
        )

    except agent_store.AgentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Agent 不存在。"
        ) from exc

    except agent_store.DefaultAgentProtectedError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except agent_store.AgentStoreError as exc:
        _raise_agent_store_error(exc)

    return _agent_response(updated)



@app.delete(
    "/api/agents/{agent_id}"
)
async def delete_agent_api(
        agent_id: str,
) -> dict[str,str]:
    normalized_id = agent_id.strip()

    try:
        agent_store.delete_agent(
            normalized_id
        )

    except agent_store.DefaultAgentProtectedError as exc:
        raise HTTPException(
            status_code=403,
            detail=str(exc),
        ) from exc

    except agent_store.AgentNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail="Agent 不存在。"
        ) from exc

    except agent_store.AgentStoreError as exc:
        _raise_agent_store_error(exc)

    return {
        "status": "deleted",
        "id": normalized_id,
    }





def _runtime_event_to_legacy_events(
        event: RuntimeEvent,
) -> list[dict[str,Any]]:
    event_data = event.data

    if isinstance(event_data,Mapping):
        data = dict(event_data)

    else:
        data = {"value": event_data}

    if event.event == "on_chain_start":
        return [
            {
                "type": "status",
                "data": {
                    "text": (
                        f"开始执行：{event.name or 'agent'}"
                    ),
                    "node": event.name,
                },
            }
        ]

    if event.event == "on_chain_error":
        return [
            {
                "type": "error",
                "data": {
                    "message": str(
                        data.get(
                            "error",
                            "chain execution failed",
                        )
                    ),
                    "node": event.name,
                },
            }
        ]

    if event.event == "on_tool_start":
        return [
            {
                "type": "tool_start",
                "data": {
                    **data,
                    "tool_name": event.name,
                },
            }
        ]

    if event.event == "on_tool_end":
        return [
            {
                "type": "tool_end",
                "data": {
                    **data,
                    "tool_name": event.name,
                },
            }
        ]

    if event.event == "on_tool_error":
        return [
            {
                "type": "tool_error",
                "data": {
                    **data,
                    "tool_name": event.name,
                },
            }
        ]

    if event.event == "on_chat_model_stream":
        chunk = data.get("chunk")
        content = getattr(chunk,"content",None)

        if content is None and isinstance(chunk,Mapping):
            content = chunk.get("content")

        if content:
            return [
                {
                    "type": "text",
                    "data": {
                        "content": str(content),
                    },
                }
            ]

    if event.event == "on_chain_end":
        output = data.get("output",data)

        if not isinstance(output,Mapping):
            return []

        converted: list[dict[str,Any]] = []

        for tool_event in output.get("tool_events",[]):
            if isinstance(tool_event,Mapping):
                converted.append(
                    {
                        "type": "tool",
                        "data": dict(tool_event),
                    }
                )

        sources = output.get("sources")

        if sources:
            converted.append(
                {
                    "type": "sources",
                    "data": {
                        "sources": sources,
                    },
                }
            )

        final_report = (
                output.get("final_report")
                or output.get("draft_report")
        )


        if final_report:
            converted.append(
                {
                    "type": "completion",
                    "data": {
                        "content": str(final_report),
                    },
                }
            )

        return converted

    return []


def _format_legacy_event(
        payload: dict[str,Any]
) -> str:
    return (
        "0:"
        + json.dumps(
            payload,
        ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    )

def _format_sse_event(
        *,
        event: str,
        data: Any,
        event_id: int,
) -> str:
    payload = {
        "type": event,
        "data": data,
    }

    return (
        f"id: {event_id}\n"
        f"event: {event}\n"
        "data: "
        + json.dumps(
            payload,
            ensure_ascii=False,
        )
        + "\n\n"
    )


async def _iter_chat_events(
        *,
        user_input: str,
        history: list[Any],
        thread_id: str,
        config: dict[str,Any]
) -> AsyncIterator[dict[str,Any]]:
    current_task = asyncio.current_task()

    if current_task is not None:
        _active_chat_tasks[thread_id] = current_task

    event_emitter = await get_emitter(thread_id)

    async def publish_event(
        payload: Mapping[str, Any],
    ) -> None:
        try:
            raw_data = payload.get("data", {})

            if isinstance(raw_data, Mapping):
                event_data = dict(raw_data)
            else:
                event_data = {
                    "value": raw_data,
                }

            await event_emitter.emit(
                str(payload.get("type", "message")),
                event_data,
            )

        except Exception:
            # 事件订阅失败不应影响主对话流。
            pass

    emitted_completion: str | None = None

    try:
        status_payload = {
            "type": "status",
            "data": {
                "text": "开始处理请求...",
                "thread_id": thread_id,
            },
        }

        await publish_event(status_payload)
        yield status_payload

        async for runtime_event in (
            chat_runtime.astream_runtime_events(
                user_input,
                messages=history,
                config=config,
            )
        ):
            if not isinstance(runtime_event,RuntimeEvent):
                runtime_event = RuntimeEvent.from_raw(
                    runtime_event
                )

            for payload in _runtime_event_to_legacy_events(
                runtime_event
            ):
                if payload["type"] == "completion":
                    content = str(
                        payload["data"].get(
                            "content",
                            ""
                        )
                    )

                    if (
                        content
                        and content == emitted_completion
                    ):
                        continue

                    emitted_completion = content

                await publish_event(payload)
                yield payload

        done_payload = {
            "type": "done",
            "data": {
                "thread_id": thread_id,
            },
        }

        await publish_event(done_payload)
        yield done_payload

    except asyncio.CancelledError:
        cancelled_payload = {
            "type": "cancelled",
            "data": {
                "thread_id": thread_id,
                "message": "任务已取消",
            },
        }

        await publish_event(cancelled_payload)
        yield cancelled_payload

    except Exception as exc:
        error_payload = {
            "type": "error",
            "data": {
                "thread_id": thread_id,
                "message": str(exc),
            },
        }

        await publish_event(error_payload)
        yield error_payload

    finally:
        if (
            current_task is not None
            and _active_chat_tasks.get(thread_id)
            is current_task
        ):
            _active_chat_tasks.pop(
                thread_id,
                None,
            )


async def _iter_legacy_chat_stream(
        **kwargs: Any
) -> AsyncIterator[str]:
    async for payload in _iter_chat_events(**kwargs):
        yield _format_legacy_event(payload)


async def _iter_sse_chat_stream(
        **kwargs: Any
) -> AsyncIterator[str]:
    yield "retry: 2000\n\n"

    event_id = 0

    async for payload in _iter_chat_events(**kwargs):
        event_id += 1

        yield _format_sse_event(
            event=payload["type"],
            data=payload["data"],
            event_id=event_id,
        )


def _as_message_event(frame: str) -> str:
    if not frame:
        return frame

    if frame.lstrip().startswith(":"):
        return frame

    lines: list[str] = []

    for line in frame.splitlines():
        if line.startswith("event:"):
            continue

        lines.append(line)

    return "\n".join(lines).rstrip("\n") + "\n\n"


@app.get(
    "/api/events/{thread_id}",
    response_model=None,
)
async def stream_thread_events(
    thread_id: str,
    request: Request,
    last_event_id: str | None = None,
) -> StreamingResponse:
    cursor = (
        last_event_id
        or request.headers.get("last-event-id")
    )

    async def event_generator():
        async for frame in event_stream_generator(
            thread_id,
            timeout=300.0,
            last_event_id=cursor,
        ):
            yield _as_message_event(frame)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/cancel/{thread_id}")
async def cancel_chat(
        thread_id: str,
        payload: CancelRequest | None = None,
) -> dict[str,Any]:
    reason = (
        payload.reason
        if payload is not None
        else "用户取消"
    )

    task = _active_chat_tasks.pop(
        thread_id,
        None,
    )

    if task is None or task.done():
        return {
            "status": "not_found",
            "thread_id": thread_id,
            "message": "任务已取消或不存在",
        }

    task.cancel()

    return {
        "status": "cancelled",
        "thread_id": thread_id,
        "reason": reason,
        "time_stamp": datetime.now(timezone.utc).isoformat(),
    }


@app.post("/api/chat/cancel-all")
async def cancel_all_chats() -> dict[str,Any]:
    active_tasks = list(
        _active_chat_tasks.items()
    )

    _active_chat_tasks.clear()

    for _, task in active_tasks:
        if not task.done():
            task.cancel()

    return {
        "status": "all_cancelled",
        "cancelled_count": len(active_tasks),
        "time_stamp": datetime.now(timezone.utc).isoformat(),
    }



@app.get(
    "/api/sessions",
    response_model=SessionListResponse,
)
async def list_sessions_api(
        limit: int = 50,
        status: str | None = None,
) -> SessionListResponse:
    try:
        sessions = session_manager.list_sessions(
            limit=limit,
            status=status,
        )

    except SessionManagerError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return SessionListResponse(
        count=len(sessions),
        sessions=[
            SessionSummaryResponse.model_validate(
                session
            )
            for session in sessions
        ],
    )


@app.get(
    "/api/sessions/{thread_id}",
    response_model=SessionSummaryResponse
)
async def get_session_api(
        thread_id: str,
) -> SessionSummaryResponse:
    try:
        session = session_manager.get_session(
            thread_id
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        )

    except SessionManagerError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return SessionSummaryResponse.model_validate(
        session
    )


@app.get(
    "/api/sessions/{thread_id}/state",
    response_model=SessionStateResponse,
)
async def get_session_state_api(
        thread_id: str,
) -> SessionStateResponse:
    try:
        state = session_manager.get_state(
            thread_id
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except (
        SessionManagerError,
        ValueError
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return SessionStateResponse.model_validate(
        state
    )

@app.delete(
    "/api/sessions/{thread_id}",
)
async def delete_session_api(
        thread_id: str,
) -> dict[str,str]:
    try:
        session_manager.delete_session(
            thread_id
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except SessionManagerError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "status": "deleted",
        "thread_id": thread_id,
    }

@app.post(
    "/api/sessions/{thread_id}/resume"
)
async def resume_session_api(
        thread_id: str,
        payload: SessionResumeRequest,
) -> dict[str,Any]:
    resume_value = (
        payload.resume_value
        if payload is not None
        else True
    )

    try:
        session_manager.get_session(
            thread_id
        )

        result = await chat_runtime.aresume(
            resume_value,
            thread_id=thread_id,
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "status": "resumed",
        "thread_id": thread_id,
        "result": (
            dict(result)
            if isinstance(result,Mapping)
            else str(result)
        ),
    }


def _mapping_list(value: Any) -> list[dict[str,Any]]:
    if not isinstance(value,list):
        return []

    return [
        dict(item)
        for item in value
        if isinstance(item,Mapping)
    ]


def _normalize_evidence_source(
        item: Mapping[str,Any]
) -> dict[str,Any]:
    return {
        "title": str(item.get("title","")),
        "url": str(item.get("url","")),
        "rawUrl": (
            item.get("rawUrl")
            or item.get("rawurl")
        ),
        "domain": item.get("domain"),
        "provider": item.get("provider"),
        "publishedDate": (
            item.get("publishedDate")
            or item.get("publisheddate")
        ),
    }


def _evidence_from_snapshot(
        snapshot: Mapping[str,Any],
) -> EvidenceResponse:
    raw_state = snapshot.get("state",{})

    if not isinstance(raw_state,Mapping):
        raw_state = {}

    artifacts = raw_state.get(
        "deepsearch_artifacts",
        {},
    )

    if not isinstance(artifacts,Mapping):
        artifacts = {}

    raw_sources = (
        raw_state.get("sources")
        or artifacts.get("sources")
        or []
    )

    sources = [
        _normalize_evidence_source(item)
        for item in _mapping_list(raw_sources)
    ]
    claims = _mapping_list(
        artifacts.get("claims",[])
    )

    quality_summary = (
        raw_state.get("quality_summary")
        or artifacts.get("quality_summary")
        or {}
    )

    if not isinstance(quality_summary,Mapping):
        quality_summary = {}

    fetched_pages = _mapping_list(
        artifacts.get("fetched_pages",[])
    )

    passages = _mapping_list(
        artifacts.get("passages",[])
    )

    return EvidenceResponse(
        sources=sources,
        claims=claims,
        quality_summary=dict(quality_summary),
        fetched_pages=fetched_pages,
        passages=passages,
    )


@app.get(
    "/api/sessions/{thread_id}/evidence",
    response_model=EvidenceResponse,
)
async def get_session_evidence_api(
        thread_id: str,
) -> EvidenceResponse:
    try:
        snapshot = session_manager.get_state(
            thread_id
        )
    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except (
        SessionManagerError,
        ValueError
    ) as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return _evidence_from_snapshot(snapshot)



@app.post(
    "/api/chat",
    response_model=None,
)
async def chat(
        payload: ChatRequest,
) -> ChatResponse | StreamingResponse:
    last_index,user_input = _get_last_user_message(
        payload.messages
    )

    history = _to_runtime_message(
        payload.messages[:last_index]
    )

    thread_id = f"thread_{uuid.uuid4().hex}"

    agent_id = (
        payload.agent_id
        or "default"
    ).strip()
    agent_record,agent_profile = (
        _resolve_agent_runtime(agent_id)
    )
    requested_model = (
        payload.model.strip()
        if payload.model
        else ""
    )
    agent_model = agent_record.model.strip()
    model_override = (
        requested_model
        or agent_model
        or None
    )
    model = (
            model_override
            or settings.primary_model
    ).strip()
    user_id = (
        payload.user_id
        or "anonymous"
    ).strip()

    search_mode = _normalize_search_mode(
        payload.search_mode
    )

    config = _build_chat_config(
        thread_id=thread_id,
        model=model,
        search_mode=search_mode,
        agent_id=agent_id,
        user_id=user_id,
        images=payload.images,
        agent_profile=agent_profile,
        model_override=model_override,
    )

    if payload.stream:
        return StreamingResponse(
            _iter_legacy_chat_stream(
                user_input=user_input,
                history=history,
                thread_id=thread_id,
                config=config,
            ),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
                "X-Thread-ID": thread_id,
            },
        )

    try:
        result = await chat_runtime.ainvoke(
            user_input,
            messages=history,
            config=config,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    final_report = str(
        result.get("final_report","")
        if isinstance(result,Mapping)
        else ""
    ).strip()

    if not final_report:
        final_report = "没有生成回复。"

    return ChatResponse(
        id=f"msg_{uuid.uuid4().hex}",
        content=final_report,
        timestamp=datetime.now(
            timezone.utc
        ).isoformat(),
    )


@app.post("/api/chat/sse")
async def chat_sse(
        payload: ChatRequest,
) -> StreamingResponse:
    last_index,user_input = _get_last_user_message(
        payload.messages
    )

    history = _to_runtime_message(
        payload.messages[:last_index]
    )

    thread_id = f"thread_{uuid.uuid4().hex}"

    agent_id = (
        payload.agent_id
        or "default"
    ).strip()
    agent_record,agent_profile = (
        _resolve_agent_runtime(agent_id)
    )
    requested_model = (
        payload.model.strip()
        if payload.model
        else ""
    )
    agent_model = agent_record.model.strip()
    model_override = (
        requested_model
        or agent_model
        or None
    )
    model = (
            model_override
            or settings.primary_model
    ).strip()
    user_id = (
        payload.user_id
        or "anonymous"
    ).strip()
    search_mode = _normalize_search_mode(
        payload.search_mode
    )
    config = _build_chat_config(
        thread_id=thread_id,
        model=model,
        search_mode=search_mode,
        agent_id=agent_id,
        user_id=user_id,
        images=payload.images,
        agent_profile=agent_profile,
        model_override=model_override,
    )

    return StreamingResponse(
        _iter_sse_chat_stream(
            user_input=user_input,
            history=history,
            thread_id=thread_id,
            config=config,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Thread-ID": thread_id,
        },
    )



@app.post("/api/research", response_model=None)
async def research_api(
        query: str,
) -> StreamingResponse:
    query = query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="查询内容不能为空。",
        )

    thread_id = f"thread_{uuid.uuid4().hex}"

    agent_id = "default"

    agent_record,agent_profile = _resolve_agent_runtime(agent_id)

    agent_model = agent_record.model.strip()
    model_override = agent_model or None
    model = (
            model_override
            or settings.primary_model
    ).strip()

    config = _build_chat_config(
        thread_id=thread_id,
        model=model,
        search_mode=_normalize_search_mode(None),
        agent_id=agent_id,
        user_id="anonymous",
        images=None,
        agent_profile=agent_profile,
        model_override=model_override,
    )

    return StreamingResponse(
        _iter_legacy_chat_stream(
            user_input=query,
            history=[],
            thread_id=thread_id,
            config=config,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Thread-ID": thread_id,
        },
    )


@app.post("/api/research/sse")
async def research_sse_api(
        payload: ResearchRequest,
) -> StreamingResponse:
    query = payload.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="查询内容不能为空。",
        )

    thread_id = f"thread_{uuid.uuid4().hex}"

    agent_id = (
        payload.agent_id
        or "default"
    ).strip()

    agent_record,agent_profile = _resolve_agent_runtime(agent_id)

    requested_model = (
        payload.model.strip()
        if payload.model
        else ""
    )
    agent_model = agent_record.model.strip()
    model_override = (
        requested_model
        or agent_model
        or None
    )
    model = (
            model_override
            or settings.primary_model
    ).strip()
    user_id = (
        payload.user_id
        or "anonymous"
    ).strip()

    search_mode = _normalize_search_mode(
        payload.search_mode
    )

    config = _build_chat_config(
        thread_id=thread_id,
        model=model,
        search_mode=search_mode,
        agent_id=agent_id,
        user_id=user_id,
        images=payload.images,
        agent_profile=agent_profile,
        model_override=model_override,
    )

    return StreamingResponse(
        _iter_sse_chat_stream(
            user_input=query,
            history=[],
            thread_id=thread_id,
            config=config,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "X-Thread-ID": thread_id,
        },
    )



@app.get(
    "/api/interrupt/{thread_id}/status"
)
async def interrupt_status_api(
        thread_id: str,
) -> dict[str,Any]:
    try:
        state = session_manager.get_state(
            thread_id
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    interrupts = state.get(
        "interrupts", [],
    )

    return {
        "thread_id": thread_id,
        "is_interrupted": bool(interrupts),
        "interrupts": interrupts,
        "available_actions": (
            ["approve","reject"]
            if interrupts
            else []
        ),
    }


@app.post(
    "/api/interrupt/{thread_id}/resume"
)
async def interrupt_resume_api(
        thread_id: str,
        payload: InterruptResumeRequest,
) -> dict[str,Any]:
    if payload.approved is not None:
        resume_value = payload.approved

    elif payload.resume_value is not None:
        resume_value = payload.resume_value

    elif payload.action == "approve":
        resume_value = True

    elif payload.action == "reject":
        resume_value = False

    else:
        raise HTTPException(
            status_code=400,
            detail=(
                "必须提供 approved、resume_value "
                "或有效的 action。"
            ),
        )

    try:
        session_manager.get_session(
            thread_id
        )

        result = await chat_runtime.aresume(
            resume_value,
            thread_id=thread_id,
        )

    except SessionNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    return {
        "status": "resumed",
        "thread_id": thread_id,
        "result": (
            dict(result)
            if isinstance(result,Mapping)
            else str(result)
        ),
    }


@app.get(
    "/api/search/cache/stats",
    response_model=SearchCacheStateResponse,
)
async def get_search_cache_stats_api():
    manager = SearchManager(
        runtime_state=get_default_search_runtime_state()
    )

    return SearchCacheStateResponse.model_validate(
        manager.cache_stats(),
    )


@app.post(
    "/api/search/cache/clear",
    response_model=SearchCacheClearResponse,
)
async def clear_search_cache_api():
    manager = SearchManager(
        runtime_state=get_default_search_runtime_state()
    )

    return SearchCacheClearResponse(
        cleared=manager.clear_cache(),
    )