from __future__ import annotations

from datetime import datetime,timezone
from time import monotonic
from typing import Any

from fastapi import FastAPI
from pydantic import BaseModel

from common.config import settings
from tools.builtin import create_default_registry
from tools.search import SearchManager


APP_STARTED_AT = monotonic()

app = FastAPI(
    title="PuzzleResearch",
    description="PuzzleResearch AI research platform",
    version="0.1.0",
)


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

    return AgentHealthResponse(
        agents_count=0,
        agent_ids=[],
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