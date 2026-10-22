from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


ResearchAgentStatus = Literal[
    "completed",
    "partial",
    "empty",
    "failed",
]


class ResearchTask(BaseModel):
    """Bounded, checkpoint-friendly task assigned to one research agent."""

    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=128)
    topic: str = Field(min_length=1, max_length=500)
    objective: str = Field(default="", max_length=1000)
    priority: int = Field(default=3, ge=1, le=5)
    context: dict[str, Any] = Field(default_factory=dict)
    allowed_tools: set[str] = Field(default_factory=lambda: {"web_search"})

    @field_validator("task_id", "topic")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must not be blank")
        return normalized

    @field_validator("allowed_tools")
    @classmethod
    def _normalize_tool_names(cls, value: set[str]) -> set[str]:
        return {name.strip() for name in value if name.strip()}


class ResearchEvidence(BaseModel):
    """Compact evidence contract shared by research and downstream agents."""

    query: str = ""
    title: str = ""
    url: str = ""
    snippet: str = Field(default="", max_length=1600)
    provider: str = ""


class ResearchAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: ResearchAgentStatus
    summary: str = ""
    findings: list[ResearchEvidence] = Field(default_factory=list)
    sources: list[str] = Field(default_factory=list)
    unresolved_questions: list[str] = Field(default_factory=list)
    tool_calls: list[str] = Field(default_factory=list)
    error: str = ""
    stop_reason: str = ""
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


@dataclass
class ResearchExecutionContext:
    """Per-task context and capability gate; tools are never inherited implicitly."""

    task: ResearchTask
    tools: Mapping[str, Callable[..., Any]]
    findings: list[ResearchEvidence] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    tool_calls: list[str] = field(default_factory=list)

    def call_tool(self, name: str, **arguments: Any) -> Any:
        if name not in self.task.allowed_tools:
            raise PermissionError(f"Research task is not allowed to use tool: {name}")
        tool = self.tools.get(name)
        if tool is None:
            raise LookupError(f"Research tool is unavailable: {name}")

        self.tool_calls.append(name)
        result = tool(**arguments)
        records = result.get("results", []) if isinstance(result, Mapping) else result
        if isinstance(records, list):
            query = str(arguments.get("query", ""))
            for item in records:
                if not isinstance(item, Mapping):
                    continue
                evidence = ResearchEvidence(
                    query=str(item.get("query") or query),
                    title=str(item.get("title", "") or "")[:500],
                    url=str(item.get("url", "") or "")[:2000],
                    snippet=str(
                        item.get("snippet")
                        or item.get("summary")
                        or item.get("content")
                        or ""
                    )[:1600],
                    provider=str(item.get("provider", "") or "")[:100],
                )
                self.findings.append(evidence)
                if evidence.url and evidence.url not in self.sources:
                    self.sources.append(evidence.url)
        return result


class ResearchAgent:
    """Runs one task with its own context, allowed tools and result contract."""

    def __init__(self, tools: Mapping[str, Callable[..., Any]]) -> None:
        self._tools = dict(tools)

    def run(
        self,
        task: ResearchTask,
        executor: Callable[[ResearchExecutionContext], Mapping[str, Any] | str | None],
    ) -> ResearchAgentResult:
        context = ResearchExecutionContext(
            task=task,
            tools={
                name: tool
                for name, tool in self._tools.items()
                if name in task.allowed_tools
            },
        )
        started = time.monotonic()
        try:
            raw = executor(context)
            payload = raw if isinstance(raw, Mapping) else {"summary": raw or ""}
            summary = str(payload.get("summary", "") or "").strip()
            unresolved = payload.get("unresolved_questions", [])
            if isinstance(unresolved, str):
                unresolved = [unresolved]
            status: ResearchAgentStatus = (
                "completed" if summary or context.findings else "empty"
            )
            return ResearchAgentResult(
                task_id=task.task_id,
                status=status,
                summary=summary,
                findings=context.findings[:5],
                sources=context.sources[:10],
                unresolved_questions=[
                    str(item).strip()[:500]
                    for item in unresolved
                    if str(item).strip()
                ][:20] if isinstance(unresolved, list) else [],
                tool_calls=context.tool_calls,
                elapsed_seconds=max(0.0, time.monotonic() - started),
            )
        except Exception as exc:
            reason = str(getattr(exc, "reason", "") or "")
            return ResearchAgentResult(
                task_id=task.task_id,
                status="partial" if context.findings else "failed",
                summary="",
                findings=context.findings[:5],
                sources=context.sources[:10],
                tool_calls=context.tool_calls,
                error=str(exc)[:1000],
                stop_reason=reason,
                elapsed_seconds=max(0.0, time.monotonic() - started),
            )
