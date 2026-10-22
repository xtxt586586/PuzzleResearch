from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.workflows.claim_verifier import ClaimCheck, ClaimStatus, ClaimVerifier


class WriterTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    evidence: list[dict[str, Any]] = Field(default_factory=list, max_length=30)
    research_summary: str = Field(default="", max_length=20000)
    guidance: str = Field(default="", max_length=4000)
    images: list[dict[str, Any]] = Field(default_factory=list, max_length=8)


class WriterAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Literal["completed", "failed"]
    report: str = ""
    tools_used: list[str] = Field(default_factory=list)
    error: str = ""
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class WriterAgent:
    """Evidence-only report writer. It receives no tool capability."""

    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, generate: Callable[[WriterTask], str]) -> None:
        self._generate = generate

    def run(self, task: WriterTask) -> WriterAgentResult:
        started = time.monotonic()
        try:
            report = str(self._generate(task) or "").strip()
            if not report:
                raise ValueError("Writer returned an empty report")
            return WriterAgentResult(
                task_id=task.task_id,
                status="completed",
                report=report,
                elapsed_seconds=max(0.0, time.monotonic() - started),
            )
        except Exception as exc:
            return WriterAgentResult(
                task_id=task.task_id,
                status="failed",
                error=str(exc)[:1000],
                elapsed_seconds=max(0.0, time.monotonic() - started),
            )


class VerifierTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=128)
    report: str = Field(min_length=1, max_length=50000)
    scraped_content: list[dict[str, Any]] = Field(default_factory=list, max_length=200)
    passages: list[dict[str, Any]] = Field(default_factory=list, max_length=500)


class VerifierAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Literal["completed", "empty", "failed"]
    claims: list[dict[str, Any]] = Field(default_factory=list)
    verified_count: int = 0
    unsupported_count: int = 0
    contradicted_count: int = 0
    tools_used: list[str] = Field(default_factory=list)
    error: str = ""
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class VerifierAgent:
    """Read-only evidence verifier with no search or mutation tools."""

    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, verifier: ClaimVerifier | None = None) -> None:
        self._verifier = verifier or ClaimVerifier()

    def run(self, task: VerifierTask) -> VerifierAgentResult:
        started = time.monotonic()
        try:
            checks = self._verifier.verify_report(
                task.report,
                task.scraped_content,
                passages=task.passages or None,
            )
            payloads = [self._serialize_check(check) for check in checks]
            return VerifierAgentResult(
                task_id=task.task_id,
                status="completed" if payloads else "empty",
                claims=payloads,
                verified_count=sum(check.status == ClaimStatus.VERIFIED for check in checks),
                unsupported_count=sum(check.status == ClaimStatus.UNSUPPORTED for check in checks),
                contradicted_count=sum(check.status == ClaimStatus.CONTRADICTED for check in checks),
                elapsed_seconds=max(0.0, time.monotonic() - started),
            )
        except Exception as exc:
            return VerifierAgentResult(
                task_id=task.task_id,
                status="failed",
                error=str(exc)[:1000],
                elapsed_seconds=max(0.0, time.monotonic() - started),
            )

    @staticmethod
    def _serialize_check(check: ClaimCheck) -> dict[str, Any]:
        return {
            "claim": check.claim,
            "status": check.status.value,
            "evidence_urls": list(check.evidence_urls),
            "evidence_passages": [dict(item) for item in check.evidence_passages],
            "score": check.score,
            "notes": check.notes,
        }

