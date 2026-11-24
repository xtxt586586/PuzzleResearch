from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from agent.workflows.claim_verifier import ClaimCheck, ClaimStatus, ClaimVerifier


class EvaluatorTask(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str = Field(min_length=1, max_length=128)
    question: str = Field(min_length=1, max_length=4000)
    report: str = Field(min_length=1, max_length=50000)
    evidence_count: int = Field(default=0, ge=0)


class EvaluatorAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Literal["completed", "failed"]
    verdict: str = ""
    evaluation: str = ""
    dimensions: dict[str, float] = Field(default_factory=dict)
    missing_topics: list[str] = Field(default_factory=list)
    error: str = ""
    elapsed_seconds: float = Field(default=0.0, ge=0.0)


class EvaluatorAgent:
    """Scores a report against its question and evidence without tool access."""

    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, evaluate: Callable[[EvaluatorTask], dict[str, Any]]) -> None:
        self._evaluate = evaluate

    def run(self, task: EvaluatorTask) -> tuple[EvaluatorAgentResult, dict[str, Any]]:
        started = time.monotonic()
        try:
            output = self._evaluate(task)
            if not isinstance(output, dict):
                raise TypeError("Evaluator must return a state update mapping")
            result = EvaluatorAgentResult(
                task_id=task.task_id,
                status=(
                    "failed"
                    if output.get("verdict") == "evaluation_failed"
                    else "completed"
                ),
                verdict=str(output.get("verdict", "")),
                evaluation=str(output.get("evaluation", ""))[:4000],
                dimensions=dict(output.get("eval_dimensions") or {}),
                missing_topics=[
                    str(item)[:500]
                    for item in (output.get("missing_topics") or [])
                    if str(item).strip()
                ][:20],
                error=(
                    str(output.get("last_error", "") or "")[:1000]
                    if output.get("verdict") == "evaluation_failed"
                    else ""
                ),
                elapsed_seconds=max(0.0, time.monotonic() - started),
            )
            return result, output
        except Exception as exc:
            return EvaluatorAgentResult(
                task_id=task.task_id,
                status="failed",
                verdict="evaluation_failed",
                error=str(exc)[:1000],
                elapsed_seconds=max(0.0, time.monotonic() - started),
            ), {
                "evaluation": "评估 Agent 执行失败；报告未通过评估。",
                "verdict": "evaluation_failed",
                "eval_dimensions": {},
                "missing_topics": [],
                "errors": [f"Evaluator Agent failed: {str(exc)[:1000]}"],
                "last_error": str(exc)[:1000],
                "current_step": 4,
                "status": "running",
            }


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

