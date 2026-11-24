from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from collections import OrderedDict
import time
from typing import Any


def _now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass
class RunMetrics:
    run_id: str
    model: str
    route: str = ""
    started_at: datetime = field(default_factory=_now)
    ended_at: datetime | None = None
    duration_ms: float = 0.0
    first_token_ms: float | None = None
    started_monotonic: float = field(
        default_factory=time.monotonic,
        repr=False,
    )
    event_count: int = 0
    model_call_count: int = 0
    nodes_started: dict[str, int] = field(default_factory=dict)
    nodes_completed: dict[str, int] = field(default_factory=dict)
    node_durations_ms: dict[str, list[float]] = field(default_factory=dict)
    tool_durations_ms: dict[str, list[float]] = field(default_factory=dict)
    phase_durations_ms: dict[str, list[float]] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False
    _active_spans: dict[tuple[str, str, str], list[float]] = field(
        default_factory=dict,
        repr=False,
    )
    _model_run_ids: set[str] = field(default_factory=set, repr=False)

    def mark_event(
        self,
        event_type: str,
        node_name: str | None = None,
        event_data: Any = None,
        event_run_id: str | None = None,
    ) -> None:
        self.event_count += 1

        if event_type in {"on_chat_model_start", "on_llm_start"}:
            model_run_id = str(event_run_id or "")
            if not model_run_id or model_run_id not in self._model_run_ids:
                self.model_call_count += 1
                if model_run_id:
                    self._model_run_ids.add(model_run_id)

        if event_type == "on_chain_end" and isinstance(event_data, dict):
            output = event_data.get("output", event_data)
            if isinstance(output, dict):
                performance = output.get("agent_performance")
                if isinstance(performance, dict):
                    for name in (
                        "search_duration_ms",
                        "generation_duration_ms",
                        "first_token_duration_ms",
                        "generation_first_token_ms",
                        "node_duration_ms",
                    ):
                        value = performance.get(name)
                        if isinstance(value, (int, float)) and value >= 0:
                            self.phase_durations_ms.setdefault(name, []).append(
                                round(float(value), 2)
                            )

        if not node_name:
            return

        bucket = (
            self.nodes_started
            if "start" in event_type
            else self.nodes_completed
        )

        bucket[node_name] = bucket.get(node_name, 0) + 1

        span_kind = None
        is_start = False
        if event_type == "on_chain_start":
            span_kind, is_start = "node", True
        elif event_type in {"on_chain_end", "on_chain_error"}:
            span_kind = "node"
        elif event_type == "on_tool_start":
            span_kind, is_start = "tool", True
        elif event_type in {"on_tool_end", "on_tool_error"}:
            span_kind = "tool"

        if span_kind is None:
            return

        span_key = (span_kind, node_name, str(event_run_id or ""))
        if is_start:
            self._active_spans.setdefault(span_key, []).append(time.monotonic())
            return

        started = self._active_spans.get(span_key)
        if not started:
            return

        elapsed_ms = max((time.monotonic() - started.pop(0)) * 1000, 0.0)
        if not started:
            self._active_spans.pop(span_key, None)
        durations = (
            self.node_durations_ms
            if span_kind == "node"
            else self.tool_durations_ms
        )
        durations.setdefault(node_name, []).append(round(elapsed_ms, 2))

    def add_error(self, message: str) -> None:
        if message:
            self.errors.append(message)

    def mark_first_token(self) -> None:
        if self.first_token_ms is None:
            self.first_token_ms = max(
                (time.monotonic() - self.started_monotonic) * 1000,
                0.0,
            )

    def finish(self, cancelled: bool = False) -> None:
        if self.ended_at is not None:
            return

        self.ended_at = _now()
        self.cancelled = cancelled
        self.duration_ms = (
            self.ended_at - self.started_at
        ).total_seconds() * 1000

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "model": self.model,
            "route": self.route,
            "started_at": self.started_at.isoformat(),
            "ended_at": (
                self.ended_at.isoformat()
                if self.ended_at
                else None
            ),
            "duration_ms": round(
                self.duration_ms,
                2,
            ),
            "first_token_ms": (
                round(self.first_token_ms, 2)
                if self.first_token_ms is not None
                else None
            ),
            "event_count": self.event_count,
            "model_call_count": self.model_call_count,
            "nodes_started": dict(self.nodes_started),
            "nodes_completed": dict(self.nodes_completed),
            "node_durations_ms": {
                name: list(durations)
                for name, durations in self.node_durations_ms.items()
            },
            "tool_durations_ms": {
                name: list(durations)
                for name, durations in self.tool_durations_ms.items()
            },
            "phase_durations_ms": {
                name: list(durations)
                for name, durations in self.phase_durations_ms.items()
            },
            "errors": list(self.errors),
            "cancelled": self.cancelled,
        }


class RunMetricsRegistry:
    def __init__(self, max_runs: int = 1000) -> None:
        if max_runs < 1:
            raise ValueError("max_runs must be at least 1")
        self.max_runs = max_runs
        self._runs: OrderedDict[str, RunMetrics] = OrderedDict()

    def _prune_finished(self) -> None:
        while len(self._runs) > self.max_runs:
            oldest_finished = next(
                (
                    run_id
                    for run_id, metrics in self._runs.items()
                    if metrics.ended_at is not None
                ),
                None,
            )
            if oldest_finished is None:
                return
            del self._runs[oldest_finished]

    def start(
        self,
        run_id: str,
        model: str,
        route: str = "",
    ) -> RunMetrics:
        metrics = RunMetrics(
            run_id=run_id,
            model=model,
            route=route,
        )
        self._runs[run_id] = metrics
        self._runs.move_to_end(run_id)
        self._prune_finished()
        return metrics

    def get(
        self,
        run_id: str,
    ) -> RunMetrics | None:
        return self._runs.get(run_id)

    def all(self) -> list[dict[str, Any]]:
        return [
            metrics.to_dict()
            for metrics in self._runs.values()
        ]

    def finish(
        self,
        run_id: str,
        cancelled: bool = False,
    ) -> RunMetrics | None:
        metrics = self._runs.get(run_id)

        if metrics is not None:
            metrics.finish(cancelled)
            self._prune_finished()

        return metrics


metrics_registry = RunMetricsRegistry()
