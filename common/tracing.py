from __future__ import annotations

import functools
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Iterator, TypeVar


F = TypeVar("F", bound=Callable[..., Any])


class SpanKind(str, Enum):
    NODE = "node"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    SEARCH = "search"
    CRAWL = "crawl"
    CUSTOM = "custom"


class SpanStatus(str, Enum):
    RUNNING = "running"
    SUCCESS = "success"
    ERROR = "error"
    CANCELLED = "cancelled"


@dataclass
class TraceSpan:
    span_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:12]
    )
    parent_id: str | None = None
    kind: SpanKind = SpanKind.CUSTOM
    name: str = ""
    start_time: float = field(
        default_factory=time.time
    )
    end_time: float | None = None
    status: SpanStatus = SpanStatus.RUNNING
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    error: str | None = None
    attributes: dict[str, Any] = field(
        default_factory=dict
    )

    @property
    def duration_ms(self) -> float:
        end_time = (
            self.end_time
            if self.end_time is not None
            else time.time()
        )
        return max(
            (end_time - self.start_time) * 1000,
            0.0,
        )

    def finish(
        self,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: str | None = None,
    ) -> None:
        self.end_time = time.time()
        self.status = status

        if error:
            self.error = error
            self.status = SpanStatus.ERROR

    def set_tokens(
        self,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        self.input_tokens = max(
            int(input_tokens or 0),
            0,
        )
        self.output_tokens = max(
            int(output_tokens or 0),
            0,
        )

    def to_dict(self) -> dict[str, Any]:
        def timestamp(value: float | None) -> str | None:
            if value is None:
                return None

            return datetime.fromtimestamp(
                value,
                timezone.utc,
            ).isoformat()

        return {
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "kind": self.kind.value,
            "name": self.name,
            "start_time": timestamp(self.start_time),
            "end_time": timestamp(self.end_time),
            "duration_ms": round(
                self.duration_ms,
                2,
            ),
            "status": self.status.value,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "error": self.error,
            "attributes": dict(self.attributes),
        }


@dataclass
class Trace:
    trace_id: str = field(
        default_factory=lambda: uuid.uuid4().hex[:16]
    )
    thread_id: str = ""
    spans: list[TraceSpan] = field(
        default_factory=list
    )
    created_at: float = field(
        default_factory=time.time
    )

    def add_span(self, span: TraceSpan) -> None:
        self.spans.append(span)

    def get_root_spans(self) -> list[TraceSpan]:
        return [
            span
            for span in self.spans
            if span.parent_id is None
        ]

    def get_children(
        self,
        parent_id: str,
    ) -> list[TraceSpan]:
        return [
            span
            for span in self.spans
            if span.parent_id == parent_id
        ]

    def build_tree(self) -> list[dict[str, Any]]:
        def build(span: TraceSpan) -> dict[str, Any]:
            item = span.to_dict()
            children = self.get_children(
                span.span_id
            )

            if children:
                item["children"] = [
                    build(child)
                    for child in children
                ]

            return item

        return [
            build(span)
            for span in self.get_root_spans()
        ]

    def get_summary(self) -> dict[str, Any]:
        llm_spans = [
            span
            for span in self.spans
            if span.kind == SpanKind.LLM_CALL
        ]
        tool_spans = [
            span
            for span in self.spans
            if span.kind == SpanKind.TOOL_CALL
        ]

        nodes: dict[str, dict[str, Any]] = {}

        for span in self.spans:
            if span.kind == SpanKind.NODE:
                nodes[span.name] = {
                    "duration_ms": round(
                        span.duration_ms,
                        2,
                    ),
                    "status": span.status.value,
                }

        models: dict[str, dict[str, int]] = {}

        for span in llm_spans:
            if not span.model:
                continue

            model_stats = models.setdefault(
                span.model,
                {
                    "calls": 0,
                    "input_tokens": 0,
                    "output_tokens": 0,
                },
            )

            model_stats["calls"] += 1
            model_stats["input_tokens"] += (
                span.input_tokens
            )
            model_stats["output_tokens"] += (
                span.output_tokens
            )

        return {
            "trace_id": self.trace_id,
            "thread_id": self.thread_id,
            "total_spans": len(self.spans),
            "total_llm_calls": len(llm_spans),
            "total_tool_calls": len(tool_spans),
            "total_input_tokens": sum(
                span.input_tokens
                for span in self.spans
            ),
            "total_output_tokens": sum(
                span.output_tokens
                for span in self.spans
            ),
            "total_duration_ms": round(
                sum(
                    span.duration_ms
                    for span in self.get_root_spans()
                ),
                2,
            ),
            "nodes": nodes,
            "models": models,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "trace_id": self.trace_id,
            "thread_id": self.thread_id,
            "created_at": datetime.fromtimestamp(
                self.created_at,
                timezone.utc,
            ).isoformat(),
            "spans": self.build_tree(),
            "summary": self.get_summary(),
        }


class TraceStore:
    def __init__(
        self,
        max_traces_per_thread: int = 10,
        max_total_traces: int = 1000,
    ) -> None:
        self._traces: dict[str, list[Trace]] = {}
        self._lock = threading.RLock()
        self._max_per_thread = max(
            int(max_traces_per_thread),
            1,
        )
        self._max_total = max(
            int(max_total_traces),
            1,
        )

    def add_trace(
        self,
        thread_id: str,
        trace: Trace,
    ) -> None:
        with self._lock:
            history = self._traces.setdefault(
                thread_id,
                [],
            )
            history.append(trace)

            if len(history) > self._max_per_thread:
                del history[
                    :-self._max_per_thread
                ]

            while self._total_count() > self._max_total:
                oldest_thread = min(
                    self._traces,
                    key=lambda key: (
                        self._traces[key][0].created_at
                    ),
                )
                self._traces[oldest_thread].pop(0)

                if not self._traces[oldest_thread]:
                    del self._traces[oldest_thread]

    def _total_count(self) -> int:
        return sum(
            len(items)
            for items in self._traces.values()
        )

    def get_latest_trace(
        self,
        thread_id: str,
    ) -> Trace | None:
        with self._lock:
            history = self._traces.get(thread_id, [])
            return history[-1] if history else None

    def get_traces(
        self,
        thread_id: str,
    ) -> list[Trace]:
        with self._lock:
            return list(
                self._traces.get(thread_id, [])
            )

    def clear_thread(
        self,
        thread_id: str,
    ) -> None:
        with self._lock:
            self._traces.pop(thread_id, None)

    def clear_all(self) -> None:
        with self._lock:
            self._traces.clear()


class TracingContext:
    def __init__(
        self,
        thread_id: str,
        store: TraceStore,
    ) -> None:
        self.thread_id = thread_id
        self.store = store
        self.trace = Trace(thread_id=thread_id)
        self._stack: list[TraceSpan] = []

    @property
    def current_span(self) -> TraceSpan | None:
        return self._stack[-1] if self._stack else None

    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.CUSTOM,
        model: str = "",
        attributes: dict[str, Any] | None = None,
    ) -> TraceSpan:
        span = TraceSpan(
            parent_id=(
                self.current_span.span_id
                if self.current_span
                else None
            ),
            kind=kind,
            name=name,
            model=model,
            attributes=dict(attributes or {}),
        )

        self.trace.add_span(span)
        self._stack.append(span)
        return span

    def end_span(
        self,
        status: SpanStatus = SpanStatus.SUCCESS,
        error: str | None = None,
    ) -> None:
        if not self._stack:
            return

        span = self._stack.pop()
        span.finish(status, error)

    def finish(self) -> Trace:
        while self._stack:
            self.end_span()

        self.store.add_trace(
            self.thread_id,
            self.trace,
        )
        return self.trace


_trace_local = threading.local()
_store_lock = threading.Lock()
_store: TraceStore | None = None


def get_trace_store() -> TraceStore:
    global _store

    with _store_lock:
        if _store is None:
            from common.config import settings

            _store = TraceStore(
                max_total_traces=getattr(
                    settings,
                    "trace_buffer_size",
                    1000,
                ),
            )

        return _store


def get_current_context() -> TracingContext | None:
    return getattr(
        _trace_local,
        "context",
        None,
    )


def set_current_context(
    context: TracingContext | None,
) -> None:
    _trace_local.context = context


@contextmanager
def trace_request(
    thread_id: str,
) -> Iterator[TracingContext | None]:
    from common.config import settings

    if not getattr(
        settings,
        "enable_tracing",
        False,
    ):
        yield None
        return

    context = TracingContext(
        thread_id,
        get_trace_store(),
    )
    previous = get_current_context()
    set_current_context(context)

    try:
        yield context
    finally:
        context.finish()
        set_current_context(previous)


def record_span(
    name: str,
    kind: SpanKind = SpanKind.CUSTOM,
    *,
    model: str = "",
    input_tokens: int = 0,
    output_tokens: int = 0,
    duration_ms: float = 0.0,
    status: SpanStatus = SpanStatus.SUCCESS,
    error: str | None = None,
    attributes: dict[str, Any] | None = None,
) -> None:
    context = get_current_context()

    if context is None:
        return

    span = context.start_span(
        name,
        kind,
        model,
        attributes,
    )
    span.set_tokens(
        input_tokens,
        output_tokens,
    )

    if duration_ms > 0:
        span.start_time -= duration_ms / 1000

    context.end_span(
        status,
        error,
    )


def get_trace(
    thread_id: str,
) -> dict[str, Any] | None:
    trace = get_trace_store().get_latest_trace(
        thread_id,
    )
    return trace.to_dict() if trace else None


def get_trace_summary(
    thread_id: str,
) -> dict[str, Any] | None:
    trace = get_trace_store().get_latest_trace(
        thread_id,
    )
    return trace.get_summary() if trace else None


def get_all_traces(
    thread_id: str,
) -> list[dict[str, Any]]:
    return [
        trace.to_dict()
        for trace in get_trace_store().get_traces(
            thread_id,
        )
    ]


def trace_node(func: F) -> F:
    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        context = get_current_context()

        if context is None:
            return func(*args, **kwargs)

        with_trace = context.start_span(
            func.__name__,
            SpanKind.NODE,
        )

        try:
            result = func(*args, **kwargs)
        except Exception as exc:
            context.end_span(
                SpanStatus.ERROR,
                str(exc),
            )
            raise
        else:
            context.end_span()
            return result

    return wrapper  # type: ignore[return-value]