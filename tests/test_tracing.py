import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from common.tracing import (
    SpanKind,
    SpanStatus,
    TraceStore,
    trace_request,
)


class TracingTests(unittest.TestCase):
    def test_nested_spans_build_tree(self):
        store = TraceStore()

        with patch.object(
            main.settings,
            "enable_tracing",
            True,
        ):
            with trace_request(
                "trace-thread-001"
            ) as context:
                self.assertIsNotNone(context)

                root = context.start_span(
                    "agent_run",
                    SpanKind.NODE,
                )
                child = context.start_span(
                    "python_executor",
                    SpanKind.TOOL_CALL,
                )

                context.end_span()
                context.end_span()

        trace = store.get_latest_trace(
            "trace-thread-001"
        )

        self.assertIsNone(trace)

    def test_trace_store_enforces_thread_limit(self):
        store = TraceStore(
            max_traces_per_thread=2,
            max_total_traces=10,
        )

        from common.tracing import Trace

        for index in range(3):
            store.add_trace(
                "trace-thread-002",
                Trace(
                    thread_id=(
                        f"trace-thread-002-{index}"
                    )
                ),
            )

        self.assertEqual(
            len(
                store.get_traces(
                    "trace-thread-002"
                )
            ),
            2,
        )

    def test_disabled_tracing_does_not_create_trace(self):
        from common.tracing import get_trace

        with patch.object(
            main.settings,
            "enable_tracing",
            False,
        ):
            with trace_request(
                "trace-disabled-001"
            ) as context:
                self.assertIsNone(context)

        self.assertIsNone(
            get_trace(
                "trace-disabled-001"
            )
        )

    def test_trace_api_rejects_when_disabled(self):
        client = TestClient(main.app)

        with patch.object(
            main.settings,
            "enable_tracing",
            False,
        ):
            response = client.get(
                "/api/traces/trace-disabled-002"
            )

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_span_error_status(self):
        store = TraceStore()

        from common.tracing import (
            Trace,
            TraceSpan,
        )

        trace = Trace(
            thread_id="trace-thread-003"
        )
        span = TraceSpan(
            kind=SpanKind.TOOL_CALL,
            name="broken_tool",
        )
        span.finish(
            SpanStatus.ERROR,
            "工具失败",
        )
        trace.add_span(span)
        store.add_trace(
            "trace-thread-003",
            trace,
        )

        payload = (
            store
            .get_latest_trace(
                "trace-thread-003"
            )
            .to_dict()
        )

        self.assertEqual(
            payload["spans"][0]["status"],
            "error",
        )
        self.assertEqual(
            payload["spans"][0]["error"],
            "工具失败",
        )


if __name__ == "__main__":
    unittest.main()