import json
import sys
import tempfile
import unittest
from types import SimpleNamespace
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from eval.benchmarks.deep_research_bench_loader import load_benchmark_tasks
from scripts.benchmark_deep_research import (
    _case_quality_targets,
    _evaluate_case_quality,
    _execute_research_case,
    _iter_sse_events,
    _parse_sse_frame,
    _percentile,
    build_benchmark_report,
    run_benchmark,
)


ROOT = Path(__file__).resolve().parents[1]


class BenchmarkLoaderTests(unittest.TestCase):
    def test_loads_sample_tasks(self):
        tasks = load_benchmark_tasks(
            ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl"
        )

        self.assertEqual(len(tasks), 3)
        self.assertEqual(tasks[0].task_id, "case_001")
        self.assertEqual(tasks[0].constraints["freshness_days"], 30)
        self.assertTrue(tasks[0].expected_fields)

    def test_respects_max_cases(self):
        tasks = load_benchmark_tasks(
            ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl",
            max_cases=2,
        )

        self.assertEqual(len(tasks), 2)

    def test_reports_invalid_json_line(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            valid_task = json.dumps({
                "id": "ok",
                "query": "研究这个主题",
                "expected_fields": ["结论"],
            }, ensure_ascii=False)
            path.write_text(valid_task + "\nnot-json\n", encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "第 2 行"):
                load_benchmark_tasks(path)

    def test_rejects_duplicate_task_ids(self):
        task = {
            "id": "duplicate",
            "query": "研究这个主题",
            "expected_fields": ["结论"],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tasks.jsonl"
            path.write_text(
                json.dumps(task, ensure_ascii=False)
                + "\n"
                + json.dumps(task, ensure_ascii=False),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "id 重复"):
                load_benchmark_tasks(path)


class BenchmarkReportTests(unittest.TestCase):
    def test_quality_targets_raise_freshness_and_complexity_thresholds(self):
        targets = _case_quality_targets(
            "近期行业数据",
            {"freshness_days": 14},
            ["数量", "趋势", "地区", "来源"],
            0.6,
            0.4,
        )

        self.assertTrue(targets["time_sensitive"])
        self.assertEqual(targets["freshness_days_constraint"], 14)
        self.assertEqual(targets["query_coverage_target"], 0.7)
        self.assertEqual(targets["freshness_ratio_target"], 0.55)

    def test_offline_report_marks_measurements_unavailable(self):
        report = build_benchmark_report(
            max_cases=2,
            bench_file=ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl",
        )

        self.assertFalse(report["execute"])
        self.assertEqual(report["summary"]["total_cases"], 2)
        self.assertEqual(report["summary"]["executed_cases"], 0)
        self.assertEqual(report["summary"]["time_sensitive_cases"], 2)
        self.assertFalse(report["metrics"]["available"])
        self.assertIsNone(report["metrics"]["avg_citation_coverage"])

    def test_run_benchmark_writes_readable_json_report(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "reports" / "benchmark.json"
            report = run_benchmark(
                output=output,
                max_cases=1,
                bench_file=ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl",
            )

            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["summary"]["total_cases"], 1)
        self.assertEqual(saved["cases"][0]["status"], "ready")
        self.assertEqual(saved["cases"][0]["id"], "case_001")

    def test_quality_evaluation_checks_query_citation_and_freshness(self):
        result = _evaluate_case_quality(
            actual={
                "status": "completed",
                "evidence_summary": {
                    "query_coverage_score": 0.8,
                    "citation_coverage": 0.75,
                    "freshness_ratio_30d": 0.6,
                },
            },
            targets={
                "query_coverage_target": 0.7,
                "freshness_ratio_target": 0.55,
                "time_sensitive": True,
            },
            min_citation_coverage=0.6,
        )

        self.assertTrue(result["quality_pass"])
        self.assertEqual(
            set(result["checks"]),
            {"query_coverage", "citation_coverage", "freshness_ratio_30d"},
        )

    def test_run_execute_mode_aggregates_metrics_without_network_calls(self):
        result = {
            "status": "completed",
            "evidence_summary": {
                "query_coverage_score": 0.8,
                "citation_coverage": 0.9,
                "freshness_ratio_30d": 0.7,
                "unsupported_claims_count": 2,
            },
        }
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.benchmark_deep_research._execute_research_case",
            new_callable=AsyncMock,
            return_value=result,
        ) as execute_case:
            report = run_benchmark(
                output=Path(directory) / "executed.json",
                max_cases=2,
                bench_file=ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl",
                execute=True,
                base_url="https://benchmark.invalid",
            )

        self.assertEqual(execute_case.await_count, 2)
        self.assertTrue(report["execute"])
        self.assertEqual(
            report["mode_applied_by"],
            "remote backend configuration",
        )
        self.assertEqual(report["summary"]["executed_cases"], 2)
        self.assertEqual(report["summary"]["quality_passed_cases"], 2)
        self.assertEqual(report["metrics"]["avg_citation_coverage"], 0.9)
        self.assertEqual(report["metrics"]["unsupported_claims_total"], 4)

    def test_run_benchmark_records_local_agent_mode_switch(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.benchmark_deep_research._execute_research_case",
            new_callable=AsyncMock,
            return_value={
                "status": "completed",
                "evidence_summary": {},
            },
        ) as execute_case:
            report = run_benchmark(
                output=Path(directory) / "hierarchical.json",
                max_cases=1,
                bench_file=ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl",
                execute=True,
                base_url="asgi",
                hierarchical_agents=True,
            )

        self.assertTrue(report["hierarchical_agents"])
        self.assertEqual(report["hierarchical_agents_applied_by"], "benchmark process")
        self.assertTrue(execute_case.await_args.kwargs["hierarchical_agents"])

    def test_execution_with_empty_evidence_does_not_claim_metrics_available(self):
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.benchmark_deep_research._execute_research_case",
            new_callable=AsyncMock,
            return_value={
                "status": "completed",
                "evidence_summary": {},
            },
        ):
            report = run_benchmark(
                output=Path(directory) / "executed.json",
                max_cases=1,
                bench_file=ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl",
                execute=True,
                base_url="https://benchmark.invalid",
            )

        self.assertFalse(report["metrics"]["available"])
        self.assertIsNone(report["metrics"]["avg_citation_coverage"])
        self.assertEqual(report["summary"]["quality_passed_cases"], 0)

    def test_agent_mode_comparison_rejects_remote_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "base_url=asgi"):
                run_benchmark(
                    output=Path(directory) / "remote.json",
                    max_cases=1,
                    bench_file=ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl",
                    execute=True,
                    base_url="https://benchmark.invalid",
                    hierarchical_agents=False,
                )

    def test_latency_percentiles_use_completed_samples(self):
        self.assertEqual(_percentile([300, 100, 200], 50), 200)
        self.assertEqual(_percentile([300, 100, 200], 95), 290)
        self.assertIsNone(_percentile([], 50))

        results = [
            {
                "status": "completed",
                "first_token_ms": first_token,
                "duration_ms": duration,
                "evidence_summary": {},
            }
            for first_token, duration in ((100, 1000), (200, 2000), (300, 4000))
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.benchmark_deep_research._execute_research_case",
            new_callable=AsyncMock,
            side_effect=results,
        ):
            report = run_benchmark(
                output=Path(directory) / "latency.json",
                max_cases=3,
                bench_file=ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl",
                execute=True,
                base_url="https://benchmark.invalid",
            )

        self.assertEqual(
            report["summary"]["latency_ms"],
            {
                "first_token_p50": 200,
                "first_token_p95": 290,
                "completion_p50": 2000,
                "completion_p95": 3800,
            },
        )


class BenchmarkExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_asgi_agent_mode_is_applied_during_request_and_restored(self):
        fake_settings = SimpleNamespace(
            deepsearch_mode="linear",
            use_hierarchical_agents=False,
        )
        original_graph = object()
        rebuilt_graph = object()
        fake_runtime = SimpleNamespace(
            graph=original_graph,
            _checkpointer=None,
            _store=None,
        )
        fake_main = SimpleNamespace(
            app=object(),
            settings=fake_settings,
            chat_runtime=fake_runtime,
            _active_api_key=lambda: "",
        )
        observed_modes = []

        def handler(request):
            observed_modes.append(fake_settings.use_hierarchical_agents)
            self.assertIs(fake_runtime.graph, rebuilt_graph)
            return httpx.Response(200, text='event: completion\ndata: {"content":"完成"}\n\nevent: done\ndata: {}\n\n')

        with (
            patch.dict(sys.modules, {"main": fake_main}),
            patch(
                "agent.core.graph.create_research_graph",
                return_value=rebuilt_graph,
            ) as create_graph,
        ):
            result = await _execute_research_case(
                "单个基准问题",
                mode="tree",
                base_url="asgi",
                model="",
                timeout_s=5,
                transport=httpx.MockTransport(handler),
                hierarchical_agents=True,
            )

        self.assertEqual(result["status"], "completed")
        self.assertTrue(observed_modes)
        self.assertTrue(all(observed_modes))
        create_graph.assert_called_once_with(checkpointer=None, store=None)
        self.assertIs(fake_runtime.graph, original_graph)
        self.assertFalse(fake_settings.use_hierarchical_agents)
        self.assertEqual(fake_settings.deepsearch_mode, "linear")

    async def test_sse_parser_unwraps_envelopes_and_ignores_comments(self):
        self.assertEqual(
            _parse_sse_frame(
                'event: completion\ndata: '
                '{"type":"completion","data":{"content":"完成"}}'
            ),
            ("completion", {"content": "完成"}),
        )
        self.assertIsNone(_parse_sse_frame(": keepalive\n"))

        async def chunks():
            yield 'event: text\ndata: {"type":"text",'
            yield '"data":{"content":"部分"}}\n\n'
            yield "event: done\ndata: {}\n\n"

        events = [event async for event in _iter_sse_events(chunks())]
        self.assertEqual(events, [("text", {"content": "部分"}), ("done", {})])

    async def test_execute_case_collects_stream_and_run_evidence(self):
        stream = """event: text
data: {"type":"text","data":{"content":"草稿"}}

event: quality_update
data: {"type":"quality_update","data":{"score":0.8}}

event: completion
data: {"type":"completion","data":{"content":"最终研究报告"}}

event: done
data: {"type":"done","data":{}}

"""
        requests = []

        def handler(request):
            requests.append(request)
            if request.method == "POST":
                self.assertEqual(request.url.path, "/api/research/sse")
                self.assertEqual(
                    json.loads(request.content)["search_mode"]["useDeepSearch"],
                    True,
                )
                return httpx.Response(
                    200,
                    headers={"X-Thread-ID": "thread-mock-001"},
                    text=stream,
                )
            if request.url.path == "/api/runs":
                return httpx.Response(
                    200,
                    json={"runs": [{"run_id": "thread-mock-001", "model_call_count": 3}]},
                )
            if request.url.path == "/api/traces/thread-mock-001/summary":
                return httpx.Response(200, json={"total_llm_calls": 3})
            self.assertEqual(request.url.path, "/api/runs/thread-mock-001")
            return httpx.Response(
                200,
                json={
                    "first_token_ms": 2100,
                    "evidence_summary": {
                        "citation_coverage": 0.9,
                        "query_coverage_score": 0.8,
                        "freshness_ratio_30d": 0.7,
                        "unsupported_claims_count": 1,
                    }
                },
            )

        result = await _execute_research_case(
            "研究主题",
            mode="tree",
            base_url="https://benchmark.invalid",
            model="deepseek-chat",
            timeout_s=10,
            expected_fields=["最终研究报告", "不存在的字段"],
            api_key="service-key",
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["thread_id"], "thread-mock-001")
        self.assertEqual(result["final_report_preview"], "最终研究报告")
        self.assertEqual(result["last_quality_update"], {"score": 0.8})
        self.assertIsNotNone(result["first_token_ms"])
        self.assertEqual(result["first_token_ms"], 2100)
        self.assertIsNotNone(result["client_first_text_ms"])
        self.assertEqual(result["evidence_summary"]["citation_coverage"], 0.9)
        self.assertEqual(result["model_call_count"], 3)
        self.assertEqual(result["expected_field_coverage"], 0.5)
        self.assertEqual(result["trace_summary"]["total_llm_calls"], 3)
        self.assertEqual(len(requests), 4)
        self.assertTrue(all(request.headers["authorization"] == "Bearer service-key" for request in requests))

    async def test_execute_case_returns_http_error_without_metrics_fetch(self):
        requests = []

        def handler(request):
            requests.append(request)
            return httpx.Response(503, text="service unavailable")

        result = await _execute_research_case(
            "研究主题",
            mode="auto",
            base_url="https://benchmark.invalid",
            model="",
            timeout_s=10,
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result["status"], "failed")
        self.assertIn("503", result["error"])
        self.assertEqual(len(requests), 1)

    async def test_truncated_stream_is_not_counted_as_completed(self):
        def handler(request):
            return httpx.Response(200, text='event: text\ndata: {"content":"部分报告"}\n\n')

        result = await _execute_research_case(
            "研究主题", mode="auto", base_url="https://benchmark.invalid",
            model="", timeout_s=5, transport=httpx.MockTransport(handler),
        )
        self.assertEqual(result["status"], "failed")
        self.assertIn("流未正常结束", result["error"])


if __name__ == "__main__":
    unittest.main()
