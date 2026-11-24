import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import httpx

from scripts.benchmark_chat_latency import execute_chat_case, run_benchmark


class ChatLatencyBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    async def test_execute_chat_case_collects_first_token_and_final_answer(self):
        request_data = {}

        def handler(request):
            if request.method == "GET":
                return httpx.Response(
                    200,
                json={
                    "first_token_ms": 120,
                    "duration_ms": 240,
                    "model": "deepseek-chat",
                    "node_durations_ms": {"agent_fast_path": [210]},
                    "tool_durations_ms": {"web_search": [80]},
                    "phase_durations_ms": {
                        "search_duration_ms": [75],
                        "generation_duration_ms": [140],
                        "generation_first_token_ms": [110],
                    },
                },
                )
            request_data["path"] = request.url.path
            request_data["payload"] = json.loads(request.content)
            return httpx.Response(
                200,
                headers={"X-Thread-ID": "thread-latency-test"},
                text=(
                    'event: text\ndata: {"type":"text","data":{"content":"部分"}}\n\n'
                    'event: completion\ndata: {"type":"completion","data":{"content":"完整回答"}}\n\n'
                    'event: done\ndata: {"type":"done","data":{}}\n\n'
                ),
            )

        result = await execute_chat_case(
            "解释缓存",
            route="direct",
            base_url="https://benchmark.invalid",
            model="deepseek-chat",
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(request_data["path"], "/api/chat/sse")
        self.assertEqual(
            request_data["payload"]["search_mode"],
            {"useWebSearch": False, "useAgent": False, "useDeepSearch": False},
        )
        self.assertEqual(request_data["payload"]["model"], "deepseek-chat")
        self.assertEqual(result["status"], "completed")
        self.assertEqual(result["thread_id"], "thread-latency-test")
        self.assertIsNotNone(result["first_token_ms"])
        self.assertEqual(result["server_first_token_ms"], 120)
        self.assertEqual(result["server_duration_ms"], 240)
        self.assertEqual(result["effective_model"], "deepseek-chat")
        self.assertEqual(result["node_durations_ms"], {"agent_fast_path": [210]})
        self.assertEqual(result["tool_durations_ms"], {"web_search": [80]})
        self.assertEqual(result["search_duration_ms"], 80)
        self.assertEqual(result["generation_duration_ms"], 140)
        self.assertEqual(result["generation_first_token_ms"], 110)
        self.assertEqual(result["answer_chars"], len("完整回答"))
        self.assertEqual(result["streamed_chunks"], 1)

    async def test_completion_only_response_does_not_claim_a_streamed_first_token(self):
        def handler(_request):
            if _request.method == "GET":
                return httpx.Response(404, json={"detail": "missing"})
            return httpx.Response(
                200,
                headers={"X-Thread-ID": "thread-completion-only"},
                text=(
                    'event: completion\ndata: {"type":"completion","data":{"content":"最终回答"}}\n\n'
                    'event: done\ndata: {"type":"done","data":{}}\n\n'
                ),
            )

        result = await execute_chat_case(
            "测试",
            route="direct",
            base_url="https://benchmark.invalid",
            transport=httpx.MockTransport(handler),
        )

        self.assertEqual(result["status"], "completed")
        self.assertIsNone(result["first_token_ms"])
        self.assertEqual(result["streamed_chunks"], 0)


class ChatLatencyReportTests(unittest.TestCase):
    def test_phase_duration_falls_back_to_node_timings(self):
        from scripts.benchmark_chat_latency import _phase_duration, _phase_max

        self.assertEqual(
            _phase_max(
                {},
                {"search": [4000, 5600]},
                "search_duration_ms",
                "search",
            ),
            5600,
        )
        self.assertEqual(
            _phase_duration(
                {},
                {"writer": [4800], "revise": [4700, 4600]},
                "generation_duration_ms",
                "writer",
                "revise",
            ),
            14100,
        )

    def test_report_summarizes_repeated_latency_samples_without_network(self):
        results = [
            {"status": "completed", "first_token_ms": 100, "duration_ms": 500, "search_duration_ms": 50, "generation_duration_ms": 200, "generation_first_token_ms": 70, "answer_chars": 20},
            {"status": "completed", "first_token_ms": 200, "duration_ms": 800, "search_duration_ms": 100, "generation_duration_ms": 300, "generation_first_token_ms": 130, "answer_chars": 25},
            {"status": "failed", "first_token_ms": None, "duration_ms": 1000, "search_duration_ms": None, "generation_duration_ms": None, "generation_first_token_ms": None, "answer_chars": 0},
        ]
        with tempfile.TemporaryDirectory() as directory, patch(
            "scripts.benchmark_chat_latency.execute_chat_case",
            new_callable=AsyncMock,
            side_effect=results,
        ):
            output = Path(directory) / "latency.json"
            report = run_benchmark(
                query="解释缓存",
                route="direct",
                repeats=3,
                output=output,
            )
            saved = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(report["completed_runs"], 2)
        self.assertEqual(report["latency_ms"]["first_token_median"], 150)
        self.assertEqual(report["latency_ms"]["completion_median"], 650)
        self.assertEqual(report["latency_ms"]["search_median"], 75)
        self.assertEqual(report["latency_ms"]["generation_median"], 250)
        self.assertEqual(report["latency_ms"]["generation_first_token_median"], 100)
        self.assertEqual(saved["route"], "direct")
        self.assertEqual(len(saved["runs"]), 3)


if __name__ == "__main__":
    unittest.main()
