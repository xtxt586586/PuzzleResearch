"""Measure client-observed first-token and completion latency for chat modes."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.benchmark_deep_research import _iter_sse_events, _percentile  # noqa: E402

ROUTES: dict[str, dict[str, bool]] = {
    "direct": {"useWebSearch": False, "useAgent": False, "useDeepSearch": False},
    "web": {"useWebSearch": True, "useAgent": False, "useDeepSearch": False},
    "agent": {"useWebSearch": False, "useAgent": True, "useDeepSearch": False},
    "deep": {"useWebSearch": True, "useAgent": True, "useDeepSearch": True},
}


def _duration_values(mapping: dict[str, Any], *names: str) -> list[float]:
    values: list[float] = []
    for name in names:
        candidate = mapping.get(name, [])
        if isinstance(candidate, list):
            values.extend(
                float(value)
                for value in candidate
                if isinstance(value, (int, float)) and value >= 0
            )
    return values


def _phase_duration(
    phases: dict[str, list[float]],
    nodes: dict[str, list[float]],
    phase_name: str,
    *node_names: str,
) -> float | None:
    phase_values = _duration_values(phases, phase_name)
    if phase_values:
        return max(phase_values)
    node_values = _duration_values(nodes, *node_names)
    return round(sum(node_values), 2) if node_values else None


def _phase_max(
    phases: dict[str, list[float]],
    nodes: dict[str, list[float]],
    phase_name: str,
    *node_names: str,
) -> float | None:
    phase_values = _duration_values(phases, phase_name)
    if phase_values:
        return max(phase_values)
    node_values = _duration_values(nodes, *node_names)
    return max(node_values, default=None)


async def execute_chat_case(
    query: str,
    *,
    route: str,
    base_url: str,
    model: str = "",
    timeout_s: float = 120.0,
    api_key: str = "",
    transport: Any = None,
) -> dict[str, Any]:
    import httpx

    if route not in ROUTES:
        raise ValueError("route 必须是 direct、web、agent 或 deep。")

    started_at = time.monotonic()
    first_token_ms: float | None = None
    client_first_token_ms: float | None = None
    server_first_token_ms: float | None = None
    server_duration_ms: float | None = None
    effective_model: str | None = None
    node_durations_ms: dict[str, list[float]] = {}
    tool_durations_ms: dict[str, list[float]] = {}
    phase_durations_ms: dict[str, list[float]] = {}
    completion_ms: float | None = None
    answer_chars = 0
    streamed_chunks = 0
    error: str | None = None
    thread_id: str | None = None
    normalized_base_url = base_url.strip()

    is_asgi = normalized_base_url.lower() == "asgi"
    if is_asgi:
        import main

        transport = transport or httpx.ASGITransport(app=main.app)
        normalized_base_url = "http://puzzleresearch.local"
        api_key = str(main._active_api_key() or api_key or "")
    elif not normalized_base_url.startswith(("http://", "https://")):
        raise ValueError("base_url 必须是 http(s) 地址或 asgi。")

    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    request_payload: dict[str, Any] = {
        "messages": [{"role": "user", "content": query.strip()}],
        "search_mode": ROUTES[route],
    }
    if model.strip():
        request_payload["model"] = model.strip()

    options: dict[str, Any] = {
        "base_url": normalized_base_url,
        "headers": headers,
        "timeout": httpx.Timeout(timeout_s),
    }
    if transport is not None:
        options["transport"] = transport

    try:
        async with httpx.AsyncClient(**options) as client:
            async with client.stream(
                "POST",
                "/api/chat/sse",
                json=request_payload,
            ) as response:
                thread_id = response.headers.get("X-Thread-ID")
                if response.status_code >= 400:
                    error = f"聊天接口返回 HTTP {response.status_code}。"
                else:
                    async for event_name, data in _iter_sse_events(response.aiter_text()):
                        if not isinstance(data, dict):
                            if event_name == "error":
                                error = str(data or "请求失败。")[:500]
                            continue
                        content = data.get("content")
                        if event_name in {"text", "token", "completion", "message"} and isinstance(content, str) and content:
                            if event_name in {"text", "token"} and client_first_token_ms is None:
                                client_first_token_ms = round((time.monotonic() - started_at) * 1000, 2)
                                if not is_asgi:
                                    first_token_ms = client_first_token_ms
                            if event_name in {"text", "token"}:
                                streamed_chunks += 1
                            if event_name == "completion":
                                answer_chars = len(content)
                                completion_ms = round((time.monotonic() - started_at) * 1000, 2)
                            else:
                                answer_chars += len(content)
                        elif event_name == "error":
                            error = str(data.get("message") or data.get("error") or data)[:500]
                        elif event_name == "done":
                            completion_ms = completion_ms or round(
                                (time.monotonic() - started_at) * 1000,
                                2,
                            )
                            break

            completion_ms = completion_ms or round(
                (time.monotonic() - started_at) * 1000,
                2,
            )

            if thread_id:
                try:
                    metrics_response = await client.get(f"/api/runs/{thread_id}")
                    if metrics_response.status_code == 200:
                        run_metrics = metrics_response.json()
                        server_first_token_ms = run_metrics.get("first_token_ms")
                        server_duration_ms = run_metrics.get("duration_ms")
                        effective_model = str(run_metrics.get("model") or "").strip() or None
                        raw_node_durations = run_metrics.get("node_durations_ms")
                        if isinstance(raw_node_durations, dict):
                            node_durations_ms = raw_node_durations
                        raw_tool_durations = run_metrics.get("tool_durations_ms")
                        if isinstance(raw_tool_durations, dict):
                            tool_durations_ms = raw_tool_durations
                        raw_phase_durations = run_metrics.get("phase_durations_ms")
                        if isinstance(raw_phase_durations, dict):
                            phase_durations_ms = raw_phase_durations
                except (httpx.HTTPError, ValueError):
                    pass
    except httpx.TimeoutException:
        error = f"请求超过 {timeout_s:g} 秒。"
    except httpx.HTTPError as exc:
        error = f"请求失败：{type(exc).__name__}。"

    if is_asgi and server_first_token_ms is not None:
        first_token_ms = float(server_first_token_ms)

    client_duration_ms = completion_ms or round(
        (time.monotonic() - started_at) * 1000,
        2,
    )
    duration_ms = (
        float(server_duration_ms)
        if is_asgi and server_duration_ms is not None
        else client_duration_ms
    )

    return {
        "status": "failed" if error else "completed" if answer_chars else "empty",
        "route": route,
        "thread_id": thread_id,
        "first_token_ms": first_token_ms,
        "client_first_token_ms": client_first_token_ms,
        "server_first_token_ms": server_first_token_ms,
        "duration_ms": round(duration_ms, 2),
        "client_duration_ms": client_duration_ms,
        "server_duration_ms": server_duration_ms,
        "node_durations_ms": node_durations_ms,
        "tool_durations_ms": tool_durations_ms,
        "phase_durations_ms": phase_durations_ms,
        "search_duration_ms": max(
            _duration_values(tool_durations_ms, "web_search"),
            default=None,
        ) or _phase_max(
            phase_durations_ms,
            node_durations_ms,
            "search_duration_ms",
            "search",
            "deep_search",
            "deep_tree_search",
        ),
        "generation_duration_ms": _phase_duration(
            phase_durations_ms,
            node_durations_ms,
            "generation_duration_ms",
            "writer",
            "deep_writer",
            "revise",
        ),
        "generation_first_token_ms": max(
            _duration_values(phase_durations_ms, "generation_first_token_ms"),
            default=None,
        ),
        "planning_duration_ms": _phase_duration(
            phase_durations_ms,
            node_durations_ms,
            "planning_duration_ms",
            "web_plan",
            "deep_plan",
            "agent_plan",
        ),
        "evaluation_duration_ms": _phase_duration(
            phase_durations_ms,
            node_durations_ms,
            "evaluation_duration_ms",
            "evaluator",
        ),
        "effective_model": effective_model,
        "streamed_chunks": streamed_chunks,
        "latency_source": "server_runtime" if is_asgi else "client_observed",
        "answer_chars": answer_chars,
        "error": error,
    }


def run_benchmark(
    *,
    query: str,
    route: str,
    repeats: int = 1,
    base_url: str = "asgi",
    model: str = "",
    timeout_s: float = 120.0,
    api_key_env: str = "PUZZLERESEARCH_API_KEY",
    output: str | Path,
) -> dict[str, Any]:
    if not query.strip():
        raise ValueError("query 不能为空。")
    if route not in ROUTES:
        raise ValueError("route 必须是 direct、web、agent 或 deep。")
    if repeats < 1 or timeout_s <= 0:
        raise ValueError("repeats 和 timeout_s 必须大于 0。")

    api_key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
    results = [
        asyncio.run(
            execute_chat_case(
                query,
                route=route,
                base_url=base_url,
                model=model,
                timeout_s=timeout_s,
                api_key=api_key,
            )
        )
        for _ in range(repeats)
    ]
    completed = [item for item in results if item["status"] == "completed"]
    first_tokens = [item["first_token_ms"] for item in completed if item["first_token_ms"] is not None]
    durations = [item["duration_ms"] for item in completed]
    search_durations = [
        item["search_duration_ms"]
        for item in completed
        if isinstance(item.get("search_duration_ms"), (int, float))
    ]
    generation_durations = [
        item["generation_duration_ms"]
        for item in completed
        if isinstance(item.get("generation_duration_ms"), (int, float))
    ]
    generation_first_tokens = [
        item["generation_first_token_ms"]
        for item in completed
        if isinstance(item.get("generation_first_token_ms"), (int, float))
    ]
    planning_durations = [
        item["planning_duration_ms"]
        for item in completed
        if isinstance(item.get("planning_duration_ms"), (int, float))
    ]
    evaluation_durations = [
        item["evaluation_duration_ms"]
        for item in completed
        if isinstance(item.get("evaluation_duration_ms"), (int, float))
    ]

    def median(values: list[float]) -> float | None:
        return round(statistics.median(values), 2) if values else None

    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "model": model,
        "models_observed": sorted(
            {item["effective_model"] for item in results if item.get("effective_model")}
        ),
        "repeats": repeats,
        "query_chars": len(query.strip()),
        "base_url": base_url,
        "latency_ms": {
            "first_token_median": median(first_tokens),
            "first_token_p95": _percentile(first_tokens, 95),
            "completion_median": median(durations),
            "completion_p95": _percentile(durations, 95),
            "search_median": median(search_durations),
            "search_p95": _percentile(search_durations, 95),
            "generation_median": median(generation_durations),
            "generation_p95": _percentile(generation_durations, 95),
            "generation_first_token_median": median(generation_first_tokens),
            "generation_first_token_p95": _percentile(generation_first_tokens, 95),
            "planning_median": median(planning_durations),
            "planning_p95": _percentile(planning_durations, 95),
            "evaluation_median": median(evaluation_durations),
            "evaluation_p95": _percentile(evaluation_durations, 95),
        },
        "completed_runs": len(completed),
        "runs": results,
    }
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="测量 PuzzleResearch 聊天首 token 和完整回答耗时。")
    parser.add_argument("--query", required=True)
    parser.add_argument("--route", choices=tuple(ROUTES), default="direct")
    parser.add_argument("--repeats", type=int, default=1, help="默认为 1，避免意外产生多次模型请求。")
    parser.add_argument("--base-url", default="asgi", help="asgi 或运行中的后端 URL。")
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-s", type=float, default=120.0)
    parser.add_argument("--api-key-env", default="PUZZLERESEARCH_API_KEY")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        report = run_benchmark(
            query=args.query,
            route=args.route,
            repeats=args.repeats,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            api_key_env=args.api_key_env,
            output=args.output,
        )
    except (OSError, ValueError) as exc:
        print(f"基准运行失败：{exc}", file=sys.stderr)
        return 2
    print(json.dumps({"latency_ms": report["latency_ms"], "completed_runs": report["completed_runs"]}, ensure_ascii=False))
    return 0 if report["completed_runs"] == args.repeats else 1


if __name__ == "__main__":
    raise SystemExit(main())
