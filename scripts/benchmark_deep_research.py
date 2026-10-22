"""Build a reproducible, offline Deep Research benchmark report."""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import sys
import time
from collections.abc import AsyncIterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval.benchmarks.deep_research_bench_loader import load_benchmark_tasks  # noqa: E402

DEFAULT_BENCH_FILE = ROOT / "eval" / "benchmarks" / "sample_tasks.jsonl"
DEFAULT_GOLDEN_FILE = ROOT / "eval" / "golden_queries.json"
_TIME_MARKERS = (
    "latest", "recent", "today", "current", "update", "news",
    "最新", "近期", "最近", "当前", "今天", "动态", "新闻",
)


def _is_time_sensitive_query(query: str, constraints: dict[str, Any]) -> bool:
    text = str(query or "").strip().lower()
    if any(marker in text for marker in _TIME_MARKERS):
        return True
    if re.search(r"\b20\d{2}\b", text):
        return True
    days = constraints.get("freshness_days")
    return type(days) in {int, float} and 0 <= days <= 30


def _case_quality_targets(
    query: str,
    constraints: dict[str, Any],
    expected_fields: list[str],
    base_query_coverage_target: float,
    base_freshness_target: float,
) -> dict[str, Any]:
    freshness_days = constraints.get("freshness_days")
    if type(freshness_days) not in {int, float}:
        freshness_days = None
    else:
        freshness_days = int(freshness_days)

    field_count = len(expected_fields)
    complexity_bonus = 0.1 if field_count >= 4 else 0.05 if field_count >= 2 else 0.0
    coverage = min(1.0, max(0.0, float(base_query_coverage_target) + complexity_bonus))
    time_sensitive = _is_time_sensitive_query(query, constraints)
    freshness = min(
        1.0,
        max(0.0, float(base_freshness_target) + (0.15 if time_sensitive else 0.0)),
    )
    return {
        "time_sensitive": time_sensitive,
        "freshness_days_constraint": freshness_days,
        "query_coverage_target": round(coverage, 3),
        "freshness_ratio_target": round(freshness, 3),
    }


def _parse_sse_frame(frame: str) -> tuple[str, Any] | None:
    event_name = ""
    data_lines: list[str] = []
    for line in str(frame or "").replace("\r\n", "\n").split("\n"):
        if not line or line.startswith(":"):
            continue
        if line.startswith("event:"):
            event_name = line[6:].strip()
        elif line.startswith("data:"):
            data_lines.append(line[5:].lstrip())

    if not data_lines:
        return None
    raw_data = "\n".join(data_lines)
    try:
        payload: Any = json.loads(raw_data)
    except json.JSONDecodeError:
        payload = raw_data

    if isinstance(payload, dict) and "type" in payload and "data" in payload:
        event_name = event_name or str(payload.get("type") or "")
        payload = payload["data"]
    if not event_name:
        return None
    return event_name, payload


async def _iter_sse_events(
    text_stream: Any,
) -> AsyncIterator[tuple[str, Any]]:
    buffer = ""
    async for chunk in text_stream:
        if not chunk:
            continue
        buffer += chunk
        buffer = buffer.replace("\r\n", "\n")
        while "\n\n" in buffer:
            frame, buffer = buffer.split("\n\n", 1)
            parsed = _parse_sse_frame(frame)
            if parsed is not None:
                yield parsed
    if buffer.strip():
        parsed = _parse_sse_frame(buffer)
        if parsed is not None:
            yield parsed


async def _execute_research_case(
    query: str,
    *,
    mode: str,
    base_url: str,
    model: str,
    timeout_s: float,
    api_key: str = "",
    transport: Any = None,
    expected_fields: list[str] | None = None,
) -> dict[str, Any]:
    """Run one deep research request and collect its final evidence metrics."""
    import httpx

    query = str(query or "").strip()
    if not query:
        return {"status": "failed", "error": "查询内容不能为空。"}

    started_at = time.monotonic()
    first_token_ms: float | None = None
    thread_id: str | None = None
    final_report = ""
    error_message: str | None = None
    last_quality_update: dict[str, Any] | None = None
    cancelled = False
    app_module = None
    previous_mode: Any = None
    restore_mode = False

    normalized_base_url = str(base_url or "").strip()
    if normalized_base_url.lower() == "asgi":
        import main as app_module

        transport = transport or httpx.ASGITransport(app=app_module.app)
        normalized_base_url = "http://puzzleresearch.local"
        api_key = str(app_module._active_api_key() or api_key or "")
        backend_settings = getattr(app_module, "settings", None)
        if backend_settings is not None and hasattr(backend_settings, "deepsearch_mode"):
            try:
                previous_mode = backend_settings.deepsearch_mode
                backend_settings.deepsearch_mode = mode
                restore_mode = True
            except (AttributeError, TypeError, ValueError):
                restore_mode = False
    elif not normalized_base_url.startswith(("http://", "https://")):
        return {"status": "failed", "error": "base_url 必须是 http(s) 地址或 asgi。"}

    headers = {"Accept": "text/event-stream"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    async def consume_response(client: Any) -> None:
        nonlocal thread_id, final_report, error_message, last_quality_update, cancelled
        nonlocal first_token_ms
        payload: dict[str, Any] = {
            "query": query,
            "search_mode": {
                "useWebSearch": False,
                "useAgent": True,
                "useDeepSearch": True,
            },
        }
        if model.strip():
            payload["model"] = model.strip()

        async with client.stream(
            "POST",
            "/api/research/sse",
            json=payload,
            headers=headers,
        ) as response:
            thread_id = response.headers.get("X-Thread-ID")
            if response.status_code >= 400:
                error_message = f"研究接口返回 HTTP {response.status_code}。"
                return

            async for event_name, data in _iter_sse_events(response.aiter_text()):
                if event_name in {"text", "message", "completion"} and isinstance(data, dict):
                    content = data.get("content")
                    if isinstance(content, str) and content:
                        if event_name == "text":
                            if first_token_ms is None:
                                first_token_ms = round(
                                    (time.monotonic() - started_at) * 1000,
                                    2,
                                )
                            final_report += content
                        else:
                            final_report = content
                elif event_name == "quality_update" and isinstance(data, dict):
                    last_quality_update = data
                elif event_name == "error":
                    if isinstance(data, dict):
                        message = data.get("message") or data.get("error")
                        error_message = str(message or data)[:1000]
                    else:
                        error_message = str(data or "研究执行失败。")[:1000]
                elif event_name == "cancelled":
                    cancelled = True
                    break
                elif event_name == "done":
                    break

    client_options: dict[str, Any] = {
        "base_url": normalized_base_url,
        "headers": headers,
        "timeout": None,
    }
    if transport is not None:
        client_options["transport"] = transport

    try:
        async with httpx.AsyncClient(**client_options) as client:
            try:
                await asyncio.wait_for(
                    consume_response(client),
                    timeout=max(1.0, float(timeout_s)),
                )
            except asyncio.TimeoutError:
                if thread_id:
                    try:
                        await client.post(
                            f"/api/chat/cancel/{thread_id}",
                            json={"reason": "基准运行超时"},
                            timeout=5.0,
                        )
                    except Exception:
                        pass
                return {
                    "status": "timeout",
                    "thread_id": thread_id,
                    "duration_ms": round((time.monotonic() - started_at) * 1000, 2),
                    "first_token_ms": first_token_ms,
                    "error": f"运行超过 {timeout_s} 秒。",
                }

            duration_ms = round((time.monotonic() - started_at) * 1000, 2)
            if error_message:
                status = "failed"
            elif cancelled:
                status = "cancelled"
            else:
                status = "completed"

            run_metrics = None
            evidence_summary = None
            model_call_count = None
            if thread_id:
                try:
                    metrics_response = await client.get(f"/api/runs/{thread_id}")
                    if metrics_response.status_code == 200:
                        run_metrics = metrics_response.json()
                        evidence = run_metrics.get("evidence_summary")
                        if isinstance(evidence, dict):
                            evidence_summary = evidence
                    registry_response = await client.get("/api/runs")
                    if registry_response.status_code == 200:
                        runs = registry_response.json().get("runs", [])
                        if isinstance(runs, list):
                            run = next(
                                (
                                    item
                                    for item in runs
                                    if isinstance(item, dict)
                                    and item.get("run_id") == thread_id
                                ),
                                None,
                            )
                            if run is not None:
                                model_call_count = _maybe_float(
                                    run.get("model_call_count")
                                )
                except (httpx.HTTPError, ValueError):
                    pass

            return {
                "status": status,
                "thread_id": thread_id,
                "duration_ms": duration_ms,
                "first_token_ms": first_token_ms,
                "final_report_chars": len(final_report),
                "final_report_preview": final_report[:600],
                "expected_field_coverage": _expected_field_coverage(
                    final_report,
                    expected_fields,
                ),
                "model_call_count": model_call_count,
                "error": error_message,
                "last_quality_update": last_quality_update,
                "run_metrics": run_metrics,
                "evidence_summary": evidence_summary,
            }
    except Exception as exc:
        return {
            "status": "failed",
            "thread_id": thread_id,
            "duration_ms": round((time.monotonic() - started_at) * 1000, 2),
            "first_token_ms": first_token_ms,
            "error": f"请求失败：{type(exc).__name__}。",
        }
    finally:
        if restore_mode and app_module is not None:
            app_module.settings.deepsearch_mode = previous_mode


def _maybe_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return result if math.isfinite(result) else None


def _percentile(values: list[float], percentile: float) -> float | None:
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * min(max(percentile, 0.0), 100.0) / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return round(ordered[lower], 2)
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 2)


def _evaluate_case_quality(
    *,
    actual: dict[str, Any],
    targets: dict[str, Any],
    min_citation_coverage: float,
) -> dict[str, Any]:
    status = actual.get("status")
    if status != "completed":
        return {"quality_pass": False, "reason": f"研究状态为 {status or '未知'}。"}
    evidence = actual.get("evidence_summary")
    if not isinstance(evidence, dict):
        return {"quality_pass": False, "reason": "运行记录缺少 evidence_summary。"}

    query_coverage = _maybe_float(evidence.get("query_coverage_score"))
    citation_coverage = _maybe_float(evidence.get("citation_coverage"))
    freshness = _maybe_float(evidence.get("freshness_ratio_30d"))
    query_target = _maybe_float(targets.get("query_coverage_target"))
    freshness_target = _maybe_float(targets.get("freshness_ratio_target"))
    checks: dict[str, Any] = {
        "query_coverage": {
            "actual": query_coverage,
            "target": query_target,
            "pass": query_coverage is not None and query_target is not None and query_coverage >= query_target,
        },
        "citation_coverage": {
            "actual": citation_coverage,
            "target": min_citation_coverage,
            "pass": citation_coverage is not None and citation_coverage >= min_citation_coverage,
        },
    }
    if bool(targets.get("time_sensitive")):
        checks["freshness_ratio_30d"] = {
            "actual": freshness,
            "target": freshness_target,
            "pass": freshness is not None and freshness_target is not None and freshness >= freshness_target,
        }
    return {
        "quality_pass": all(check["pass"] for check in checks.values()),
        "checks": checks,
    }


def _expected_field_coverage(
    report: str,
    expected_fields: list[str] | None,
) -> float | None:
    if not expected_fields:
        return None
    text = str(report or "").casefold()
    fields = [str(field).strip().casefold() for field in expected_fields if str(field).strip()]
    if not fields:
        return None
    return round(sum(field in text for field in fields) / len(fields), 4)


def _load_golden_entries(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_file():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Golden 数据文件格式错误：{exc.msg}。") from exc
    if not isinstance(raw, list):
        raise ValueError("Golden 数据文件顶层必须是 JSON 数组。")
    entries: dict[str, dict[str, Any]] = {}
    for item in raw:
        if isinstance(item, dict) and isinstance(item.get("id"), str):
            entry_id = item["id"].strip()
            if entry_id:
                entries[entry_id] = item
    return entries


def build_benchmark_report(
    *,
    max_cases: int = 5,
    mode: str = "auto",
    bench_file: str | Path = DEFAULT_BENCH_FILE,
    golden_file: str | Path = DEFAULT_GOLDEN_FILE,
    min_query_coverage: float = 0.6,
    min_freshness_ratio: float = 0.4,
) -> dict[str, Any]:
    if mode not in {"auto", "tree", "linear"}:
        raise ValueError("mode 必须是 auto、tree 或 linear。")
    for name, value in (
        ("min_query_coverage", min_query_coverage),
        ("min_freshness_ratio", min_freshness_ratio),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} 必须在 0 到 1 之间。")

    tasks = load_benchmark_tasks(bench_file, max_cases=max_cases)
    golden = _load_golden_entries(Path(golden_file))
    cases: list[dict[str, Any]] = []
    coverage_targets: list[float] = []
    freshness_targets: list[float] = []

    for task in tasks:
        targets = _case_quality_targets(
            task.query,
            task.constraints,
            task.expected_fields,
            min_query_coverage,
            min_freshness_ratio,
        )
        coverage_targets.append(targets["query_coverage_target"])
        freshness_targets.append(targets["freshness_ratio_target"])
        cases.append({
            **task.to_dict(),
            "mode": mode,
            "golden_available": task.task_id in golden,
            "quality_targets": targets,
            "status": "ready",
        })

    def average(values: list[float]) -> float:
        return round(sum(values) / len(values), 3) if values else 0.0

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": mode,
        "execute": False,
        "benchmark_file": str(Path(bench_file)),
        "cases": cases,
        "summary": {
            "total_cases": len(cases),
            "executed_cases": 0,
            "quality_passed_cases": 0,
            "success_rate": None,
            "failure_rate": None,
            "avg_model_calls": None,
            "avg_expected_field_coverage": None,
            "golden_covered": sum(case["golden_available"] for case in cases),
            "time_sensitive_cases": sum(
                case["quality_targets"]["time_sensitive"] for case in cases
            ),
            "avg_query_coverage_target": average(coverage_targets),
            "avg_freshness_ratio_target": average(freshness_targets),
            "quality_gate_defaults": {
                "min_query_coverage": min_query_coverage,
                "min_freshness_ratio": min_freshness_ratio,
            },
        },
        "metrics": {
            "available": False,
            "avg_citation_coverage": None,
            "avg_freshness_ratio_30d": None,
            "avg_query_coverage_score": None,
            "unsupported_claims_total": None,
            "avg_expected_field_coverage": None,
            "avg_model_calls": None,
        },
    }


def run_benchmark(
    *,
    output: str | Path,
    max_cases: int = 5,
    mode: str = "auto",
    bench_file: str | Path = DEFAULT_BENCH_FILE,
    golden_file: str | Path = DEFAULT_GOLDEN_FILE,
    min_query_coverage: float = 0.6,
    min_freshness_ratio: float = 0.4,
    execute: bool = False,
    base_url: str = "asgi",
    model: str = "",
    timeout_s: float = 180.0,
    min_citation_coverage: float = 0.6,
    api_key_env: str = "PUZZLERESEARCH_API_KEY",
) -> dict[str, Any]:
    report = build_benchmark_report(
        max_cases=max_cases,
        mode=mode,
        bench_file=bench_file,
        golden_file=golden_file,
        min_query_coverage=min_query_coverage,
        min_freshness_ratio=min_freshness_ratio,
    )
    if execute:
        if timeout_s <= 0:
            raise ValueError("timeout_s 必须大于 0。")
        if not 0.0 <= min_citation_coverage <= 1.0:
            raise ValueError("min_citation_coverage 必须在 0 到 1 之间。")

        api_key = os.environ.get(api_key_env, "").strip() if api_key_env else ""
        evidence_summaries: list[dict[str, Any]] = []
        passed_cases = 0
        for case in report["cases"]:
            result = asyncio.run(
                _execute_research_case(
                    case["query"],
                    mode=mode,
                    base_url=base_url,
                    model=model,
                    timeout_s=timeout_s,
                    api_key=api_key,
                    expected_fields=case.get("expected_fields", []),
                )
            )
            case["execution"] = result
            case["status"] = result.get("status", "failed")
            case.update(
                _evaluate_case_quality(
                    actual=result,
                    targets=case["quality_targets"],
                    min_citation_coverage=min_citation_coverage,
                )
            )
            if case["quality_pass"]:
                passed_cases += 1
            evidence = result.get("evidence_summary")
            if isinstance(evidence, dict):
                evidence_summaries.append(evidence)

        def average_metric(name: str) -> float | None:
            values = [
                value
                for value in (
                    _maybe_float(summary.get(name))
                    for summary in evidence_summaries
                )
                if value is not None
            ]
            return round(sum(values) / len(values), 4) if values else None

        unsupported_counts = [
            value
            for value in (
                _maybe_float(summary.get("unsupported_claims_count"))
                for summary in evidence_summaries
            )
            if value is not None
        ]
        unsupported_total = (
            sum(int(value) for value in unsupported_counts)
            if unsupported_counts
            else None
        )
        model_call_counts = [
            value
            for value in (
                _maybe_float(case.get("execution", {}).get("model_call_count"))
                for case in report["cases"]
            )
            if value is not None
        ]
        field_coverages = [
            value
            for value in (
                _maybe_float(case.get("execution", {}).get("expected_field_coverage"))
                for case in report["cases"]
            )
            if value is not None
        ]
        completed_cases = sum(
            case.get("status") == "completed" for case in report["cases"]
        )
        executed_cases = len(report["cases"])
        metric_names = (
            "citation_coverage",
            "freshness_ratio_30d",
            "query_coverage_score",
            "unsupported_claims_count",
            "model_call_count",
        )
        metrics_available = any(
            _maybe_float(summary.get(name)) is not None
            for summary in evidence_summaries
            for name in metric_names
        )
        report.update({
            "execute": True,
            "base_url": base_url,
            "model": model,
            "timeout_s": timeout_s,
            "mode_applied_by": (
                "benchmark process"
                if str(base_url).strip().lower() == "asgi"
                else "remote backend configuration"
            ),
        })
        report["summary"].update({
            "executed_cases": len(report["cases"]),
            "quality_passed_cases": passed_cases,
            "success_rate": round(completed_cases / executed_cases, 4) if executed_cases else None,
            "failure_rate": round(1 - completed_cases / executed_cases, 4) if executed_cases else None,
            "avg_model_calls": round(sum(model_call_counts) / len(model_call_counts), 4) if model_call_counts else None,
            "avg_expected_field_coverage": round(sum(field_coverages) / len(field_coverages), 4) if field_coverages else None,
            "latency_ms": {
                "first_token_p50": _percentile(
                    [
                        value
                        for case in report["cases"]
                        if case.get("status") == "completed"
                        if (value := _maybe_float(case["execution"].get("first_token_ms"))) is not None
                    ],
                    50,
                ),
                "first_token_p95": _percentile(
                    [
                        value
                        for case in report["cases"]
                        if case.get("status") == "completed"
                        if (value := _maybe_float(case["execution"].get("first_token_ms"))) is not None
                    ],
                    95,
                ),
                "completion_p50": _percentile(
                    [
                        value
                        for case in report["cases"]
                        if case.get("status") == "completed"
                        if (value := _maybe_float(case["execution"].get("duration_ms"))) is not None
                    ],
                    50,
                ),
                "completion_p95": _percentile(
                    [
                        value
                        for case in report["cases"]
                        if case.get("status") == "completed"
                        if (value := _maybe_float(case["execution"].get("duration_ms"))) is not None
                    ],
                    95,
                ),
            },
            "quality_gate_defaults": {
                **report["summary"]["quality_gate_defaults"],
                "min_citation_coverage": min_citation_coverage,
            },
        })
        report["metrics"] = {
            "available": metrics_available,
            "citation_coverage": average_metric("citation_coverage"),
            "unsupported_claims": unsupported_total,
            "avg_citation_coverage": average_metric("citation_coverage"),
            "avg_freshness_ratio_30d": average_metric("freshness_ratio_30d"),
            "avg_query_coverage_score": average_metric("query_coverage_score"),
            "unsupported_claims_total": unsupported_total,
            "avg_expected_field_coverage": round(sum(field_coverages) / len(field_coverages), 4) if field_coverages else None,
            "avg_model_calls": round(sum(model_call_counts) / len(model_call_counts), 4) if model_call_counts else None,
        }

    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="生成离线 Deep Research 基准报告。"
    )
    parser.add_argument("--max-cases", type=int, default=5)
    parser.add_argument("--mode", choices=("auto", "tree", "linear"), default="auto")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bench-file", type=Path, default=DEFAULT_BENCH_FILE)
    parser.add_argument("--golden-file", type=Path, default=DEFAULT_GOLDEN_FILE)
    parser.add_argument("--min-query-coverage", type=float, default=0.6)
    parser.add_argument("--min-freshness-ratio", type=float, default=0.4)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="实际调用研究接口并统计运行证据（会消耗模型和搜索 API）。",
    )
    parser.add_argument(
        "--base-url",
        default="asgi",
        help="asgi（本地应用；使用本地模式设置）或后端 URL（模式需在服务端配置）。",
    )
    parser.add_argument("--model", default="")
    parser.add_argument("--timeout-s", type=float, default=180.0)
    parser.add_argument("--min-citation-coverage", type=float, default=0.6)
    parser.add_argument(
        "--api-key-env",
        default="PUZZLERESEARCH_API_KEY",
        help="后端 API Key 所在环境变量名；不会把密钥写入报告。",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        report = run_benchmark(
            output=args.output,
            max_cases=args.max_cases,
            mode=args.mode,
            bench_file=args.bench_file,
            golden_file=args.golden_file,
            min_query_coverage=args.min_query_coverage,
            min_freshness_ratio=args.min_freshness_ratio,
            execute=args.execute,
            base_url=args.base_url,
            model=args.model,
            timeout_s=args.timeout_s,
            min_citation_coverage=args.min_citation_coverage,
            api_key_env=args.api_key_env,
        )
    except (OSError, ValueError) as exc:
        print(f"基准报告生成失败：{exc}", file=sys.stderr)
        return 2
    print(f"基准报告已生成：{args.output}（{report['summary']['total_cases']} 个案例）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
