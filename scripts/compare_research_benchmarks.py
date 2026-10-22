"""Compare two saved Deep Research benchmark reports without network access."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


METRICS = (
    ("success_rate", "成功率"),
    ("failure_rate", "失败率"),
    ("quality_pass_rate", "质量通过率"),
    ("first_token_p50", "首 token P50 (ms)"),
    ("completion_p50", "完成耗时 P50 (ms)"),
    ("completion_p95", "完成耗时 P95 (ms)"),
    ("avg_model_calls", "平均模型调用次数"),
    ("avg_expected_field_coverage", "期望字段覆盖率"),
    ("avg_query_coverage_score", "查询覆盖率"),
    ("avg_citation_coverage", "引用覆盖率"),
    ("avg_freshness_ratio_30d", "30 天新鲜度"),
)


def _number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _report_metric(report: dict[str, Any], name: str) -> float | None:
    summary = report.get("summary") or {}
    metrics = report.get("metrics") or {}
    if name in summary:
        return _number(summary.get(name))
    if name == "quality_pass_rate":
        passed = _number(summary.get("quality_passed_cases"))
        executed = _number(summary.get("executed_cases"))
        return passed / executed if passed is not None and executed else None
    cases = report.get("cases") or []
    if name == "avg_model_calls" and isinstance(cases, list):
        values = [
            _number(
                ((case.get("execution") or {}).get("run_metrics") or {}).get(
                    "model_call_count"
                )
            )
            for case in cases
            if isinstance(case, dict)
        ]
        available = [value for value in values if value is not None]
        return sum(available) / len(available) if available else None
    if name == "avg_expected_field_coverage" and isinstance(cases, list):
        ratios = []
        for case in cases:
            if not isinstance(case, dict):
                continue
            fields = case.get("expected_fields") or []
            preview = str((case.get("execution") or {}).get("final_report_preview") or "").casefold()
            normalized = [str(field).strip().casefold() for field in fields if str(field).strip()]
            if normalized and preview:
                ratios.append(sum(field in preview for field in normalized) / len(normalized))
        return sum(ratios) / len(ratios) if ratios else None
    return _number(metrics.get(name))


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
) -> dict[str, Any]:
    rows = []
    for key, label in METRICS:
        before = _report_metric(baseline, key)
        after = _report_metric(candidate, key)
        rows.append({
            "metric": key,
            "label": label,
            "baseline": before,
            "candidate": after,
            "delta": round(after - before, 4) if before is not None and after is not None else None,
        })
    return {
        "baseline_mode": baseline.get("mode"),
        "candidate_mode": candidate.get("mode"),
        "baseline_execute": bool(baseline.get("execute")),
        "candidate_execute": bool(candidate.get("execute")),
        "metrics": rows,
        "note": (
            "耗时和调用次数的 delta 越小越好；质量、成功率和覆盖率的 delta 越大越好。"
        ),
    }


def compare_files(
    baseline_path: str | Path,
    candidate_path: str | Path,
    output: str | Path,
) -> dict[str, Any]:
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    candidate = json.loads(Path(candidate_path).read_text(encoding="utf-8"))
    if not isinstance(baseline, dict) or not isinstance(candidate, dict):
        raise ValueError("两个基准报告都必须是 JSON 对象。")
    report = compare_reports(baseline, candidate)
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="比较两份 Deep Research 基准报告。")
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    compare_files(args.baseline, args.candidate, args.output)
    print(f"对比报告已生成：{args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
