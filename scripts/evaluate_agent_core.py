"""Run a small, deterministic, no-model core evaluation and write evidence."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORE_CASES = {
    "followup_dispatch": "tests.test_coordinator.CoordinatorTests.test_evaluation_gap_returns_to_coordinator_for_followup_dispatch",
    "research_branch_isolation": "tests.test_research_tree_explorer.ResearchTreeExplorerTests.test_async_branches_keep_findings_and_sources_isolated",
    "approval_resume": "tests.test_runtime.AgentRuntimeAsyncTests.test_ainvoke_pauses_and_aresume_completes_tool_approval",
    "duplicate_resume": "tests.test_sessions_api.SessionApiTests.test_concurrent_duplicate_resume_only_runs_once",
    "sse_streaming": "tests.test_api.PublicApiTests.test_chat_sse_streams_model_tokens_before_completion",
}


def git_revision() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
            text=True, timeout=5, check=True,
        ).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    results = []
    for name, test_path in CORE_CASES.items():
        started = time.monotonic()
        suite = unittest.defaultTestLoader.loadTestsFromName(test_path)
        import io
        output = io.StringIO()
        result = unittest.TextTestRunner(stream=output, verbosity=1).run(suite)
        passed = result.wasSuccessful() and result.testsRun == 1 and not result.skipped
        results.append({
            "name": name, "test": test_path, "passed": passed,
            "duration_ms": round((time.monotonic() - started) * 1000, 2),
            "tests_run": result.testsRun, "skipped": len(result.skipped),
            "diagnostics": None if passed else output.getvalue()[-4000:],
        })
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed_count = sum(case["passed"] for case in results)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "mock_core_tests", "real_model_calls": False,
        "git_revision": git_revision(),
        "note": "受控模拟测试；不能作为真实模型任务成功率或报告质量。工作区可能有未提交修改。",
        "summary": {"cases": len(results), "passed": passed_count,
                    "failed": len(results) - passed_count},
        "cases": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"报告：{args.output}")
    return 0 if passed_count == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
