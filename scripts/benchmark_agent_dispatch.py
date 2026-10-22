"""Offline benchmark for coordinator dispatch and bounded research-agent retry.

This deterministic harness measures orchestration mechanics only. It does not
call a model, search provider, or application API, so its timings are not
representative of production model latency or answer quality.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agent.workflows.coordinator import coordinator_node  # noqa: E402
from agent.workflows.research_agent import ResearchAgent, ResearchTask  # noqa: E402

DEFAULT_TOPICS = [
    "市场规模与增长",
    "主要竞争者与方案",
    "成本、风险与限制",
]


def _run_agent(task: dict[str, Any], search: Any) -> dict[str, Any]:
    agent = ResearchAgent({"web_search": search})
    contract = ResearchTask.model_validate(task)

    def execute(context: Any) -> dict[str, str]:
        context.call_tool(
            "web_search",
            query=contract.topic,
            max_results=5,
        )
        return {"summary": f"完成：{contract.topic}"}

    return agent.run(contract, execute).model_dump(mode="json")


def _coordinator_dispatch(state: dict[str, Any]) -> dict[str, Any]:
    return coordinator_node(state, {})


def run_comparison(*, delay_ms: float = 40.0) -> dict[str, Any]:
    if delay_ms < 0 or delay_ms > 5000:
        raise ValueError("delay_ms 必须在 0 到 5000 之间。")

    delay_seconds = delay_ms / 1000.0

    def run_single_worker() -> dict[str, Any]:
        calls: dict[str, int] = {}
        lock = threading.Lock()

        def search(*, query: str, max_results: int) -> dict[str, Any]:
            with lock:
                calls[query] = calls.get(query, 0) + 1
                call_number = calls[query]
            time.sleep(delay_seconds)
            if query == DEFAULT_TOPICS[-1] and call_number == 1:
                raise RuntimeError("simulated transient provider error")
            return {"results": [{
                "title": query,
                "url": f"https://benchmark.invalid/{len(calls)}-{call_number}",
                "snippet": f"Deterministic evidence for {query}",
            }]}

        started = time.perf_counter()
        statuses = []
        for index, topic in enumerate(DEFAULT_TOPICS):
            task = {
                "task_id": f"linear-{index}",
                "topic": topic,
                "objective": f"Collect evidence for {topic}",
                "allowed_tools": ["web_search"],
            }
            result = _run_agent(task, search)
            statuses.append(result)
            if result["status"] == "failed":
                statuses.append(_run_agent(task, search))
        elapsed = time.perf_counter() - started
        completed = [item for item in statuses if item["status"] == "completed"]
        return {
            "elapsed_ms": round(elapsed * 1000, 2),
            "dispatched_tasks": len(DEFAULT_TOPICS),
            "completed_tasks": len({item["task_id"] for item in completed}),
            "retried_tasks": sum(1 for count in calls.values() if count > 1),
            "failed_attempts": sum(item["status"] == "failed" for item in statuses),
            "evidence_count": sum(len(item["findings"]) for item in completed),
            "covered_topics": sorted({
                finding["query"] for item in completed for finding in item["findings"]
            }),
            "search_calls": sum(calls.values()),
        }

    def run_coordinated_agents() -> dict[str, Any]:
        calls: dict[str, int] = {}
        lock = threading.Lock()

        def search(*, query: str, max_results: int) -> dict[str, Any]:
            with lock:
                calls[query] = calls.get(query, 0) + 1
                call_number = calls[query]
            time.sleep(delay_seconds)
            if query == DEFAULT_TOPICS[-1] and call_number == 1:
                raise RuntimeError("simulated transient provider error")
            return {"results": [{
                "title": query,
                "url": f"https://benchmark.invalid/{len(calls)}-{call_number}",
                "snippet": f"Deterministic evidence for {query}",
            }]}

        start = time.perf_counter()
        initial = _coordinator_dispatch({
            "input": "固定离线基准主题",
            "route": "deep",
            "research_plan": list(DEFAULT_TOPICS),
            "deepsearch_epoch": 1,
            "deepsearch_max_epochs": 3,
            "revision_count": 0,
            "max_revisions": 2,
        })
        tasks = initial.get("research_tasks", [])
        worker_limit = max(1, min(len(tasks), 3))
        with ThreadPoolExecutor(max_workers=worker_limit) as pool:
            results = list(pool.map(lambda task: _run_agent(task, search), tasks))

        retry_state = {
            "input": "固定离线基准主题",
            "route": "deep",
            "research_plan": list(DEFAULT_TOPICS),
            "research_tasks": tasks,
            "research_agent_results": results,
            "search_results": [{"url": "https://benchmark.invalid/seed"}],
            "draft_report": "临时草稿",
            "final_report": "临时草稿",
            "verdict": "revise",
            "missing_topics": [],
            "deepsearch_epoch": 1,
            "deepsearch_max_epochs": 3,
            "revision_count": 0,
            "max_revisions": 2,
        }
        retry_dispatch = _coordinator_dispatch(retry_state)
        retry_tasks = retry_dispatch.get("research_tasks", [])
        retry_count = len(retry_tasks)
        if retry_tasks:
            results.extend(_run_agent(task, search) for task in retry_tasks)

        elapsed = time.perf_counter() - start
        completed_by_task = {
            item["task_id"]: item
            for item in results
            if item["status"] == "completed"
        }
        covered_topics = sorted({
            finding["query"]
            for item in completed_by_task.values()
            for finding in item["findings"]
        })
        return {
            "elapsed_ms": round(elapsed * 1000, 2),
            "dispatched_tasks": len(tasks) + retry_count,
            "completed_tasks": len(completed_by_task),
            "retried_tasks": retry_count,
            "failed_attempts": sum(item["status"] == "failed" for item in results),
            "evidence_count": sum(len(item["findings"]) for item in completed_by_task.values()),
            "covered_topics": covered_topics,
            "search_calls": sum(calls.values()),
            "max_concurrency": worker_limit,
            "retry_epoch": retry_dispatch.get("deepsearch_epoch"),
        }

    linear = run_single_worker()
    coordinated = run_coordinated_agents()
    return {
        "benchmark": "coordinator_agent_dispatch_offline_simulation",
        "real_model_calls": False,
        "real_search_calls": False,
        "delay_ms_per_mock_search": delay_ms,
        "topics": list(DEFAULT_TOPICS),
        "single_worker_baseline": linear,
        "coordinator_agents": coordinated,
        "comparison": {
            "coverage_equal": linear["covered_topics"] == coordinated["covered_topics"],
            "agent_speedup_factor": (
                round(linear["elapsed_ms"] / coordinated["elapsed_ms"], 3)
                if coordinated["elapsed_ms"] > 0
                else None
            ),
            "simulated_latency_is_production_representative": False,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="离线比较 Coordinator 多 Agent 派单机制")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--delay-ms", type=float, default=40.0)
    args = parser.parse_args()
    report = run_comparison(delay_ms=args.delay_ms)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
