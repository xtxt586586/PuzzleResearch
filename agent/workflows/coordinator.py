from __future__ import annotations

import json
import hashlib
from collections.abc import Mapping
from typing import Any, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.core.model_factory import ModelFactory
from agent.core.state import AgentState
from agent.prompts import get_prompt_manager
from agent.workflows.research_budget import (
    add_budget_tokens,
    budget_state_fields,
    estimate_tokens,
    model_call_tokens,
    new_research_budget,
    refresh_research_budget,
)
from common.config import settings


CoordinatorAction = Literal[
    "plan",
    "research",
    "synthesize",
    "reflect",
    "complete",
]

_ACTIONS = {
    "plan",
    "research",
    "synthesize",
    "reflect",
    "complete",
}

_COORDINATOR_ACTION_CONTRACT = """可选 action：
- plan：尚无研究计划时制定计划
- research：针对知识缺口继续搜索
- synthesize：根据现有证据修订或综合报告
- reflect：重新审视研究方向，并继续补充研究
- complete：研究已满足要求，或预算已耗尽

只返回 JSON：{"action":"...","reasoning":"...","priority_topics":["..."]}
"""


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        return "\n".join(
            str(block.get("text", ""))
            for block in content
            if isinstance(block, Mapping)
        ).strip()

    return str(content or "").strip()


def _parse_decision(response: Any) -> tuple[str, str, list[str]]:
    content = _response_text(response)
    decoder = json.JSONDecoder()

    for index, character in enumerate(content):
        if character != "{":
            continue

        try:
            payload, _ = decoder.raw_decode(content[index:])
        except json.JSONDecodeError:
            continue

        if not isinstance(payload, dict):
            continue

        action = str(payload.get("action", "")).strip().lower()
        if action not in _ACTIONS:
            continue

        raw_topics = payload.get("priority_topics", [])
        if isinstance(raw_topics, str):
            raw_topics = [raw_topics]
        if not isinstance(raw_topics, list):
            raw_topics = []

        topics = list(
            dict.fromkeys(
                str(topic).strip()[:200]
                for topic in raw_topics
                if str(topic).strip()
            )
        )[:8]

        return (
            action,
            str(payload.get("reasoning", "")).strip()[:1000],
            topics,
        )

    raise ValueError("协调器返回了无效的决策 JSON")


def _build_research_tasks(
    state: Mapping[str, Any],
    topics: list[str],
    retry_counts: Mapping[str, int] | None = None,
) -> list[dict[str, Any]]:
    """Create stable, bounded task contracts without an additional model call."""
    overall_topic = str(state.get("input", "")).strip()[:500]
    missing = {
        " ".join(str(topic).split()).casefold()
        for topic in (state.get("missing_topics") or [])
        if str(topic).strip()
    }
    try:
        task_limit = max(1, min(8, int(getattr(settings, "deepsearch_tree_max_branches", 4))))
    except (TypeError, ValueError):
        task_limit = 4
    tasks: list[dict[str, Any]] = []
    seen_topics: set[str] = set()
    for topic in topics:
        normalized = " ".join(str(topic).split())[:500]
        canonical_topic = normalized.casefold()
        if not normalized or canonical_topic in seen_topics:
            continue
        if len(tasks) >= task_limit:
            break
        seen_topics.add(canonical_topic)
        index = len(tasks)
        digest = hashlib.sha1(
            f"{overall_topic.casefold()}|{canonical_topic}|"
            f"{max(int(state.get('deepsearch_epoch', 1) or 1), 1)}".encode("utf-8")
        ).hexdigest()[:12]
        is_gap = canonical_topic in missing
        tasks.append({
            "task_id": f"research-{digest}",
            "topic": normalized,
            "objective": (
                f"补足研究缺口并收集可核查证据：{normalized}"
                if is_gap
                else f"针对研究问题收集可核查证据：{normalized}"
            ),
            "priority": 5 if is_gap else max(1, 4 - min(index, 3)),
            "context": {
                "overall_topic": overall_topic,
                "focus_topic": normalized,
                "related_topics": [
                    str(item)[:500]
                    for item in (state.get("research_plan") or [])[:8]
                    if " ".join(str(item).split()).casefold() != canonical_topic
                ],
                "known_gaps": [str(item)[:500] for item in (state.get("missing_topics") or [])[:8]],
                "epoch": max(int(state.get("deepsearch_epoch", 1) or 1), 1),
                "retry_count": max(
                    0,
                    int((retry_counts or {}).get(normalized.casefold(), 0) or 0),
                ),
            },
            "allowed_tools": ["web_search"],
        })
    return tasks


def _research_budget_for_state(
    state: Mapping[str, Any],
    config: RunnableConfig | None,
) -> dict[str, Any]:
    configurable = (
        config.get("configurable", {})
        if isinstance(config, Mapping)
        else {}
    )
    if not isinstance(configurable, Mapping):
        configurable = {}

    def configured(name: str, default: Any, cast: Any) -> Any:
        try:
            return cast(configurable.get(name, getattr(settings, name, default)))
        except (TypeError, ValueError):
            return cast(default)

    artifacts = state.get("deepsearch_artifacts") or {}
    if not isinstance(artifacts, Mapping):
        artifacts = {}
    budget = new_research_budget(
        artifacts,
        max_seconds=max(configured("deepsearch_max_seconds", 0.0, float), 0.0),
        max_tokens=max(configured("deepsearch_max_tokens", 0, int), 0),
        initial_tokens=estimate_tokens(state.get("input", "")),
    )
    stop_reason = str(
        state.get("budget_stop_reason")
        or artifacts.get("budget_stop_reason")
        or ""
    ).strip()
    if stop_reason and not budget["stop_reason"]:
        budget["stop_reason"] = stop_reason
    return budget


def coordinator_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Choose the next Deep Research action while enforcing hard budgets."""

    research_plan = state.get("research_plan") or []
    search_results = state.get("search_results") or []
    report = str(
        state.get("draft_report") or state.get("final_report") or ""
    ).strip()
    missing_topics = [
        str(topic).strip()
        for topic in (state.get("missing_topics") or [])
        if str(topic).strip()
    ]
    assigned_tasks = {
        str(task.get("task_id", "")): task
        for task in (state.get("research_tasks") or [])
        if isinstance(task, Mapping) and task.get("task_id")
    }
    retry_counts: dict[str, int] = {}
    retry_topics: list[str] = []
    for agent_result in state.get("research_agent_results") or []:
        if not isinstance(agent_result, Mapping):
            continue
        if str(agent_result.get("status", "")).strip().lower() not in {"failed", "empty"}:
            continue
        if agent_result.get("stop_reason"):
            continue
        task = assigned_tasks.get(str(agent_result.get("task_id", "")))
        if not isinstance(task, Mapping):
            continue
        context = task.get("context") or {}
        retry_count = int(context.get("retry_count", 0) or 0) if isinstance(context, Mapping) else 0
        topic = str(task.get("topic", "")).strip()
        if topic and retry_count < 1 and topic.casefold() not in retry_counts:
            retry_counts[topic.casefold()] = retry_count + 1
            retry_topics.append(topic)

    epoch = max(int(state.get("deepsearch_epoch", 1)), 1)
    max_epochs = max(int(state.get("deepsearch_max_epochs", 3)), 1)
    revision_count = max(int(state.get("revision_count", 0)), 0)
    max_revisions = max(int(state.get("max_revisions", 2)), 0)
    can_research = epoch < max_epochs
    can_revise = revision_count < max_revisions
    budget = _research_budget_for_state(state, config)

    def budget_safe_action() -> tuple[str, str]:
        if not report:
            if search_results:
                return "synthesize", "预算已耗尽，使用已有证据生成报告。"
            return "complete", "预算已耗尽且没有可用证据，停止研究。"
        return "complete", "研究预算已耗尽，不再调用模型修订现有报告。"

    if state.get("is_cancelled") or state.get("status") in {
        "cancelled",
        "failed",
    }:
        return {
            "coordinator_action": "complete",
            "coordinator_reasoning": "任务已取消或失败，停止协调。",
            **budget_state_fields(budget),
        }

    if budget["stop_reason"]:
        action, reasoning = budget_safe_action()
        topics = []
    elif not research_plan:
        action = "plan"
        reasoning = "研究尚未规划，先生成研究计划。"
        topics: list[str] = []
    elif not report:
        if search_results:
            action = "synthesize"
            reasoning = "已有搜索证据但尚无报告，进入综合阶段。"
            topics = []
        elif retry_topics and can_research:
            action = "research"
            reasoning = "研究分支无可用证据，按上限重试失败任务。"
            topics = retry_topics
        elif retry_topics:
            action = "complete"
            reasoning = "研究轮次已耗尽，不再重试失败分支。"
            topics = []
        elif state.get("research_agent_results"):
            action = "complete"
            reasoning = "研究分支重试后仍未产出证据，停止自动派单。"
            topics = []
        else:
            action = "research"
            reasoning = "研究计划已具备，交由独立研究 Agent 并行收集证据。"
            topics = [str(topic).strip() for topic in research_plan if str(topic).strip()]
    elif str(state.get("verdict", "pass")).strip().lower() == "pass" and not missing_topics:
        action = "complete"
        reasoning = "评估通过且没有未解决的研究缺口。"
        topics = []
    elif retry_topics and can_research:
        action = "research"
        reasoning = "重试一次失败或无结果的研究分支，并与评估缺口一并补充。"
        topics = list(dict.fromkeys([*retry_topics, *missing_topics]))
    elif not can_research and not can_revise:
        action = "complete"
        reasoning = "研究轮次和修订预算均已耗尽。"
        topics = []
    elif not can_research:
        action = "synthesize"
        reasoning = "研究轮次已耗尽，使用现有证据修订报告。"
        topics = []
    elif not can_revise:
        action = "research"
        reasoning = "修订预算已耗尽，优先补充证据后再评估。"
        topics = missing_topics
    else:
        fallback_action = "research" if can_research else "synthesize"
        messages = [
            SystemMessage(
                content=(
                    f"{get_prompt_manager(config).get_prompt('research_coordinator')}\n\n"
                    f"{_COORDINATOR_ACTION_CONTRACT}"
                )
            ),
            HumanMessage(
                content=json.dumps(
                    {
                        "topic": state.get("input", ""),
                        "plan": research_plan,
                        "search_result_count": len(search_results),
                        "report_exists": bool(report),
                        "evaluation": state.get("evaluation", ""),
                        "verdict": state.get("verdict", "revise"),
                        "missing_topics": missing_topics,
                        "quality_score": state.get("quality_overall_score"),
                        "quality_gap_count": state.get("quality_gap_count"),
                        "citation_coverage": state.get("citation_coverage_score"),
                        "epoch": epoch,
                        "max_epochs": max_epochs,
                        "revision_count": revision_count,
                        "max_revisions": max_revisions,
                    },
                    ensure_ascii=False,
                )
            ),
        ]
        budget = refresh_research_budget(budget)
        max_tokens = max(int(budget.get("max_tokens", 0) or 0), 0)
        would_exceed_tokens = (
            max_tokens
            and int(budget.get("tokens_used", 0) or 0) + estimate_tokens(messages) >= max_tokens
        )
        if would_exceed_tokens and not budget["stop_reason"]:
            budget["stop_reason"] = "token_budget_exceeded"

        if budget["stop_reason"]:
            action, reasoning = budget_safe_action()
            topics = []
        else:
            try:
                model = ModelFactory.for_task(
                    "routing",
                    temperature=0.0,
                    config=config,
                )
                response = model.invoke(messages, config=config)
                budget = add_budget_tokens(
                    budget,
                    model_call_tokens(messages, response),
                )
                action, reasoning, topics = _parse_decision(response)
            except Exception as exc:
                action = fallback_action
                reasoning = f"协调器不可用，按预算规则继续：{exc}"
                topics = missing_topics if action == "research" else []

        if action in {"research", "reflect", "plan"} and not can_research:
            action = "synthesize" if can_revise else "complete"
            topics = []
        elif action == "synthesize" and not can_revise:
            action = "research" if can_research else "complete"
            topics = missing_topics if action == "research" else []
        elif action == "complete" and (
            str(state.get("verdict", "revise")).strip().lower() != "pass"
            or missing_topics
        ):
            action = fallback_action
            topics = missing_topics if action == "research" else []

        budget = refresh_research_budget(budget)
        if budget["stop_reason"]:
            action, budget_reasoning = budget_safe_action()
            reasoning = f"{reasoning} {budget_reasoning}".strip()
            topics = []

    result: dict[str, Any] = {
        "coordinator_action": action,
        "coordinator_reasoning": reasoning,
    }
    artifacts = dict(state.get("deepsearch_artifacts") or {})
    artifacts["research_budget"] = refresh_research_budget(budget)
    artifacts["budget_stop_reason"] = artifacts["research_budget"]["stop_reason"]
    summary = artifacts.get("quality_summary")
    quality_summary = dict(summary) if isinstance(summary, Mapping) else {}
    quality_summary.update({
        **budget_state_fields(artifacts["research_budget"]),
        "stop_reason": artifacts["research_budget"]["stop_reason"],
        "tokens_used": artifacts["research_budget"]["tokens_used"],
        "elapsed_seconds": artifacts["research_budget"]["elapsed_seconds"],
        "searches_used": artifacts["research_budget"]["searches_used"],
        "max_seconds": artifacts["research_budget"]["max_seconds"],
        "max_tokens": artifacts["research_budget"]["max_tokens"],
    })
    artifacts["quality_summary"] = quality_summary
    result["deepsearch_artifacts"] = artifacts
    result.update(budget_state_fields(artifacts["research_budget"]))
    if topics and action in {"research", "reflect"}:
        result["missing_topics"] = topics
    if action == "research":
        task_topics = topics or [
            str(topic).strip()
            for topic in (state.get("missing_topics") or research_plan)
            if str(topic).strip()
        ]
        task_state = dict(state)
        if retry_topics or (report and search_results):
            next_epoch = min(epoch + 1, max_epochs)
            task_state["deepsearch_epoch"] = next_epoch
            result["deepsearch_epoch"] = next_epoch
        result["research_tasks"] = _build_research_tasks(
            task_state,
            task_topics,
            retry_counts=retry_counts,
        )
    return result


def coordinator_destination(state: AgentState) -> str:
    """Map a coordinator decision to a bounded graph transition."""

    if state.get("is_cancelled") or state.get("status") in {
        "cancelled",
        "failed",
    }:
        return "finalize"

    action = str(state.get("coordinator_action", "complete")).strip().lower()
    artifacts = state.get("deepsearch_artifacts") or {}
    budget = (
        artifacts.get("research_budget")
        or artifacts.get("linear_budget")
        or artifacts.get("tree_budget")
        or {}
        if isinstance(artifacts, Mapping)
        else {}
    )
    budget_stop_reason = str(
        state.get("budget_stop_reason")
        or (budget.get("stop_reason", "") if isinstance(budget, Mapping) else "")
        or ""
    ).strip()
    if budget_stop_reason:
        if action == "synthesize":
            if state.get("draft_report") or state.get("final_report"):
                return "revise"
            if state.get("search_results"):
                return "deep_writer"
            return "human_review"
        return "human_review"

    if action == "plan":
        return "deep_plan" if not state.get("research_plan") else "plan_review"
    if action in {"research", "reflect"}:
        return "plan_review" if state.get("research_tasks") else "deep_followup"
    if action == "synthesize":
        if state.get("draft_report") or state.get("final_report"):
            return "revise"
        return "deep_writer" if state.get("search_results") else "deep_plan"
    return "human_review"
