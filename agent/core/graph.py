from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, Literal

from langgraph.graph import END,StateGraph
from langgraph.types import Send
from langchain_core.runnables import RunnableConfig

from agent.core.state import AgentState, merge_unique_items
from common.config import settings
from agent.workflows.coordinator import (
    coordinator_destination,
    coordinator_node,
)
from agent.workflows.nodes import (
    direct_answer_node,
    finalize_node,
    human_review_node,
    router_node,
    clarify_node,
    web_plan_node,
    deep_plan_node,
    deep_followup_plan_node,
    plan_review_node,
    sources_review_node,
    draft_review_node,
    search_node,
    deep_linear_search_node,
    deep_tree_search_node,
    writer_node,
    deep_writer_node,
    evaluator_node,
    revise_node,
    agent_tool_node,
    agent_plan_node,
    agent_fast_path_node,
    should_use_agent_fast_path,
)
from agent.workflows.research_budget import (
    DEFAULT_DEEPSEARCH_MAX_SECONDS,
    DEFAULT_DEEPSEARCH_MAX_TOKENS,
)
from agent.workflows.research_tree import ResearchTree


def _attempted_research_task_ids(state: AgentState) -> set[str]:
    attempted: set[str] = set()
    for item in (state.get("research_agent_results") or []):
        if not isinstance(item, Mapping) or not item.get("task_id"):
            continue
        attempted.add(str(item["task_id"]).strip())
    for item in (state.get("research_task_checkpoints") or []):
        if not isinstance(item, Mapping) or not item.get("task_id"):
            continue
        result = item.get("result")
        if not isinstance(result, Mapping):
            continue
        results = [
            candidate for candidate in (result.get("research_agent_results") or [])
            if isinstance(candidate, Mapping)
            and str(candidate.get("task_id", "")).strip() == str(item["task_id"]).strip()
        ]
        if results:
            attempted.add(str(item["task_id"]).strip())
    return attempted


def fan_out_research_tasks(state: AgentState) -> list[Send] | str:
    """Dispatch unfinished coordinator tasks as checkpointable LangGraph workers."""
    artifacts = state.get("deepsearch_artifacts") or {}
    budget = artifacts.get("research_budget") or artifacts.get("tree_budget") or {}
    budget_stop_reason = str(
        state.get("budget_stop_reason")
        or artifacts.get("budget_stop_reason")
        or (budget.get("stop_reason", "") if isinstance(budget, Mapping) else "")
    ).strip()
    if budget_stop_reason:
        return "sources_review"

    tasks = [
        item
        for item in (state.get("research_tasks") or [])
        if isinstance(item, Mapping)
        and str(item.get("task_id", "")).strip()
        and str(item.get("topic", "")).strip()
    ]
    attempted = _attempted_research_task_ids(state)
    attempted.update(
        str(task_id).strip()
        for task_id in (state.get("research_task_attempted_ids") or [])
        if str(task_id).strip()
    )
    pending = [
        dict(task)
        for task in tasks
        if str(task.get("task_id", "")).strip() not in attempted
    ]
    if not pending:
        return "sources_review"
    try:
        parallel_limit = max(
            1,
            int(
                state.get(
                    "research_task_parallel_limit",
                    getattr(settings, "deepsearch_tree_parallel_branches", 3),
                )
            ),
        )
    except (TypeError, ValueError):
        parallel_limit = 3
    pending = pending[:parallel_limit]
    return [
        Send(
            "research_task_worker",
            {**dict(state), "active_research_task": task},
        )
        for task in pending
    ]


def research_task_worker_node(
    state: AgentState,
    config: RunnableConfig,
) -> dict[str, Any]:
    """Run one isolated research task and persist its result as a worker write."""
    task = state.get("active_research_task")
    if not isinstance(task, Mapping):
        raise ValueError("研究 worker 缺少 active_research_task")
    task = dict(task)
    task_id = str(task.get("task_id", "")).strip()
    topic = str(task.get("topic", "")).strip()
    if not task_id or not topic:
        raise ValueError("研究 worker 的 task_id/topic 不能为空")

    configurable = config.get("configurable", {}) if isinstance(config, Mapping) else {}
    configurable = dict(configurable) if isinstance(configurable, Mapping) else {}
    all_tasks = [
        item for item in (state.get("research_tasks") or [])
        if isinstance(item, Mapping) and item.get("task_id")
    ]
    worker_count = max(len(all_tasks), 1)
    artifacts = dict(state.get("deepsearch_artifacts") or {})
    prior_budget = artifacts.get("research_budget") or artifacts.get("tree_budget") or {}
    prior_budget = dict(prior_budget) if isinstance(prior_budget, Mapping) else {}

    def positive_number(value: Any, fallback: float, cast: Any) -> Any:
        try:
            parsed = cast(value)
        except (TypeError, ValueError, OverflowError):
            parsed = fallback
        return parsed if parsed > 0 else fallback

    max_seconds = positive_number(
        prior_budget.get(
            "max_seconds",
            configurable.get("deepsearch_max_seconds", getattr(settings, "deepsearch_max_seconds", 0)),
        ),
        DEFAULT_DEEPSEARCH_MAX_SECONDS,
        float,
    )
    max_tokens = positive_number(
        prior_budget.get(
            "max_tokens",
            configurable.get("deepsearch_max_tokens", getattr(settings, "deepsearch_max_tokens", 0)),
        ),
        DEFAULT_DEEPSEARCH_MAX_TOKENS,
        int,
    )
    elapsed = max(float(prior_budget.get("elapsed_seconds", 0) or 0), 0.0)
    tokens = max(int(prior_budget.get("tokens_used", 0) or 0), 0)
    # Keep room for planning, coordination, writing, and evaluation calls.
    research_token_ceiling = max(1, int(max_tokens * 0.6))
    token_share = max(1, (research_token_ceiling - tokens) // worker_count)
    seconds_share = max(0.1, (max_seconds - elapsed) / worker_count)

    # Each concurrent branch receives a disjoint slice of the remaining run budget.
    branch_budget = {
        **prior_budget,
        "elapsed_seconds": 0.0,
        "tokens_used": max(1, len(str(state.get("input", ""))) // 4),
        "searches_used": 0,
        "stop_reason": "",
        "started_at_epoch": 0.0,
        "last_updated_epoch": 0.0,
    }
    artifacts["research_budget"] = branch_budget
    branch_state = {
        **dict(state),
        "research_plan": [topic],
        "research_tasks": [task],
        "research_tree": {},
        "research_agent_results": [],
        "research_task_checkpoints": [],
        "deepsearch_artifacts": artifacts,
        "active_research_task": task,
    }
    branch_config = dict(config or {})
    branch_config["configurable"] = {
        **configurable,
        "deepsearch_max_seconds": seconds_share,
        "deepsearch_max_tokens": token_share,
    }
    result = deep_tree_search_node(branch_state, branch_config)
    return {
        "research_task_checkpoints": [{
            "task_id": task_id,
            "result": result,
        }]
    }


def research_task_aggregate_node(state: AgentState) -> dict[str, Any]:
    """Merge durable branch outputs into the existing deep-research contract."""
    all_tasks = [
        item for item in (state.get("research_tasks") or [])
        if isinstance(item, Mapping) and item.get("task_id")
    ]
    checkpoint_by_id = {
        str(item.get("task_id")): item
        for item in (state.get("research_task_checkpoints") or [])
        if isinstance(item, Mapping) and item.get("task_id")
    }
    tasks = [
        task for task in all_tasks
        if str(task["task_id"]) in checkpoint_by_id
    ]
    branches = [
        checkpoint_by_id[str(task["task_id"])].get("result", {})
        for task in tasks
        if str(task["task_id"]) in checkpoint_by_id
        and isinstance(checkpoint_by_id[str(task["task_id"])].get("result"), Mapping)
    ]
    branches = [dict(item) for item in branches if isinstance(item, Mapping)]

    search_results: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    fetched_pages: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    task_results: dict[str, dict[str, Any]] = {
        str(item.get("task_id")): dict(item)
        for item in (state.get("research_agent_results") or [])
        if isinstance(item, Mapping) and item.get("task_id")
    }
    tree_payload = dict(state.get("research_tree") or {})
    tree_nodes: dict[str, dict[str, Any]] = {
        str(item.get("id")): dict(item)
        for item in (tree_payload.get("nodes") or [])
        if isinstance(item, Mapping) and item.get("id")
    }
    notes = list(state.get("summary_notes") or [])
    for branch in branches:
        search_results.extend(
            item for item in (branch.get("search_results") or [])
            if isinstance(item, dict)
        )
        sources.extend(
            item for item in (branch.get("sources") or [])
            if isinstance(item, dict)
        )
        fetched_pages.extend(
            item for item in (branch.get("fetched_pages") or [])
            if isinstance(item, dict)
        )
        passages.extend(
            item for item in (branch.get("passages") or [])
            if isinstance(item, dict)
        )
        for item in branch.get("research_agent_results") or []:
            if isinstance(item, Mapping) and item.get("task_id"):
                task_results[str(item["task_id"])] = dict(item)
        branch_tree = branch.get("research_tree") or {}
        if isinstance(branch_tree, Mapping):
            if not tree_payload:
                tree_payload = dict(branch_tree)
            for item in branch_tree.get("nodes") or []:
                if isinstance(item, Mapping) and item.get("id"):
                    node_id = str(item["id"])
                    node = dict(item)
                    previous = tree_nodes.get(node_id)
                    if previous is not None:
                        node["children_ids"] = list(dict.fromkeys([
                            *(previous.get("children_ids") or []),
                            *(node.get("children_ids") or []),
                        ]))
                    tree_nodes[node_id] = node
        for note in branch.get("summary_notes") or []:
            if note and note not in notes:
                notes.append(note)

    if tree_payload:
        tree_payload["nodes"] = list(tree_nodes.values())
        tree = ResearchTree.from_dict(tree_payload)
        tree._refresh_root()
        tree_payload = tree.to_dict()

    current_ids = {str(task["task_id"]) for task in tasks}
    base_artifacts = dict(state.get("deepsearch_artifacts") or {})
    branch_artifacts = [
        item.get("deepsearch_artifacts") or {}
        for item in branches
        if isinstance(item.get("deepsearch_artifacts"), Mapping)
    ]
    merged_artifacts = dict(branch_artifacts[0]) if branch_artifacts else base_artifacts
    base_budget = base_artifacts.get("research_budget") or {}
    base_budget = dict(base_budget) if isinstance(base_budget, Mapping) else {}
    budgets = [
        (item.get("research_budget") or item.get("tree_budget") or {})
        for item in branch_artifacts
    ]
    budgets = [dict(item) for item in budgets if isinstance(item, Mapping)]
    if budgets:
        base_tokens = max(int(base_budget.get("tokens_used", 0) or 0), 0)
        base_searches = max(int(base_budget.get("searches_used", 0) or 0), 0)
        combined = dict(budgets[0])
        combined["tokens_used"] = base_tokens + sum(
            max(int(item.get("tokens_used", 0) or 0) - max(1, len(str(state.get("input", ""))) // 4), 0)
            for item in budgets
        )
        combined["searches_used"] = base_searches + sum(
            max(int(item.get("searches_used", 0) or 0), 0) for item in budgets
        )
        combined["elapsed_seconds"] = float(base_budget.get("elapsed_seconds", 0) or 0) + max(
            float(item.get("elapsed_seconds", 0) or 0) for item in budgets
        )
        combined["max_tokens"] = max(int(base_budget.get("max_tokens", combined.get("max_tokens", 0)) or 0), 0)
        combined["max_seconds"] = max(float(base_budget.get("max_seconds", combined.get("max_seconds", 0)) or 0), 0.0)
        branch_stop_reasons = [
            {
                "task_id": str(task.get("task_id", "")),
                "reason": str(
                    (branch.get("deepsearch_artifacts") or {})
                    .get("research_budget", {})
                    .get("stop_reason", "")
                ),
            }
            for task, branch in zip(tasks, branches)
            if str(
                (branch.get("deepsearch_artifacts") or {})
                .get("research_budget", {})
                .get("stop_reason", "")
            )
        ]
        combined["branch_stop_reasons"] = branch_stop_reasons
        combined["stop_reason"] = str(base_budget.get("stop_reason", "") or "")
        if not combined["stop_reason"]:
            if combined["max_seconds"] and combined["elapsed_seconds"] >= combined["max_seconds"]:
                combined["stop_reason"] = "time_budget_exceeded"
            elif combined["max_tokens"] and combined["tokens_used"] >= combined["max_tokens"]:
                combined["stop_reason"] = "token_budget_exceeded"
            elif (
                getattr(settings, "deepsearch_tree_max_searches", 0)
                and combined["searches_used"] >= int(settings.deepsearch_tree_max_searches)
            ):
                combined["stop_reason"] = "search_budget_exceeded"
        combined["last_updated_epoch"] = time.time()
        merged_artifacts["research_budget"] = combined
        merged_artifacts["tree_budget"] = dict(combined)
        merged_artifacts["budget_stop_reason"] = combined["stop_reason"]

    result: dict[str, Any] = {
        "research_plan": [str(item.get("topic", "")) for item in all_tasks if item.get("topic")],
        "search_results": merge_unique_items([], search_results),
        "sources": merge_unique_items([], sources),
        "fetched_pages": fetched_pages,
        "passages": passages,
        "research_tree": tree_payload,
        "research_agent_results": list(task_results.values()),
        "research_task_attempted_ids": sorted(current_ids),
        "deepsearch_artifacts": merged_artifacts,
        "summary_notes": notes,
        "deepsearch_mode": "tree",
        "current_branch_id": tree_payload.get("root_id") if tree_payload else None,
        "current_step": 2,
        "status": "running",
        "research_task_checkpoints": [
            {"task_id": task_id, "_clear": True} for task_id in current_ids
        ],
    }
    history = list(state.get("deepsearch_query_history") or [])
    result["deepsearch_query_history"] = [
        topic for topic in result["research_plan"]
        if topic.casefold() not in {str(item).casefold() for item in history}
    ]
    if budgets:
        combined_budget = merged_artifacts["research_budget"]
        result.update({
            "budget_stop_reason": combined_budget["stop_reason"],
            "deepsearch_tokens_used": combined_budget["tokens_used"],
            "deepsearch_elapsed_seconds": combined_budget["elapsed_seconds"],
        })
        quality = dict(merged_artifacts.get("quality_summary") or {})
        quality.update({
            "branch_count": len(tree_payload.get("nodes", [])),
            "result_count": len(result["search_results"]),
            "source_count": len(result["sources"]),
            "tokens_used": combined_budget["tokens_used"],
            "elapsed_seconds": combined_budget["elapsed_seconds"],
            "searches_used": combined_budget["searches_used"],
            "max_tokens": combined_budget["max_tokens"],
            "max_seconds": combined_budget["max_seconds"],
            "budget_stop_reason": combined_budget["stop_reason"],
        })
        result["quality_summary"] = quality
        merged_artifacts["quality_summary"] = quality
    return result


def continue_research_task_batches(state: AgentState):
    """Run remaining coordinator tasks in bounded checkpointed batches."""
    return fan_out_research_tasks(state)


def create_research_graph(checkpointer=None, store=None):
    """
    创建 PuzzleResearch 主研究图。

        主要路径：

        router
          ├── direct → direct_answer → finalize
          ├── clarify → clarify → finalize
          ├── web → web_plan
          │          ↓
          │      parallel search
          │          ↓
          │        writer
          │
        ├── deep → deep_plan
        │          ↓
        │      budgeted linear search or tree search
        │          ↓
        │        deep_writer
        │          ↓
        │        evaluator
          │
          └── agent → agent_plan ⇄ agent_tool → finalize
                                  ↑
                         可选 interrupt/resume
    """

    workflow = StateGraph(AgentState)
    use_hierarchical_agents = bool(
        getattr(settings, "use_hierarchical_agents", False)
    )

    #注册节点
    workflow.add_node("router",router_node)
    workflow.add_node("direct_answer",direct_answer_node)
    workflow.add_node("finalize",finalize_node)
    workflow.add_node("human_review",human_review_node)
    workflow.add_node("web_plan",web_plan_node)
    workflow.add_node("deep_plan",deep_plan_node)
    workflow.add_node("deep_followup",deep_followup_plan_node)
    workflow.add_node("plan_review", plan_review_node)
    workflow.add_node("sources_review", sources_review_node)
    workflow.add_node("draft_review", draft_review_node)
    workflow.add_node("search",search_node)
    workflow.add_node("deep_tree_search",deep_tree_search_node)
    if use_hierarchical_agents:
        workflow.add_node("research_task_worker", research_task_worker_node)
        workflow.add_node(
            "research_task_aggregate",
            research_task_aggregate_node,
            defer=True,
        )
    workflow.add_node("deep_search",deep_linear_search_node)
    workflow.add_node("clarify",clarify_node)
    workflow.add_node("agent_tool",agent_tool_node)
    workflow.add_node("writer",writer_node)
    workflow.add_node("deep_writer",deep_writer_node)
    workflow.add_node("evaluator",evaluator_node)
    workflow.add_node("revise",revise_node)
    workflow.add_node("agent_plan",agent_plan_node)
    workflow.add_node("agent_fast_path", agent_fast_path_node)
    if use_hierarchical_agents:
        workflow.add_node("coordinator", coordinator_node)

    #设置入口节点
    workflow.set_entry_point("router")

    def route_decision(state: AgentState) -> str:
        """
        根据 Router 写入的 route 字段选择下一个节点。

        direct、clarify、web、deep 和 agent 路径进入各自节点。
        agent 路径先由 agent_plan 生成工具调用计划，
        再由 agent_tool 执行工具。
        启用工具审批时，会在执行前通过 interrupt 暂停。
        """

        if state.get("is_cancelled"):
            return "finalize"


        route = state.get("route", "clarify")

        allowed_routes = {
            "direct",
            "agent",
            "web",
            "deep",
            "clarify"
        }

        if route not in allowed_routes:
            return "clarify"

        if route == "clarify":
            return "clarify"

        if route == "web":
            if should_use_agent_fast_path(state):
                return "agent_fast_path"
            return "web_plan"

        if route == "deep":
            if use_hierarchical_agents:
                return "coordinator"
            return "deep_plan"

        if route == "agent":
            if should_use_agent_fast_path(state):
                return "agent_fast_path"
            return "agent_plan"

        return "direct_answer"


    def agent_plan_decision(state: AgentState) -> str:
        """
        判断规划节点是进入工具执行，还是直接结束。
        """

        current_status = state.get("status")

        if (
            current_status in {
                "failed",
                "cancelled",
            }
            or state.get("is_cancelled")
        ):
            return "finalize"

        if state.get("agent_finished"):
            return "finalize"

        return "agent_tool"

    def agent_tool_decision(state: AgentState) -> str:
        """
        判断工具执行后是结束，还是回到规划节点继续执行。
        """

        current_status = state.get("status")

        if (
            current_status in {
                "failed",
                "cancelled",
            }
            or state.get("is_cancelled")
        ):
            return "finalize"

        if state.get("agent_finished"):
            return "human_review"

        pending_tool_calls = state.get("pending_tool_calls") or []

        tool_call_count = max(
            int(state.get("tool_call_count", 0)),
            0,
        )

        tool_call_limit = max(
            int(
                state.get(
                    "tool_call_limit",
                    getattr(settings,"tool_call_limit",20),
                )
            ),
            0,
        )

        if tool_call_count >= tool_call_limit:
            return "finalize"

        if pending_tool_calls:
            return "agent_tool"

        return "agent_plan"

    def plan_review_dispatch(state: AgentState, config: RunnableConfig):
        if (
            state.get("is_cancelled")
            or state.get("status") in {"cancelled", "failed"}
        ):
            return "finalize"
        if state.get("route") == "web":
            return fan_out_searches(state)
        if state.get("route") == "deep":
            if use_hierarchical_agents and state.get("research_tasks"):
                configurable = config.get("configurable", {}) if isinstance(config, Mapping) else {}
                try:
                    parallel_limit = int(
                        configurable.get(
                            "deepsearch_tree_parallel_branches",
                            getattr(settings, "deepsearch_tree_parallel_branches", 3),
                        )
                    )
                except (TypeError, ValueError):
                    parallel_limit = 3
                return fan_out_research_tasks({
                    **dict(state),
                    "research_task_parallel_limit": max(parallel_limit, 1),
                })
            return fan_out_deep_searches(state)
        return "finalize"

    def sources_review_dispatch(state: AgentState) -> str:
        if (
            state.get("is_cancelled")
            or state.get("status") in {"cancelled", "failed"}
        ):
            return "finalize"
        return "deep_writer" if state.get("route") == "deep" else "writer"

    def draft_review_dispatch(state: AgentState) -> str:
        if (
            state.get("is_cancelled")
            or state.get("status") in {"cancelled", "failed"}
        ):
            return "finalize"
        return "evaluator"



    #添加决策边
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "direct_answer": "direct_answer",
            "clarify": "clarify",
            "web_plan": "web_plan",
            "deep_plan": "deep_plan",
            "agent_plan": "agent_plan",
            "agent_fast_path": "agent_fast_path",
            "finalize": "finalize",
            **(
                {"coordinator": "coordinator"}
                if use_hierarchical_agents
                else {}
            ),
        },
    )

    if use_hierarchical_agents:
        workflow.add_conditional_edges(
            "coordinator",
            coordinator_destination,
            {
                "deep_plan": "deep_plan",
                "deep_followup": "deep_followup",
                "plan_review": "plan_review",
                "deep_writer": "deep_writer",
                "revise": "revise",
                "human_review": "human_review",
                "finalize": "finalize",
            },
        )

    #添加普通边
    workflow.add_edge("direct_answer","human_review")
    workflow.add_edge("clarify","finalize")
    workflow.add_conditional_edges(
        "agent_plan",
        agent_plan_decision,
        {
            "agent_tool": "agent_tool",
            "finalize": "finalize",
        },
    )
    workflow.add_conditional_edges(
        "agent_fast_path",
        lambda state: (
            "agent_plan"
            if state.get("agent_fast_path_fallback")
            else "finalize"
            if state.get("status") in {"failed", "cancelled"}
            else "human_review"
        ),
        {
            "agent_plan": "agent_plan",
            "human_review": "human_review",
            "finalize": "finalize",
        },
    )
    workflow.add_conditional_edges(
        "agent_tool",
        agent_tool_decision,
        {
            "agent_tool": "agent_tool",
            "finalize": "finalize",
            "agent_plan": "agent_plan",
            "human_review": "human_review",
        },
    )
    workflow.add_edge("web_plan", "plan_review")
    if use_hierarchical_agents:
        workflow.add_edge("deep_plan", "coordinator")
        workflow.add_edge("deep_followup", "coordinator")
    else:
        workflow.add_edge("deep_plan", "plan_review")
        workflow.add_edge("deep_followup", "plan_review")
    workflow.add_conditional_edges(
        "plan_review",
        plan_review_dispatch,
        [
            "finalize",
            "search",
            "deep_search",
            "deep_tree_search",
            "sources_review",
            *(["research_task_worker", "research_task_aggregate"] if use_hierarchical_agents else []),
        ],
    )

    workflow.add_edge("search", "sources_review")
    workflow.add_edge("deep_search", "sources_review")
    workflow.add_edge("deep_tree_search", "sources_review")
    if use_hierarchical_agents:
        workflow.add_edge("research_task_worker", "research_task_aggregate")
        workflow.add_conditional_edges(
            "research_task_aggregate",
            continue_research_task_batches,
            ["research_task_worker", "research_task_aggregate", "sources_review"],
        )
    workflow.add_conditional_edges(
        "sources_review",
        sources_review_dispatch,
        ["finalize", "writer", "deep_writer"],
    )
    workflow.add_edge("writer", "draft_review")
    workflow.add_edge("deep_writer", "draft_review")
    workflow.add_conditional_edges(
        "draft_review",
        draft_review_dispatch,
        ["finalize", "evaluator"],
    )
    def after_evaluation(state: AgentState) -> str:
        if use_hierarchical_agents and state.get("route") == "deep":
            return "coordinator"
        return evaluation_decision(state)

    evaluator_targets = {
        "deep_followup": "deep_followup",
        "revise": "revise",
        "human_review": "human_review",
    }
    if use_hierarchical_agents:
        evaluator_targets["coordinator"] = "coordinator"

    workflow.add_conditional_edges(
        "evaluator",
        after_evaluation,
        evaluator_targets,
    )
    # Match Weaver: human draft review applies to the writer's initial draft;
    # automated revisions return directly to evaluation.
    workflow.add_edge("revise", "evaluator")
    workflow.add_edge("human_review","finalize")
    workflow.add_edge("finalize",END)

    compile_kwargs = {}

    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

    if store is not None:
        compile_kwargs["store"] = store

    return workflow.compile(**compile_kwargs)

def _collect_research_queries(
        state: AgentState
) -> list[str]:
    """
    清理并去重研究查询。
    """

    raw_queries = state.get("research_plan") or []

    queries: list[str] = []
    seen: set[str] = set()

    for item in raw_queries:
        query = str(item).strip()

        if query and query not in seen:
            queries.append(query)
            seen.add(query)

    if not queries:
        user_input = state.get("input","").strip()
        if user_input:
            queries = [user_input]

    if not queries:
        raise ValueError("没有可分发的搜索查询")

    return queries


def _fan_out_to_search_node(
        state: AgentState,
        target_node: str,
        search_mode: str,
) -> list[Send]:
    """
    将多个研究查询分发到指定搜索节点。
    """

    queries = _collect_research_queries(state)

    print(
        "[fan_out]",
        f"target={target_node}",
        f"branches={len(queries)}",
        f"queries={queries}",
    )

    return [
        Send(
            target_node,
            {
                "query": query,
                "search_mode": search_mode,
                "domain": state.get("domain", "general"),
                "domain_config": state.get("domain_config", {}),
            },
        )
        for query in queries
    ]


def fan_out_searches(
        state: AgentState
) -> list[Send]:
    """
    Web Research 使用的并行搜索分发。
    """

    return _fan_out_to_search_node(
        state,
        "search",
        "web",
    )


def fan_out_deep_searches(
        state: AgentState
) -> list[Send]:
    """
    树模式使用专用探索节点，线性模式使用预算受控的批处理节点。
    """
    if (
        bool(state.get("research_tasks"))
        or (
        bool(state.get("tree_exploration_enabled"))
        and str(state.get("deepsearch_mode", "")).strip().lower() == "tree"
        )
    ):
        return [
            Send(
                "deep_tree_search",
                dict(state),
            )
        ]

    return [
        Send(
            "deep_search",
            {
                **dict(state),
                "search_mode": "deep",
            },
        )
    ]



def evaluation_decision(
        state: AgentState
) -> Literal["deep_followup","revise","human_review"]:
    """
    根据评估结果决定修订或结束。

    即使评估结果要求修订，
    达到最大修订次数后也必须结束，
    防止图进入无限循环。
    """

    verdict = str(
        state.get("verdict","pass")
    ).strip().lower()

    raw_missing_topics = state.get("missing_topics") or []
    if isinstance(raw_missing_topics, str):
        raw_missing_topics = [raw_missing_topics]

    has_missing_topics = any(
        str(topic).strip()
        for topic in raw_missing_topics
    )

    # 缺失主题与 pass 矛盾，优先遵守质量契约，避免报告带着已知缺口结束。
    if verdict == "pass" and has_missing_topics:
        verdict = "revise"


    route = str(
        state.get("route","")
    ).strip().lower()

    deep_epoch = max(
        int(state.get("deepsearch_epoch", 1)),
        1,
    )

    deep_max_epochs = max(
        int(state.get("deepsearch_max_epochs", 3)),
        1,
    )

    artifacts = state.get("deepsearch_artifacts") or {}
    if not isinstance(artifacts, dict):
        artifacts = {}
    budget = (
        artifacts.get("research_budget")
        or artifacts.get("linear_budget")
        or artifacts.get("tree_budget")
        or {}
    )
    if not isinstance(budget, dict):
        budget = {}
    budget_stop_reason = str(
        state.get("budget_stop_reason")
        or artifacts.get("budget_stop_reason")
        or budget.get("stop_reason")
        or ""
    ).strip()


    revision_count = int(
        state.get("revision_count", 0)
    )

    max_revisions = max(
        int(state.get("max_revisions", 2)),
        0,
    )


    if route == "deep" and budget_stop_reason:
        print(
            "[evaluation_decision]",
            "route=human_review",
            f"budget_stop_reason={budget_stop_reason}",
            "budget_exhausted=True",
        )
        return "human_review"


    if (
        route == "deep"
        and verdict == "revise"
        and deep_epoch < deep_max_epochs
        and not budget_stop_reason
    ):
        print(
            "[evaluation_decision]",
            "route=deep_followup",
            f"deep_epoch={deep_epoch}",
            f"deep_max_epochs={deep_max_epochs}",
        )
        return "deep_followup"



    if (
        verdict == "revise"
        and revision_count < max_revisions
    ):
        print(
            "[evaluation_decision]",
            "route=revise",
            f"revision_count={revision_count}",
            f"max_revisions={max_revisions}",
        )
        return "revise"

    print(
        "[evaluation_decision]",
        "route=finalize",
        f"verdict={verdict}",
        f"revision_count={revision_count}",
        f"max_revisions={max_revisions}",
    )

    return "human_review"
