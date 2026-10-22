from __future__ import annotations

from typing import Literal

from langgraph.graph import END,StateGraph
from langgraph.types import Send

from agent.core.state import AgentState
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

    def plan_review_dispatch(state: AgentState):
        if (
            state.get("is_cancelled")
            or state.get("status") in {"cancelled", "failed"}
        ):
            return "finalize"
        if state.get("route") == "web":
            return fan_out_searches(state)
        if state.get("route") == "deep":
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
        ["finalize", "search", "deep_search", "deep_tree_search"],
    )

    workflow.add_edge("search", "sources_review")
    workflow.add_edge("deep_search", "sources_review")
    workflow.add_edge("deep_tree_search", "sources_review")
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
