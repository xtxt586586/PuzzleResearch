from __future__ import annotations

from typing import Literal

from langgraph.graph import END,StateGraph
from langgraph.types import Send

from agent.core.state import AgentState
from agent.workflows.nodes import (
    direct_answer_node,
    finalize_node,
    router_node,
    clarify_node,
    web_plan_node,
    search_node,
    writer_node,
    evaluator_node,
    revise_node,
)


def create_research_graph():
    """
    创建 PuzzleResearch 主研究图。

    当前阶段：
        router
          ├── direct
          ├── agent
          ├── web
          ├── deep
          └── clarify
                ↓
          direct_answer
                ↓
             finalize
    """
    workflow = StateGraph(AgentState)

    #注册节点
    workflow.add_node("router",router_node)
    workflow.add_node("direct_answer",direct_answer_node)
    workflow.add_node("finalize",finalize_node)
    workflow.add_node("web_plan",web_plan_node)
    workflow.add_node("search",search_node)
    workflow.add_node("clarify",clarify_node)
    workflow.add_node("writer",writer_node)
    workflow.add_node("evaluator",evaluator_node)
    workflow.add_node("revise",revise_node)

    #设置入口节点
    workflow.set_entry_point("router")

    def route_decision(state: AgentState) -> str:
        """
        根据 Router 写入的 route 字段选择下一个节点。

        当前阶段只实现两条实际路径：

        - clarify → clarify
        - 其他合法路径 → direct_answer

        web、agent、deep 后续再分别接入真实节点。
        """

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
            return "web_plan"

        # 当前阶段所有路径都进入 direct_answer。
        # 后续会把 agent/web/deep 替换成真正的节点。
        return "direct_answer"

    #添加决策边
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "direct_answer": "direct_answer",
            "clarify": "clarify",
            "web_plan": "web_plan",
        },
    )

    #添加普通边
    workflow.add_edge("direct_answer","finalize")
    workflow.add_edge("clarify","finalize")
    workflow.add_conditional_edges(
        "web_plan",
        fan_out_searches,
        ["search"],
    )
    workflow.add_edge("search","writer")
    workflow.add_edge("writer","evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        evaluation_decision,
        {
            "revise": "revise",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("revise","evaluator")
    workflow.add_edge("finalize",END)

    return workflow.compile()



def fan_out_searches(
        state: AgentState
) -> list[Send]:
    """
    根据 research_plan 创建并行搜索分支。
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

    print(
        "[fan_out_searches]",
        f"branches={len(queries)}",
        f"queries={queries}",
    )

    return [
        Send(
            "search",
            {
                "query": query,
            },
        )
        for query in queries
    ]



def evaluation_decision(
        state: AgentState
) -> Literal["revise","finalize"]:
    """
    根据评估结果决定修订或结束。

    即使评估结果要求修订，
    达到最大修订次数后也必须结束，
    防止图进入无限循环。
    """

    verdict = str(
        state.get("verdict","pass")
    ).strip().lower()

    revision_count = int(
        state.get("revision_count", 0)
    )

    max_revisions = max(
        int(state.get("max_revisions", 2)),
        0,
    )

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

    return "finalize"