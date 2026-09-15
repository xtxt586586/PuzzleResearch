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
    deep_plan_node,
    deep_followup_plan_node,
    search_node,
    writer_node,
    deep_writer_node,
    evaluator_node,
    revise_node,
    agent_tool_node,
    agent_plan_node,
)


def create_research_graph(checkpointer=None):
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
          └── deep → deep_plan
                     ↓
                 parallel deep_search
                     ↓
                 deep_writer
                     ↓
                 evaluator
                     ├── pass → finalize
                     ├── revise → evaluator
                     └── 缺口且仍有预算
                         → deep_followup
                         → deep_search
    """

    workflow = StateGraph(AgentState)

    #注册节点
    workflow.add_node("router",router_node)
    workflow.add_node("direct_answer",direct_answer_node)
    workflow.add_node("finalize",finalize_node)
    workflow.add_node("web_plan",web_plan_node)
    workflow.add_node("deep_plan",deep_plan_node)
    workflow.add_node("deep_followup",deep_followup_plan_node)
    workflow.add_node("search",search_node)
    workflow.add_node("deep_search",search_node)
    workflow.add_node("clarify",clarify_node)
    workflow.add_node("agent_tool",agent_tool_node)
    workflow.add_node("writer",writer_node)
    workflow.add_node("deep_writer",deep_writer_node)
    workflow.add_node("evaluator",evaluator_node)
    workflow.add_node("revise",revise_node)
    workflow.add_node("agent_plan",agent_plan_node)

    #设置入口节点
    workflow.set_entry_point("router")

    def route_decision(state: AgentState) -> str:
        """
        根据 Router 写入的 route 字段选择下一个节点。

        direct、clarify、web 和 deep 路径进入各自节点。
        agent 路径当前进入能力边界提示节点，
        因为工具执行节点尚未接入。
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

        if route == "deep":
            return "deep_plan"

        if route == "agent":
            return "agent_plan"

        return "direct_answer"

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
        },
    )

    #添加普通边
    workflow.add_edge("direct_answer","finalize")
    workflow.add_edge("clarify","finalize")
    workflow.add_edge("agent_plan", "agent_tool")
    workflow.add_edge("agent_tool", "finalize")
    workflow.add_conditional_edges(
        "web_plan",
        fan_out_searches,
        ["search"],
    )
    workflow.add_conditional_edges(
        "deep_plan",
        fan_out_deep_searches,
        ["deep_search"],
    )
    workflow.add_conditional_edges(
        "deep_followup",
        fan_out_deep_searches,
        ["deep_search"],
    )

    workflow.add_edge("search","writer")
    workflow.add_edge("deep_search","deep_writer")
    workflow.add_edge("writer","evaluator")
    workflow.add_edge("deep_writer","evaluator")
    workflow.add_conditional_edges(
        "evaluator",
        evaluation_decision,
        {
            "deep_followup": "deep_followup",
            "revise": "revise",
            "finalize": "finalize",
        },
    )
    workflow.add_edge("revise","evaluator")
    workflow.add_edge("finalize",END)

    compile_kwargs = {}

    if checkpointer is not None:
        compile_kwargs["checkpointer"] = checkpointer

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
    Deep Research 使用的并行搜索分发。
    """
    return _fan_out_to_search_node(
        state,
        "deep_search",
        "deep",
    )



def evaluation_decision(
        state: AgentState
) -> Literal["deep_followup","revise","finalize"]:
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


    revision_count = int(
        state.get("revision_count", 0)
    )

    max_revisions = max(
        int(state.get("max_revisions", 2)),
        0,
    )


    if (
        route == "deep"
        and verdict == "revise"
        and deep_epoch < deep_max_epochs
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

    return "finalize"
