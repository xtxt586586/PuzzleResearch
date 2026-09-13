from __future__ import annotations

from langgraph.graph import END,StateGraph

from agent.core.state import AgentState
from agent.workflows.nodes import (
    direct_answer_node,
    finalize_node,
    router_node,
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

    #设置入口节点
    workflow.set_entry_point("router")

    def route_decision(state: AgentState) -> str:
        """
        根据路由节点写入的 route 字段选择下一个节点。
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

        # 当前阶段所有路径都进入 direct_answer。
        # 后续会把 agent/web/deep 替换成真正的节点。
        return "direct_answer"

    #添加决策边
    workflow.add_conditional_edges(
        "router",
        route_decision,
        {
            "direct_answer": "direct_answer",
        },
    )

    #添加普通边
    workflow.add_edge("direct_answer","finalize")
    workflow.add_edge("finalize",END)

    return workflow.compile()

    # #编译工作流
    # graph = workflow.compile()
    #
    # return graph