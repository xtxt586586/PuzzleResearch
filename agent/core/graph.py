from __future__ import annotations

from langgraph.graph import END,StateGraph

from agent.core.state import AgentState
from agent.workflows.nodes import (
    direct_answer_node,
    finalize_node,
)


def create_research_graph():
    """
    创建 PuzzleResearch 主研究图。

    当前阶段包含最基础的直接回答流程。
    后续将在这张图上逐步添加：

    - router
    - planner
    - web search
    - deepsearch
    - writer
    - evaluator
    - reviser
    - human review
    """
    workflow = StateGraph(AgentState)

    #注册节点
    workflow.add_node("direct_answer",direct_answer_node)
    workflow.add_node("finalize",finalize_node)

    #设置入口节点
    workflow.set_entry_point("direct_answer")

    #添加普通边
    workflow.add_edge("direct_answer","finalize")
    workflow.add_edge("finalize",END)

    #编译工作流
    graph = workflow.compile()

    return graph