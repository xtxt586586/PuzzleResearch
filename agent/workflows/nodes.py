from __future__ import annotations

from typing import Dict,Any

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.core.model_factory import ModelFactory
from agent.core.state import AgentState
from agent.core.smart_router import smart_route


def _extract_response_text(response: Any) -> str:
    """
    从模型响应中提取文本。

    ChatOpenAI 通常返回 AIMessage，
    但保留这个辅助函数可以兼容不同响应对象。
    """
    if hasattr(response,"content"):
        content = response.content

        if isinstance(content,str):
            return content

        return str(content)

    return str(response)


def router_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    路由节点。

    读取用户问题，调用 SmartRouter，
    把路由结果写入共享 AgentState。
    """

    user_input = state.get("input","").strip()

    result = smart_route(
        user_input,
        config=config,
    )

    print(
        "[route_node]",
        f"route={result['route']}",
        f"confidence={result['routing_confidence']:.2f}"
    )

    return result


def direct_answer_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    直接回答节点。

    这个节点不联网、不搜索、不调用工具，
    只使用 LLM 根据用户输入生成回答。
    """
    user_input = state.get("input","").strip()

    if not user_input:
        raise ValueError("AgentState.input 不能为空")

    llm = ModelFactory.for_task(
        "writing",
        temperature=0.7,
    )

    messages = [
        SystemMessage(
            content=(
                "你是 PuzzleResearch 的直接问答 Agent。"
                "请准确、清晰、简洁地回答用户问题。"
                "当前节点不联网、不搜索、不调用外部工具。"
            )
        ),
        HumanMessage(content=user_input),
    ]

    response = llm.invoke(
        messages,
        config=config,
    )

    answer = _extract_response_text(response).strip()

    if not answer:
        raise ValueError("模型返回了空回答")

    return {
        "draft_report": answer,
        "final_answer": answer,
        "messages": [
            AIMessage(content=answer)
        ],
        "is_complete": False,
        "status": "running",
    }


def finalize_node(
    state: AgentState,
) -> Dict[str, Any]:
    """
    结束节点。

    当前阶段暂时负责把报告标记为完成。
    后续会替换成完整的 human_review_node，
    负责人工审核、中断和恢复。
    """
    report = state.get("final_report") or state.get("draft_report","")

    return {
        "final_report": report,
        "is_complete": True,
        "status": "completed",
    }
