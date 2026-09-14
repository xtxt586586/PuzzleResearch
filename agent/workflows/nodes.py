from __future__ import annotations

import json
import re
from typing import Dict,Any

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langchain_core.runnables import RunnableConfig

from agent.core.model_factory import ModelFactory
from agent.core.state import AgentState,QueryState
from agent.core.smart_router import smart_route
from common.config import settings
from tools.search import SearchManager


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


def _parse_json_object(content: str) -> Dict[str, Any]:
    """
    从模型文本中提取 JSON 对象。

    兼容：
    1. 纯 JSON；
    2. Markdown JSON 代码块；
    3. JSON 前后带少量解释文字。
    """

    content = content.strip()

    fenced_match = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if fenced_match:
        json_text = fenced_match.group(1)
    else:
        start = content.find("{")
        end = content.rfind("}")

        if start == -1 or end == -1 or start >= end:
            raise ValueError(
                f"模型返回中没有找到有效 JSON：{content}"
            )

        json_text = content[start:end + 1]

    data = json.loads(json_text)

    if not isinstance(data, dict):
        raise ValueError("模型返回的 JSON 不是对象")

    return data


def _parse_plan_response(
    response: Any,
) -> tuple[list[str], str]:
    """
    解析规划模型返回的查询计划。
    """

    content = _extract_response_text(response)
    data = _parse_json_object(content)

    raw_queries = data.get("queries", [])

    if isinstance(raw_queries, str):
        raw_queries = [raw_queries]

    if not isinstance(raw_queries, list):
        raw_queries = []

    queries: list[str] = []
    seen: set[str] = set()

    for item in raw_queries:
        query = str(item).strip()

        if query and query not in seen:
            queries.append(query)
            seen.add(query)

    reasoning = str(
        data.get("reasoning", "")
    ).strip()

    return queries, reasoning


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


def web_plan_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str,Any]:
    """
    Web Research 规划节点。

    优先使用 Router 提供的 suggested_queries。
    如果 Router 没有提供查询建议，则调用规划模型生成查询。
    """

    user_input = state.get("input","").strip()

    if not user_input:
        raise ValueError("AgentState.input 不能为空")

    max_queries = max(
        int(getattr(settings,"web_max_queries",3)),
        1,
    )

    suggested_queries = state.get("suggested_queries") or []

    queries: list[str] = []
    seen: set[str] = set()

    for item in suggested_queries:
        query = str(item).strip()

        if query and query not in seen:
            queries.append(query)
            seen.add(query)

    reasoning = ""

    # Router 已经给出搜索建议时，优先复用，减少一次模型调用。
    if queries:
        queries = queries[:max_queries]
        reasoning = "复用 Router 提供的搜索建议。"

    else:
        llm = ModelFactory.for_task(
            "planning",
            temperature=0.2,
        )

        messages = [
            SystemMessage(
                content=(
                    "你是 PuzzleResearch 的 Web Research Planner。"
                    "请把用户问题拆解成适合搜索引擎执行的查询词。"
                    "最多生成 3 个查询。"
                    "查询应该互补，不要重复。"
                    "只返回 JSON，不要输出 Markdown 代码块或解释文字。"
                    "\n\n"
                    "JSON 格式："
                    "\n"
                    "{"
                    '"queries": ["查询一", "查询二"], '
                    '"reasoning": "规划原因"'
                    "}"
                )
            ),
            HumanMessage(
                content=f"用户问题：\n{user_input}"
            ),
        ]

        try:
            response = llm.invoke(
                messages,
                config=config,
            )

            queries, reasoning = _parse_plan_response(response)
            queries = queries[:max_queries]

        except Exception as exc:
            print(f"{web_plan_node}查询规划失败，使用原问题：{exc}")
            queries = [user_input]
            reasoning = "规划模型失败，直接使用用户原始问题搜索。"

    if not queries:
        queries = [user_input]
        reasoning = "没有生成有效查询，直接使用用户原始问题搜索。"

    print(
        "[web_plan_node]",
        f"queries={queries}",
        f"reasoning={reasoning}",
    )

    return {
        "research_plan": queries,
        "current_step": 1,
        "status": "running",
    }


def search_node(
        state: QueryState,
) -> Dict[str,Any]:
    """
    执行单个查询的搜索任务。

    该节点由 LangGraph Send 分发执行。
    每个节点实例只处理一个 query。
    """


    query = state.get("query","").strip()

    if not query:
        return {
            "errors": ["搜索分支没有收到有效 query"]
        }

    max_results = max(
        int(getattr(settings,"web_results_per_query",5)),
        1,
    )

    manager = SearchManager()

    try:
        results = manager.search(
            query,
            max_results=max_results,
        )

    except Exception as exc:
        error_message = (
            f"查询「{query}」失败：{exc}"
        )
        print(f"[search_node] {error_message}")

        return {
            "errors": [error_message],
        }

    if not results:
        error_message = f"查询「{query}」没有得到任何结果"
        print(f"[search_node] {error_message}")

        return {
            "errors": [error_message],
        }

    search_results: list[Dict[str,Any]] = []
    sources: list[Dict[str,str]] = []

    for item in results:
        normalized_item = {
            "query": query,
            "title": str(item.get("title","")),
            "url": str(item.get("url","")),
            "snippet": str(item.get("snippet","")),
            "content": str(item.get("content","")),
            "provider": str(item.get("provider","")),
        }

        search_results.append(normalized_item)

        sources.append(
            {
                "title": normalized_item["title"],
                "url": normalized_item["url"],
                "provider": normalized_item["provider"],
                "query": query,
            }
        )

    print(
        "[search_node]",
        f"query={query}",
        f"results={len(search_results)}",
    )

    return {
        "search_results": search_results,
        "sources": sources,
    }


def writer_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str,Any]:
    """
    根据搜索结果生成最终研究回答。
    """

    user_input = state.get("input","").strip()
    search_results = state.get("search_results") or []

    if not user_input:
        raise ValueError("AgentState.input 不能为空")

    if not search_results:
        raise ValueError("Writer 没有收到搜索结果")

    context_blocks: list[str] = []

    for index,item in enumerate(
        search_results[:12],
        start=1,
    ):
        title = str(item.get("title","")).strip()
        url = str(item.get("url","")).strip()
        snippet = str(item.get("snippet","")).strip()
        content = str(item.get("content","")).strip()

        excerpt = content or snippet
        excerpt = excerpt[:2000]

        context_blocks.append(
            f"【来源 {index}】\n"
            f"标题：{title}\n"
            f"链接：{url}\n"
            f"内容：{excerpt}\n"
        )

    search_context = "\n\n".join(context_blocks)

    llm = ModelFactory.for_task(
        "writing",
        temperature=0.4,
    )

    messages = [
        SystemMessage(
            content=(
                "你是 PuzzleResearch 的 Web Research Writer。"
                "请根据用户问题和搜索结果撰写准确、清晰的回答。"
                "只能使用提供的搜索结果，不要编造来源或事实。"
                "引用来源时使用 [来源1]、[来源2] 这样的格式。"
                "如果不同来源的信息存在冲突，请明确指出。"
                "回答应直接解决用户问题，不要描述内部工作流程。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：\n{user_input}\n"
                f"搜索结果：\n{search_context}\n"
            )
        ),
    ]

    response = llm.invoke(
        messages,
        config=config,
    )

    answer = _extract_response_text(response).strip()

    if not answer:
        raise ValueError("Writer 模型返回了空回答")

    reference_lines: list[str] = []
    seen_urls: set[str] = set()

    for index,item in enumerate(
        search_results[:12],
        start=1,
    ):
        title = str(item.get("title","")).strip()
        url = str(item.get("url","")).strip()

        if not url or url in seen_urls:
            continue

        reference_lines.append(
            f"[来源 {index}] {title}:{url}\n"
        )
        seen_urls.add(url)

    if reference_lines:
        answer += (
            "\n\n参考来源：\n"
            + "\n".join(reference_lines)
        )

    return {
        "draft_report": answer,
        "final_report": answer,
        "messages": [
            AIMessage(content=answer)
        ],
        "current_step": 3,
        "is_complete": False,
        "status": "running",
    }


def evaluator_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str, Any]:
    """
    报告评估 Agent。

    负责检查 Writer 生成的报告是否：
    - 回答了用户问题；
    - 覆盖了主要内容；
    - 有足够的来源支持；
    - 存在明显遗漏。
    """

    user_input = state.get("input","").strip()
    report = (
        state.get("draft_report","")
        or state.get("final_report","")
        or ""
    ).strip()

    if not user_input:
        raise ValueError("Evaluator 缺少用户问题")

    if not report:
        raise ValueError("Evaluator 没有收到报告")

    sources = state.get("sources") or []

    sources_lines: list[str] = []

    for index,source in enumerate(
        sources[:12],
        start=1,
    ):
        title = str(source.get("title","")).strip()
        url = str(source.get("url","")).strip()

        sources_lines.append(
            f"[来源 {index}] {title}:{url}"
        )

    source_context = "\n".join(sources_lines)

    llm = ModelFactory.for_task(
        "evaluation",
        temperature=0.0,
    )

    messages = [
        SystemMessage(
            content=(
                "你是 PuzzleResearch 的报告评估 Agent。"
                "请检查报告是否准确回答用户问题，"
                "是否覆盖主要内容，"
                "是否合理使用搜索来源。"
                "请只返回 JSON，不要输出 Markdown 代码块。"
                "\n\n"
                "JSON 格式："
                "\n"
                "{"
                '"verdict": "pass", '
                '"evaluation": "总体评价", '
                '"eval_dimensions": {'
                '"accuracy": 0.0, '
                '"completeness": 0.0, '
                '"citation": 0.0'
                "}, "
                '"missing_topics": []'
                "}"
                "\n\n"
                "verdict 只能是 pass 或 revise。"
                "各项评分必须是 0 到 1 之间的数字。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：\n{user_input}\n"
                f"报告：\n{report}\n"
                f"可用来源：\n{source_context}\n"
            )
        ),
    ]

    try:
        response = llm.invoke(
            messages,
            config=config,
        )

        data = _parse_json_object(
            _extract_response_text(response)
        )

        raw_verdict = str(
            data.get("verdict","revise")
        ).strip().lower()

        if raw_verdict in {
            "pass",
            "通过",
            "合格",
            "ok",
        }:
            verdict = "pass"
        else:
            verdict = "revise"

        evaluation = str(
            data.get("evaluation","")
        ).strip()

        raw_dimensions = data.get(
            "eval_dimensions",
            {},
        )

        dimensions: Dict[str,float] = {}

        if isinstance(raw_dimensions,dict):
            for name,value in raw_dimensions.items():
                try:
                    score = float(value)
                except (ValueError,TypeError):
                    continue

                dimensions[str(name)] = max(
                    0.0,
                    min(1.0, score),
                )

        raw_missing_topics = data.get(
            "missing_topics",
            [],
        )

        if isinstance(raw_missing_topics,str):
            raw_missing_topics = [raw_missing_topics]

        missing_topics = [
            str(item).strip()
            for item in raw_missing_topics
            if str(item).strip()
        ]

        if not evaluation:
            evaluation = "评估模型没有提供文字说明。"

        print(
            "[evaluator_node]",
            f"verdict={verdict}",
            f"missing_topics={missing_topics}",
        )

        return {
            "evaluation": evaluation,
            "verdict": verdict,
            "eval_dimensions": dimensions,
            "missing_topics": missing_topics,
            "current_step": 4,
            "status": "running",
        }

    except Exception as exc:
        error_message = f"报告评估失败： {exc}"
        print(f"报告评估失败： {exc}")
        return {
            "evaluation": (
                "自动评估失败，本轮暂不阻断报告。"
            ),
            "verdict": "pass",
            "eval_dimensions": {},
            "missing_topics": [],
            "errors": [error_message],
            "current_step": 4,
            "status": "running",
        }


def clarify_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    澄清节点。

    当 Router 判断用户请求信息不足时，
    生成一个简洁的澄清问题。

    当前阶段：
    - 生成澄清问题；
    - 将问题返回给用户；
    - 结束本次图执行。

    后续阶段会通过 interrupt 和 thread resume
    支持用户回答后继续执行原任务。
    """

    user_input = state.get("input","").strip()

    #空输入不需要再次调用模型，直接返回固定问题。
    if not user_input:
        question = "请告诉我你希望完成什么任务？"

        return {
            "clarification_question": question,
            "draft_report": question,
            "final_report": question,
            "messages": [
                AIMessage(content=question),
            ],
            "is_complete": False,
            "status": "running",
        }

    llm = ModelFactory.for_task(
        "clarification",
        temperature=0.0,
    )

    messages = [
        SystemMessage(
            content=(
                "你是 PuzzleResearch 的澄清助手。"
                "用户的请求信息不足，无法直接执行。"
                "请根据用户原始请求提出一个简洁、具体的问题，"
                "帮助用户补充任务目标、对象或必要条件。"
                "只输出一个问题，不要解释原因。"
            )
        ),
        HumanMessage(
            content=f"用户原始提问：\n{user_input}"
        ),
    ]

    try:
        response = llm.invoke(
            messages,
            config=config,
        )

        question = _extract_response_text(response).strip()

    except Exception as exc:
        print(f"{clarify_node}澄清问题生成失败： {exc}")
        question = "请补充说明你希望处理的具体内容和目标。"

    if not question:
        question = "请补充说明你希望处理的具体内容和目标。"

    return {
        "clarification_question": question,
        "draft_report": question,
        "final_report": question,
        "messages": [
            AIMessage(content=question),
        ],
        "is_complete": False,
        "status": "running",
    }


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
        "final_report": answer,
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
