from __future__ import annotations

import json
import time
import re
from typing import Dict,Any

from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from agent.core.model_factory import ModelFactory
from agent.core.state import AgentState,QueryState
from agent.core.profile import AgentProfile
from agent.core.smart_router import smart_route
from common.config import settings
from tools.search import SearchManager
from tools.registry import ToolRegistry
from tools.builtin import create_default_registry


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



def _append_source_references(
    answer: str,
    search_results: list[Dict[str, Any]],
    max_sources: int = 12,
) -> str:
    """
    为报告追加去重后的来源列表。
    """

    reference_lines: list[str] = []
    seen_urls: set[str] = set()

    for item in search_results[:max_sources]:
        title = str(item.get("title", "")).strip()
        url = str(item.get("url", "")).strip()

        if not url or url in seen_urls:
            continue

        reference_number = len(reference_lines) + 1
        reference_lines.append(
            f"[来源 {reference_number}] {title}：{url}"
        )
        seen_urls.add(url)

    if not reference_lines:
        return answer

    return (
        answer
        + "\n\n参考来源：\n"
        + "\n".join(reference_lines)
    )


def _select_deep_search_results(
    search_results: list[Dict[str, Any]],
    max_sources: int = 20,
) -> list[Dict[str,Any]]:
    """
    从 Deep Research 的全部结果中选择上下文。

    结果较少时全部保留；
    结果较多时保留前半部分和最新一轮结果。
    """

    if max_sources <= 0:
        return []

    if len(search_results) <= max_sources:
        return list(search_results)

    head_count = max_sources // 2
    tail_count = max_sources - head_count

    return (
        list(search_results[:head_count])
        + list(search_results[-tail_count:])
    )





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

    return {
        **result,
        "messages": [HumanMessage(content=user_input)],
    }


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
        "revision_count": 0,
        "max_revisions": max(
            int(getattr(settings,"max_revisions",2)),
            0,
        ),
        "current_step": 1,
        "status": "running",
    }


def deep_plan_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str,Any]:
    """
    Deep Research 规划节点。

    与 Web Research 的区别：
    - 查询数量更多；
    - 查询角度更加互补；
    - 目标是多来源综合研究；
    - 结果会交给专门的 Deep Writer。
    """
    user_input = state.get("input","").strip()

    if not user_input:
        raise ValueError("AgentState.input 不能为空")

    max_queries = max(
        int(getattr(settings,"deepsearch_query_num",5)),
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

    if queries:
        queries = queries[:max_queries]
        reasoning = "复用 Router 提供的深度研究查询建议。"

    else:
        llm = ModelFactory.for_task(
            "planning",
            temperature=0.2,
        )

        messages = [
            SystemMessage(
                content=(
                    "你是 PuzzleResearch 的 Deep Research Planner。"
                    "请为用户问题设计多角度、多来源的研究查询。"
                    "查询应该互补，不能只是同一句话的重复。"
                    "可以从定义、原理、实践、优缺点、争议、"
                    "官方资料等不同角度设计查询。"
                    f"最多生成 {max_queries} 个查询。"
                    "只返回 JSON，不要输出 Markdown 代码块。"
                    "\n\n"
                    "JSON 格式："
                    "\n"
                    "{"
                    '"queries": ["查询一", "查询二"], '
                    '"reasoning": "研究规划原因"'
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
            print(f"[deep_plan_node] 规划失败，使用原问题：{exc}")
            queries = [user_input]
            reasoning = "深度规划失败，使用用户原始问题搜索。"

    if not queries:
        queries = [user_input]
        reasoning = "没有生成有效查询，使用用户原始问题搜索。"


    max_epochs = max(
        int(getattr(settings,"deepsearch_max_epochs",3)),
        1,
    )


    print(
        "[deep_plan_node]",
        f"queries={queries}",
        f"reasoning={reasoning}",
    )
    return {
        "research_plan": queries,
        "revision_count": 0,
        "max_revisions": max(
            int(getattr(settings,"max_revisions",2)),
            0,
        ),
        "deepsearch_mode": "linear",
        "deepsearch_epoch": 1,
        "deepsearch_max_epochs": max_epochs,
        "deepsearch_query_history": queries,
        "deepsearch_artifacts": {
            "phase": "planned",
            "query_count": len(queries),
            "reasoning": reasoning,
        },
        "current_step": 1,
        "status": "running",
    }



def deep_followup_plan_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str,Any]:
    """
    根据 Evaluator 发现的知识缺口，生成下一轮研究查询。
    """

    user_input = state.get("input","").strip()

    if not user_input:
        raise ValueError("AgentState.input 不能为空")

    missing_topics = [
        str(item).strip()
        for item in (state.get("missing_topics") or [])
        if str(item).strip()
    ]

    if not missing_topics:
        evaluation = str(
            state.get("evaluation","")
        ).strip()

        missing_topics = [
            evaluation or "补充验证当前报告中的关键事实"
        ]

    current_epoch = max(
        int(state.get("deepsearch_epoch", 1)),
        1,
    )

    max_epochs = max(
        int(getattr(settings,"deepsearch_max_epochs",3)),
        1,
    )

    if current_epoch >= max_epochs:
        raise ValueError("已经达到 Deep Research 的最大轮数")

    max_queries = max(
        int(getattr(settings,"deepsearch_query_num",5)),
        1,
    )

    query_history = {
        str(item).strip()
        for item in (state.get("deepsearch_query_history") or [])
        if str(item).strip()
    }

    missing_topic_context = "\n".join(
        f"- {topic}"
        for topic in missing_topics
    )

    history_text = "\n".join(
        f"- {query}"
        for query in query_history
    )

    llm = ModelFactory.for_task(
        "planning",
        temperature=0.2,
    )

    messages = [
        SystemMessage(
            content=(
                "你是 PuzzleResearch 的后续研究规划 Agent。"
                "请根据报告评估器发现的知识缺口，"
                "生成下一轮用于补充证据的搜索查询。"
                "查询必须针对缺失主题，不能重复已经使用过的查询。"
                f"最多生成 {max_queries} 个查询。"
                "只返回 JSON，不要输出 Markdown 代码块。"
                "\n\n"
                "JSON 格式："
                "\n"
                "{"
                '"queries": ["查询一", "查询二"], '
                '"reasoning": "后续研究原因"'
                "}"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：\n{user_input}\n\n"
                f"缺失主题：\n{missing_topic_context}\n\n"
                f"历史查询：\n{history_text}"
            )
        ),
    ]

    candidates: list[str] = []
    reasoning = ""

    try:
        response = llm.invoke(
            messages,
            config=config,
        )

        candidates, reasoning = _parse_plan_response(response)

    except Exception as exc:
        print(f"[deep_followup_plan_node] \n规划失败，使用缺口生成查询：{exc}")


    queries: list[str] = []

    for query in candidates:
        if query not in query_history:
            queries.append(query)

        if len(queries) >= max_queries:
            break


    if not queries:
        for topic in missing_topics:
            fallback_query = (
                f"{user_input} {topic}"
            ).strip()

            if fallback_query not in query_history:
                queries.append(fallback_query)

            if len(queries) >= max_queries:
                break

    if not queries:
        raise ValueError(
            "后续研究没有生成新的查询。"
        )

    next_epoch = current_epoch + 1

    print(
        "[deep_followup_plan_node]",
        f"epoch={next_epoch}"
        f"queries={queries}",
        f"reasoning={reasoning}",
    )

    return {
        "research_plan": queries,
        "deepsearch_epoch": next_epoch,
        "deepsearch_query_history": queries,
        "deepsearch_artifacts": {
            "phase": "followup_planned",
            "epoch": next_epoch,
            "query_count": len(queries),
            "missing_topics": missing_topics,
            "reasoning": reasoning,
        },
        "current_step": 2,
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

    search_mode = str(
        state.get("search_mode", "web")
    ).strip().lower()

    if search_mode == "deep":
        result_limit = getattr(
            settings,
            "deepsearch_results_per_query",
            5,
        )
    else:
        result_limit = getattr(
            settings,
            "web_results_per_query",
            5,
        )

    max_results = max(int(result_limit), 1)

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

    answer = _append_source_references(
        answer,
        search_results,
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



def deep_writer_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str, Any]:
    """
    Deep Research 综合写作节点。

    负责：
    - 综合多个查询返回的搜索结果；
    - 对不同来源进行交叉比较；
    - 明确指出来源之间的差异；
    - 生成结构化的深度报告。
    """

    user_input = state.get("input","").strip()
    search_results = state.get("search_results") or []

    if not user_input:
        raise ValueError("Deep Writer 缺少用户问题")

    if not search_results:
        raise ValueError("Deep Writer 没有收到搜索结果")


    selected_results = _select_deep_search_results(
        search_results,
        max_sources=20,
    )

    context_blocks: list[str] = []

    for index,item in enumerate(
        selected_results,
        start=1,
    ):
        title = str(item.get("title","")).strip()
        url = str(item.get("url","")).strip()
        query = str(item.get("query","")).strip()
        snippet = str(item.get("snippet","")).strip()
        content = str(item.get("content","")).strip()

        excerpt = (content or snippet)[:2200]

        context_blocks.append(
            f"【来源 {index}】\n"
            f"所属查询：{query}\n"
            f"标题：{title}\n"
            f"链接：{url}\n"
            f"内容：{excerpt}\n"
        )

    research_context = "\n\n".join(context_blocks)

    llm = ModelFactory.for_task(
        "writing",
        temperature=0.3,
    )

    messages = [
        SystemMessage(
            content=(
                "你是 PuzzleResearch 的 Deep Research Writer。"
                "请根据用户问题和多个搜索来源撰写深度研究报告。"
                "报告必须综合多个来源，而不是简单罗列搜索结果。"
                "请区分事实、推断和观点。"
                "如果不同来源存在冲突，请明确指出。"
                "不要编造搜索结果中没有出现的事实。"
                "引用来源时使用 [来源1]、[来源2] 格式。"
                "建议使用清晰的标题和小节。"
                "直接输出完整报告，不要描述内部工作流程。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：\n{user_input}\n"
                f"多来源研究材料：\n{research_context}\n"
            )
        ),
    ]

    response = llm.invoke(
        messages,
        config=config,
    )

    answer = _extract_response_text(response).strip()

    if not answer:
        raise ValueError("Deep Writer 模型返回了空回答")

    answer = _append_source_references(
        answer,
        selected_results,
        max_sources=20,
    )

    return {
        "draft_report": answer,
        "final_report": answer,
        "messages": [
            AIMessage(content=answer)
        ],
        "deepsearch_artifacts": {
            "phase": "synthesized",
            "epoch": state.get("deepsearch_epoch",1),
            "result_count": len(search_results),
            "context_result_count": len(selected_results),
        },
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


    if state.get("route") == "deep":
        source_items = _select_deep_search_results(
            sources,
            max_sources=20,
        )
    else:
        source_items = list(sources[:12])

    sources_lines: list[str] = []


    for index,source in enumerate(
        source_items,
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

        # 评估器不能同时声称通过并报告已知遗漏。
        if verdict == "pass" and missing_topics:
            verdict = "revise"

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



def revise_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str,Any]:
    """
    根据评估结果修订研究报告。

    Revision Agent 不重新搜索，而是：
    - 保留原报告中的正确内容；
    - 补充评估器指出的遗漏主题；
    - 修正不准确的表述；
    - 继续使用已有搜索来源。
    """

    user_input = state.get("input","").strip()
    report = (
        state.get("draft_report","")
        or state.get("final_report","")
        or ""
    ).strip()

    search_results = state.get("search_results") or []
    missing_topics = state.get("missing_topics") or []

    if not user_input:
        raise ValueError("Revision Agent 缺少用户问题")

    if not report:
        raise ValueError("Revision Agent 缺少收到原始报告")

    revision_count = int(
        state.get("revision_count", 0)
    ) + 1

    context_blocks: list[str] = []

    if state.get("route") == "deep":
        source_items = _select_deep_search_results(
            search_results,
            max_sources=20,
        )
    else:
        source_items = list(search_results[:12])

    for index, item in enumerate(
            source_items,
            start=1,
    ):
        title = str(item.get("title","")).strip()
        url = str(item.get("url","")).strip()
        snippet = str(item.get("snippet","")).strip()
        content = str(item.get("content","")).strip()

        excerpt = (content or snippet)[:2000]

        context_blocks.append(
            f"【来源 {index}】\n"
            f"标题：{title}\n"
            f"链接：{url}\n"
            f"内容：{excerpt}\n"
        )

    source_context = "\n\n".join(context_blocks)

    missing_topic_text = "\n".join(
        f"- {str(topic).strip()}"
        for topic in missing_topics
        if str(topic).strip()
    )

    llm = ModelFactory.for_task(
        "writing",
        temperature=0.3,
    )

    messages = [
        SystemMessage(
            content=(
                "你是 PuzzleResearch 的报告修订 Agent。"
                "请根据评估意见修订原始报告。"
                "保留原报告中正确的内容，"
                "补充缺失主题，修正不准确的表述。"
                "只能使用提供的搜索来源，不能编造事实。"
                "引用来源时使用 [来源1]、[来源2] 格式。"
                "请直接返回修订后的完整报告正文。"
                "不要输出修订说明。"
                "不要单独添加参考来源列表，程序会自动追加。"
            )
        ),
        HumanMessage(
            content=(
                f"用户问题：\n{user_input}\n\n"
                f"原始报告：\n{report}\n\n"
                f"评估指出的遗漏主题：\n"
                f"{missing_topic_text or '无明确遗漏主题'}\n\n"
                f"搜索来源：\n{source_context}"
            )
        ),
    ]

    response = llm.invoke(
        messages,
        config=config,
    )

    revised_report = _extract_response_text(
        response
    ).strip()

    if not revised_report:
        raise ValueError("Revision Agent 返回了空报告")

    revised_report = _append_source_references(
        revised_report,
        source_items,
        max_sources=len(source_items),
    )

    print(
        "[revise_node]",
        f"revision_count={revision_count}",
        f"missing_topics={missing_topics}",
    )

    return {
        "draft_report": revised_report,
        "final_report": revised_report,
        "revision_count": revision_count,
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



def _get_agent_profile(
        config: RunnableConfig,
) -> AgentProfile | None:
    """
    获取当前 Agent 的运行 Profile。

    没有注入 Profile 时返回 None，
    由旧的 state/config 默认逻辑继续生效。
    """

    configurable = {}

    if config:
        configurable = config.get(
            "configurable",
            {},
        ) or {}

    profile = configurable.get("agent_profile")

    if profile is None:
        return None

    if not isinstance(profile, AgentProfile):
        raise TypeError(
            "configurable.agent_profile 必须是 AgentProfile。"
        )

    return profile



def _get_agent_tool_registry(
        config: RunnableConfig,
) -> ToolRegistry:
    """
    获取当前 Agent 使用的工具注册表。

    未注入自定义注册表时，使用默认工具注册表。
    """

    configurable = {}

    if config:
        configurable = config.get(
            "configurable",
            {},
        ) or {}

    registry = configurable.get("tool_registry")

    if registry is None:
        return create_default_registry()

    if not isinstance(registry, ToolRegistry):
        raise TypeError(
            "configurable.tool_registry 必须是 ToolRegistry。"
        )

    return registry


def _make_tool_event(
        event_type: str,
        *,
        tool_name: str,
        call_count: int,
        attempt: int,
        error: str = "",
        duration_ms: float | None = None,
) -> Dict[str,Any]:
    """
    创建统一格式的工具执行事件。

    事件只保存执行元数据和错误信息，
    具体参数与结果保存在 tool_results 中。
    """

    event = {
        "event_type": event_type,
        "tool_name": tool_name,
        "call_count": call_count,
        "attempt": attempt,
    }

    if error:
        event["error"] = error

    if duration_ms is not None:
        event["duration_ms"] = round(
            max(float(duration_ms), 0.0),
            3,
        )

    return event


def _tool_call_signature(
        tool_name: str,
        arguments: Any,
) -> str:
    """
    创建稳定的工具调用签名。

    sort_keys=True 可以忽略参数字典的插入顺序，
    确保内容相同的调用得到相同签名。
    """

    return json.dumps(
        {
            "tool_name": tool_name,
            "arguments": arguments,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def _has_prior_tool_call(
        state: AgentState,
        tool_name: str,
        arguments: Dict[str,Any],
) -> bool:
    """
    判断当前工具调用是否已经执行过。
    """

    current_signature = _tool_call_signature(
        tool_name,
        arguments,
    )

    for item in state.get("tool_results") or []:
        if not isinstance(item,dict):
            continue

        prior_tool_name = str(
            item.get("tool_name","")
        ).strip()

        prior_arguments = item.get(
            "arguments",
            {},
        )

        if _tool_call_signature(
            prior_tool_name,
            prior_arguments,
        ) == current_signature:
            return True

    return False



def _get_cached_tool_results(
        state: AgentState,
        tool_name: str,
        arguments: Dict[str,Any],
) -> Dict[str,Any] | None:
    """
    查找相同工具和参数的成功执行结果。

    失败结果不缓存，避免后续调用永久复用失败信息。
    """

    current_signature = _tool_call_signature(
        tool_name,
        arguments,
    )

    for item in state.get("tool_results") or []:
        if not isinstance(item,dict):
            continue

        prior_tool_name = str(
            item.get("tool_name","")
        ).strip()

        prior_arguments = item.get(
            "arguments",
            {},
        )

        if _tool_call_signature(
            prior_tool_name,
            prior_arguments,
        ) != current_signature:
            continue

        prior_result = item.get("result")

        if not isinstance(prior_result,dict):
            continue

        if str(
            prior_result.get("error","")
        ).strip():
            continue

        return {
            "tool_name": prior_tool_name,
            "arguments": prior_arguments,
            "result": prior_result,
        }


    return None



def _limit_tool_result(
        tool_result: Any,
        max_chars: int | None,
) -> Dict[str, Any]:
    """
    限制工具 output 和 error 文本长度。

    只裁剪结果文本，不修改 return_code、
    timed_out 等结构化字段。
    """

    if isinstance(tool_result,dict):
        normalized_result = dict(tool_result)

    else:
        normalized_result = {
            "output": tool_result,
            "error": "",
        }

    if max_chars is None:
        return normalized_result

    marker = "\n...[工具结果已截断]"

    for field_name in {"output","error"}:
        value = normalized_result.get(field_name)

        if value is None:
            continue

        text = str(value)

        if len(text) <= max_chars:
            normalized_result[field_name] = text
            continue

        if max_chars <= len(marker):
            clipped_text = marker[:max_chars]
        else:
            clipped_text = (
                    text[:max_chars - len(marker)]
                    + marker
            )

        normalized_result[field_name] = clipped_text
        normalized_result["result_truncated"] = True

    return normalized_result




def agent_plan_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str,Any]:
    """
    生成并校验工具调用计划。

    该节点只负责规划，不执行工具。
    """

    user_input = state.get("input","").strip()

    if not user_input:
        raise ValueError("Agent 工具规划节点缺少用户输入。")


    try:
        profile = _get_agent_profile(config)
    except TypeError as exc:
        message = f"Agent Profile 配置无效: {exc}"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    tool_call_count = max(
        int(state.get("tool_call_count", 0)),
        0
    )

    configured_tool_call_limit = (
        profile.tool_call_limit
        if profile is not None
        and profile.tool_call_limit is not None
        else state.get(
            "tool_call_limit",
            getattr(settings,"tool_call_limit",20),
        )
    )


    tool_call_limit = max(
        int(configured_tool_call_limit),
        0,
    )

    if tool_call_count >= tool_call_limit:
        message = f"工具调用次数已达到限制： {tool_call_limit}"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    try:
        registry = _get_agent_tool_registry(config)
    except TypeError as exc:
        message = f"工具注册表配置无效: {exc}"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    available_tools = registry.list_tools()

    state_enabled_tools = state.get("enabled_tools")

    if (
        profile is not None
        and profile.enabled_tools is not None
    ):
        enabled_tools = {
            tool.name: tool.name in profile.enabled_tools
            for tool in available_tools
        }
    elif state_enabled_tools is None:
        enabled_tools = {
            tool.name: True
            for tool in available_tools
        }
    else:
        enabled_tools = state_enabled_tools

    if (
        profile is not None
        and profile.blocked_tools is not None
    ):
        for blocked_tool_name in profile.blocked_tools:
            enabled_tools[blocked_tool_name] = False

    tool_descriptions = "\n".join(
        (
            f"- {tool.name}"
            f"({tool.format_argument_contract()}): "
            f"{tool.description}"
        )
        for tool in available_tools
        if enabled_tools.get(tool.name, False)
    )


    prior_tool_results = state.get("tool_results") or []

    if prior_tool_results:
        tool_history = json.dumps(
            prior_tool_results[-5:],
            ensure_ascii=False,
            default=str,
        )

    else:
        tool_history = "尚未执行任何工具"


    planner_messages = [
        SystemMessage(
            content=(
                "你是 PuzzleResearch 的多轮工具调用规划器。"
                "请根据用户请求和工具执行历史决定下一步。"
                "如果还需要工具，返回 action=tool。"
                "如果任务已经完成，返回 action=finish。"
                "不要选择不在工具列表中的工具。"
                f"\n当前可用工具：\n{tool_descriptions}\n"
                "只能返回 JSON，不要输出 Markdown 代码块或解释文字。"
                "\n\n"
                "调用工具时的 JSON 格式："
                "\n"
                "{"
                '"action": "tool", '
                '"tool_name": "工具名称", '
                '"arguments": {"参数名": "参数值"}'
                "}"
                "\n\n"
                "完成任务时的 JSON 格式："
                "\n"
                "{"
                '"action": "finish", '
                '"final_report": "最终回答"'
                "}"
            )
        ),
        HumanMessage(
            content=(
                f"用户请求：\n{user_input}\n\n"
                f"工具执行历史：\n{tool_history}"
            )
        ),
    ]

    try:
        llm = ModelFactory.for_task(
            "planning",
            temperature=0.0,
        )

        response = llm.invoke(
            planner_messages,
            config=config,
        )

        data = _parse_json_object(
            _extract_response_text(response)
        )


        action = str(
            data.get("action","tool")
        ).strip().lower()

        if action == "finish":
            final_report = str(
                data.get("final_report","")
            ).strip()

            if not final_report:
                raise ValueError("完成动作缺少 final_report")

            return {
                "pending_tool_calls": [],
                "final_report": final_report,
                "draft_report": final_report,
                "messages": [
                    AIMessage(content=final_report),
                ],
                "agent_finished": True,
                "is_complete": False,
                "status": "running",
            }

        if action != "tool":
            raise ValueError(f"不支持的 Agent action: {action}")


        tool_name = str(
            data.get("tool_name","")
        ).strip()

        arguments = data.get("arguments", {})
        if not tool_name:
            raise ValueError("工具规划结果缺少 tool_name")

        if not isinstance(arguments,dict):
            raise ValueError("工具规划结果 arguments 必须是 JSON 对象")

    except Exception as exc:
        message = f"工具调用规划失败： {exc}"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }


    if enabled_tools is not None and not enabled_tools.get(
        tool_name,
        False,
    ):
        message = f"工具未启用：{tool_name}"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    try:
        tool = registry.get(tool_name)
    except (KeyError,ValueError) as exc:
        message = str(exc)

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }


    if (
        profile is not None
        and profile.deduplicate_tool_calls is True
        and _has_prior_tool_call(
            state,
            tool_name,
            arguments,
        )
    ):
        message = f"检测到重复工具调用：{tool_name}"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }


    try:
        tool.validate_arguments(arguments)
    except ValueError as exc:
        message = str(exc)

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }


    pending_call = {
        "tool_name": tool.name,
        "arguments": arguments,
    }

    approval_enabled = (
        profile.require_tool_approval
        if (
            profile is not None
            and profile.require_tool_approval is not None
        )
        else bool(
            getattr(settings, "tool_approval", False)
        )
    )


    approval_required = (
        approval_enabled
        and not state.get("tool_approved", False)
    )


    return {
        "pending_tool_calls": [pending_call],
        "tool_call_count": tool_call_count,
        "tool_call_limit": tool_call_limit,
        "agent_finished": False,
        "is_complete": False,
        "status": "paused" if approval_required else "running",
    }




def agent_tool_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str, Any]:
    """
    执行已经批准的工具调用。
    """

    if state.get("status") in {
        "failed",
        "cancelled",
    }:
        return {}

    try:
        profile = _get_agent_profile(config)
    except TypeError as exc:
        message = f"Agent Profile 配置无效: {exc}"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    pending_calls = state.get("pending_tool_calls") or []

    if not pending_calls:
        message = "没有待处理的工具调用。"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    pending_call = pending_calls[0]

    tool_name = str(
        pending_call.get("tool_name", "")
    ).strip()

    arguments = pending_call.get("arguments", {})

    if not tool_name:
        raise ValueError("待执行工具缺少 tool_name")

    if not isinstance(arguments, dict):
        raise ValueError("工具 arguments 必须是 JSON 对象")

    tool_call_count = max(
        int(state.get("tool_call_count", 0)),
        0,
    )

    configured_tool_call_limit = (
        profile.tool_call_limit
        if (
            profile is not None
            and profile.tool_call_limit is not None
        )
        else state.get(
            "tool_call_limit",
            getattr(settings, "tool_call_limit", 20),
        )
    )

    tool_call_limit = max(
        int(configured_tool_call_limit),
        0,
    )

    if tool_call_count >= tool_call_limit:
        message = f"工具调用次数已达到限制： {tool_call_limit}"

        return {
            "pending_tool_calls": [],
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    try:
        registry = _get_agent_tool_registry(config)
        tool = registry.get(tool_name)
        tool.validate_arguments(arguments)
    except (KeyError, TypeError,ValueError) as exc:
        message = str(exc)

        return {
            "pending_tool_calls": [],
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    if (
        profile is not None
        and profile.blocked_tools is not None
        and tool_name in profile.blocked_tools
    ):
        message = f"工具被 Agent Profile 禁止：{tool_name}"

        return {
            "pending_tool_calls": [],
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    if (
        profile is not None
        and profile.deduplicate_tool_calls is True
        and _has_prior_tool_call(
            state,
            tool_name,
            arguments,
        )
    ):
        message = f"检测到重复工具调用：{tool_name}"

        return {
            "pending_tool_calls": [],
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
        }

    tool_approval_enabled = (
        profile.require_tool_approval
        if (
            profile is not None
            and profile.require_tool_approval is not None
        )
        else bool(
            getattr(settings, "tool_approval", False)
        )
    )

    if (
        tool_approval_enabled
        and not state.get("tool_approved", False)
    ):
        approval = interrupt(
            {
                "type": "tool_approval",
                "tool_name": tool_name,
                "arguments": arguments,
                "message": f"是否批准执行工具：{tool_name}?",
            }
        )

        approved = (
            approval is True
            or approval == "approve"
            or (
                isinstance(approval, dict)
                and approval.get("approved") is True
            )
        )

        if not approved:
            message = f"工具调用未获批准：{tool_name}"

            return {
                "pending_tool_calls": [],
                "final_report": message,
                "is_complete": False,
                "status": "cancelled",
            }

    # ===== 新增：解析工具结果长度上限 =====
    configured_tool_result_limit = (
        profile.max_tool_result_chars
        if (
            profile is not None
            and profile.max_tool_result_chars is not None
        )
        else None
    )

    if (
        profile is not None
        and profile.cache_tool_results is True
    ):
        cached_call = _get_cached_tool_results(
            state,
            tool_name,
            arguments,
        )

        if cached_call is not None:
            # ===== 修改：对缓存结果重新施加长度限制 =====
            cached_result = _limit_tool_result(
                cached_call["result"],
                configured_tool_result_limit,
            )

            cached_call = {
                **cached_call,
                "result": cached_result,
            }

            output = str(
                cached_result.get("output", "")
            ).strip()

            if not output:
                output = "（缓存结果没有标准输出）"

            report = (
                f"工具「{tool_name}」复用缓存结果 ：\n"
                f"{output}"
            )

            cache_event = _make_tool_event(
                "tool_cache_hit",
                tool_name=tool_name,
                call_count=tool_call_count + 1,
                attempt=1,
                duration_ms=0.0,
            )

            return {
                "pending_tool_calls": [],
                "tool_results": [
                    cached_call
                ],
                "tool_events": [cache_event],
                "tool_call_count": tool_call_count + 1,
                "final_report": report,
                "draft_report": report,
                "messages": [
                    AIMessage(content=report)
                ],
                "is_complete": False,
                "status": "running",
            }

    configured_tool_retry_limit = (
        profile.tool_retry_limit
        if (
            profile is not None
            and profile.tool_retry_limit is not None
        )
        else 0
    )

    retry_limit = max(
        int(configured_tool_retry_limit),
        0,
    )

    retry_count = 0
    attempt = 1
    execution_events = [
        _make_tool_event(
            "tool_started",
            tool_name=tool_name,
            call_count=tool_call_count + 1,
            attempt=attempt,
        )
    ]

    while True:
        attempt_started_at = time.perf_counter()

        try:
            tool_result = registry.invoke(
                tool_name,
                arguments,
            )
            break

        except Exception as exc:
            error_text = str(exc)

            if retry_count >= retry_limit:
                execution_events.append(
                    _make_tool_event(
                        "tool_failed",
                        tool_name=tool_name,
                        call_count=tool_call_count + 1,
                        attempt=attempt,
                        error=error_text,
                        duration_ms=(
                            time.perf_counter() - attempt_started_at
                        ) * 1000,
                    )
                )

                message = (
                    f"工具「{tool_name}」执行失败，"
                    f"已重试 {retry_count} 次：{error_text}"
                )

                return {
                    "pending_tool_calls": [],
                    "tool_results": [
                        {
                            **pending_call,
                            # ===== 修改：对执行失败结果施加长度限制 =====
                            "result": _limit_tool_result(
                                {
                                    "error": str(exc),
                                },
                                configured_tool_result_limit,
                            ),
                        }
                    ],
                    "tool_events": execution_events,
                    "tool_call_count": tool_call_count + 1,
                    "final_report": message,
                    "errors": [message],
                    "last_error": message,
                    "is_complete": False,
                    "status": "failed",
                }

            execution_events.append(
                _make_tool_event(
                    "tool_retry",
                    tool_name=tool_name,
                    call_count=tool_call_count + 1,
                    attempt=attempt,
                    error=error_text,
                    duration_ms=(
                        time.perf_counter() - attempt_started_at
                    ) * 1000,
                )
            )

            retry_count += 1
            attempt += 1

    if not isinstance(tool_result, dict):
        tool_result = {
            "output": tool_result,
            "error": "",
        }

    # ===== 新增：对成功执行结果施加长度限制 =====
    tool_result = _limit_tool_result(
        tool_result,
        configured_tool_result_limit,
    )

    result_error = str(
        tool_result.get("error", "")
    ).strip()

    execution_events.append(
        _make_tool_event(
            "tool_failed" if result_error else "tool_completed",
            tool_name=tool_name,
            call_count=tool_call_count + 1,
            attempt=attempt,
            error=result_error,
            duration_ms=(
                time.perf_counter() - attempt_started_at
            ) * 1000,
        )
    )

    if isinstance(tool_result, dict):
        result_error = str(tool_result.get("error", "")).strip()

        output = str(tool_result.get("output", "")).strip()

        if not output:
            output = "（工具没有产生标准输出）"

        report = (
            f"工具「{tool_name}」执行结果：\n"
            f"{output}"
        )

        if result_error:
            report += (
                "\n\n错误信息："
                f"{result_error}"
            )

        else:
            # ===== 修改：统一使用 output，而不是整个 tool_result =====
            report = (
                f"工具「{tool_name}」执行结果：\n"
                f"{output}"
            )

        result_state: Dict[str, Any] = {
            "pending_tool_calls": [],
            "tool_results": [
                {
                    **pending_call,
                    "result": tool_result,
                }
            ],
            "tool_events": execution_events,
            "tool_call_count": tool_call_count + 1,
            "final_report": report,
            "draft_report": report,
            "messages": [
                AIMessage(content=report)
            ],
            "is_complete": False,
            "status": "failed" if result_error else "running",
        }

        if result_error:
            result_state.update(
                {
                    "errors": [result_error],
                    "last_error": result_error,
                }
            )

        print(
            "[agent_tool_node]",
            f"tool={tool_name}",
            f"call_count={tool_call_count + 1}",
            f"has_error={bool(result_error)}",
        )

        return result_state





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

    根据当前状态保留失败、取消和暂停语义，
    并将正常完成的执行标记为 completed。

    工具审批的 interrupt/resume 由 agent_tool_node 负责；
    本节点暂不负责完整的 human review 流程。
    """
    report = state.get("final_report") or state.get("draft_report","")

    current_status = state.get("status")

    is_paused = (
        current_status == "paused"
        or state.get("route") == "clarify"
        or bool(state.get("needs_clarification"))
    )

    if current_status in {
        "failed",
        "cancelled",
    }:
        return {
            "final_report": report,
            "is_complete": False,
            "status": "failed",
        }

    return {
        "final_report": report,
        "is_complete": not is_paused,
        "status": "paused" if is_paused else "completed",
    }
