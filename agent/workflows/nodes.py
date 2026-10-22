from __future__ import annotations

import json
import time
import re
import asyncio
import logging
import random
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from typing import Dict,Any
from collections.abc import Mapping

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage,HumanMessage,SystemMessage
from langgraph.types import interrupt
from langchain_core.runnables import RunnableConfig

from agent.core.model_factory import ModelFactory
from agent.core.middleware import clear_tool_results_for_context
from agent.core.state import AgentState,QueryState
from agent.parsers import XMLToolParser
from agent.core.profile import AgentProfile
from agent.core.smart_router import smart_route
from agent.prompts import get_prompt_manager
from common.config import settings
from tools.search import (
    SearchManager,
    get_default_search_runtime_state,
)
from tools.registry import ToolRegistry
from tools.builtin import create_default_registry
from agent.core.event_stream import (
    get_emitter_sync,
)
from agent.workflows.stuck_middleware import detect_stuck, inject_stuck_hint
from agent.workflows.research_tree import (
    ResearchTree,
    TreeExplorationBudgetExceeded,
    TreeExplorer,
)

from agent.workflows.knowledge_gap import (
    analyze_research_quality,
    normalize_knowledge_gap_analysis,
)
from agent.workflows.research_budget import (
    add_budget_tokens,
    budget_state_fields,
    estimate_tokens,
    model_call_tokens,
    new_research_budget,
    refresh_research_budget,
    set_budget_elapsed,
)
from agent.workflows.domain_router import (
    DomainClassifier,
    ResearchDomain,
    build_provider_profile,
)
from agent.workflows.result_aggregator import ResultAggregator
from agent.workflows.evidence_passages import passages_from_fetched_pages
from agent.workflows.query_strategy import (
    analyze_query_coverage,
    backfill_diverse_queries,
    build_quality_diagnostics,
    merge_query_history,
    new_query_history_entries,
)
from agent.workflows.compressor import ResearchCompressor
from agent.workflows.viz_planner import (
    finalize_report_visualizations,
)
from agent.workflows.claim_verifier import ClaimStatus, ClaimVerifier
from agent.workflows.role_agents import (
    VerifierAgent,
    VerifierTask,
    WriterAgent,
    WriterTask,
)
from agent.workflows import quality_assessor
from tools.research import ContentFetcher


_TODO_TOOL_NAMES = frozenset(
    {"create_tasks", "view_tasks", "update_task", "get_next_task"}
)
_DEFAULT_TODO_SYSTEM_PROMPT = (
    "任务清单仅用于复杂、多步骤或用户明确要求跟踪进度的目标；"
    "简单问题直接回答，不要创建清单。"
    "开始复杂任务时创建清晰、可执行的任务项；执行中及时更新状态，"
    "每完成一项就立即标记完成，遇到阻塞则保留未完成状态并说明原因。"
)


logger = logging.getLogger(__name__)


def _build_model_user_content(
    text: str,
    images: Any,
) -> str | list[dict[str, Any]]:
    """Build OpenAI-compatible text and image content for a user message."""

    normalized_text = str(text or "")
    parts: list[dict[str, Any]] = []

    if normalized_text:
        parts.append({"type": "text", "text": normalized_text})

    if isinstance(images, list):
        for image in images:
            if not isinstance(image, Mapping):
                continue

            data = image.get("data") or image.get("url") or ""
            if not isinstance(data, str) or not data.strip():
                continue

            mime = str(image.get("mime") or "image/png").strip().lower()
            if not mime.startswith("image/"):
                continue

            data = data.strip()
            data_url = (
                data
                if data.startswith("data:")
                else f"data:{mime};base64,{data}"
            )
            parts.append(
                {
                    "type": "image_url",
                    "image_url": {"url": data_url},
                }
            )

    if not parts:
        return normalized_text

    if len(parts) == 1 and parts[0]["type"] == "text":
        return normalized_text

    if not normalized_text:
        parts.insert(
            0,
            {"type": "text", "text": "请分析我上传的图片。"},
        )

    return parts


def _research_budget_for(
    state: Mapping[str, Any],
    config: RunnableConfig | None,
    *,
    initial_tokens: int = 0,
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
        initial_tokens=initial_tokens,
    )
    external_reason = str(
        state.get("budget_stop_reason")
        or artifacts.get("budget_stop_reason")
        or ""
    ).strip()
    if external_reason and not budget["stop_reason"]:
        budget["stop_reason"] = external_reason
    return budget


def _store_research_budget(
    artifacts: Mapping[str, Any] | None,
    budget: Mapping[str, Any],
    *,
    mode: str | None = None,
) -> dict[str, Any]:
    result = dict(artifacts or {})
    refreshed = refresh_research_budget(budget)
    result["research_budget"] = refreshed
    result["budget_stop_reason"] = refreshed["stop_reason"]
    existing_quality = result.get("quality_summary")
    quality_summary = (
        dict(existing_quality)
        if isinstance(existing_quality, Mapping)
        else {}
    )
    quality_summary.update({
        **budget_state_fields(refreshed),
        "stop_reason": refreshed["stop_reason"],
        "tokens_used": int(refreshed.get("tokens_used", 0) or 0),
        "elapsed_seconds": float(refreshed.get("elapsed_seconds", 0.0) or 0.0),
        "searches_used": int(refreshed.get("searches_used", 0) or 0),
        "max_seconds": float(refreshed.get("max_seconds", 0.0) or 0.0),
        "max_tokens": int(refreshed.get("max_tokens", 0) or 0),
    })
    result["quality_summary"] = quality_summary
    if mode in {"linear", "tree"}:
        alias_key = f"{mode}_budget"
        alias = dict(result.get(alias_key) or {})
        alias.update({
            key: value
            for key, value in refreshed.items()
            if key not in {"started_at_epoch", "last_updated_epoch"}
        })
        result[alias_key] = alias
    return result


def _record_research_model_usage(
    budget: Mapping[str, Any],
    messages: Any,
    response: Any,
) -> dict[str, Any]:
    return add_budget_tokens(
        budget,
        model_call_tokens(messages, response),
    )


def _research_budget_allows_call(
    budget: Mapping[str, Any],
    messages: Any,
) -> tuple[dict[str, Any], bool]:
    refreshed = refresh_research_budget(budget)
    if refreshed["stop_reason"]:
        return refreshed, False

    max_tokens = max(int(refreshed.get("max_tokens", 0) or 0), 0)
    if max_tokens and int(refreshed.get("tokens_used", 0) or 0) + estimate_tokens(messages) >= max_tokens:
        refreshed["stop_reason"] = "token_budget_exceeded"
        return refreshed, False

    return refreshed, True


def _budget_limited_research_report(
    topic: str,
    results: list[Mapping[str, Any]],
    stop_reason: str,
) -> str:
    lines = [
        f"# {topic}",
        "",
        f"研究因预算限制提前停止（{stop_reason}）。以下仅整理当前已获取的证据，不能视为完整研究结论。",
        "",
        "## 已收集证据",
    ]
    for index, item in enumerate(results[:8], start=1):
        title = str(item.get("title", "")).strip() or f"来源 {index}"
        excerpt = str(
            item.get("content")
            or item.get("snippet")
            or item.get("summary")
            or ""
        ).strip()
        if len(excerpt) > 700:
            excerpt = excerpt[:697].rstrip() + "..."
        url = str(item.get("url", "")).strip()
        citation = f"[来源{index}]"
        lines.append(f"### {index}. {title}")
        lines.append(excerpt or "当前来源未提供可用摘要文本。")
        lines.append(citation)
        if url:
            lines.append(url)
        lines.append("")
    if not results:
        lines.append("预算耗尽前没有取得可引用的搜索材料。")
    return "\n".join(lines).strip()


def _extract_response_text(response: Any) -> str:
    """
    从模型响应中提取文本。

    ChatOpenAI 通常返回 AIMessage，
    但保留这个辅助函数可以兼容不同响应对象。
    """
    if isinstance(response, Mapping):
        content = response.get("content")
        if isinstance(content, str):
            return content
        choices = response.get("choices")
        if isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, Mapping) else None
            if isinstance(message, Mapping) and isinstance(message.get("content"), str):
                return message["content"]

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



def _configured_route(
        config: RunnableConfig,
) -> str | None:
    if not isinstance(config,Mapping):
        return None

    configurable = config.get("configurable",{})

    if not isinstance(configurable,Mapping):
        return None

    search_mode = configurable.get("search_mode")

    if isinstance(search_mode,Mapping):
        mode = str(
            search_mode.get("mode","")
        ).strip().lower()

        if mode in {
            "direct",
            "web",
            "deep",
            "agent",
            "clarify",
        }:
            return mode

        if bool(search_mode.get("useDeepSearch")):
            return "deep"

        if bool(search_mode.get("useAgent")):
            return "agent"

        if bool(search_mode.get("useWebSearch")):
            return "web"

    if isinstance(search_mode,str):
        mode = str(search_mode).strip().lower()

        if mode in {
            "direct",
            "web",
            "deep",
            "agent",
            "clarify",
        }:
            return mode

    return None



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

    forced_route = _configured_route(config)

    if state.get("is_cancelled"):
        result = {
            "route": "direct",
            "routing_reasoning": "请求已取消，跳过模型路由。",
            "routing_confidence": 1.0,
            "suggested_queries": [],
            "clarification_question": "",
            "needs_clarification": False,
        }
    elif forced_route is not None:
        result = {
            "route": forced_route,
            "routing_reasoning": (
                "客户端通过 search_mode 明确指定执行路径。"
            ),
            "routing_confidence": 1.0,
            "suggested_queries": [],
            "clarification_question": "",
            "needs_clarification": (
                forced_route == "clarify"
            ),
        }
    else:
        result = smart_route(
            user_input,
            config=config,
        )

    print(
        "[router_node]",
        f"route={result['route']}",
        f"confidence={result['routing_confidence']:.2f}",
    )

    domain_result: dict[str, Any] = {}
    if bool(getattr(settings, "domain_routing_enabled", False)) and result["route"] in {"web", "deep"}:
        domain_llm = None
        try:
            domain_llm = ModelFactory.for_task(
                "routing",
                temperature=0.3,
                config=config,
            )
        except Exception:
            domain_llm = None
        classification = DomainClassifier(
            llm=domain_llm,
            config=config,
        ).classify(user_input, state.get("images"))
        domain_result = {
            "domain": classification.domain.value,
            "domain_config": classification.to_dict(),
        }
        print(
            "[domain_router]",
            f"domain={classification.domain.value}",
            f"confidence={classification.confidence:.2f}",
        )

    return {
        **result,
        **domain_result,
        "messages": [
            HumanMessage(
                content=_build_model_user_content(
                    user_input,
                    state.get("images"),
                )
            )
        ],
    }


def _hitl_checkpoint_enabled(
    config: RunnableConfig | None,
    checkpoint: str,
) -> bool:
    configurable = (
        config.get("configurable", {})
        if isinstance(config, Mapping)
        else {}
    )
    if not isinstance(configurable, Mapping):
        configurable = {}

    configured = configurable.get(
        "hitl_checkpoints",
        getattr(settings, "hitl_checkpoints", ""),
    )
    if isinstance(configured, str):
        checkpoints = {
            item.strip().lower()
            for item in configured.split(",")
            if item.strip()
        }
    elif isinstance(configured, (list, tuple, set)):
        checkpoints = {
            str(item).strip().lower()
            for item in configured
            if str(item).strip()
        }
    else:
        checkpoints = set()
    return checkpoint.strip().lower() in checkpoints


def _normalize_reviewed_queries(
    value: Any,
    *,
    max_queries: int,
) -> list[str]:
    candidate = value
    if isinstance(candidate, Mapping):
        candidate = candidate.get(
            "queries",
            candidate.get("content", []),
        )

    if isinstance(candidate, str):
        text = candidate.strip()
        if text.startswith(("[", "{")):
            try:
                parsed = json.loads(text)
            except (TypeError, ValueError):
                parsed = None
            if parsed is not None:
                return _normalize_reviewed_queries(
                    parsed,
                    max_queries=max_queries,
                )
        values = text.splitlines()
    elif isinstance(candidate, (list, tuple)):
        values = list(candidate)
    else:
        values = []

    queries: list[str] = []
    seen: set[str] = set()
    for item in values:
        query = re.sub(r"^\s*(?:(?:[-*+])|(?:\d+[.)]))\s*", "", str(item)).strip()
        key = query.casefold()
        if query and key not in seen:
            queries.append(query)
            seen.add(key)
        if len(queries) >= max(max_queries, 1):
            break
    return queries


def _reviewed_tree_fields(
    state: AgentState,
    queries: list[str],
) -> dict[str, Any]:
    if (
        state.get("route") != "deep"
        or str(state.get("deepsearch_mode", "")).strip().lower() != "tree"
    ):
        return {}

    tree = ResearchTree.from_dict(state.get("research_tree"))
    epoch = max(int(state.get("deepsearch_epoch", 1) or 1), 1)
    removed_ids = {
        node.id
        for node in tree.nodes.values()
        if node.depth > 0
        and node.epoch == epoch
        and str(getattr(node.status, "value", node.status)).lower() == "pending"
    }
    for node_id in removed_ids:
        tree.nodes.pop(node_id, None)
    for node in tree.nodes.values():
        node.children_ids = [
            child_id
            for child_id in node.children_ids
            if child_id not in removed_ids
        ]

    branch_limit = max(
        len(queries),
        int(getattr(settings, "deepsearch_tree_max_branches", 4) or 4),
        1,
    )
    tree.max_branches = branch_limit
    added = tree.add_queries(
        queries,
        epoch=epoch,
        max_branches=branch_limit,
    )
    if not added:
        return {
            "tree_exploration_enabled": False,
            "deepsearch_mode": "linear",
        }
    return {
        "research_tree": tree.to_dict(),
        "tree_exploration_enabled": True,
        "deepsearch_mode": "tree",
    }


def _content_review_decision(value: Any) -> tuple[str, str | None]:
    if value is True:
        return "approve", None
    if value is False:
        return "reject", None
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"approve", "approved", "accept"}:
            return "approve", None
        if normalized in {"reject", "rejected", "deny", "cancel"}:
            return "reject", None
        return "edit", value
    if isinstance(value, Mapping):
        action = str(
            value.get("type")
            or value.get("action")
            or value.get("decision")
            or ""
        ).strip().lower()
        if value.get("approved") is True or action in {
            "approve", "approved", "accept"
        }:
            return "approve", None
        if value.get("approved") is False or action in {
            "reject", "rejected", "deny", "cancel"
        }:
            return "reject", None
        if "content" in value and isinstance(value.get("content"), str):
            return "edit", value["content"]
    return "invalid", None


def _format_source_review_snapshot(state: AgentState) -> str:
    raw_sources = state.get("search_results") or state.get("sources") or []
    lines: list[str] = []
    seen_urls: set[str] = set()
    for item in raw_sources:
        if not isinstance(item, Mapping):
            continue
        url = str(item.get("url", "")).strip()
        key = url.rstrip("/").casefold()
        if key and key in seen_urls:
            continue
        if key:
            seen_urls.add(key)
        title = str(item.get("title", "")).strip() or "未命名来源"
        query = str(item.get("query", "")).strip()
        excerpt = str(
            item.get("fetched_content")
            or item.get("content")
            or item.get("snippet")
            or ""
        ).strip()
        entry = [f"- {title}"]
        if query:
            entry.append(f"  查询：{query}")
        if url:
            entry.append(f"  链接：{url}")
        if excerpt:
            entry.append(f"  摘要：{excerpt[:500]}")
        lines.extend(entry)
        if len(seen_urls) >= 12 or len(lines) >= 72:
            break
    return "\n".join(lines)[:6000]


def sources_review_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Pause after retrieval so a person can inspect sources and guide writing."""
    if not _hitl_checkpoint_enabled(config, "sources"):
        return {}

    snapshot = _format_source_review_snapshot(state)
    if not snapshot:
        return {}

    result = interrupt(
        {
            "type": "sources_review",
            "checkpoint": "sources",
            "message": "请检查已收集的来源；可补充写作要求后继续，或取消本次研究。",
            "content": str(state.get("human_guidance", "") or ""),
            "snapshot": snapshot,
            "available_actions": ["approve", "edit", "reject"],
        }
    )
    decision, guidance = _content_review_decision(result)
    if decision == "reject":
        return {
            "is_cancelled": True,
            "is_complete": False,
            "status": "cancelled",
            "final_report": "来源审核后取消了本次研究。",
        }
    if decision == "approve":
        return {"status": "running"}
    if decision == "edit" and guidance is not None:
        return {"human_guidance": guidance, "status": "running"}
    message = "来源审核响应无效，已停止执行。"
    return {
        "is_complete": False,
        "status": "failed",
        "final_report": message,
        "last_error": message,
        "errors": [message],
    }


def draft_review_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Pause after writing so a person can approve or edit the draft."""
    if not _hitl_checkpoint_enabled(config, "draft"):
        return {}

    draft = str(state.get("draft_report") or state.get("final_report") or "").strip()
    if not draft:
        return {}

    result = interrupt(
        {
            "type": "draft_review",
            "checkpoint": "draft",
            "message": "请审核初稿，可直接修改正文后交给评估器，或取消本次研究。",
            "content": draft,
            "available_actions": ["approve", "edit", "reject"],
        }
    )
    decision, revised_draft = _content_review_decision(result)
    if decision == "reject":
        return {
            "is_cancelled": True,
            "is_complete": False,
            "status": "cancelled",
            "final_report": "初稿审核后取消了本次研究。",
        }
    if decision == "approve":
        return {"status": "running"}
    if decision == "edit" and revised_draft and revised_draft.strip():
        return {
            "draft_report": revised_draft.strip(),
            "final_report": revised_draft.strip(),
            "status": "running",
        }
    message = "初稿审核响应无效或内容为空，已停止执行。"
    return {
        "is_complete": False,
        "status": "failed",
        "final_report": message,
        "last_error": message,
        "errors": [message],
    }


def plan_review_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Optionally pause before search so a person can approve or edit the plan."""
    queries = _normalize_reviewed_queries(
        state.get("research_plan") or [state.get("input", "")],
        max_queries=(
            int(getattr(settings, "web_max_queries", 3))
            if state.get("route") == "web"
            else int(getattr(settings, "deepsearch_query_num", 5))
        ),
    )
    if not _hitl_checkpoint_enabled(config, "plan"):
        return {"research_plan": queries}

    review_result = interrupt(
        {
            "type": "plan_review",
            "checkpoint": "plan",
            "message": "请审核研究计划。每行一条查询，可批准、修改或取消研究。",
            "content": "\n".join(queries),
            "queries": queries,
            "available_actions": ["approve", "edit", "reject"],
        }
    )

    decision = ""
    edited_plan: Any = None
    if review_result is True:
        decision = "approve"
    elif review_result is False or review_result == "reject":
        decision = "reject"
    elif isinstance(review_result, str):
        normalized = review_result.strip().lower()
        if normalized in {"reject", "rejected", "deny", "cancel"}:
            decision = "reject"
        elif normalized in {"approve", "approved", "accept"}:
            decision = "approve"
        elif review_result.strip():
            decision = "edit"
            edited_plan = review_result
    elif isinstance(review_result, Mapping):
        action = str(
            review_result.get("type")
            or review_result.get("action")
            or review_result.get("decision")
            or ""
        ).strip().lower()
        if review_result.get("approved") is False or action in {
            "reject", "rejected", "deny", "cancel"
        }:
            decision = "reject"
        elif "queries" in review_result or "content" in review_result or action in {
            "edit", "edit_plan", "modify"
        }:
            decision = "edit"
            edited_plan = review_result
        elif review_result.get("approved") is True or action in {
            "approve", "approved", "accept"
        }:
            decision = "approve"

    if decision not in {"approve", "edit", "reject"}:
        message = "研究计划审核响应无效，已停止执行。"
        return {
            "is_complete": False,
            "status": "failed",
            "final_report": message,
            "last_error": message,
            "errors": [message],
        }

    if decision == "reject":
        return {
            "is_cancelled": True,
            "is_complete": False,
            "status": "cancelled",
            "final_report": "研究计划已取消，未执行搜索。",
        }

    if decision == "edit":
        queries = _normalize_reviewed_queries(
            edited_plan,
            max_queries=(
                int(getattr(settings, "web_max_queries", 3))
                if state.get("route") == "web"
                else int(getattr(settings, "deepsearch_query_num", 5))
            ),
        )
        if not queries:
            return {
                "is_complete": False,
                "status": "failed",
                "final_report": "修改后的研究计划不能为空。",
                "last_error": "修改后的研究计划不能为空。",
                "errors": ["修改后的研究计划不能为空。"],
            }

    return {
        "research_plan": queries,
        "plan_reviewed": True,
        "status": "running",
        **_reviewed_tree_fields(state, queries),
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
            config=config,
        )

        messages = [
            SystemMessage(
                content=(
                    f"{get_prompt_manager(config).get_prompt('web_planner')}"
                    f"最多生成 {max_queries} 个查询。"
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
                content=_build_model_user_content(
                    f"用户问题：\n{user_input}",
                    state.get("images"),
                )
            ),
        ]

        messages = _apply_agent_system_prompt(
            messages,
            config,
        )

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

    research_budget = _research_budget_for(
        state,
        config,
        initial_tokens=estimate_tokens(user_input),
    )

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
        research_budget = add_budget_tokens(
            research_budget,
            estimate_tokens(queries),
        )

    else:
        llm = ModelFactory.for_task(
            "planning",
            temperature=0.2,
            config=config,
        )

        messages = [
            SystemMessage(
                content=(
                    f"{get_prompt_manager(config).get_prompt('deep_research_planner')}"
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
                content=_build_model_user_content(
                    f"用户问题：\n{user_input}",
                    state.get("images"),
                )
            ),
        ]

        messages = _apply_agent_system_prompt(
            messages,
            config,
        )

        research_budget = refresh_research_budget(research_budget)
        prompt_tokens = estimate_tokens(messages)
        max_tokens = int(research_budget["max_tokens"] or 0)
        if research_budget["stop_reason"]:
            queries = [user_input]
            reasoning = "研究预算在规划前已耗尽，使用原始问题并停止后续研究。"
        elif max_tokens and int(research_budget["tokens_used"]) + prompt_tokens >= max_tokens:
            research_budget["stop_reason"] = "token_budget_exceeded"
            queries = [user_input]
            reasoning = "规划请求将超出 token 预算，使用原始问题并停止后续研究。"
        else:
            try:
                response = llm.invoke(
                    messages,
                    config=config,
                )

                research_budget = add_budget_tokens(
                    research_budget,
                    model_call_tokens(messages, response),
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

    queries = backfill_diverse_queries(
        topic=user_input,
        existing_queries=queries,
        historical_queries=list(state.get("deepsearch_query_history") or []),
        query_num=max_queries,
    )
    query_coverage = analyze_query_coverage(queries)


    max_epochs = max(
        int(getattr(settings,"deepsearch_max_epochs",3)),
        1,
    )

    runtime_config = (
        config.get("configurable", {})
        if isinstance(config, Mapping)
        else {}
    )

    requested_mode = str(
        runtime_config.get(
            "deepsearch_mode",
            getattr(
                settings,
                "deepsearch_mode",
                "auto",
            ),
        )
        or "auto"
    ).strip().lower()

    tree_enabled = (
        bool(
            getattr(
                settings,
                "deepsearch_tree_enabled",
                True,
            )
        )
        and requested_mode
        in {
            "auto",
            "tree",
        }
    )

    research_tree = ResearchTree.from_queries(
        user_input,
        queries,
        epoch=1,
        max_branches=int(
            getattr(
                settings,
                "deepsearch_tree_max_branches",
                4,
            )
            or 4
        ),
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
        "research_tree": research_tree.to_dict(),
        "current_branch_id": research_tree.root_id,
        "tree_exploration_enabled": tree_enabled,
        "deepsearch_mode": (
            "tree"
            if tree_enabled
            else "linear"
        ),
        "deepsearch_epoch": 1,
        "deepsearch_max_epochs": max_epochs,
        "deepsearch_query_history": queries,
        "deepsearch_artifacts": _store_research_budget({
            "phase": "planned",
            "query_count": len(queries),
            "reasoning": reasoning,
            "query_coverage": query_coverage,
        }, research_budget),
        **budget_state_fields(research_budget),
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

    research_budget = _research_budget_for(state, config)

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

    query_history = merge_query_history(
        [],
        state.get("deepsearch_query_history") or [],
    )
    query_history_keys = {query.casefold() for query in query_history}

    missing_topic_context = "\n".join(
        f"- {topic}"
        for topic in missing_topics
    )

    gap_analysis = state.get("knowledge_gap_analysis")
    suggested_gap_queries = (
        gap_analysis.get("suggested_queries", [])
        if isinstance(gap_analysis, dict)
        else []
    )
    if not isinstance(suggested_gap_queries, list):
        suggested_gap_queries = []
    suggested_query_context = "\n".join(
        f"- {query}"
        for query in suggested_gap_queries
        if str(query).strip()
    ) or "无"

    history_text = "\n".join(
        f"- {query}"
        for query in query_history
    )

    llm = ModelFactory.for_task(
        "planning",
        temperature=0.2,
        config=config,
    )

    messages = [
        SystemMessage(
            content=(
                f"{get_prompt_manager(config).get_prompt('deep_followup_planner')}"
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
            content=_build_model_user_content(
                (
                    f"用户问题：\n{user_input}\n\n"
                    f"缺失主题：\n{missing_topic_context}\n\n"
                    f"知识缺口分析建议查询：\n{suggested_query_context}\n\n"
                    f"历史查询：\n{history_text}"
                ),
                state.get("images"),
            )
        ),
    ]

    messages = _apply_agent_system_prompt(
        messages,
        config,
    )

    candidates: list[str] = []
    reasoning = ""

    research_budget, can_call_model = _research_budget_allows_call(
        research_budget,
        messages,
    )
    if can_call_model:
        try:
            response = llm.invoke(
                messages,
                config=config,
            )
            research_budget = _record_research_model_usage(
                research_budget,
                messages,
                response,
            )
            candidates, reasoning = _parse_plan_response(response)

        except Exception as exc:
            print(f"[deep_followup_plan_node] \n规划失败，使用缺口生成查询：{exc}")
    else:
        reasoning = "研究预算已耗尽，跳过后续规划模型调用。"


    queries: list[str] = []

    for query in candidates:
        normalized_query = str(query or "").strip()
        query_key = normalized_query.casefold()
        if normalized_query and query_key not in query_history_keys:
            queries.append(normalized_query)
            query_history_keys.add(query_key)

        if len(queries) >= max_queries:
            break


    if not queries:
        for suggested_query in suggested_gap_queries:
            fallback_query = str(suggested_query or "").strip()
            query_key = fallback_query.casefold()
            if fallback_query and query_key not in query_history_keys:
                queries.append(fallback_query)
                query_history_keys.add(query_key)

            if len(queries) >= max_queries:
                break

    if not queries:
        for topic in missing_topics:
            fallback_query = (
                f"{user_input} {topic}"
            ).strip()

            query_key = fallback_query.casefold()
            if fallback_query and query_key not in query_history_keys:
                queries.append(fallback_query)
                query_history_keys.add(query_key)

            if len(queries) >= max_queries:
                break

    queries = backfill_diverse_queries(
        topic=user_input,
        existing_queries=queries,
        historical_queries=query_history,
        query_num=max_queries,
    )

    if not queries:
        if research_budget.get("stop_reason"):
            artifacts = _store_research_budget(
                state.get("deepsearch_artifacts"),
                research_budget,
            )
            artifacts.update({
                "phase": "followup_skipped",
                "epoch": current_epoch,
                "missing_topics": missing_topics,
                "reasoning": reasoning,
            })
            return {
                "research_plan": [],
                "deepsearch_artifacts": artifacts,
                **budget_state_fields(research_budget),
                "current_step": 2,
                "status": "running",
            }
        raise ValueError(
            "后续研究没有生成新的查询。"
        )

    next_query_history = merge_query_history(query_history, queries)

    next_epoch = current_epoch + 1

    research_tree = ResearchTree.from_dict(
        state.get("research_tree")
    )

    research_tree.epoch = next_epoch

    research_tree.add_queries(
        queries,
        epoch=next_epoch,
        max_branches=int(
            getattr(
                settings,
                "deepsearch_tree_max_branches",
                4,
            )
            or 4
        ),
    )

    artifacts = dict(state.get("deepsearch_artifacts") or {})
    artifacts = _store_research_budget(artifacts, research_budget)
    artifacts.update({
        "phase": "followup_planned",
        "epoch": next_epoch,
        "query_count": len(queries),
        "query_history_count": len(next_query_history),
        "queries": queries,
        "missing_topics": missing_topics,
        "reasoning": reasoning,
    })

    print(
        "[deep_followup_plan_node]",
        f"epoch={next_epoch}"
        f"queries={queries}",
        f"reasoning={reasoning}",
    )

    return {
        "research_plan": queries,
        "research_tree": research_tree.to_dict(),
        "current_branch_id": research_tree.root_id,
        "deepsearch_epoch": next_epoch,
        "deepsearch_query_history": queries,
        "deepsearch_artifacts": artifacts,
        **budget_state_fields(research_budget),
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

    domain_value = str(state.get("domain", "general") or "general").strip().lower()
    try:
        domain = ResearchDomain(domain_value)
    except ValueError:
        domain = ResearchDomain.GENERAL
    domain_config = state.get("domain_config") or {}
    provider_profile = build_provider_profile(
        domain_config.get("suggested_sources", [])
        if isinstance(domain_config, dict)
        else [],
        domain=domain,
    )

    manager = SearchManager(
        runtime_state=get_default_search_runtime_state()
    )

    try:
        results = manager.search(
            query,
            max_results=max_results,
            strategy=(
                str(getattr(settings, "search_strategy", "fallback"))
                if search_mode == "deep"
                else None
            ),
            provider_profile=(provider_profile if search_mode == "deep" else None),
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
            "score": float(item.get("score") or 0.0),
            "published_date": item.get("published_date"),
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

    fetched_pages: list[Dict[str, Any]] = []
    passages: list[Dict[str, Any]] = []
    if search_mode == "deep" and bool(
        getattr(settings, "deepsearch_enable_research_fetcher", False)
    ):
        pages = ContentFetcher().fetch_many(
            [item["url"] for item in search_results if item.get("url")]
        )
        fetched_pages = [page.to_dict() for page in pages]
        passages = passages_from_fetched_pages(fetched_pages)
        page_by_url = {
            str(page.get("url", "")): page
            for page in fetched_pages
        }
        for item in search_results:
            page = page_by_url.get(str(item.get("url", "")))
            if page and page.get("text"):
                item["fetched_content"] = str(page["text"])

    print(
        "[search_node]",
        f"query={query}",
        f"results={len(search_results)}",
    )

    return {
        "search_results": search_results,
        "sources": sources,
        "fetched_pages": fetched_pages,
        "passages": passages,
    }


def deep_linear_search_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """Run linear Deep Research queries sequentially under a shared budget."""
    topic = str(state.get("input", "") or "").strip()
    queries = list(state.get("research_plan") or []) or [topic]
    queries = list(dict.fromkeys(
        str(query or "").strip()
        for query in queries
        if str(query or "").strip()
    ))

    artifacts = dict(state.get("deepsearch_artifacts") or {})
    budget = _research_budget_for(state, config)
    searches_used = int(budget.get("searches_used", 0) or 0)
    search_results: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    fetched_pages: list[dict[str, Any]] = []
    passages: list[dict[str, Any]] = []
    errors: list[str] = []

    for query in queries:
        budget = refresh_research_budget(budget)
        if budget["stop_reason"]:
            break

        budget = add_budget_tokens(
            budget,
            _estimate_deepsearch_tokens(query),
        )
        if budget["stop_reason"]:
            break

        result = search_node({
            "query": query,
            "search_mode": "deep",
            "domain": state.get("domain", "general"),
            "domain_config": state.get("domain_config", {}),
        })
        try:
            from agent.workflows.browser_visualizer import visualize_urls_from_results

            visualize_urls_from_results(
                state=state,
                config=config,
                results=result.get("search_results", []),
                max_urls=1,
                reason="deepsearch:linear",
            )
        except Exception:
            logger.debug("Deep Research 浏览器预览启动失败", exc_info=True)
        errors.extend(str(item) for item in result.get("errors", []) or [])
        search_results.extend(
            dict(item)
            for item in result.get("search_results", []) or []
            if isinstance(item, Mapping)
        )
        sources.extend(
            dict(item)
            for item in result.get("sources", []) or []
            if isinstance(item, Mapping)
        )
        fetched_pages.extend(
            dict(item)
            for item in result.get("fetched_pages", []) or []
            if isinstance(item, Mapping)
        )
        passages.extend(
            dict(item)
            for item in result.get("passages", []) or []
            if isinstance(item, Mapping)
        )
        searches_used += 1
        budget["searches_used"] = searches_used
        budget = add_budget_tokens(
            budget,
            _estimate_search_result_tokens(result.get("search_results", []) or []),
        )

        budget = refresh_research_budget(budget)
        if budget["stop_reason"]:
            break

    budget["searches_used"] = searches_used
    budget = refresh_research_budget(budget)
    artifacts.update({
        "phase": "linear_searched",
        "mode": "linear",
        "result_count": len(search_results),
    })
    artifacts = _store_research_budget(
        artifacts,
        budget,
        mode="linear",
    )
    if errors:
        artifacts["search_errors"] = errors

    return {
        "research_plan": queries,
        "search_results": search_results,
        "sources": sources,
        "fetched_pages": fetched_pages,
        "passages": passages,
        "errors": errors,
        "deepsearch_artifacts": artifacts,
        **budget_state_fields(budget),
        "quality_summary": dict(artifacts.get("quality_summary") or {}),
        "current_step": 2,
        "status": "running",
    }


def _estimate_deepsearch_tokens(text: Any) -> int:
    value = str(text or "")
    return max(1, len(value) // 4) if value else 0


def _estimate_search_result_tokens(results: list[Any]) -> int:
    total = 0
    for result in results:
        if not isinstance(result, Mapping):
            continue
        total += _estimate_deepsearch_tokens(result.get("title", ""))
        excerpt = (
            result.get("content")
            or result.get("snippet")
            or result.get("summary")
            or ""
        )
        total += _estimate_deepsearch_tokens(str(excerpt)[:600])
    return total


def deep_tree_search_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str, Any]:
    """执行树状 Deep Research，并输出与线性搜索相同的结果契约。"""

    user_input = str(state.get("input", "")).strip()
    if not user_input:
        raise ValueError("树状 Deep Research 缺少用户问题")
    assigned_tasks = [
        item for item in (state.get("research_tasks") or [])
        if isinstance(item, Mapping) and str(item.get("topic", "")).strip()
    ]

    configurable = (
        config.get("configurable", {})
        if isinstance(config, Mapping)
        else {}
    )
    if not isinstance(configurable, Mapping):
        configurable = {}

    thread_id = str(
        configurable.get("thread_id")
        or state.get("thread_id")
        or ""
    ).strip()
    event_emitter = None
    if thread_id:
        try:
            event_emitter = get_emitter_sync(thread_id)
        except Exception:
            event_emitter = None

    def publish_tree_event(
        event_type: str,
        data: dict[str, Any],
    ) -> None:
        if event_emitter is None:
            return
        event_emitter.emit_sync(event_type, data)

    def configured_int(name: str, default: int) -> int:
        try:
            value = configurable.get(name, getattr(settings, name, default))
            return int(value)
        except (TypeError, ValueError):
            return default

    def configured_float(name: str, default: float) -> float:
        try:
            value = configurable.get(name, getattr(settings, name, default))
            return float(value)
        except (TypeError, ValueError):
            return default

    max_depth = max(
        configured_int("deepsearch_tree_max_depth", 2),
        0,
    )
    max_branches = max(
        configured_int("deepsearch_tree_max_branches", 4),
        1,
    )
    queries_per_branch = max(
        configured_int("deepsearch_tree_queries_per_branch", 3),
        1,
    )
    parallel_branches = max(
        configured_int("deepsearch_tree_parallel_branches", 3),
        0,
    )
    per_query_results = max(
        configured_int("deepsearch_results_per_query", 5),
        1,
    )
    max_searches = max(
        configured_int("deepsearch_tree_max_searches", 0),
        0,
    )
    research_budget = _research_budget_for(
        state,
        config,
        initial_tokens=estimate_tokens(user_input),
    )
    max_seconds = float(research_budget["max_seconds"])
    max_tokens = int(research_budget["max_tokens"])
    if research_budget["stop_reason"]:
        artifacts = _store_research_budget(
            state.get("deepsearch_artifacts"),
            research_budget,
            mode="tree",
        )
        artifacts.update({
            "phase": "tree_skipped",
            "mode": "tree",
        })
        return {
            "search_results": list(state.get("search_results") or []),
            "sources": list(state.get("sources") or []),
            "fetched_pages": list(state.get("fetched_pages") or []),
            "passages": list(state.get("passages") or []),
            "research_tree": state.get("research_tree") or {},
            "deepsearch_artifacts": artifacts,
            **budget_state_fields(research_budget),
            "quality_summary": dict(artifacts.get("quality_summary") or {}),
            "deepsearch_mode": "tree",
            "current_step": 2,
            "status": "running",
        }

    initial_elapsed_seconds = max(
        float(research_budget.get("elapsed_seconds", 0.0) or 0.0),
        0.0,
    )
    initial_tokens_used = max(
        int(research_budget.get("tokens_used", 0) or 0),
        0,
    )

    search_runs: list[dict[str, Any]] = []

    def tree_search(
        payload: dict[str, Any],
        config_payload: Mapping[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        query = str(payload.get("query", "")).strip()
        if not query:
            return []

        result = search_node(
            {
                "query": query,
                "search_mode": "deep",
                "domain": state.get("domain", "general"),
                "domain_config": state.get("domain_config", {}),
            }
        )
        errors = result.get("errors") or []
        if errors:
            raise RuntimeError("；".join(str(item) for item in errors))

        results = [
            dict(item)
            for item in (result.get("search_results") or [])
            if isinstance(item, Mapping)
        ]
        try:
            from agent.workflows.browser_visualizer import visualize_urls_from_results

            visualize_urls_from_results(
                state=state,
                config=config_payload or config,
                results=results,
                max_urls=1,
                reason="deepsearch:tree",
            )
        except Exception:
            logger.debug("树状研究浏览器预览启动失败", exc_info=True)
        search_runs.append(
            {
                "query": query,
                "results": results,
                "fetched_pages": list(result.get("fetched_pages") or []),
                "passages": list(result.get("passages") or []),
            }
        )
        return results

    def build_result(
        tree: ResearchTree | None,
        *,
        mode: str,
        error: str = "",
    ) -> Dict[str, Any]:
        findings = tree.get_all_findings() if tree is not None else []
        search_results: list[dict[str, Any]] = []
        sources: list[dict[str, str]] = []
        seen_result_keys: set[str] = set()
        seen_source_urls: set[str] = set()

        for finding in findings:
            if not isinstance(finding, Mapping):
                continue
            raw_result = finding.get("result")
            if not isinstance(raw_result, Mapping):
                continue
            item = dict(raw_result)
            query = str(finding.get("query", "")).strip()
            if query:
                item["query"] = query
            branch_id = str(finding.get("branch_id", "")).strip()
            if branch_id:
                item["branch_id"] = branch_id

            url = str(item.get("url", "")).strip()
            title = str(item.get("title", "")).strip()
            result_key = url.rstrip("/") or title or f"{query}:{len(search_results)}"
            if result_key in seen_result_keys:
                continue
            seen_result_keys.add(result_key)
            search_results.append(item)

            if url and url not in seen_source_urls:
                seen_source_urls.add(url)
                sources.append(
                    {
                        "title": title,
                        "url": url,
                        "provider": str(item.get("provider", "")),
                        "query": query,
                    }
                )

        research_plan: list[str] = []
        if tree is not None:
            for node in tree.nodes.values():
                for query in node.queries or ([node.query] if node.query else []):
                    normalized = str(query).strip()
                    if normalized and normalized not in research_plan:
                        research_plan.append(normalized)
        resolved_plan = (
            [str(item["topic"]).strip() for item in assigned_tasks]
            if assigned_tasks
            else research_plan or list(state.get("research_plan") or [])
        )
        query_history_delta = new_query_history_entries(
            state.get("deepsearch_query_history") or [],
            resolved_plan,
        )

        fetched_pages = [
            page
            for run in search_runs
            for page in run.get("fetched_pages", [])
            if isinstance(page, Mapping)
        ]
        passages = [
            passage
            for run in search_runs
            for passage in run.get("passages", [])
            if isinstance(passage, Mapping)
        ]

        tree_payload = tree.to_dict() if tree is not None else {}
        artifacts = dict(state.get("deepsearch_artifacts") or {})
        artifacts.update(
            {
                "phase": "tree_searched",
                "mode": mode,
                "result_count": len(search_results),
                "branch_count": len(tree.nodes) if tree is not None else 0,
                "research_tree": tree_payload,
            }
        )
        if error:
            artifacts["error"] = error

        return {
            "research_plan": resolved_plan,
            "deepsearch_query_history": query_history_delta,
            "search_results": search_results,
            "sources": sources,
            "fetched_pages": fetched_pages,
            "passages": passages,
            "research_tree": tree_payload,
            "current_branch_id": tree.root_id if tree is not None else None,
            "deepsearch_artifacts": artifacts,
            "deepsearch_mode": mode,
            "current_step": 2,
            "status": "running",
        }

    def linear_fallback(reason: str) -> Dict[str, Any]:
        """Use the same bounded linear path if tree models are unavailable."""
        fallback = deep_linear_search_node(state, config)
        artifacts = dict(fallback.get("deepsearch_artifacts") or {})
        artifacts.update({
            "phase": "linear_fallback",
            "mode": "linear",
            "fallback_reason": reason,
        })
        fallback.update({
            "deepsearch_query_history": new_query_history_entries(
                state.get("deepsearch_query_history") or [],
                fallback.get("research_plan") or [],
            ),
            "deepsearch_mode": "linear",
            "deepsearch_artifacts": artifacts,
        })
        return fallback

    try:
        planner_llm = ModelFactory.for_task(
            "planning",
            temperature=0.2,
            config=config,
        )
        researcher_llm = ModelFactory.for_task(
            "research",
            temperature=0.2,
            config=config,
        )
        writer_llm = ModelFactory.for_task(
            "writing",
            temperature=0.3,
            config=config,
        )
    except Exception as exc:
        print(f"[deep_tree_search_node] 树模式模型不可用，回退线性搜索：{exc}")
        return linear_fallback(str(exc))

    explorer = TreeExplorer(
        planner_llm=planner_llm,
        researcher_llm=researcher_llm,
        writer_llm=writer_llm,
        search_func=tree_search,
        config=config,
        max_depth=max_depth,
        max_branches=max_branches,
        queries_per_branch=queries_per_branch,
        event_callback=publish_tree_event,
        max_searches=max_searches,
        max_seconds=max_seconds,
        max_tokens=max_tokens,
        initial_elapsed_seconds=initial_elapsed_seconds,
        initial_tokens_used=initial_tokens_used,
        count_topic_tokens=False,
    )

    tree: ResearchTree | None = None
    tree_error = ""
    try:
        current_epoch = int(state.get("deepsearch_epoch", 1) or 1)
        previous_tree = state.get("research_tree")
        if assigned_tasks:
            if previous_tree and current_epoch > 1:
                tree = ResearchTree.from_dict(previous_tree)
                explorer.tree = tree
                tree.epoch = current_epoch
                tree.add_queries(
                    [str(item["topic"]) for item in assigned_tasks],
                    epoch=current_epoch,
                    max_branches=max_branches,
                )
            else:
                tree = ResearchTree.from_queries(
                    user_input,
                    [str(item["topic"]) for item in assigned_tasks],
                    epoch=current_epoch,
                    max_branches=max_branches,
                )
            explorer.tree = tree
            for node in tree.get_pending_nodes():
                if node.depth == 1 and node.topic == tree.topic and node.query:
                    node.topic = node.query
            task_by_topic = {
                str(item["topic"]).strip().casefold(): dict(item)
                for item in assigned_tasks
            }
            pending_nodes = [
                node
                for node in tree.get_pending_nodes()
                if node.topic.casefold() in task_by_topic
            ]
            for node in pending_nodes:
                node_task = task_by_topic.get(node.topic.casefold())
                if node_task:
                    node.agent_result = {"assigned_task": node_task}
            workers = min(max(parallel_branches, 1), len(pending_nodes))
            if workers > 1:
                with ThreadPoolExecutor(max_workers=workers) as executor:
                    futures = [
                        executor.submit(
                            explorer.explore_branch,
                            node,
                            {**dict(state), "active_research_task": task_by_topic.get(node.topic.casefold(), {})},
                            per_query_results,
                        )
                        for node in pending_nodes
                    ]
                    for future in futures:
                        future.result()
            else:
                for node in pending_nodes:
                    explorer.explore_branch(
                        node,
                        {**dict(state), "active_research_task": task_by_topic.get(node.topic.casefold(), {})},
                        per_query_results,
                    )
        elif (current_epoch > 1 or state.get("plan_reviewed")) and previous_tree:
            tree = ResearchTree.from_dict(previous_tree)
            explorer.tree = tree
            pending_nodes = tree.get_pending_nodes()
            if not pending_nodes:
                tree.add_queries(
                    list(state.get("research_plan") or []),
                    epoch=current_epoch,
                    max_branches=max_branches,
                )
                pending_nodes = tree.get_pending_nodes()
            for node in pending_nodes:
                if node.depth == 1 and node.topic == tree.topic and node.query:
                    node.topic = node.query
                explorer.explore_branch(
                    node,
                    dict(state),
                    per_query_results=per_query_results,
                )
        elif parallel_branches > 0:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                tree = asyncio.run(
                    explorer.run_async(user_input, dict(state), decompose_root=True)
                )
            else:
                with ThreadPoolExecutor(max_workers=1) as executor:
                    tree = executor.submit(
                        asyncio.run,
                        explorer.run_async(user_input, dict(state), decompose_root=True),
                    ).result()
        else:
            tree = explorer.run(user_input, dict(state), decompose_root=True)
    except TreeExplorationBudgetExceeded as exc:
        tree = tree or explorer.tree
        tree_error = getattr(exc, "reason", "") or str(exc)
    except Exception as exc:
        tree = tree or explorer.tree
        tree_error = str(exc)

    if tree is None:
        return linear_fallback(tree_error or "树探索没有生成结果")

    print(
        "[deep_tree_search_node]",
        f"branches={len(tree.nodes)}",
        f"results={len(explorer.get_all_findings())}",
        f"error={tree_error}",
    )
    payload = build_result(
        tree,
        mode="tree",
        error=tree_error,
    )
    merged_summary = explorer.get_final_summary()
    budget_stats = explorer.get_budget_stats()
    research_budget["tokens_used"] = max(
        int(research_budget.get("tokens_used", 0) or 0),
        int(budget_stats.get("tokens_used", 0) or 0),
    )
    research_budget["searches_used"] = (
        int(research_budget.get("searches_used", 0) or 0)
        + int(budget_stats.get("searches_used", 0) or 0)
    )
    if budget_stats.get("stop_reason") and not research_budget.get("stop_reason"):
        research_budget["stop_reason"] = budget_stats["stop_reason"]
    research_budget = set_budget_elapsed(
        research_budget,
        float(budget_stats.get("elapsed_seconds", 0.0) or 0.0),
    )
    payload["deepsearch_artifacts"] = _store_research_budget(
        payload["deepsearch_artifacts"],
        research_budget,
        mode="tree",
    )
    payload["deepsearch_artifacts"]["tree_budget"].update(budget_stats)
    payload["research_agent_results"] = [
        dict(node.agent_result)
        for node in tree.nodes.values()
        if getattr(node, "depth", 0) > 0 and getattr(node, "agent_result", None)
    ]
    budget_stats = research_budget
    payload.update(budget_state_fields(research_budget))
    quality_diagnostics = build_quality_diagnostics(
        user_input,
        payload["research_plan"],
        search_runs,
        min_known_results=getattr(
            settings,
            "deepsearch_freshness_warning_min_known",
            3,
        ),
        min_fresh_ratio=getattr(
            settings,
            "deepsearch_freshness_warning_min_ratio",
            0.4,
        ),
    )
    quality_summary = {
        "branch_count": len(tree.nodes),
        "result_count": len(payload["search_results"]),
        "source_count": len(payload["sources"]),
        "budget_stop_reason": budget_stats["stop_reason"],
        "tokens_used": budget_stats["tokens_used"],
        "elapsed_seconds": budget_stats["elapsed_seconds"],
        "searches_used": budget_stats["searches_used"],
        "max_seconds": budget_stats["max_seconds"],
        "max_tokens": budget_stats["max_tokens"],
        **quality_diagnostics,
    }
    payload["quality_summary"] = quality_summary
    payload["deepsearch_artifacts"].update(
        {
            "query_coverage": quality_diagnostics["query_coverage"],
            "freshness_summary": quality_diagnostics["freshness_summary"],
            "quality_summary": quality_summary,
        }
    )
    if merged_summary:
        payload["summary_notes"] = [
            *list(state.get("summary_notes") or []),
            merged_summary,
        ]
        payload["deepsearch_artifacts"]["tree_summary"] = merged_summary
    publish_tree_event(
        "research_tree_update",
        {
            "tree": payload["research_tree"],
            "quality": {
                "branch_count": len(tree.nodes),
                "result_count": len(payload["search_results"]),
                "source_count": len(payload["sources"]),
                "budget_stop_reason": budget_stats["stop_reason"],
                "tokens_used": budget_stats["tokens_used"],
                "elapsed_seconds": budget_stats["elapsed_seconds"],
                **quality_diagnostics,
            },
            "mode": "tree",
        },
    )
    publish_tree_event(
        "quality_update",
        {
            "stage": "tree_search",
            "branch_count": len(tree.nodes),
            "result_count": len(payload["search_results"]),
            "source_count": len(payload["sources"]),
            "budget_error": tree_error or budget_stats["stop_reason"],
            "budget_stop_reason": budget_stats["stop_reason"],
            "tokens_used": budget_stats["tokens_used"],
            "elapsed_seconds": budget_stats["elapsed_seconds"],
            "searches_used": budget_stats["searches_used"],
            "max_seconds": budget_stats["max_seconds"],
            "max_tokens": budget_stats["max_tokens"],
            **quality_diagnostics,
        },
    )
    return payload


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
        content = str(
            item.get("fetched_content")
            or item.get("content", "")
        ).strip()

        excerpt = content or snippet
        excerpt = excerpt[:2000]

        context_blocks.append(
            f"【来源 {index}】\n"
            f"标题：{title}\n"
            f"链接：{url}\n"
            f"内容：{excerpt}\n"
        )

    search_context = "\n\n".join(context_blocks)
    human_guidance = str(state.get("human_guidance", "") or "").strip()

    llm = ModelFactory.for_task(
        "writing",
        temperature=0.4,
        config=config,
    )

    messages = [
        SystemMessage(
            content=(
                f"{get_prompt_manager(config).get_prompt('web_writer')}"
                "引用来源时使用 [来源1]、[来源2] 这样的格式。"
                "仅根据提供的搜索材料作答；如果来源信息冲突，明确指出。"
                "回答直接解决用户问题，不描述内部工作流程。"
            )
        ),
        HumanMessage(
            content=_build_model_user_content(
                (
                    f"用户问题：\n{user_input}\n"
                    f"搜索结果：\n{search_context}\n"
                    f"人工补充要求：\n{human_guidance or '无'}\n"
                ),
                state.get("images"),
            )
        ),
    ]

    messages = _apply_agent_system_prompt(
        messages,
        config,
    )

    if isinstance(llm, BaseChatModel):
        answer = "".join(
            _extract_response_text(chunk)
            for chunk in llm.stream(
                messages,
                config=config,
            )
        ).strip()
    else:
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

    research_budget = _research_budget_for(state, config)
    budget_mode = str(
        state.get("deepsearch_mode")
        or (state.get("deepsearch_artifacts") or {}).get("mode", "")
    ).strip().lower()
    if budget_mode not in {"linear", "tree"}:
        budget_mode = None

    if not search_results:
        artifacts = _store_research_budget(
            state.get("deepsearch_artifacts"),
            research_budget,
            mode=budget_mode,
        )
        stop_reason = str(research_budget.get("stop_reason", ""))
        answer = "本轮没有获得可验证的搜索结果，因此无法生成有来源支持的研究结论。"
        if stop_reason:
            answer += f"研究已因预算限制提前停止（{stop_reason}）。"
        elif state.get("errors"):
            answer += "请检查搜索服务后重试。"

        quality_diagnostics = build_quality_diagnostics(
            user_input,
            list(state.get("research_plan") or []),
            [],
        )
        quality_summary = {
            **quality_diagnostics,
            "result_count": 0,
            "source_count": 0,
            "budget_stop_reason": stop_reason,
            "tokens_used": research_budget.get("tokens_used", 0),
            "elapsed_seconds": research_budget.get("elapsed_seconds", 0.0),
            "searches_used": research_budget.get("searches_used", 0),
            "max_seconds": research_budget.get("max_seconds", 0.0),
            "max_tokens": research_budget.get("max_tokens", 0),
        }
        artifacts.update({
            "phase": "synthesized",
            "result_count": 0,
            "quality_summary": quality_summary,
        })
        return {
            "draft_report": answer,
            "final_report": answer,
            "messages": [AIMessage(content=answer)],
            "quality_summary": quality_summary,
            "deepsearch_artifacts": artifacts,
            **budget_state_fields(research_budget),
            "current_step": 3,
            "is_complete": False,
            "status": "running",
        }


    aggregated = ResultAggregator(
        max_results_per_query=10,
    ).aggregate(
        search_results,
        original_query=user_input,
    )
    selected_results = aggregated.to_dicts(limit=20)
    if not selected_results:
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
    tree_summary = str(
        (state.get("deepsearch_artifacts") or {}).get("tree_summary", "")
    ).strip()
    human_guidance = str(state.get("human_guidance", "") or "").strip()
    if tree_summary:
        research_context = (
            "【树状研究合并摘要】\n"
            f"{tree_summary}\n\n"
            + research_context
        )

    if research_budget.get("stop_reason"):
        answer = _budget_limited_research_report(
            user_input,
            selected_results,
            str(research_budget["stop_reason"]),
        )
        writer_agent_result = {
            "task_id": f"writer-{state.get('thread_id') or 'session'}-{state.get('deepsearch_epoch', 1)}",
            "status": "skipped",
            "report": answer,
            "tools_used": [],
            "error": str(research_budget["stop_reason"]),
            "elapsed_seconds": 0.0,
        }
    else:
        llm = ModelFactory.for_task(
            "writing",
            temperature=0.3,
            config=config,
        )
        writer_task = WriterTask(
            task_id=f"writer-{state.get('thread_id') or 'session'}-{state.get('deepsearch_epoch', 1)}",
            question=user_input,
            evidence=selected_results,
            research_summary=tree_summary,
            guidance=human_guidance,
            images=[item for item in (state.get("images") or []) if isinstance(item, dict)],
        )

        def generate_report(task: WriterTask) -> str:
            nonlocal research_budget
            task_context_blocks: list[str] = []
            if task.research_summary:
                task_context_blocks.append(
                    f"【树状研究合并摘要】\n{task.research_summary}"
                )
            for index, item in enumerate(task.evidence, start=1):
                title = str(item.get("title", "")).strip()
                url = str(item.get("url", "")).strip()
                query = str(item.get("query", "")).strip()
                snippet = str(item.get("snippet", "")).strip()
                content = str(item.get("content", "")).strip()
                task_context_blocks.append(
                    f"【来源 {index}】\n"
                    f"所属查询：{query}\n"
                    f"标题：{title}\n"
                    f"链接：{url}\n"
                    f"内容：{(content or snippet)[:2200]}"
                )
            task_evidence_context = "\n\n".join(task_context_blocks)
            messages = [
                SystemMessage(
                    content=(
                        f"{get_prompt_manager(config).get_prompt('deep_writer')}"
                        "引用来源时使用 [来源1]、[来源2] 格式。"
                        "建议使用清晰的标题和小节。"
                        "直接输出完整报告，不要描述内部工作流程。"
                    )
                ),
                HumanMessage(
                    content=_build_model_user_content(
                        (
                            f"用户问题：\n{task.question}\n"
                            f"多来源研究材料：\n{task_evidence_context}\n"
                            f"人工补充要求：\n{task.guidance or '无'}\n"
                        ),
                        task.images,
                    )
                ),
            ]
            messages = _apply_agent_system_prompt(messages, config)
            response = llm.invoke(messages, config=config)
            research_budget = _record_research_model_usage(
                research_budget,
                messages,
                response,
            )
            return _extract_response_text(response)

        writer_result = WriterAgent(generate_report).run(writer_task)
        writer_agent_result = writer_result.model_dump(mode="json")
        if writer_result.status != "completed":
            raise RuntimeError(writer_result.error or "Writer Agent failed")
        answer = writer_result.report

    if not answer:
        raise ValueError("Deep Writer 模型返回了空回答")

    answer = _append_source_references(
        answer,
        selected_results,
        max_sources=20,
    )

    research_tree = ResearchTree.from_dict(
        state.get("research_tree")
    )

    research_tree.record_results(
        search_results
    )
    research_tree.finalize()

    tree_payload = research_tree.to_dict()

    compressed = ResearchCompressor().compress(
        user_input,
        [{"results": selected_results}],
        summary_notes=list(state.get("summary_notes") or []),
    )

    runs_by_query: dict[str, list[dict[str, Any]]] = {}
    for item in search_results:
        if not isinstance(item, dict):
            continue
        query = str(item.get("query", "")).strip()
        if query:
            runs_by_query.setdefault(query, []).append(item)
    quality_diagnostics = build_quality_diagnostics(
        user_input,
        list(state.get("research_plan") or []),
        [{"query": query, "results": items} for query, items in runs_by_query.items()],
        min_known_results=getattr(
            settings,
            "deepsearch_freshness_warning_min_known",
            3,
        ),
        min_fresh_ratio=getattr(
            settings,
            "deepsearch_freshness_warning_min_ratio",
            0.4,
        ),
    )
    quality_summary = dict(quality_diagnostics)
    quality_summary.update({
        "budget_stop_reason": research_budget.get("stop_reason", ""),
        "tokens_used": research_budget.get("tokens_used", 0),
        "elapsed_seconds": research_budget.get("elapsed_seconds", 0.0),
        "searches_used": research_budget.get("searches_used", 0),
        "max_seconds": research_budget.get("max_seconds", 0.0),
        "max_tokens": research_budget.get("max_tokens", 0),
    })
    existing_artifacts = dict(state.get("deepsearch_artifacts") or {})
    artifacts = {
        **existing_artifacts,
        "phase": "synthesized",
        "epoch": state.get("deepsearch_epoch", 1),
        "result_count": len(search_results),
        "context_result_count": len(selected_results),
        "research_tree": tree_payload,
        "fetched_pages": list(state.get("fetched_pages") or []),
        "passages": list(state.get("passages") or []),
        "query_coverage": quality_diagnostics["query_coverage"],
        "freshness_summary": quality_diagnostics["freshness_summary"],
        "quality_summary": quality_summary,
        "compressed_knowledge": compressed.to_dict(),
        "writer_agent": writer_agent_result,
    }
    artifacts = _store_research_budget(
        artifacts,
        research_budget,
        mode=budget_mode,
    )

    return {
        "draft_report": answer,
        "final_report": answer,
        "writer_agent_result": writer_agent_result,
        "messages": [
            AIMessage(content=answer)
        ],
        "research_tree": tree_payload,
        "deepsearch_artifacts": artifacts,
        **budget_state_fields(research_budget),
        "quality_summary": quality_summary,
        "compressed_knowledge": compressed.to_dict(),
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

    research_budget = _research_budget_for(state, config)
    budget_mode = str(
        state.get("deepsearch_mode")
        or (state.get("deepsearch_artifacts") or {}).get("mode", "")
    ).strip().lower()
    if budget_mode not in {"linear", "tree"}:
        budget_mode = None
    artifacts = _store_research_budget(
        state.get("deepsearch_artifacts"),
        research_budget,
        mode=budget_mode,
    )

    if state.get("route") == "deep" and research_budget.get("stop_reason"):
        quality_summary = dict(state.get("quality_summary") or {})
        quality_summary.update({
            **budget_state_fields(research_budget),
            "budget_stop_reason": research_budget["stop_reason"],
            "evaluation_skipped": True,
        })
        artifacts["quality_summary"] = quality_summary
        return {
            "evaluation": "研究预算已耗尽，跳过模型评估与自动修订；当前报告仅包含已收集证据。",
            "verdict": "budget_exhausted",
            "eval_dimensions": {},
            "missing_topics": [],
            "knowledge_gap_analysis": None,
            "quality_summary": quality_summary,
            "deepsearch_artifacts": artifacts,
            **budget_state_fields(research_budget),
            "current_step": 4,
            "status": "running",
        }

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
        config=config,
    )

    messages = [
        SystemMessage(
            content=(
                f"{get_prompt_manager(config).get_prompt('report_evaluator')}"
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
                '"missing_topics": [], '
                '"knowledge_gap_analysis": {'
                '"overall_coverage": 0.0, '
                '"confidence": 0.0, '
                '"gaps": [], '
                '"suggested_queries": [], '
                '"covered_aspects": [], '
                '"analysis": ""'
                "}"
                "}"
                "\n\n"
                "verdict 只能是 pass 或 revise。"
                "各项评分必须是 0 到 1 之间的数字。"
                "knowledge_gap_analysis 用于评估主题覆盖率、缺口优先级和补查查询；"
                "coverage 与 confidence 均为 0 到 1。"
                "请按主题相关性检查定义、背景、核心内容、应用案例、优缺点、"
                "相关比较、趋势以及专家观点或数据证据；不相关方面不强行补充。"
                "gaps 中 importance 只能是 high、medium 或 low。"
            )
        ),
        HumanMessage(
            content=_build_model_user_content(
                (
                    f"用户问题：\n{user_input}\n"
                    f"报告：\n{report}\n"
                    f"可用来源：\n{source_context}\n"
                ),
                state.get("images"),
            )
        ),
    ]

    messages = _apply_agent_system_prompt(
        messages,
        config,
    )

    try:
        response = llm.invoke(
            messages,
            config=config,
        )
        research_budget = _record_research_model_usage(
            research_budget,
            messages,
            response,
        )
        artifacts = _store_research_budget(
            artifacts,
            research_budget,
            mode=budget_mode,
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

        knowledge_gap_analysis = None
        current_deep_epoch = max(
            int(state.get("deepsearch_epoch", 1) or 1),
            1,
        )
        max_deep_epochs = max(
            int(
                state.get(
                    "deepsearch_max_epochs",
                    getattr(settings, "deepsearch_max_epochs", 3),
                )
                or 1
            ),
            1,
        )
        if (
            state.get("route") == "deep"
            and bool(getattr(settings, "deepsearch_use_gap_analysis", True))
            and current_deep_epoch < max_deep_epochs
        ):
            gap_value = data.get("knowledge_gap_analysis")
            query_history = state.get("deepsearch_query_history") or []
            if isinstance(query_history, (list, tuple)):
                executed_queries = "\n".join(
                    f"- {str(query).strip()}"
                    for query in query_history
                    if str(query).strip()
                )[:2000]
            else:
                executed_queries = ""
            gap_messages = [
                SystemMessage(
                    content=(
                        "你是研究知识缺口分析器。只返回 JSON 对象，字段为："
                        "overall_coverage (0 到 1)、confidence (0 到 1)、"
                        "gaps (包含 aspect、importance、reason)、suggested_queries、"
                        "covered_aspects、analysis。按用户问题判断主题覆盖，不要把"
                        "报告没有提及但与问题无关的方面列为缺口。"
                    )
                ),
                HumanMessage(
                    content=(
                        f"用户问题：\n{user_input[:2000]}\n\n"
                        f"已执行查询：\n{executed_queries or '暂无'}\n\n"
                        f"研究报告：\n{report[:12000]}\n\n"
                        f"可用来源：\n{source_context[:4000]}"
                    )
                ),
            ]
            gap_messages = _apply_agent_system_prompt(
                gap_messages,
                config,
            )
            research_budget, gap_call_allowed = _research_budget_allows_call(
                research_budget,
                gap_messages,
            )
            if gap_call_allowed:
                try:
                    gap_llm = ModelFactory.for_task(
                        "gap_analysis",
                        temperature=0.0,
                        config=config,
                    )
                    gap_response = gap_llm.invoke(
                        gap_messages,
                        config=config,
                    )
                    research_budget = _record_research_model_usage(
                        research_budget,
                        gap_messages,
                        gap_response,
                    )
                    artifacts = _store_research_budget(
                        artifacts,
                        research_budget,
                        mode=budget_mode,
                    )
                    parsed_gap = _parse_json_object(
                        _extract_response_text(gap_response)
                    )
                    gap_value = parsed_gap.get(
                        "knowledge_gap_analysis",
                        parsed_gap,
                    )
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "独立知识缺口分析失败，使用评估结果作为降级：%s",
                        exc,
                    )
            else:
                logging.getLogger(__name__).info(
                    "研究预算不足，跳过独立知识缺口分析：%s",
                    research_budget.get("stop_reason", "token_budget_exceeded"),
                )

            knowledge_gap_analysis = normalize_knowledge_gap_analysis(
                gap_value,
                coverage_threshold=float(
                    getattr(settings, "deepsearch_gap_coverage_threshold", 0.8)
                    or 0.8
                ),
            )
            if knowledge_gap_analysis is not None:
                high_priority_aspects = [
                    gap["aspect"]
                    for gap in knowledge_gap_analysis["gaps"]
                    if gap["importance"] == "high"
                ]
                missing_topics = list(dict.fromkeys([
                    *missing_topics,
                    *high_priority_aspects,
                ]))
                if not knowledge_gap_analysis["is_sufficient"]:
                    verdict = "revise"

        # 评估器不能同时声称通过并报告已知遗漏。
        if verdict == "pass" and missing_topics:
            verdict = "revise"

        quality = analyze_research_quality(
            report=report,
            sources=source_items,
            missing_topics=missing_topics,
            model_dimensions=dimensions,
            citation_threshold=float(
                getattr(
                    settings,
                    "deepsearch_citation_threshold",
                    0.6,
                )
                or 0.6
            ),
        )

        missing_topics = list(
            dict.fromkeys(
                [
                    *missing_topics,
                    *quality.missing_topics,
                ]
            )
        )

        if missing_topics:
            verdict = "revise"

        claim_payloads: list[dict[str, Any]] = []
        verifier_agent_result: dict[str, Any] = {
            "task_id": f"verifier-{state.get('thread_id') or 'session'}-{state.get('deepsearch_epoch', 1)}",
            "status": "empty",
            "claims": [],
            "verified_count": 0,
            "unsupported_count": 0,
            "contradicted_count": 0,
            "tools_used": [],
            "error": "",
            "elapsed_seconds": 0.0,
        }
        contradicted_count = 0
        unsupported_count = 0
        if state.get("route") == "deep":
            scraped_content = list(state.get("scraped_content") or [])
            search_results = list(state.get("search_results") or [])
            if not scraped_content and search_results:
                scraped_content = [{"results": search_results}]

            verifier = ClaimVerifier(
                min_overlap_tokens=int(
                    getattr(
                        settings,
                        "deepsearch_claim_verifier_min_overlap_tokens",
                        2,
                    )
                ),
                max_evidence_per_claim=int(
                    getattr(
                        settings,
                        "deepsearch_claim_verifier_max_evidence_per_claim",
                        3,
                    )
                ),
            )
            verifier_task = VerifierTask(
                task_id=f"verifier-{state.get('thread_id') or 'session'}-{state.get('deepsearch_epoch', 1)}",
                report=report,
                scraped_content=scraped_content,
                passages=(
                    list(state.get("passages") or [])
                    if bool(getattr(settings, "deepsearch_claim_verifier_use_passages", True))
                    else []
                ),
            )
            verifier_agent_result = VerifierAgent(verifier).run(verifier_task).model_dump(mode="json")
            claim_payloads = list(verifier_agent_result["claims"])
            contradicted_count = int(verifier_agent_result["contradicted_count"])
            unsupported_count = int(verifier_agent_result["unsupported_count"])
            if verifier_agent_result["status"] == "failed":
                verdict = "revise"
                missing_topics = list(dict.fromkeys([*missing_topics, "引用证据核查失败"] ))
                evaluation = f"{evaluation} Verifier Agent 执行失败，不能确认报告主张均有证据支持。".strip()

            max_contradicted = max(
                0,
                int(
                    getattr(
                        settings,
                        "claim_verifier_gate_max_contradicted",
                        0,
                    )
                ),
            )
            max_unsupported = max(
                0,
                int(
                    getattr(
                        settings,
                        "claim_verifier_gate_max_unsupported",
                        0,
                    )
                ),
            )
            if contradicted_count > max_contradicted or unsupported_count > max_unsupported:
                verdict = "revise"
                evaluation = (
                    f"{evaluation} Claim verifier 检测到"
                    f" {contradicted_count} 条矛盾主张和"
                    f" {unsupported_count} 条缺少证据的主张。"
                ).strip()

        if not evaluation:
            evaluation = "评估模型没有提供文字说明。"

        print(
            "[evaluator_node]",
            f"verdict={verdict}",
            f"missing_topics={missing_topics}",
        )

        enhanced_quality = None
        try:
            scraped_content = list(state.get("scraped_content") or [])
            if not scraped_content and state.get("search_results"):
                scraped_content = [{"results": list(state.get("search_results") or [])}]
            known_sources = [
                item.get("url")
                for item in sources
                if isinstance(item, dict) and item.get("url")
            ]
            enhanced_quality = quality_assessor.QualityAssessor(
                llm=llm,
                config=dict(config or {}),
            ).assess(
                report,
                scraped_content,
                known_sources,
            )
            dimensions.update({
                "claim_support": enhanced_quality.claim_support_score,
                "source_diversity": enhanced_quality.source_diversity_score,
                "contradiction_free": enhanced_quality.contradiction_free_score,
                "citation_accuracy": enhanced_quality.citation_accuracy_score,
                "citation_coverage": enhanced_quality.citation_coverage_score,
            })
            quality = enhanced_quality
            if enhanced_quality.overall_score < 0.5 and verdict == "pass":
                verdict = "revise"
            citation_gate = float(
                getattr(
                    settings,
                    "citation_gate_min_coverage",
                    getattr(settings, "deepsearch_citation_threshold", 0.6),
                )
                or 0.6
            )
            if verdict == "pass" and enhanced_quality.citation_coverage_score < citation_gate:
                verdict = "revise"
                evaluation = (
                    f"{evaluation} Citation gate: coverage "
                    f"{enhanced_quality.citation_coverage_score:.2f} "
                    f"below threshold {citation_gate:.2f}."
                ).strip()
            if enhanced_quality.recommendations:
                evaluation = (
                    f"{evaluation} Quality recommendations: "
                    f"{'; '.join(enhanced_quality.recommendations)}"
                ).strip()
        except Exception as exc:
            print(f"质量评估扩展跳过：{exc}")

        artifacts = _store_research_budget(
            artifacts,
            research_budget,
            mode=budget_mode,
        )
        if knowledge_gap_analysis is not None:
            artifacts["knowledge_gap_analysis"] = knowledge_gap_analysis
        artifacts["claims"] = claim_payloads
        artifacts["verifier_agent"] = verifier_agent_result
        artifacts["claim_verifier"] = {
            "contradicted_count": contradicted_count,
            "unsupported_count": unsupported_count,
            "verified_count": sum(
                item.get("status") == ClaimStatus.VERIFIED.value
                for item in claim_payloads
            ),
        }
        previous_quality = artifacts.get("quality_summary")
        quality_summary = {
            **(
                previous_quality
                if isinstance(previous_quality, dict)
                else {}
            ),
            **quality.to_dict(),
        }
        quality_summary.update({
            "budget_stop_reason": research_budget["stop_reason"],
            "tokens_used": research_budget["tokens_used"],
            "elapsed_seconds": research_budget["elapsed_seconds"],
            "searches_used": research_budget["searches_used"],
            "max_seconds": research_budget["max_seconds"],
            "max_tokens": research_budget["max_tokens"],
        })
        artifacts["quality_summary"] = quality_summary

        return {
            "evaluation": evaluation,
            "verdict": verdict,
            "eval_dimensions": dimensions,
            "missing_topics": missing_topics,
            "knowledge_gap_analysis": knowledge_gap_analysis,
            "quality_overall_score": (
                quality.overall_score
            ),
            "quality_gap_count": (
                quality.gap_count
                if hasattr(quality, "gap_count")
                else len(missing_topics)
                + len(getattr(quality, "missing_citations", []) or [])
                + len(getattr(quality, "contradictions", []) or [])
            ),
            "citation_coverage_score": (
                quality.citation_coverage
                if hasattr(quality, "citation_coverage")
                else quality.citation_coverage_score
            ),
            "quality_summary": quality_summary,
            "deepsearch_artifacts": artifacts,
            "verifier_agent_result": verifier_agent_result,
            **budget_state_fields(research_budget),
            "current_step": 4,
            "status": "running",
        }

    except Exception as exc:
        error_message = f"报告评估失败： {exc}"
        print(f"报告评估失败： {exc}")
        artifacts = _store_research_budget(
            artifacts,
            research_budget,
            mode=budget_mode,
        )
        return {
            "evaluation": (
                "自动评估失败，本轮暂不阻断报告。"
            ),
            "verdict": "pass",
            "eval_dimensions": {},
            "missing_topics": [],
            "knowledge_gap_analysis": None,
            "deepsearch_artifacts": artifacts,
            **budget_state_fields(research_budget),
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

    research_budget = _research_budget_for(state, config)
    budget_mode = str(
        state.get("deepsearch_mode")
        or (state.get("deepsearch_artifacts") or {}).get("mode", "")
    ).strip().lower()
    if budget_mode not in {"linear", "tree"}:
        budget_mode = None

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
        config=config,
    )

    messages = [
        SystemMessage(
            content=(
                f"{get_prompt_manager(config).get_prompt('report_reviser')}"
                "只能使用提供的搜索来源，不能编造事实。"
                "引用来源时使用 [来源1]、[来源2] 格式。"
                "请直接返回修订后的完整报告正文。"
                "不要输出修订说明。"
                "不要单独添加参考来源列表，程序会自动追加。"
            )
        ),
        HumanMessage(
            content=_build_model_user_content(
                (
                    f"用户问题：\n{user_input}\n\n"
                    f"原始报告：\n{report}\n\n"
                    f"评估指出的遗漏主题：\n"
                    f"{missing_topic_text or '无明确遗漏主题'}\n\n"
                    f"人工补充要求：\n"
                    f"{str(state.get('human_guidance', '') or '').strip() or '无'}\n\n"
                    f"搜索来源：\n{source_context}"
                ),
                state.get("images"),
            )
        ),
    ]

    messages = _apply_agent_system_prompt(
        messages,
        config,
    )

    response = llm.invoke(
        messages,
        config=config,
    )
    research_budget = _record_research_model_usage(
        research_budget,
        messages,
        response,
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
        "deepsearch_artifacts": _store_research_budget(
            state.get("deepsearch_artifacts"),
            research_budget,
            mode=budget_mode,
        ),
        **budget_state_fields(research_budget),
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
        config=config,
    )

    messages = [
        SystemMessage(
            content=(
                f"{get_prompt_manager(config).get_prompt('clarifier')}"
                "只输出一个问题，不要解释原因。"
            )
        ),
        HumanMessage(
            content=_build_model_user_content(
                f"用户原始提问：\n{user_input}",
                state.get("images"),
            )
        ),
    ]

    messages = _apply_agent_system_prompt(
        messages,
        config,
    )

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


def _profile_allows_rag_search(
        profile: AgentProfile | None,
) -> bool:
    if profile is None or profile.enabled_tools is None:
        return False

    rag_names = {"rag", "rag_search"}
    enabled = bool(profile.enabled_tools & rag_names)
    blocked = bool((profile.blocked_tools or frozenset()) & rag_names)
    return enabled and not blocked


def _apply_agent_system_prompt(
        messages: list[Any],
        config: RunnableConfig,
) -> list[Any]:
    """
    应用当前 Agent 的系统提示。
    """
    profile = _get_agent_profile(config)

    if profile is None:
        return messages

    system_prompt = profile.system_prompt.strip()

    if not system_prompt:
        return messages

    return [
        SystemMessage(
            content=(
                "Agent Profile 系统提示：\n"
                + system_prompt
            )
        ),
        *messages,
    ]




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
        thread_id = str(
            configurable.get(
                "thread_id",
                "default",
            )
            or "default"
        ).strip()

        return create_default_registry(
            thread_id=thread_id,
            include_browser=True,
            owner_id=(
                str(configurable.get("user_id") or "").strip()
                or None
            ),
            include_mcp=bool(configurable.get("mcp_enabled", False)),
            mcp_tools=(
                configurable.get("mcp_tools", [])
                if isinstance(configurable.get("mcp_tools", []), list)
                else []
            ),
        )

    if not isinstance(registry, ToolRegistry):
        raise TypeError(
            "configurable.tool_registry 必须是 ToolRegistry。"
        )

    return registry


def should_use_agent_fast_path(state: Mapping[str, Any]) -> bool:
    """Select short, single-question web checks for the reduced-call path."""
    user_input = str(state.get("input", "") or "").strip()
    if not user_input or state.get("images") or len(user_input) > 180:
        return False
    if any(
        cue in user_input.lower()
        for cue in ("详细分析", "深入研究", "研究报告", "全面比较", "comprehensive", "in depth")
    ):
        return False
    return any(
        cue in user_input.lower()
        for cue in (
            "查一下", "查证", "核实", "确认是否", "是真的吗", "最新版本",
            "当前价格", "现在的", "简单对比", "哪个更快", "verify", "fact-check",
            "latest version", "current price", "quick comparison",
        )
    )


def agent_fast_path_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str, Any]:
    """Answer a narrowly scoped web check with one search and one synthesis call."""
    user_input = str(state.get("input", "") or "").strip()
    fallback = {"agent_fast_path_fallback": True}
    if not should_use_agent_fast_path(state):
        return fallback
    base_call_count = max(int(state.get("tool_call_count", 0) or 0), 0)
    recorded_tool_call = None
    recorded_tool_events: list[dict[str, Any]] = []

    try:
        profile = _get_agent_profile(config)
        if profile is not None:
            if profile.enabled_tools is not None and "web_search" not in profile.enabled_tools:
                return fallback
            if profile.blocked_tools and "web_search" in profile.blocked_tools:
                return fallback

        enabled_tools = state.get("enabled_tools")
        if isinstance(enabled_tools, Mapping) and not enabled_tools.get("web_search", True):
            return fallback

        registry = _get_agent_tool_registry(config)
        search_tool = registry.get("web_search")
        if _tool_requires_approval(search_tool, profile):
            return fallback

        call_limit = (
            profile.tool_call_limit
            if profile is not None and profile.tool_call_limit is not None
            else int(state.get("tool_call_limit", getattr(settings, "tool_call_limit", 20)))
        )
        call_count = max(int(state.get("tool_call_count", 0) or 0), 0)
        if call_limit <= call_count:
            return fallback

        started_at = time.perf_counter()
        search_started_at = time.perf_counter()
        _emit_tool_stream_event(
            config,
            "tool_started",
            {"tool": "web_search", "attempt": 1, "call_count": call_count + 1},
        )
        try:
            tool_result = registry.invoke(
                "web_search",
                {"query": user_input, "max_results": 3},
            )
        except Exception as exc:
            search_duration_ms = (time.perf_counter() - search_started_at) * 1000
            error_result = {"output": "", "error": str(exc)}
            _emit_tool_stream_event(
                config,
                "tool_error",
                {
                    "tool": "web_search",
                    "result": error_result,
                    "success": False,
                    "attempt": 1,
                    "call_count": call_count + 1,
                    "duration_ms": search_duration_ms,
                },
            )
            return {
                **fallback,
                "tool_results": [{
                    "tool_name": "web_search",
                    "arguments": {"query": user_input, "max_results": 3},
                    "result": error_result,
                }],
                "tool_events": [_make_tool_event(
                    "tool_failed",
                    tool_name="web_search",
                    call_count=call_count + 1,
                    attempt=1,
                    error=str(exc),
                    duration_ms=search_duration_ms,
                )],
                "tool_call_count": call_count + 1,
            }
        search_duration_ms = (time.perf_counter() - search_started_at) * 1000
        if not isinstance(tool_result, Mapping):
            tool_result = {"output": str(tool_result), "error": ""}
        tool_result = dict(tool_result)
        result_error = str(tool_result.get("error", "") or "").strip()
        output = str(tool_result.get("output", "") or "").strip()
        results = tool_result.get("results")
        has_evidence = bool(output and isinstance(results, list) and results)
        tool_events = [
            _make_tool_event(
                "tool_failed" if result_error else "tool_completed",
                tool_name="web_search",
                call_count=call_count + 1,
                attempt=1,
                error=result_error,
                duration_ms=search_duration_ms,
            )
        ]
        tool_call = {
            "tool_name": "web_search",
            "arguments": {"query": user_input, "max_results": 3},
            "result": tool_result,
        }
        recorded_tool_call = tool_call
        recorded_tool_events = tool_events
        if not has_evidence or result_error:
            _emit_tool_stream_event(
                config,
                "tool_error" if result_error else "tool_result",
                {
                    "tool": "web_search",
                    "result": tool_result,
                    "success": not bool(result_error),
                    "attempt": 1,
                    "call_count": call_count + 1,
                    "duration_ms": search_duration_ms,
                },
            )
            return {
                **fallback,
                "tool_results": [tool_call],
                "tool_events": tool_events,
                "tool_call_count": call_count + 1,
            }
        _emit_tool_stream_event(
            config,
            "tool_result",
            {
                "tool": "web_search",
                "result": tool_result,
                "success": True,
                "attempt": 1,
                "call_count": call_count + 1,
                "duration_ms": search_duration_ms,
            },
        )

        messages = [
            SystemMessage(
                content=(
                    "你正在回答一个简短的联网核验问题。仅依据提供的搜索证据回答，"
                    "优先采用可靠且相互印证的来源；明确证据不足之处。"
                    "只回答用户问的范围，保持简洁，不要重复列出搜索摘要。"
                )
            ),
            HumanMessage(
                content=(
                    f"用户问题：\n{user_input}\n\n"
                    f"搜索证据：\n{output[:6000]}"
                )
            ),
        ]
        messages = _apply_agent_system_prompt(messages, config)
        generation_started_at = time.perf_counter()
        llm = ModelFactory.for_task(
            "writing",
            temperature=0.2,
            config=config,
        )
        answer_parts: list[str] = []
        first_token_duration_ms = None
        generation_first_token_ms = None
        stream_method = getattr(llm, "stream", None)
        if callable(stream_method):
            try:
                for chunk in stream_method(messages, config=config):
                    chunk_text = _extract_response_text(chunk)
                    if not chunk_text:
                        continue
                    if first_token_duration_ms is None:
                        first_token_duration_ms = round(
                            (time.perf_counter() - started_at) * 1000,
                            2,
                        )
                        generation_first_token_ms = round(
                            (time.perf_counter() - generation_started_at) * 1000,
                            2,
                        )
                    answer_parts.append(chunk_text)
            except Exception as exc:
                if answer_parts:
                    partial_answer = "".join(answer_parts).strip()
                    generation_duration_ms = (
                        time.perf_counter() - generation_started_at
                    ) * 1000
                    return {
                        "agent_fast_path_fallback": False,
                        "agent_fast_path": True,
                        "agent_performance": {
                            "search_duration_ms": round(search_duration_ms, 2),
                            "generation_duration_ms": round(generation_duration_ms, 2),
                            "first_token_duration_ms": first_token_duration_ms,
                            "generation_first_token_ms": generation_first_token_ms,
                            "node_duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
                        },
                        "tool_results": [tool_call],
                        "tool_events": tool_events,
                        "tool_call_count": call_count + 1,
                        "draft_report": partial_answer,
                        "final_report": partial_answer,
                        "messages": [AIMessage(content=partial_answer)],
                        "errors": [f"流式回答中断：{exc}"],
                        "last_error": f"流式回答中断：{exc}",
                        "status": "failed",
                        "is_complete": False,
                    }
                logger.info("Agent fast-path streaming unavailable; using invoke")
        if answer_parts:
            answer = "".join(answer_parts).strip()
        else:
            response = llm.invoke(messages, config=config)
            answer = _extract_response_text(response).strip()
        generation_duration_ms = (time.perf_counter() - generation_started_at) * 1000
        if not answer:
            return {
                **fallback,
                "tool_results": [tool_call],
                "tool_events": tool_events,
                "tool_call_count": call_count + 1,
            }

        _emit_tool_stream_event(
            config,
            "tool_progress",
            {
                "tool": "web_search",
                "status": "completed",
                "duration_ms": search_duration_ms,
                "fast_path": True,
            },
        )
        return {
            "agent_fast_path_fallback": False,
            "agent_finished": True,
            "agent_fast_path": True,
            "agent_performance": {
                "search_duration_ms": round(search_duration_ms, 2),
                "generation_duration_ms": round(generation_duration_ms, 2),
                "first_token_duration_ms": first_token_duration_ms,
                "generation_first_token_ms": generation_first_token_ms,
                "node_duration_ms": round((time.perf_counter() - started_at) * 1000, 2),
            },
            "tool_results": [tool_call],
            "tool_events": tool_events,
            "tool_call_count": call_count + 1,
            "draft_report": answer,
            "final_report": answer,
            "messages": [AIMessage(content=answer)],
            "status": "running",
            "is_complete": False,
        }
    except Exception as exc:
        logger.info("Agent fast path failed; using regular tool loop: %s", exc)
        if recorded_tool_call is not None:
            return {
                **fallback,
                "tool_results": [recorded_tool_call],
                "tool_events": recorded_tool_events,
                "tool_call_count": base_call_count + 1,
            }
        return fallback


def _emit_tool_stream_event(
        config: RunnableConfig,
        event_type: str,
        data: dict[str,Any],
) -> None:
    configurable = {}

    if config:
        configurable = config.get(
            "configurable",
            {},
        ) or {}

    thread_id = str(
        configurable.get("thread_id","")
    ).strip()

    if not thread_id:
        return

    try:
        emitter = get_emitter_sync(thread_id)

        emitter.emit_sync(
            event_type,
            data,
        )

    except Exception:
        pass



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




def _tool_definition_to_openai_schema(
    tool: ToolDefinition,
) -> dict[str, Any]:
    """Build a provider-compatible function schema from a registered tool."""
    if tool.parameters_schema is not None:
        parameters = dict(tool.parameters_schema)
        parameters.setdefault("type", "object")
        parameters.setdefault("properties", {})
        parameters.setdefault("required", [])
    else:
        properties: dict[str, Any] = {}
        required: list[str] = []
        json_types = {
            "str": "string",
            "string": "string",
            "int": "integer",
            "integer": "integer",
            "float": "number",
            "number": "number",
            "bool": "boolean",
            "boolean": "boolean",
            "list": "array",
            "array": "array",
            "dict": "object",
            "mapping": "object",
            "object": "object",
        }
        for parameter in tool.argument_schema():
            name = str(parameter["name"])
            definition: dict[str, Any] = {}
            annotation = str(parameter.get("annotation") or "").strip()
            normalized_annotation = annotation.lower().replace("typing.", "")
            json_type = json_types.get(normalized_annotation)
            if json_type:
                definition["type"] = json_type
            if parameter.get("description"):
                definition["description"] = str(parameter["description"])
            if "default" in parameter:
                definition["default"] = parameter["default"]
            properties[name] = definition
            if parameter.get("required"):
                required.append(name)

        parameters = {
            "type": "object",
            "properties": properties,
            "required": required,
            "additionalProperties": False,
        }

    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": parameters,
        },
    }


def _extract_native_tool_calls(response: Any) -> list[dict[str, Any]]:
    """Normalize LangChain and OpenAI-compatible tool-call response shapes."""
    raw_calls: Any = None
    additional_kwargs: Mapping[str, Any] = {}

    if isinstance(response, Mapping):
        raw_calls = response.get("tool_calls")
        additional = response.get("additional_kwargs")
        if isinstance(additional, Mapping):
            additional_kwargs = additional
        choices = response.get("choices")
        if not raw_calls and isinstance(choices, list) and choices:
            choice = choices[0]
            message = choice.get("message") if isinstance(choice, Mapping) else None
            if isinstance(message, Mapping):
                raw_calls = message.get("tool_calls")
    else:
        raw_calls = getattr(response, "tool_calls", None)
        additional = getattr(response, "additional_kwargs", None)
        if isinstance(additional, Mapping):
            additional_kwargs = additional

    if not raw_calls:
        raw_calls = additional_kwargs.get("tool_calls")
    if not raw_calls and isinstance(additional_kwargs.get("function_call"), Mapping):
        raw_calls = [{"function": additional_kwargs["function_call"]}]
    if not isinstance(raw_calls, (list, tuple)):
        return []

    normalized_calls: list[dict[str, Any]] = []
    for raw_call in raw_calls:
        if not isinstance(raw_call, Mapping):
            raise ValueError("模型原生工具调用必须是对象")
        function = raw_call.get("function")
        if not isinstance(function, Mapping):
            function = {}
        name = str(raw_call.get("name") or function.get("name") or "").strip()
        raw_arguments = raw_call.get(
            "args",
            raw_call.get("arguments", function.get("arguments", {})),
        )
        if isinstance(raw_arguments, str):
            raw_arguments = json.loads(raw_arguments) if raw_arguments.strip() else {}
        if not name:
            raise ValueError("模型原生工具调用缺少工具名称")
        if not isinstance(raw_arguments, Mapping):
            raise ValueError(f"工具「{name}」的原生调用参数必须是 JSON 对象")
        normalized_calls.append(
            {
                "tool_name": name,
                "arguments": dict(raw_arguments),
            }
        )

    return normalized_calls


def _select_agent_tools(
    llm: Any,
    tools: list[ToolDefinition],
    *,
    user_input: str,
    tool_history: str,
    config: RunnableConfig | None = None,
) -> list[ToolDefinition]:
    """Filter a large enabled toolset, falling back to the complete set."""
    if not bool(getattr(settings, "agent_tool_selector_enabled", True)):
        return tools

    try:
        max_tools = max(
            int(getattr(settings, "agent_tool_selector_max_tools", 3)),
            1,
        )
    except (TypeError, ValueError):
        max_tools = 3

    configured_always_include = getattr(
        settings,
        "agent_tool_selector_always_include_list",
        None,
    )
    if not isinstance(configured_always_include, (list, tuple, set)):
        configured_always_include = str(
            getattr(settings, "agent_tool_selector_always_include", "") or ""
        ).split(",")
    always_include_names = {
        str(name).strip()
        for name in configured_always_include
        if str(name).strip()
    }
    if bool(getattr(settings, "enable_todo_middleware", True)):
        always_include_names.update(_TODO_TOOL_NAMES)
    available_names = {tool.name for tool in tools}
    always_include_names &= available_names
    selectable_tools = [
        tool for tool in tools if tool.name not in always_include_names
    ]
    if len(selectable_tools) <= max_tools:
        return tools

    bind_structured_output = getattr(llm, "with_structured_output", None)
    if not callable(bind_structured_output):
        return tools

    tool_names = [tool.name for tool in selectable_tools]
    schema = {
        "type": "object",
        "title": "AgentToolSelection",
        "properties": {
            "tools": {
                "type": "array",
                "items": {"type": "string", "enum": tool_names},
            }
        },
        "required": ["tools"],
        "additionalProperties": False,
    }
    tool_catalog = [
        {"name": tool.name, "description": tool.description}
        for tool in selectable_tools
    ]
    selection_messages = [
        SystemMessage(
            content=(
                f"{get_prompt_manager(config).get_prompt('tool_selector')}"
                "按相关度排序。"
                f"最多选择 {max_tools} 个。只选择工具目录中的名称。"
                '仅输出 JSON 对象：{"tools":["工具名"]}。\n'
                f"工具目录：{json.dumps(tool_catalog, ensure_ascii=False)}"
            )
        ),
        HumanMessage(
            content=(
                f"用户请求：\n{user_input}\n\n"
                f"已执行工具及结果：\n{tool_history}"
            )
        ),
    ]

    base_url = str(getattr(settings, "deepseek_base_url", "") or "").lower()
    methods = (
        ("json_mode", "function_calling", "json_schema")
        if "deepseek" in base_url
        else ("json_schema", "function_calling", "json_mode")
    )
    last_error: Exception | None = None
    for method in methods:
        try:
            selection_model = bind_structured_output(schema, method=method)
            response = selection_model.invoke(selection_messages)
            if isinstance(response, Mapping):
                selection = response
            else:
                selection = _parse_json_object(_extract_response_text(response))

            selected_names = selection.get("tools")
            if not isinstance(selected_names, list):
                raise ValueError("工具选择响应缺少 tools 数组")

            selected = []
            for name in selected_names:
                if not isinstance(name, str) or name not in tool_names:
                    raise ValueError(f"工具选择响应包含无效工具：{name!r}")
                if name not in selected and len(selected) < max_tools:
                    selected.append(name)
            if not selected:
                raise ValueError("工具选择响应没有选出候选工具")

            selected_names_set = set(selected) | always_include_names
            return [tool for tool in tools if tool.name in selected_names_set]
        except Exception as exc:
            last_error = exc

    logger.warning(
        "Agent 工具筛选不可用，改用完整工具集：%s",
        last_error,
    )
    return tools


def _tool_requires_approval(
    tool: Any,
    profile: AgentProfile | None,
) -> bool:
    if getattr(tool, "name", "") in {
        "execute_command",
        "install_package",
    }:
        return True

    if (
        profile is not None
        and profile.require_tool_approval is not None
    ):
        return bool(profile.require_tool_approval)

    return bool(
        getattr(settings, "tool_approval", False)
        and getattr(tool, "requires_approval", False)
    )


def agent_plan_node(
        state: AgentState,
        config: RunnableConfig,
) -> Dict[str,Any]:
    """
    生成并校验工具调用计划。

    该节点只负责规划，不执行工具。
    """

    if state.get("is_cancelled"):
        message = "Agent 任务已取消。"

        return {
            "pending_tool_calls": [],
            "final_report": message,
            "is_complete": False,
            "status": "cancelled",
        }


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

    if any(tool.name == "rag_search" for tool in available_tools):
        enabled_tools["rag_search"] = _profile_allows_rag_search(profile)

    enabled_tool_definitions = [
        tool
        for tool in available_tools
        if enabled_tools.get(tool.name, True)
    ]
    todo_enabled = bool(getattr(settings, "enable_todo_middleware", True))
    custom_todo_description = str(
        getattr(settings, "todo_tool_description", "") or ""
    ).strip()
    if todo_enabled and custom_todo_description:
        enabled_tool_definitions = [
            replace(tool, description=custom_todo_description)
            if tool.name == "create_tasks"
            else tool
            for tool in enabled_tool_definitions
        ]

    prior_tool_results = state.get("tool_results") or []

    if prior_tool_results:
        context_tool_results = clear_tool_results_for_context(
            prior_tool_results[-5:]
        )
        tool_history = json.dumps(
            context_tool_results,
            ensure_ascii=False,
            default=str,
        )

    else:
        tool_history = "尚未执行任何工具"

    prior_messages = state.get("messages") or []
    if detect_stuck(prior_messages, threshold=1):
        hinted_messages = inject_stuck_hint(prior_messages)
        tool_history = f"{tool_history}\n\n{hinted_messages[-1].content}"


    xml_tool_calling_enabled = bool(
        getattr(settings, "agent_xml_tool_calling", False)
    )
    if xml_tool_calling_enabled:
        tool_call_instruction = (
            "如果需要调用一个或多个工具，可以使用模型原生工具调用，"
            "也可以返回 XML 格式的工具调用；"
            "每个 invoke 的 name 必须是工具名称，每个 parameter 的 name 必须是参数名。"
            "参数值使用文本，复杂对象使用 JSON 文本。"
            "不要使用 Markdown 代码围栏或在工具调用前后添加解释。"
            "格式示例："
            "<function_calls><invoke name=\"工具名\">"
            "<parameter name=\"参数名\">参数值</parameter>"
            "</invoke></function_calls>。"
            "多个调用可放在同一个 function_calls 中，按顺序执行。"
            "任务完成时返回 JSON finish 动作或直接给出最终回答。"
        )
    else:
        tool_call_instruction = (
            "优先使用模型原生工具调用接口。若不可用，调用工具时只能返回 JSON，"
            "不要输出 Markdown 代码块。格式："
            '{"action": "tool", "tool_name": "工具名称", '
            '"arguments": {"参数名": "参数值"}}。'
            "完成任务时返回 JSON finish 动作或直接给出最终回答："
            '{"action": "finish", "final_report": "最终回答"}。'
        )

    try:
        llm = ModelFactory.for_task(
            "planning",
            temperature=0.0,
            config=config,
        )
        planner_tools = _select_agent_tools(
            llm,
            enabled_tool_definitions,
            user_input=user_input,
            tool_history=tool_history,
            config=config,
        )
        planner_tool_names = {tool.name for tool in planner_tools}
        tool_descriptions = json.dumps(
            [tool.to_schema() for tool in planner_tools],
            ensure_ascii=False,
            default=str,
        )
        todo_system_prompt = str(
            getattr(settings, "todo_system_prompt", "") or ""
        ).strip()
        if todo_enabled and not todo_system_prompt:
            todo_system_prompt = _DEFAULT_TODO_SYSTEM_PROMPT
        todo_instruction = (
            f"\n\n任务清单规则：\n{todo_system_prompt}"
            if todo_enabled
            else ""
        )
        planner_messages = [
            SystemMessage(
                content=(
                    f"{get_prompt_manager(config).get_agent_planner_prompt()}\n\n"
                    "不要选择不在工具列表中的工具。"
                    f"\n当前可用工具（JSON Schema）：\n"
                    f"{tool_descriptions}\n"
                    f"{todo_instruction}\n"
                    f"\n{tool_call_instruction}"
                )
            ),
            HumanMessage(
                content=_build_model_user_content(
                    (
                        f"用户请求：\n{user_input}\n\n"
                        f"工具执行历史：\n{tool_history}"
                    ),
                    state.get("images"),
                )
            ),
        ]
        configurable = (
            config.get("configurable", {})
            if isinstance(config, Mapping)
            else {}
        )
        if not isinstance(configurable, Mapping):
            configurable = {}
        profile = _get_agent_profile(config)
        profile_browser_context = (
            profile.browser_context_helper
            if profile is not None
            else None
        )
        browser_context_enabled = (
            profile_browser_context
            if profile_browser_context is not None
            else configurable.get(
                "enable_browser_context_helper",
                getattr(settings, "enable_browser_context_helper", False),
            )
        )
        if bool(browser_context_enabled):
            thread_id = str(
                configurable.get("thread_id")
                or state.get("thread_id")
                or ""
            ).strip()
            if thread_id:
                try:
                    from agent.workflows.browser_context_helper import (
                        build_browser_context_hint,
                    )

                    browser_hint = build_browser_context_hint(thread_id)
                except Exception:
                    browser_hint = None
                    logger.debug("读取浏览器上下文失败", exc_info=True)
                if browser_hint:
                    planner_messages.insert(
                        1,
                        SystemMessage(content=browser_hint),
                    )
        planner_messages = _apply_agent_system_prompt(
            planner_messages,
            config,
        )
        native_tool_schemas = [
            _tool_definition_to_openai_schema(tool)
            for tool in planner_tools
        ]
        native_tool_calling_available = False
        bind_tools = getattr(llm, "bind_tools", None)
        if native_tool_schemas and callable(bind_tools):
            llm = bind_tools(native_tool_schemas)
            native_tool_calling_available = True

        response = llm.invoke(
            planner_messages,
            config=config,
        )

        response_text = _extract_response_text(response)
        native_calls = _extract_native_tool_calls(response)
        xml_calls = (
            XMLToolParser().parse_content(response_text)
            if xml_tool_calling_enabled and not native_calls
            else []
        )
        xml_tool_calls_detected = bool(xml_calls)

        if native_calls:
            action = "tool"
            proposed_calls = native_calls
        elif xml_calls:
            action = "tool"
            proposed_calls = [
                {
                    "tool_name": call.function_name,
                    "arguments": call.parameters,
                }
                for call in xml_calls
            ]
        else:
            try:
                data = _parse_json_object(response_text)
            except ValueError:
                if not native_tool_calling_available or not response_text.strip():
                    raise
                data = {
                    "action": "finish",
                    "final_report": response_text.strip(),
                }
            action = str(
                data.get("action", "tool")
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
                "xml_tool_calls_detected": False,
            }

        if action != "tool":
            raise ValueError(f"不支持的 Agent action: {action}")

        if not xml_calls and not native_calls:
            tool_name = str(
                data.get("tool_name", "")
            ).strip()
            arguments = data.get("arguments", {})
            if not tool_name:
                raise ValueError("工具规划结果缺少 tool_name")
            if not isinstance(arguments, dict):
                raise ValueError("工具规划结果 arguments 必须是 JSON 对象")
            proposed_calls = [
                {
                    "tool_name": tool_name,
                    "arguments": arguments,
                }
            ]

    except Exception as exc:
        message = f"工具调用规划失败： {exc}"

        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
            "xml_tool_calls_detected": False,
        }

    remaining_calls = tool_call_limit - tool_call_count
    if len(proposed_calls) > remaining_calls:
        message = (
            f"本批工具调用数量 {len(proposed_calls)} 超出剩余调用额度 "
            f"{remaining_calls}，已拒绝整批调用。"
        )
        return {
            "final_report": message,
            "errors": [message],
            "last_error": message,
            "is_complete": False,
            "status": "failed",
            "xml_tool_calls_detected": xml_tool_calls_detected,
        }

    pending_calls = []
    seen_signatures = set()
    for proposed_call in proposed_calls:
        tool_name = str(
            proposed_call.get("tool_name", "")
        ).strip()
        arguments = proposed_call.get("arguments", {})

        if not tool_name:
            message = "工具规划结果缺少 tool_name"
            return {
                "final_report": message,
                "errors": [message],
                "last_error": message,
                "is_complete": False,
                "status": "failed",
                "xml_tool_calls_detected": xml_tool_calls_detected,
            }
        if not isinstance(arguments, dict):
            message = f"工具「{tool_name}」arguments 必须是 JSON 对象"
            return {
                "final_report": message,
                "errors": [message],
                "last_error": message,
                "is_complete": False,
                "status": "failed",
                "xml_tool_calls_detected": xml_tool_calls_detected,
            }
        if not enabled_tools.get(tool_name, False):
            message = f"工具未启用：{tool_name}"
            return {
                "final_report": message,
                "errors": [message],
                "last_error": message,
                "is_complete": False,
                "status": "failed",
                "xml_tool_calls_detected": xml_tool_calls_detected,
            }
        if tool_name not in planner_tool_names:
            message = f"工具未进入本轮候选集：{tool_name}"
            return {
                "final_report": message,
                "errors": [message],
                "last_error": message,
                "is_complete": False,
                "status": "failed",
                "xml_tool_calls_detected": xml_tool_calls_detected,
            }

        try:
            tool = registry.get(tool_name)
        except (KeyError, TypeError, ValueError) as exc:
            message = str(exc)
            return {
                "final_report": message,
                "errors": [message],
                "last_error": message,
                "is_complete": False,
                "status": "failed",
                "xml_tool_calls_detected": xml_tool_calls_detected,
            }

        signature = _tool_call_signature(tool.name, arguments)
        if (
            profile is not None
            and profile.deduplicate_tool_calls is True
            and (
                signature in seen_signatures
                or _has_prior_tool_call(state, tool.name, arguments)
            )
        ):
            message = f"检测到重复工具调用：{tool.name}"
            return {
                "final_report": message,
                "errors": [message],
                "last_error": message,
                "is_complete": False,
                "status": "failed",
                "xml_tool_calls_detected": xml_tool_calls_detected,
            }

        try:
            tool.validate_arguments(arguments)
        except (TypeError, ValueError) as exc:
            message = str(exc)
            return {
                "final_report": message,
                "errors": [message],
                "last_error": message,
                "is_complete": False,
                "status": "failed",
                "xml_tool_calls_detected": xml_tool_calls_detected,
            }

        seen_signatures.add(signature)
        pending_calls.append(
            {
                "tool_name": tool.name,
                "arguments": arguments,
            }
        )

    approval_required = bool(
        pending_calls
        and not state.get("tool_approved", False)
        and _tool_requires_approval(
            registry.get(pending_calls[0]["tool_name"]),
            profile,
        )
    )


    return {
        "pending_tool_calls": pending_calls,
        "agent_tool_candidates": sorted(planner_tool_names),
        "tool_call_count": tool_call_count,
        "tool_call_limit": tool_call_limit,
        "agent_finished": False,
        "is_complete": False,
        "status": "paused" if approval_required else "running",
        "xml_tool_calls_detected": xml_tool_calls_detected,
    }




def _agent_tool_candidate_names(state: Mapping[str, Any]) -> set[str]:
    raw_candidates = state.get("agent_tool_candidates")
    if not isinstance(raw_candidates, (list, tuple, set, frozenset)):
        return set()

    return {
        str(candidate).strip()
        for candidate in raw_candidates
        if str(candidate).strip()
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

    if state.get("is_cancelled"):
        message = "工具调用已取消。"

        return {
            "pending_tool_calls": [],
            "final_report": message,
            "is_complete": False,
            "status": "cancelled",
        }



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
        tool_name == "rag_search"
        and not _profile_allows_rag_search(profile)
    ):
        message = f"工具未启用：{tool_name}"

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

    if (
        _tool_requires_approval(tool, profile)
        and not state.get("tool_approved", False)
    ):
        allowed_tool_names = sorted(_agent_tool_candidate_names(state))
        if not allowed_tool_names:
            # Older checkpoints do not contain the candidate list. Keep their
            # approval editable only for the tool already selected.
            allowed_tool_names = [tool_name]

        approval = interrupt(
            {
                "type": "tool_approval",
                "tool_name": tool_name,
                "arguments": arguments,
                "message": f"是否批准执行工具：{tool_name}?",
                "allowed_decisions": ["approve", "edit", "reject"],
                "allowed_tools": allowed_tool_names,
            }
        )

        decision_type = "approve" if approval is True or approval == "approve" else ""
        if isinstance(approval, Mapping):
            if approval.get("approved") is True:
                decision_type = (
                    "edit"
                    if "edited_action" in approval
                    else "approve"
                )
            else:
                decision_type = str(
                    approval.get("type") or approval.get("action") or ""
                ).strip().lower()

        if decision_type == "reject" or approval is False or approval == "reject":
            message = f"工具调用未获批准：{tool_name}"
            if isinstance(approval, Mapping) and approval.get("message"):
                message = str(approval["message"]).strip() or message

            return {
                "pending_tool_calls": [],
                "final_report": message,
                "is_complete": False,
                "status": "cancelled",
            }

        if decision_type == "edit":
            edited_action = (
                approval.get("edited_action")
                if isinstance(approval, Mapping)
                else None
            )
            if not isinstance(edited_action, Mapping):
                message = "编辑审批缺少 edited_action 对象。"
                return {
                    "pending_tool_calls": [],
                    "final_report": message,
                    "errors": [message],
                    "last_error": message,
                    "is_complete": False,
                    "status": "failed",
                }

            edited_tool_name = str(edited_action.get("name") or "").strip()
            edited_arguments = edited_action.get("args")
            if not edited_tool_name or not isinstance(edited_arguments, Mapping):
                message = "编辑后的工具调用必须包含工具名称和 JSON 对象参数。"
                return {
                    "pending_tool_calls": [],
                    "final_report": message,
                    "errors": [message],
                    "last_error": message,
                    "is_complete": False,
                    "status": "failed",
                }

            allowed_candidates = _agent_tool_candidate_names(state)
            if not allowed_candidates:
                allowed_candidates = {tool_name}
            if edited_tool_name not in allowed_candidates:
                message = f"编辑后的工具未进入本轮候选集：{edited_tool_name}"
                return {
                    "pending_tool_calls": [],
                    "final_report": message,
                    "errors": [message],
                    "last_error": message,
                    "is_complete": False,
                    "status": "failed",
                }

            try:
                edited_tool = registry.get(edited_tool_name)
                edited_tool.validate_arguments(dict(edited_arguments))
            except (KeyError, TypeError, ValueError) as exc:
                message = str(exc)
                return {
                    "pending_tool_calls": [],
                    "final_report": message,
                    "errors": [message],
                    "last_error": message,
                    "is_complete": False,
                    "status": "failed",
                }

            effective_enabled_tools = state.get("enabled_tools") or {}
            if isinstance(effective_enabled_tools, Mapping) and not effective_enabled_tools.get(
                edited_tool_name, True
            ):
                message = f"工具未启用：{edited_tool_name}"
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
                and profile.enabled_tools is not None
                and edited_tool_name not in profile.enabled_tools
            ):
                message = f"工具未启用：{edited_tool_name}"
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
                and edited_tool_name in profile.blocked_tools
            ):
                message = f"工具被 Agent Profile 禁止：{edited_tool_name}"
                return {
                    "pending_tool_calls": [],
                    "final_report": message,
                    "errors": [message],
                    "last_error": message,
                    "is_complete": False,
                    "status": "failed",
                }
            if edited_tool_name == "rag_search" and not _profile_allows_rag_search(profile):
                message = f"工具未启用：{edited_tool_name}"
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
                and _has_prior_tool_call(state, edited_tool_name, dict(edited_arguments))
            ):
                message = f"检测到重复工具调用：{edited_tool_name}"
                return {
                    "pending_tool_calls": [],
                    "final_report": message,
                    "errors": [message],
                    "last_error": message,
                    "is_complete": False,
                    "status": "failed",
                }

            tool_name = edited_tool_name
            arguments = dict(edited_arguments)
            tool = edited_tool
            pending_call = {
                **pending_call,
                "tool_name": tool_name,
                "arguments": arguments,
            }
        elif decision_type != "approve":
            message = "无法识别的工具审批决定。"
            return {
                "pending_tool_calls": [],
                "final_report": message,
                "errors": [message],
                "last_error": message,
                "is_complete": False,
                "status": "failed",
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

            _emit_tool_stream_event(
                config,
                "tool_result",
                {
                    "tool": tool.name,
                    "result": cached_result,
                    "success": True,
                    "cached": True,
                    "attempt": 1,
                    "call_count": tool_call_count + 1,
                    "duration_ms": 0.0,
                },
            )

            cache_event = _make_tool_event(
                "tool_cache_hit",
                tool_name=tool_name,
                call_count=tool_call_count + 1,
                attempt=1,
                duration_ms=0.0,
            )

            return {
                "pending_tool_calls": pending_calls[1:],
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

    if profile is not None and profile.tool_retry_limit is not None:
        configured_tool_retry_limit = profile.tool_retry_limit
    elif bool(getattr(settings, "tool_retry", True)):
        try:
            configured_tool_retry_limit = max(
                int(getattr(settings, "tool_retry_max_attempts", 3)) - 1,
                0,
            )
        except (TypeError, ValueError):
            configured_tool_retry_limit = 2
    else:
        configured_tool_retry_limit = 0

    retry_limit = max(
        int(configured_tool_retry_limit),
        0,
    )


    _emit_tool_stream_event(
        config,
        "tool_start",
        {
            "tool": tool_name,
            "args": arguments,
            "call_count": tool_call_count + 1,
            "attempt": 1,
        },
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

                _emit_tool_stream_event(
                    config,
                    "tool_error",
                    {
                        "tool": tool.name,
                        "error": error_text,
                        "attempt": attempt,
                        "call_count": tool_call_count + 1,
                    },
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

            _emit_tool_stream_event(
                config,
                "tool_progress",
                {
                    "tool": tool.name,
                    "status": "retrying",
                    "attempt": attempt,
                    "call_count": tool_call_count + 1,
                    "error": error_text,
                },
            )

            retry_count += 1
            attempt += 1
            try:
                initial_delay = max(
                    float(getattr(settings, "tool_retry_initial_delay", 1.0)),
                    0.0,
                )
                backoff_factor = max(
                    float(getattr(settings, "tool_retry_backoff", 1.5)),
                    0.0,
                )
                max_delay = max(
                    float(getattr(settings, "tool_retry_max_delay", 60.0)),
                    0.0,
                )
            except (TypeError, ValueError):
                initial_delay, backoff_factor, max_delay = 1.0, 1.5, 60.0
            retry_delay = min(
                initial_delay * max(backoff_factor, 1.0) ** (retry_count - 1),
                max_delay,
            )
            if bool(getattr(settings, "tool_retry_jitter", True)):
                retry_delay = random.uniform(
                    retry_delay * 0.75,
                    retry_delay * 1.25,
                )
            if retry_delay > 0:
                time.sleep(retry_delay)

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

    duration_ms = (
        time.perf_counter() - attempt_started_at
    ) * 1000

    if result_error:
        _emit_tool_stream_event(
            config,
            "tool_error",
            {
                "tool": tool.name,
                "error": result_error,
                "result": tool_result,
                "attempt": attempt,
                "call_count": tool_call_count + 1,
                "duration_ms": duration_ms,
            },
        )

    else:
        _emit_tool_stream_event(
            config,
            "tool_result",
            {
                "tool": tool.name,
                "result": tool_result,
                "success": True,
                "attempt": attempt,
                "call_count": tool_call_count + 1,
                "duration_ms": duration_ms,
            },
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
            "pending_tool_calls": pending_calls[1:],
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
        config=config,
    )

    user_content = _build_model_user_content(
        user_input,
        state.get("images"),
    )
    image_prompt = ""
    if isinstance(user_content, list) and any(
        isinstance(part, Mapping) and part.get("type") == "image_url"
        for part in user_content
    ):
        image_prompt = (
            f"\n\n{get_prompt_manager(config).get_prompt('multimodal_image_analyzer')}"
        )

    state_messages = list(state.get("messages") or [])
    if state_messages and getattr(state_messages[-1], "type", "") == "human":
        state_messages = state_messages[:-1]
    conversation_messages = [
        message
        for message in state_messages
        if getattr(message, "type", "") in {"human", "ai"}
    ]

    messages = [
        SystemMessage(
            content=(
                f"{get_prompt_manager(config).get_prompt('direct_answer')}"
                "当前节点不联网、不搜索、不调用外部工具。"
                f"{image_prompt}"
            )
        ),
        *conversation_messages,
        HumanMessage(content=user_content),
    ]

    messages = _apply_agent_system_prompt(
        messages,
        config,
    )

    generation_started_at = time.perf_counter()
    generation_first_token_ms = None
    answer_parts: list[str] = []
    stream_method = getattr(llm, "stream", None)
    if callable(stream_method):
        for chunk in stream_method(messages, config=config):
            chunk_text = _extract_response_text(chunk)
            if chunk_text:
                if generation_first_token_ms is None:
                    generation_first_token_ms = round(
                        (time.perf_counter() - generation_started_at) * 1000,
                        2,
                    )
                answer_parts.append(chunk_text)

    if answer_parts:
        answer = "".join(answer_parts).strip()
        generation_duration_ms = (
            time.perf_counter() - generation_started_at
        ) * 1000
    elif not callable(stream_method):
        response = llm.invoke(
            messages,
            config=config,
        )
        answer = _extract_response_text(response).strip()
        generation_duration_ms = (
            time.perf_counter() - generation_started_at
        ) * 1000
        generation_first_token_ms = round(generation_duration_ms, 2)
    else:
        raise ValueError("模型流式接口未返回文本")
    if not answer:
        raise ValueError("模型返回了空回答")

    return {
        "draft_report": answer,
        "final_report": answer,
        "messages": [
            AIMessage(content=answer)
        ],
        "agent_performance": {
            "generation_duration_ms": round(generation_duration_ms, 2),
            "generation_first_token_ms": generation_first_token_ms,
        },
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

    if current_status == "failed":
        return {
            "final_report": report,
            "is_complete": False,
            "status": "failed",
        }

    if (
        current_status == "cancelled"
        or state.get("is_cancelled")
    ):
        return {
            "final_report": report,
            "is_complete": False,
            "status": "cancelled",
        }

    deepsearch_artifacts = state.get("deepsearch_artifacts") or {}
    if state.get("route") == "deep" and not is_paused:
        deepsearch_artifacts = dict(deepsearch_artifacts)
        try:
            report, charts = finalize_report_visualizations(
                report,
                deepsearch_artifacts.get("compressed_knowledge")
            )
            deepsearch_artifacts["generated_artifacts"] = charts
        except Exception as exc:
            logger.warning(
                "[finalize] Report visualization skipped (%s)",
                type(exc).__name__,
            )
            deepsearch_artifacts.pop("generated_artifacts", None)

    result = {
        "final_report": report,
        "is_complete": not is_paused,
        "status": "paused" if is_paused else "completed",
    }
    if state.get("route") == "deep":
        result["deepsearch_artifacts"] = deepsearch_artifacts
    return result


def human_review_node(
    state: AgentState,
    config: RunnableConfig,
) -> Dict[str, Any]:
    """
    最终人工审核节点。

    human_review 未启用时直接放行。
    human_review 启用时通过 LangGraph interrupt 暂停，
    等待人工批准或修改报告。
    """

    configurable = {}

    if isinstance(config,Mapping):
        configurable = (
            config.get("configurable", {})
            or {}
        )

    review_required = _hitl_checkpoint_enabled(config, "final") or bool(
        configurable.get(
            "human_review",
            getattr(
                settings,
                "human_review",
                False,
            ),
        )
    )

    report = str(
        state.get("final_report")
        or state.get("draft_report")
        or ""
    ).strip()

    if not review_required:
        return {
            "final_report": report,
            "status": "running",
        }

    review_result = interrupt(
        {
            "type": "human_review",
            "checkpoint": "final",
            "instruction":(
                "请审核最终报告。"
                "可以直接批准，也可以返回修改后的内容。"
            ),
            "content": report,
            "available_actions": [
                "approve",
                "edit",
                "reject",
            ],
        }
    )

    if isinstance(review_result, str):
        decision = review_result.strip().lower()
        if decision in {"reject", "rejected", "deny"}:
            return {
                "final_report": "报告未通过人工审核。",
                "is_complete": False,
                "status": "cancelled",
            }
        if (
            decision not in {"approve", "approved", "accept"}
            and review_result.strip()
        ):
            report = review_result.strip()
        return {
            "final_report": report,
            "draft_report": report,
            "status": "running",
        }

    if review_result is False:
        return {
            "final_report": "报告未通过人工审核。",
            "is_complete": False,
            "status": "cancelled",
        }

    if review_result is True:
        return {
            "final_report": report,
            "draft_report": report,
            "status": "running",
        }

    if isinstance(review_result, Mapping):
        decision = str(
            review_result.get("type")
            or review_result.get("action")
            or ""
        ).strip().lower()
        if (
            review_result.get("approved") is False
            or decision in {"reject", "rejected", "deny"}
        ):
            message = review_result.get("message")
            return {
                "final_report": (
                    message.strip()
                    if isinstance(message, str) and message.strip()
                    else "报告未通过人工审核。"
                ),
                "is_complete": False,
                "status": "cancelled",
            }

        edited_content = review_result.get("content")

        if isinstance(
            edited_content, str
        ) and edited_content.strip():
            report = edited_content.strip()

        elif isinstance(review_result,str):
            if review_result.strip():
                report = review_result.strip()

        return {
            "final_report": report,
            "draft_report": report,
            "status": "running",
        }

    raise ValueError("人工审核结果必须是布尔值、文本或对象。")
