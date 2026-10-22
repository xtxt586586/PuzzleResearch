from __future__ import annotations

import operator
from typing import Annotated,Any,Dict,List,Literal,Optional,TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from agent.core.message_utils import is_conversation_summary, summarize_messages
from agent.core.middleware import maybe_strip_tool_messages
from common.config import settings


def capped_add_messages(
        existing: List[BaseMessage] | None,
        new: List[BaseMessage] | None,
) -> List[BaseMessage]:
    """
    合并消息，并限制消息历史长度。

    LangGraph 在遇到 Annotated[List[BaseMessage], reducer] 时，
    会使用这个函数合并旧状态和节点返回的新消息。

    保留策略：

    - 保留最前面的若干条消息；
    - 保留最后的若干条消息；
    - 删除中间过旧的消息；
    - 避免 Agent 长时间运行后上下文无限增长。
    """

    existing_messages = existing or []
    new_messages = new or []

    # 使用 LangGraph 官方的消息合并函数。
    # 它会正确处理消息 ID、消息覆盖和消息追加。
    merged = add_messages(existing_messages, new_messages)
    merged = maybe_strip_tool_messages(merged)

    trim_enabled = bool(getattr(settings, "trim_messages", True))

    if not trim_enabled:
        return merged

    keep_first = max(
        int(getattr(settings, "trim_messages_keep_first", 1)),
        0,
    )
    keep_last = max(
        int(getattr(settings, "trim_messages_keep_last", 8)),
        0,
    )

    summary_enabled = bool(getattr(settings, "summary_messages", False))
    summaries = [message for message in merged if is_conversation_summary(message)]
    history = [message for message in merged if not is_conversation_summary(message)]

    if summary_enabled:
        trigger = max(int(getattr(settings, "summary_messages_trigger", 20)), 1)
        if len(history) > trigger and len(history) > keep_first + keep_last:
            head = history[:keep_first] if keep_first else []
            tail = history[-keep_last:] if keep_last else []
            middle_end = len(history) - keep_last if keep_last else len(history)
            middle = history[keep_first:middle_end]
            summary = summarize_messages([*summaries, *middle])
            return [*head, summary, *tail]

        if summaries:
            insertion_index = min(keep_first, len(history))
            return [
                *history[:insertion_index],
                *summaries,
                *history[insertion_index:],
            ]

        # 摘要尚未达到触发阈值前先保留历史，避免过早丢弃可摘要内容。
        return history

    if len(merged) <= keep_first + keep_last:
        return merged

    if keep_first + keep_last == 0:
        return []

    head = merged[:keep_first] if keep_first else []
    tail = merged[-keep_last:] if keep_last else []
    return head + tail


def merge_unique_items(
    existing: List[Dict[str, Any]] | None,
    new: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
    """合并并按 URL 去重并行节点产生的结果。"""

    merged: List[Dict[str, Any]] = []
    seen_urls: set[str] = set()

    for item in (existing or []) + (new or []):
        url = str(item.get("url", "")).strip()

        if url:
            key = url.rstrip("/")
            if key in seen_urls:
                continue
            seen_urls.add(key)

        merged.append(item)

    return merged


ExecutionStatus = Literal[
    "pending",
    "running",
    "paused",
    "completed",
    "failed",
    "cancelled",
]


class AgentState(TypedDict,total=False):
    """
    PuzzleResearch 的主 Agent 状态。

    total=False 的含义是：
    图中的节点可以只返回部分字段，
    不需要每次都重新返回完整状态。
    """

    # ========== 输入和输出 ==========

    input: str
    images: List[Dict[str,Any]]

    draft_report: str
    final_report: str

    # ========== 用户和会话 ==========

    user_id: str
    thread_id: str
    agent_id: str

    # ========== 执行控制 ==========

    messages: Annotated[
        List[BaseMessage],
        capped_add_messages,
    ]

    # 完整 UI 对话归档，不参与模型上下文裁剪。
    conversation_history: List[Dict[str, str]]

    research_plan: List[str]
    research_tasks: List[Dict[str, Any]]
    research_agent_results: List[Dict[str, Any]]
    plan_reviewed: bool
    human_guidance: str
    current_step: int
    status: ExecutionStatus
    is_complete: bool

    started_at: str
    ended_at: str

    # ========== 路由信息 ==========

    route: str
    routing_reasoning: str
    routing_confidence: float
    suggested_queries: List[str]

    # ========== 澄清信息 ==========

    needs_clarification: bool
    clarification_question: str

    # ========== 研究数据 ==========

    scraped_content: Annotated[
        List[Dict[str,Any]],
        operator.add,
    ]

    fetched_pages: Annotated[
        List[Dict[str,Any]],
        operator.add,
    ]

    passages: Annotated[
        List[Dict[str,Any]],
        operator.add,
    ]

    # 搜索节点写入的统一搜索结果。
    # 使用自定义 reducer，便于并行搜索分支合并并按 URL 去重。
    search_results: Annotated[
        List[Dict[str,Any]],
        merge_unique_items,
    ]

    code_results: Annotated[
        List[Dict[str,Any]],
        operator.add,
    ]

    summary_notes: List[str]
    sources: Annotated[
        List[Dict[str,str]],
        merge_unique_items,
    ]

    # ========== 质量控制 ==========

    evaluation: str
    verdict: str
    eval_dimensions: Dict[str,float]
    missing_topics: List[str]
    knowledge_gap_analysis: Dict[str,Any]

    revision_count: int
    max_revisions: int

    quality_overall_score: float
    quality_gap_count: int
    citation_coverage_score: float
    quality_summary: Dict[str,Any]
    claim_verifier_counts: Dict[str,Any]

    # ========== 工具控制 ==========

    tool_approved: bool
    agent_finished: bool
    pending_tool_calls: List[Dict[str,Any]]
    agent_tool_candidates: List[str]
    tool_call_count: int
    tool_call_limit: int
    enabled_tools: Dict[str,bool]

    tool_results: Annotated[
        List[Dict[str,Any]],
        operator.add,
    ]

    tool_events: Annotated[
        List[Dict[str,Any]],
        operator.add,
    ]

    # ========== 取消和错误 ==========

    cancel_token_id: Optional[str]
    is_cancelled: bool

    errors: Annotated[
        List[str],
        operator.add,
    ]

    last_error: str

    # ========== 树状研究 ==========

    research_tree: Dict[str,Any]
    current_branch_id: Optional[str]
    tree_exploration_enabled: bool

    # ========== 层级 Agent ==========

    coordinator_action: str
    coordinator_reasoning: str

    # ========== 压缩知识 ==========

    compressed_knowledge: Dict[str,Any]

    # ========== 领域路由 ==========
    domain: str
    domain_config: Dict[str,Any]

    # ========== 子 Agent 上下文 ==========

    sub_agent_contexts: Dict[str,Dict[str,Any]]

    # ========== Token 统计 ==========

    total_input_tokens: int
    total_output_tokens: int

    # ========== Deep Research 扩展字段 ==========

    deepsearch_artifacts: Dict[str,Any]
    deepsearch_mode: str
    budget_stop_reason: str

    #当前是第几轮 Deep Research。
    deepsearch_epoch: int

    #最多允许执行多少条研究。
    deepsearch_max_epochs: int

    #已经使用过的查询，用于避免后续轮次重复搜索。
    deepsearch_query_history: Annotated[
        List[str],
        operator.add,
    ]

    deepsearch_tokens_used: int
    deepsearch_elapsed_seconds: float

    # ========== Agent 扩展字段 ==========

    continuation_needed: bool
    xml_tool_calls_detected: bool
    agent_fast_path: bool
    agent_fast_path_fallback: bool
    agent_performance: Dict[str, Any]


class ResearchPlan(TypedDict):
    """
    研究计划的数据结构。
    """

    queries: List[str]
    reasoning: str


class SearchResult(TypedDict):
    """
    单次搜索任务的数据结构。
    """

    query: str
    results: List[Dict[str,Any]]
    timestamp: str


class CodeExecution(TypedDict):
    """
    Python 代码执行结果的数据结构。
    """

    code: str
    output: str
    error: Optional[str]
    image: Optional[str]


class QueryState(TypedDict):
    """
    单个并行搜索分支使用的状态。
    """

    query: str
    search_mode: Literal["web", "deep"]
    domain: str
    domain_config: Dict[str, Any]
