from __future__ import annotations

import operator
from typing import Annotated,Any,Dict,List,Literal,Optional,TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

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
    merged = add_messages(existing_messages,new_messages)

    # 当前配置类中还没有正式加入这些字段，
    # 因此使用 getattr 提供默认值，避免导入时出错。
    trim_enabled = bool(getattr(settings,"trim_messages",True))

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

    # 消息数量没有超过限制时，不需要裁剪。
    if len(merged) <= keep_first + keep_last:
        return merged

    # 如果配置为不保留任何消息，则返回空列表。
    if keep_last + keep_first == 0:
        return []

    head = merged[:keep_first] if keep_first else []
    tail = merged[-keep_last:] if keep_last else []

    return head + tail


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

    research_plan: List[str]
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

    # 搜索节点写入的统一搜索结果。
    # 使用 operator.add，便于未来并行搜索分支合并结果。
    search_results: Annotated[
        List[Dict[str,Any]],
        operator.add,
    ]

    code_results: Annotated[
        List[Dict[str,Any]],
        operator.add,
    ]

    summary_notes: List[str]
    sources: Annotated[
        List[Dict[str,str]],
        operator.add,
    ]

    # ========== 质量控制 ==========

    evaluation: str
    verdict: str
    eval_dimensions: Dict[str,float]
    missing_topics: List[str]

    revision_count: int
    max_revisions: int

    quality_overall_score: float
    quality_gap_count: int
    citation_coverage_score: float
    quality_summary: Dict[str,Any]
    claim_verifier_counts: Dict[str,Any]

    # ========== 工具控制 ==========

    tool_approved: bool
    pending_tool_calls: List[Dict[str,Any]]
    tool_call_count: int
    tool_call_limit: int
    enabled_tools: Dict[str,bool]

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