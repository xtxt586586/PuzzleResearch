from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    """
       PuzzleResearch 全局配置。

       配置优先级：
       1. 系统环境变量
       2. 项目根目录下的 .env
       3. 代码中的默认值
    """

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    # ========== 应用基础配置 ==========

    app_name: str = "PuzzleResearch"
    app_env: str = "dev"
    debug: bool = True
    host: str = "127.0.0.1"
    port: int = 8001

    # Optional API authentication for public HTTP endpoints.
    # An empty value keeps local development in anonymous mode.
    api_key: str = ""

    internal_api_key: str = Field(
        default="",
        description="WEAVER_INTERNAL_API_KEY",
    )

    # HTTP rate limiting is opt-in for local development and can be enabled
    # explicitly for tests or production deployments.
    rate_limit_enabled: bool | None = None
    rate_limit_general_per_minute: int = 60
    rate_limit_chat_per_minute: int = 20
    rate_limit_window_seconds: int = 60
    rate_limit_max_buckets: int = 10_000

    enable_mcp: bool = Field(
        default=False,
        description="启用全局 MCP 工具。",
    )
    mcp_servers: str = Field(
        default="{}",
        description="JSON 格式的 MCP server 配置。",
    )
    mcp_allowed_commands: str = Field(
        default="",
        description="允许启动的 MCP stdio 命令，以逗号分隔；默认全部拒绝。",
    )
    mcp_allowed_sse_hosts: str = Field(
        default="localhost,127.0.0.1,::1",
        description="允许连接的 MCP SSE 主机，以逗号分隔。",
    )
    mcp_allowed_cwd_roots: str = Field(
        default="",
        description="允许作为 MCP stdio cwd 的目录，以逗号分隔。",
    )

    auth_user_header: str = Field(
        default="X-Auth-User",
        validation_alias="WEAVE_AUTH_USER_HEADER",
        description="可信用户身份请求头。",
    )

    @property
    def rate_limit_enabled_effective(self) -> bool:
        configured = self.rate_limit_enabled
        if configured is not None:
            return bool(configured)
        return self.app_env.strip().lower() in {"prod", "production"}

    enable_tracing: bool = False
    trace_buffer_size: int = 1000
    enable_prometheus: bool = False

    # ========== DeepSeek 配置 ==========

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"

    primary_model: str = "deepseek-chat"
    reasoning_model: str = "deepseek-reasoner"

    planner_model: str = ""
    researcher_model: str = ""
    writer_model: str = ""
    evaluator_model: str = ""
    critic_model: str = ""

    model_timeout: int = 60

    # 可选的 XML 文本工具调用协议，关闭时仅接受默认 JSON 计划格式。
    agent_xml_tool_calling: bool = False

    # Agent 工具规划提示词风格，可通过环境变量 PROMPT_STYLE 设置。
    prompt_style: str = "enhanced"
    agent_planner_prompt_file: str = ""
    prompt_directory: str = ""

    # 工具数量较多时，先筛选与当前请求相关的候选工具。
    agent_tool_selector_enabled: bool = True
    agent_tool_selector_max_tools: int = 3
    agent_tool_selector_always_include: str = ""

    # 复杂 Agent 请求使用会话任务清单跟踪执行进度。
    enable_todo_middleware: bool = True
    todo_system_prompt: str = ""
    todo_tool_description: str = ""

    @property
    def agent_tool_selector_always_include_list(self) -> list[str]:
        return [
            name.strip()
            for name in self.agent_tool_selector_always_include.split(",")
            if name.strip()
        ]

    # ========== 消息上下文管理 ==========

    trim_messages: bool = True

    # 保留最前面的消息，通常用于保留系统消息
    trim_messages_keep_first: int = 1

    # 保留最后面的消息，通常用于保存最近对话
    trim_messages_keep_last: int = 8

    # 可选地从模型上下文中移除工具结果消息，默认保留完整工具历史。
    strip_tool_messages: bool = False

    # 工具结果上下文清理：超出阈值时用占位符替换较旧结果，并保留最近结果。
    context_edit_trigger_tokens: int = 1_000
    context_edit_keep_tools: int = 3

    # 当前阶段不启用中间消息摘要
    summary_messages: bool = False

    # 未来启用摘要时，达到该数量触发
    summary_messages_trigger: int = 20

    # 为空时使用 primary_model，避免摘要功能依赖额外的付费模型服务。
    summary_messages_model: str = ""
    summary_messages_word_limit: int = 200

    # ========== 搜索服务配置 ==========

    tavily_api_key: str = ""
    brave_api_key: str = ""
    serper_api_key: str = ""
    arxiv_enabled: bool = True
    semantic_scholar_api_key: str = ""
    pubmed_email: str = ""
    pubmed_api_key: str = ""
    hackernews_enabled: bool = True
    reddit_client_id: str = ""
    reddit_client_secret: str = ""
    reddit_user_agent: str = "PuzzleResearch/1.0"
    twitter_bearer_token: str = ""

    # 多个搜索引擎使用逗号分隔
    search_engines: str = "tavily,duckduckgo"
    search_cache_max_size: int = 200
    search_cache_ttl_seconds: float = 1800.0
    search_cache_similarity_threshold: float = 0.9
    search_strategy: str = "fallback"
    search_enable_freshness_ranking: bool = True
    search_freshness_half_life_days: float = 30.0
    search_freshness_weight: float = 0.35
    search_reliability_max_retries: int = 2
    search_reliability_retry_backoff_seconds: float = 0.5
    search_reliability_circuit_breaker_failures: int = 3
    search_reliability_circuit_breaker_reset_seconds: float = 60.0
    search_parallel_max_workers: int = 8
    search_parallel_timeout_seconds: float = 30.0
    domain_routing_enabled: bool = False

    # Web Research 配置
    web_max_queries: int = 3
    web_results_per_query: int = 5

    # ========== 数据库和缓存 ==========

    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"
    storage_fallback_to_memory: bool = True

    # ========== RAG 配置 ==========

    rag_enabled: bool = False
    rag_store_path: str = "./data/chroma"
    rag_collection_name: str = "puzzleresearch_documents"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    rag_embedding_provider: str = "local"
    rag_embedding_model: str = ""
    rag_embedding_api_key: str = ""
    rag_embedding_base_url: str = ""
    rag_embedding_dimensions: int = 0

    # ========== 支持对话长期记忆 ==========

    memory_store_backend: str = "memory"
    memory_store_url: str = ""
    memory_store_path: str = "./data/support_memory.json"
    memory_user_id: str = "default_user"
    memory_max_entries: int = 20
    enable_memory: bool = True
    mem0_api_key: str = ""

    # ========== 普通聊天与研究长期记忆 ==========

    enable_chat_memory: bool = True
    chat_memory_store_path: str = "./data/chat_memory.json"
    chat_memory_max_entries: int = 50

    # ========== 音频服务配置 ==========

    enable_browser_use: bool = False
    enable_browser_context_helper: bool = False
    enable_computer_use: bool = False

    dashscope_api_key: str = ""
    audio_max_bytes: int = 50 * 1024 * 1024

    # ========== 截图存储配置 ==========

    screenshot_store_path: str = "./data/screenshots"
    screenshot_retention_hours: int = 24
    screenshot_max_bytes: int = 10 * 1024 * 1024

    # ========== 触发器配置 ==========

    triggers_enabled: bool = True
    trigger_store_path: str = "./data/triggers.json"
    trigger_poll_seconds: float = 1.0
    trigger_history_limit: int = 100

    # ========== Agent 安全配置 ==========

    tool_approval: bool = False
    human_review: bool = False
    hitl_checkpoints: str = ""
    max_revisions: int = 2
    tool_call_limit: int = 20
    tool_retry: bool = True
    tool_retry_max_attempts: int = 3
    tool_retry_backoff: float = 1.5
    tool_retry_initial_delay: float = 1.0
    tool_retry_max_delay: float = 60.0
    tool_retry_jitter: bool = True

    # ========== Deep Research 配置 ==========

    use_hierarchical_agents: bool = True
    deepsearch_mode: str = "auto"
    deepsearch_max_epochs: int = 3
    deepsearch_query_num: int = 5
    deepsearch_results_per_query: int = 5
    deepsearch_use_gap_analysis: bool = True
    deepsearch_gap_coverage_threshold: float = 0.8
    deepsearch_enable_research_fetcher: bool = False
    deepsearch_max_seconds: float = 300.0
    deepsearch_max_tokens: int = 50_000
    deepsearch_tree_enabled: bool = True
    deepsearch_tree_max_depth: int = 2
    deepsearch_tree_max_branches: int = 4
    deepsearch_tree_queries_per_branch: int = 3
    deepsearch_tree_parallel_branches: int = 3
    deepsearch_tree_max_searches: int = 0
    deepsearch_visualize_browser: bool = True
    deepsearch_freshness_warning_min_known: int = 3
    deepsearch_freshness_warning_min_ratio: float = 0.4
    deepsearch_citation_threshold: float = 0.6
    citation_gate_min_coverage: float = 0.6
    claim_verifier_gate_max_contradicted: int = 0
    claim_verifier_gate_max_unsupported: int = 0
    deepsearch_claim_verifier_use_passages: bool = True
    deepsearch_claim_verifier_min_overlap_tokens: int = 2
    deepsearch_claim_verifier_max_evidence_per_claim: int = 3

    # ========== Research source fetching ==========

    reader_fallback_mode: str = "both"
    reader_public_base: str = "https://r.jina.ai"
    reader_self_hosted_base: str = ""
    research_fetch_timeout_s: float = 25.0
    research_fetch_max_bytes: int = 2_000_000
    research_fetch_concurrency: int = 6
    research_fetch_concurrency_per_domain: int = 2
    research_fetch_cache_ttl_s: float = 0.0
    research_fetch_cache_max_entries: int = 256
    research_fetch_cache_store_errors: bool = False
    research_fetch_render_mode: str = "off"
    research_fetch_render_min_chars: int = 200
    research_fetch_extract_markdown: bool = True

    @property
    def search_engines_list(self) -> list[str]:
        """
        将逗号分隔的搜索服务字符串转换为列表。

        例如：
        "tavily,duckduckgo"
        转换为：
        ["tavily", "duckduckgo"]
        """
        return [
            engine.strip().lower()
            for engine in self.search_engines.split(",")
            if engine.strip()
        ]

    @property
    def effective_planner_model(self) -> str:
        """
        获取实际使用的规划模型。

        如果没有单独配置 planner_model，
        则使用 reasoning_model。
        """
        return self.planner_model or self.reasoning_model

    @property
    def effective_researcher_model(self) -> str:
        """
        获取实际使用的研究模型。
        """
        return self.researcher_model or self.primary_model

    @property
    def effective_writer_model(self) -> str:
        """
        获取实际使用的写作模型。
        """
        return self.writer_model or self.primary_model

    @property
    def effective_evaluator_model(self) -> str:
        """
        获取实际使用的评估模型。
        """
        return self.evaluator_model or self.reasoning_model


@lru_cache
def get_settings() -> Settings:
    """
    创建全局配置对象。

    lru_cache 可以确保整个进程中复用同一个 Settings 实例，
    避免每次导入配置时重复读取环境变量和 .env 文件。
    """
    return Settings()


settings = get_settings()
