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

    auth_user_header: str = Field(
        default="X-Auth-User",
        validation_alias="WEAVE_AUTH_USER_HEADER",
        description="可信用户身份请求头。",
    )

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

    # ========== 消息上下文管理 ==========

    trim_messages: bool = True

    # 保留最前面的消息，通常用于保留系统消息
    trim_messages_keep_first: int = 1

    # 保留最后面的消息，通常用于保存最近对话
    trim_messages_keep_last: int = 8

    # 当前阶段不启用中间消息摘要
    summary_messages: bool = False

    # 未来启用摘要时，达到该数量触发
    summary_messages_trigger: int = 20

    # ========== 搜索服务配置 ==========

    tavily_api_key: str = ""
    brave_api_key: str = ""
    serper_api_key: str = ""

    # 多个搜索引擎使用逗号分隔
    search_engines: str = "tavily,duckduckgo"

    # Web Research 配置
    web_max_queries: int = 3
    web_results_per_query: int = 5

    # ========== 数据库和缓存 ==========

    database_url: str = ""
    redis_url: str = "redis://localhost:6379/0"

    # ========== RAG 配置 ==========

    rag_enabled: bool = False
    rag_store_path: str = "./data/chroma"
    rag_collection_name: str = "puzzleresearch_documents"
    embedding_model: str = "BAAI/bge-small-zh-v1.5"

    # ========== Agent 安全配置 ==========

    tool_approval: bool = False
    human_review: bool = False
    max_revisions: int = 2
    tool_call_limit: int = 20

    # ========== Deep Research 配置 ==========

    deepsearch_mode: str = "auto"
    deepsearch_max_epochs: int = 3
    deepsearch_query_num: int = 5
    deepsearch_results_per_query: int = 5
    deepsearch_max_seconds: float = 0.0
    deepsearch_max_tokens: int = 0

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
