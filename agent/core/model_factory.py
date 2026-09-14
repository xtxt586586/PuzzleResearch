from __future__ import annotations

from typing import Literal

from langchain_openai import ChatOpenAI

from common.config import settings


TaskType = Literal[
    "routing",
    "planning",
    "research",
    "writing",
    "evaluation",
    "critic",
    "clarification"
]


class ModelFactory:
    """
    DeepSeek 模型工厂。

    统一负责：

    - 创建 ChatOpenAI 兼容客户端；
    - 注入 DeepSeek API Key；
    - 注入 DeepSeek Base URL；
    - 选择不同任务对应的模型；
    - 统一设置 temperature 和 timeout。
    """

    _PLACEHOLDER_KEYS = {
        "",
        "your-deepseek-api-key",
        "your_deepseek_api_key",
        "your_api_key",
    }

    @classmethod
    def _get_api_key(cls) -> str:
        """
        获取并验证 DeepSeek API Key。

        不能只判断字符串是否为空，
        因为 .env.example 中的占位符也是非空字符串。
        """
        api_key = settings.deepseek_api_key.strip()

        if api_key.lower() in cls._PLACEHOLDER_KEYS:
            raise ValueError(
                "DeepSeek API Key 未配置。"
                "请在项目根目录 .env 文件中填写真实的 DEEPSEEK_API_KEY。"
            )

        return api_key

    @classmethod
    def create(
            cls,
            model: str | None = None,
            *,
            temperature: float = 0.7,
            timeout: int | None = None,
    ) -> ChatOpenAI:
        """
        创建一个 DeepSeek ChatOpenAI 客户端。

        这里使用 ChatOpenAI，
        是因为 DeepSeek 提供 OpenAI 兼容接口。
        """
        api_key = cls._get_api_key()

        model_name = (
            model.strip()
            if isinstance(model, str) and model.strip()
            else settings.primary_model
        )

        request_timeout = timeout or settings.model_timeout

        return ChatOpenAI(
            model=model_name,
            api_key=api_key,
            base_url=settings.deepseek_base_url,
            temperature=temperature,
            timeout=request_timeout,
        )

    @classmethod
    def model_name_for_task(cls,task: TaskType) -> str:
        """
        根据任务类型选择模型。

        不同 Agent 角色可以使用不同模型，
        但它们仍然通过同一个模型工厂创建。
        """
        if task == "routing":
            return settings.reasoning_model

        if task == "planning":
            return settings.effective_planner_model

        if task == "research":
            return settings.effective_researcher_model

        if task == "writing":
            return settings.effective_writer_model

        if task == "evaluation":
            return settings.effective_evaluator_model

        if task == "critic":
            return settings.critic_model or settings.reasoning_model

        if task == "clarification":
            return settings.reasoning_model

        raise ValueError(f"不支持的任务类型：{task}")

    @classmethod
    def for_task(
            cls,
            task: TaskType,
            *,
            temperature: float | None = None,
            timeout: int | None = None,
    ) -> ChatOpenAI:
        """
        创建指定任务使用的模型。

        例如：

        ModelFactory.for_task("planning")
        ModelFactory.for_task("writing")
        ModelFactory.for_task("evaluation")
        """
        model_name = cls.model_name_for_task(task)

        if temperature is None:
            if task in {"routing","evaluation","critic"}:
                temperature = 0.0
            elif task == "planning":
                temperature = 0.8
            elif task == "research":
                temperature = 0.2
            else:
                temperature = 0.7

        return cls.create(
            model=model_name,
            temperature=temperature,
            timeout=timeout,
        )