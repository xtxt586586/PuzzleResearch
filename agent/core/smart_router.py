from __future__ import annotations

import json
import re

from typing import Any,Dict,List,Literal,Optional

from langchain_core.messages import HumanMessage,SystemMessage
from langchain_core.runnables import RunnableConfig
from pydantic import Field,BaseModel

from agent.core.model_factory import ModelFactory
from agent.prompts import get_prompt_manager


RouteType = Literal[
    "direct",
    "agent",
    "web",
    "deep",
    "clarify",
]


class RouteDecision(BaseModel):
    """
    LLM 返回的结构化路由结果。
    """

    route: RouteType = Field(
        description=(
            "执行路径: "
            "direct=直接回答，"
            "agent=工具调用，"
            "web=快速网页搜索，"
            "deep=深度研究，"
            "clarify=需要澄清"
        )
    )

    reasoning: str = Field(
        description="简要说明为什么选择这条路径"
    )

    confidence: float = Field(
        default=0.8,
        ge=0.0,
        le=1.0,
        description="路由置信度，范围为 0 到 1",
    )

    suggested_queries: List[str] = Field(
        default_factory=list,
        description="适用于 web 或 deep 路径的搜索建议",
    )

    clarification_question: str = Field(
        default="",
        description="clarify 路径的需要询问用户的问题",
    )


ROUTE_SYSTEM_PROMPT = """
你是 PuzzleResearch 的任务路由器。

请根据用户问题选择最合适的执行路径。

路径说明：

1. direct
适合简单知识问答、解释概念、简单计算。
例如：
- 什么是 LangGraph？
- Python 中的列表是什么？
- 2 加 2 等于多少？

2. agent
适合需要调用工具或执行动作的任务。
例如：
- 执行一段 Python 代码
- 读取一个文件
- 操作浏览器
- 创建一个 Excel 文件

3. web
适合需要少量实时信息的任务。
例如：
- 今天的天气
- 最新新闻
- 当前某个产品的价格

4. deep
适合需要多来源、多角度和综合分析的复杂研究。
例如：
- 比较 PostgreSQL 和 MySQL 的适用场景
- 分析某个行业的市场趋势
- 研究某项技术的优缺点和发展历史

5. clarify
适合信息不足、含义不明确或无法判断目标的请求。
例如：
- 帮我处理一下
- 修改这个
- 做得更好一点

只根据用户真实需求选择路径。
不要因为问题中出现“分析”两个字就自动选择 deep。
请返回符合要求的结构化结果。
"""

ROUTER_JSON_FALLBACK_PROMPT = """
当前模型可能不支持严格的 response_format。
请直接返回一个 JSON 对象，不要使用 Markdown 代码块，不要添加解释文字。

JSON 必须包含以下字段：

{
  "route": "direct",
  "reasoning": "选择该路径的原因",
  "confidence": 0.8,
  "suggested_queries": [],
  "clarification_question": ""
}

route 只能是：
direct、agent、web、deep、clarify。

confidence 必须是 0 到 1 之间的数字。
"""


class SmartRouter:
    """
    使用 LLM 对用户请求进行分类。

    该类只负责：
    1. 接收用户问题；
    2. 调用路由模型；
    3. 返回结构化的 RouteDecision；
    4. 在模型调用失败时进行安全降级。
    """

    def __init__(
            self,
            *,
            fallback_route: RouteType = "direct",
    ) -> None:
        self.fallback_route = fallback_route

    @staticmethod
    def _is_transient_model_error(error: BaseException) -> bool:
        transient_error_names = {
            "APITimeoutError",
            "APIConnectionError",
            "RateLimitError",
            "InternalServerError",
            "TimeoutError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "ConnectError",
            "ReadError",
            "WriteError",
            "RemoteProtocolError",
        }
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if type(current).__name__ in transient_error_names:
                return True
            current = current.__cause__ or current.__context__
        return False

    @staticmethod
    def _explicit_decision(query: str) -> RouteDecision | None:
        """识别无歧义的用户意图，避免明显请求被模型误分类。"""

        normalized = query.lower()

        agent_markers = (
            "执行代码",
            "运行代码",
            "执行一段",
            "运行一段",
            "调用工具",
            "读取文件",
            "打开文件",
            "操作浏览器",
            "创建 excel",
            "创建文件",
        )
        if any(marker in normalized for marker in agent_markers) or re.search(
            r"(?:执行|运行).{0,20}(?:python|代码|脚本)",
            normalized,
        ):
            return RouteDecision(
                route="agent",
                reasoning="用户明确要求执行代码、调用工具或操作外部资源。",
                confidence=0.99,
            )

        deep_markers = (
            "深入研究",
            "深度研究",
            "深度分析",
            "多来源",
            "多源分析",
            "多角度分析",
            "全面分析",
            "系统分析",
            "优缺点",
        )
        compares_multiple_subjects = bool(
            re.search(
                r"(?:比较|对比).{0,40}(?:和|与|vs)"
                r"|(?:和|与).{0,40}(?:比较|对比)",
                normalized,
            )
        )
        if any(marker in normalized for marker in deep_markers) or compares_multiple_subjects:
            return RouteDecision(
                route="deep",
                reasoning="用户要求多角度或多来源研究，需要综合分析。",
                confidence=0.99,
            )

        search_is_negated = bool(
            re.search(r"(?:不要|无需|不需要|不用)\s*(?:联网|搜索|检索)", query)
        )
        web_markers = (
            "搜索",
            "联网",
            "查找",
            "检索",
            "最新",
            "实时",
            "今天",
            "当前",
            "新闻",
            "官方文档",
        )
        if not search_is_negated and any(
            marker in normalized for marker in web_markers
        ):
            return RouteDecision(
                route="web",
                reasoning="用户明确要求联网或获取实时、最新的网页信息。",
                confidence=0.99,
            )

        return None

    def route(
            self,
            query: str,
            *,
            config: Optional[RunnableConfig] = None,
    ) -> RouteDecision:
        """
        对用户输入进行路由判断。

        调用顺序：

        1. 尝试 LangChain 结构化输出；
        2. DeepSeek 不支持时，降级为普通文本 JSON；
        3. 两种方式都失败时，返回 fallback_route。
        """

        query = query.strip()

        if not query:
            return RouteDecision(
                route="clarify",
                reasoning="用户输入为空",
                confidence=1.0,
                clarification_question="请告诉我你希望完成什么任务？",
            )

        explicit_decision = self._explicit_decision(query)
        if explicit_decision is not None:
            return explicit_decision

        try:
            routing_prompt = get_prompt_manager(config).get_prompt("task_router")
            llm = ModelFactory.for_task(
                "routing",
                temperature=0.0,
                config=config,
            )

            messages = [
                SystemMessage(content=routing_prompt),
                HumanMessage(content=query),
            ]

            # 第一优先级：使用 LangChain 结构化输出。
            try:

                structured_llm = llm.with_structured_output(RouteDecision)

                if config is None:
                    response = structured_llm.invoke(messages)
                else:
                    response = structured_llm.invoke(
                        messages,
                        config=config,
                    )

                return self._normalize_decision(response)

            except Exception as structured_error:
                if self._is_transient_model_error(structured_error):
                    raise
                print(
                    "[SmartRoute] 结构化输出不可用，"
                    f"降级为普通 JSON: {structured_error}"
                )

            # 第二优先级：普通文本 JSON。
            fallback_messages = [
                SystemMessage(
                    content=(
                        routing_prompt
                        + "\n\n"
                        + ROUTER_JSON_FALLBACK_PROMPT
                    )
                ),
                HumanMessage(content=query),
            ]

            if config is None:
                fallback_response = llm.invoke(fallback_messages)
            else:
                fallback_response = llm.invoke(
                    fallback_messages,
                    config=config,
                )

            return self._parse_json_response(fallback_response)

        except Exception as exc:

            return RouteDecision(
                route=self.fallback_route,
                reasoning=f"路由模型调用失败：{exc}",
                confidence=0.0,
            )


    @staticmethod
    def _response_to_text(response: Any) -> str:
        """
        将模型返回结果转换成普通字符串。
        """

        content = getattr(response,"content",response)

        if isinstance(content,str):
            return content

        if isinstance(content,list):
            parts: list[str] = []

            for item in content:
                if isinstance(item,dict):
                    text = item.get("text","")
                    if text:
                        parts.append(str(text))
                else:
                    parts.append(str(item))

            return "".join(parts)

        return str(content)

    @classmethod
    def _parse_json_response(cls, response: Any) -> RouteDecision:
        """
        解析普通文本形式的 JSON 路由结果。

        兼容以下返回格式：

        1. 纯 JSON；
        2. ```json ... ```；
        3. JSON 前后带有少量解释文字。
        """

        content = cls._response_to_text(response).strip()

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

        return cls._normalize_decision(data)


    @staticmethod
    def _normalize_decision(response: Any) -> RouteDecision:
        """
        将模型返回结果统一转换为 RouteDecision。

        这样可以兼容：
        - Pydantic 对象；
        - 字典；
        - 某些兼容模型返回的普通对象。
        """

        if isinstance(response,RouteDecision):
            return response

        if isinstance(response,dict):
            return RouteDecision.model_validate(response)

        if hasattr(response,"model_dump"):
            return RouteDecision.model_validate(
                response.model_dump()
            )

        raise TypeError(
            f"无法识别的路由模型返回类型：{type(response).__name__}"
        )


def smart_route(
        query: str,
        *,
        config: Optional[RunnableConfig] = None,
) -> Dict[str,Any]:
    """
    提供一个简单的函数式调用入口。

    节点代码可以调用这个函数，
    不需要直接管理 SmartRouter 实例。
    """

    decision = SmartRouter().route(
        query,
        config=config,
    )

    return {
        "route": decision.route,
        "routing_reasoning": decision.reasoning,
        "routing_confidence": float(decision.confidence),
        "suggested_queries": decision.suggested_queries,
        "clarification_question": decision.clarification_question,
        "needs_clarification": decision.route == "clarify",
    }
