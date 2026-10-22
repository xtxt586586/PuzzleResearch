from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Any, Callable, Literal

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field


class PlannerTask(BaseModel):
    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    task_id: str = Field(min_length=1, max_length=128)
    goal: str = Field(min_length=1, max_length=4000)
    mode: Literal["web", "deep", "followup"]
    instructions: str = Field(min_length=1, max_length=12000)
    context: Any = ""
    messages: Any = None
    max_queries: int = Field(default=5, ge=1, le=100)


class PlannedQuery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=500)
    priority: int = Field(ge=1, le=20)


class PlannerAgentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: str
    status: Literal["completed", "failed"]
    queries: list[PlannedQuery] = Field(default_factory=list)
    reasoning: str = ""
    error: str = ""
    model_tokens: int = Field(default=0, ge=0)


class PlannerAgent:
    """Creates a bounded query plan; this role has no tool capability."""

    allowed_tools: frozenset[str] = frozenset()

    def __init__(self, model_factory: Callable[..., Any]) -> None:
        self._model_factory = model_factory

    def run(
        self,
        task: PlannerTask,
        model: Any = None,
        config: Any = None,
    ) -> PlannerAgentResult:
        messages = task.messages or [
            SystemMessage(
                content=(
                    f"{task.instructions}\n最多生成 {task.max_queries} 个查询。"
                    "只返回 JSON 对象，不要输出 Markdown。格式："
                    '{"queries": ["查询一", "查询二"], "reasoning": "规划原因"}'
                )
            ),
            HumanMessage(content=task.context or task.goal),
        ]
        try:
            if model is None:
                model = self._model_factory(
                    "planning",
                    temperature=0.2,
                    config=config,
                )
            response = model.invoke(messages, config=config)
            payload = self._parse_payload(response)
            raw_queries = payload.get("queries", [])
            if isinstance(raw_queries, str):
                raw_queries = [raw_queries]
            if not isinstance(raw_queries, list):
                raw_queries = []

            queries: list[PlannedQuery] = []
            seen: set[str] = set()
            for item in raw_queries:
                query = " ".join(str(item or "").strip().split())[:500]
                key = query.casefold()
                if query and key not in seen:
                    seen.add(key)
                    queries.append(PlannedQuery(query=query, priority=max(1, 20 - len(queries))))
                if len(queries) >= task.max_queries:
                    break

            if not queries:
                raise ValueError("规划模型没有返回有效查询。")
            return PlannerAgentResult(
                task_id=task.task_id,
                status="completed",
                queries=queries,
                reasoning=str(payload.get("reasoning", "") or "").strip()[:1000],
                model_tokens=self._token_usage(messages, response),
            )
        except Exception as exc:
            return PlannerAgentResult(
                task_id=task.task_id,
                status="failed",
                error=str(exc)[:1000],
            )

    @staticmethod
    def _parse_payload(response: Any) -> dict[str, Any]:
        content = getattr(response, "content", response)
        if isinstance(content, list):
            content = "\n".join(
                str(item.get("text", ""))
                for item in content
                if isinstance(item, Mapping)
            )
        text = str(content or "").strip()
        fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.I | re.S)
        if fenced:
            text = fenced.group(1)
        else:
            start, end = text.find("{"), text.rfind("}")
            if start < 0 or end < start:
                raise ValueError("规划模型返回中没有有效 JSON。")
            text = text[start:end + 1]
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ValueError("规划模型返回的 JSON 必须是对象。")
        return payload

    @staticmethod
    def _token_usage(messages: list[Any], response: Any) -> int:
        for usage in (
            getattr(response, "usage_metadata", None),
            getattr(response, "response_metadata", {}).get("token_usage")
            if isinstance(getattr(response, "response_metadata", {}), Mapping)
            else None,
        ):
            if isinstance(usage, Mapping):
                try:
                    total = usage.get("total_tokens")
                    if total is not None:
                        return max(0, int(total))
                    return max(
                        0,
                        int(usage.get("input_tokens", usage.get("prompt_tokens", 0)) or 0)
                        + int(usage.get("output_tokens", usage.get("completion_tokens", 0)) or 0),
                    )
                except (TypeError, ValueError, OverflowError):
                    pass
        text = "".join(str(getattr(message, "content", "")) for message in messages)
        text += str(getattr(response, "content", response) or "")
        return (len(text) + 3) // 4
