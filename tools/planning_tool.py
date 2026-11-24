from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.core.model_factory import ModelFactory
from agent.prompts import get_prompt_manager
from tools.registry import ToolDefinition


MAX_PLAN_GOAL_CHARS = 4_000
MAX_PLAN_STEPS = 10


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            item.get("text", "")
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ]
        return "\n".join(parts).strip()
    return str(content or "").strip()


def _parse_steps(text: str, max_steps: int) -> list[str]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate, flags=re.I)

    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        payload = None

    raw_steps = payload.get("steps") if isinstance(payload, dict) else payload
    if isinstance(raw_steps, list):
        steps = [item.strip() for item in raw_steps if isinstance(item, str) and item.strip()]
    else:
        steps = []
        for line in candidate.splitlines():
            normalized = re.sub(r"^\s*(?:\d+[.)、]|[-*•])\s*", "", line).strip()
            if normalized:
                steps.append(normalized)

    unique_steps: list[str] = []
    seen: set[str] = set()
    for step in steps:
        key = step.casefold()
        if key not in seen:
            unique_steps.append(step)
            seen.add(key)
        if len(unique_steps) >= max_steps:
            break
    return unique_steps


def plan_steps(goal: str, max_steps: int = 5) -> dict[str, Any]:
    normalized_goal = str(goal or "").strip()
    if not normalized_goal:
        raise ValueError("规划目标不能为空。")
    if len(normalized_goal) > MAX_PLAN_GOAL_CHARS:
        raise ValueError(f"规划目标不能超过 {MAX_PLAN_GOAL_CHARS} 个字符。")
    if type(max_steps) is not int or not 1 <= max_steps <= MAX_PLAN_STEPS:
        raise ValueError(f"max_steps 必须是 1 到 {MAX_PLAN_STEPS} 之间的整数。")

    model = ModelFactory.for_task("planning", temperature=0.2)
    response = model.invoke(
        [
            SystemMessage(
                content=(
                    f"{get_prompt_manager().get_prompt('task_planner')}\n\n"
                    "输出约束：只返回 JSON 对象，格式为"
                    '{"steps": ["步骤一", "步骤二"]}。'
                )
            ),
            HumanMessage(
                content=(
                    f"目标：\n{normalized_goal}\n\n"
                    f"最多生成 {max_steps} 步。"
                )
            ),
        ]
    )
    steps = _parse_steps(_response_text(response), max_steps)
    if not steps:
        raise ValueError("规划模型没有返回有效步骤。")

    output = "\n".join(
        f"{index}. {step}"
        for index, step in enumerate(steps, start=1)
    )
    return {
        "success": True,
        "goal": normalized_goal,
        "steps": steps,
        "output": output,
        "error": "",
    }


def build_planning_tool_definition() -> ToolDefinition:
    return ToolDefinition(
        name="plan_steps",
        description=(
            "将一个目标拆分为有顺序、可执行的步骤并返回步骤列表；只生成计划，"
            "不会执行任务。max_steps 默认为 5，范围为 1 到 10。"
        ),
        handler=plan_steps,
    )
