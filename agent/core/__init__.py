"""
PuzzleResearch 核心 Agent 公共接口。

使用延迟导入，避免 graph、runtime 和 workflow 之间产生循环导入。
"""

from __future__ import annotations

import importlib
from typing import Any


__all__ = [
    "create_research_graph",
    "create_checkpointer",
    "AgentState",
    "QueryState",
    "ResearchPlan",
    "CodeExecution",
    "SearchResult",
    "RuntimeEvent",
    "AgentRuntime",
    "AgentProfile",
    "RouteDecision",
    "SmartRouter",
    "smart_route",
]


_SYMBOL_TO_MODULE: dict[str, str] = {
    "create_research_graph": "agent.core.graph",
    "create_checkpointer": "agent.core.checkpointer",
    "AgentState": "agent.core.state",
    "QueryState": "agent.core.state",
    "ResearchPlan": "agent.core.state",
    "CodeExecution": "agent.core.state",
    "SearchResult": "agent.core.state",
    "RuntimeEvent": "agent.core.events",
    "AgentRuntime": "agent.core.runtime",
    "AgentProfile": "agent.core.profile",
    "RouteDecision": "agent.core.smart_router",
    "SmartRouter": "agent.core.smart_router",
    "smart_route": "agent.core.smart_router",
}


def __getattr__(name: str) -> Any:
    module_name = _SYMBOL_TO_MODULE.get(name)

    if module_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )

    module = importlib.import_module(module_name)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(
        set(
            list(globals().keys())
            + list(_SYMBOL_TO_MODULE.keys())
        )
    )