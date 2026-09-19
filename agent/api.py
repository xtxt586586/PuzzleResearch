"""
PuzzleResearch 的稳定公共 Agent API。

外部应用应优先从这里导入核心对象，
避免直接依赖内部模块路径。
"""

from agent.core.events import RuntimeEvent
from agent.core.graph import create_research_graph
from agent.core.profile import AgentProfile
from agent.core.runtime import AgentRuntime
from agent.core.smart_router import (
    RouteDecision,
    SmartRouter,
    smart_route,
)
from agent.core.state import (
    AgentState,
    CodeExecution,
    QueryState,
    ResearchPlan,
    SearchResult,
)

__all__ = [
    "AgentProfile",
    "AgentRuntime",
    "AgentState",
    "CodeExecution",
    "QueryState",
    "ResearchPlan",
    "RouteDecision",
    "RuntimeEvent",
    "SearchResult",
    "SmartRouter",
    "create_research_graph",
    "smart_route",
]