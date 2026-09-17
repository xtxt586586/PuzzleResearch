from __future__ import annotations

from typing import Any,Mapping

from langgraph.types import Command

from agent.core.graph import create_research_graph
from agent.core.profile import AgentProfile
from tools.registry import ToolRegistry


class AgentRuntime:
    """
    Agent 运行时门面。

    统一负责：
    - 创建研究图；
    - 注入 AgentProfile；
    - 注入 ToolRegistry；
    - 管理 thread_id；
    - 执行普通调用和审批恢复。
    """

    def __init__(
            self,
            *,
            checkpointer: Any = None,
            profile: AgentProfile | None = None,
            registry: ToolRegistry | None = None,
    ):
        if profile is not None and not isinstance(
            profile,
            AgentProfile,
        ):
            raise TypeError(
                "profile 必须是 AgentProfile"
            )

        if registry is not None and not isinstance(
            registry,
            ToolRegistry
        ):
            raise TypeError(
                "registry 必须是 ToolRegistry"
            )

        self._checkpointer = checkpointer
        self._profile = profile
        self._registry = registry

        self.graph = create_research_graph(
            checkpointer=checkpointer,
        )


    def _require_checkpointer(
            self,
            operation: str,
    ) -> None:
        if self._checkpointer is None:
            raise ValueError(
                f"{operation} 需要配置 checkpointer"
            )



    def _build_config(
            self,
            *,
            config: Mapping[str, Any] | None = None,
            thread_id: str | None = None,
    ) -> dict[str,Any]:
        runtime_config = dict(config or {})

        configurable = dict(
            runtime_config.get("configurable", {}),
        )

        if thread_id is not None:
            normalized_thread_id = thread_id.strip()

            if not normalized_thread_id:
                raise ValueError(
                    "thread_id 不能为空"
                )

            configurable["thread_id"] = (
                normalized_thread_id
            )

        if (
            self._checkpointer is not None
            and not configurable.get("thread_id")
        ):
            raise ValueError(
                "启用 checkpointer 时必须提供 thread_id"
            )

        if (
            self._profile is not None
            and "agent_profile" not in configurable
        ):
            configurable["agent_profile"] = self._profile

        if (
            self._registry is not None
            and "tool_registry" not in configurable
        ):
            configurable["tool_registry"] = self._registry

        runtime_config["configurable"] = configurable

        return runtime_config


    def _build_payload(
            self,
            user_input: str,
            messages: list[Any] | None,
    ) -> dict[str, Any]:
        if not isinstance(user_input,str):
            raise ValueError(
                "user_input 必须是字符串。"
            )

        if not user_input.strip():
            raise ValueError(
                "user_input 不能为空。"
            )

        return {
            "input": user_input,
            "messages": list(messages or []),
        }



    def invoke(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
    ) -> dict[str,Any]:
        """
        执行一次新的 Agent 请求。
        """

        if not isinstance(user_input,str):
            raise ValueError(
                "user_input 必须是字符串。"
            )

        if not user_input.strip():
            raise ValueError(
                "user_input 不能为空。"
            )

        payload = self._build_payload(
            user_input,
            messages,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return self.graph.invoke(
            payload,
            config=runtime_config,
        )


    async def ainvoke(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
    ) -> dict[str,Any]:
        """
        异步执行一次新的 Agent 请求。
        """

        payload = self._build_payload(
            user_input,
            messages,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return await self.graph.ainvoke(
            payload,
            config=runtime_config,
        )


    def resume(
            self,
            resume_value: Any,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
    ) -> dict[str,Any]:
        """
        恢复一个需要审批或人工介入的线程。
        """

        self._require_checkpointer("resume")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return self.graph.invoke(
            Command(resume=resume_value),
            config=runtime_config,
        )

    async def aresume(
            self,
            resume_value: Any,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
    ) -> dict[str,Any]:
        """
        异步恢复一个需要审批或人工介入的线程。
        """

        self._require_checkpointer("aresume")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return await self.graph.ainvoke(
            Command(resume=resume_value),
            config=runtime_config,
        )


    def get_state(
            self,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
            subgraphs: bool = False,
    ) -> Any:
        """
        查询指定线程的最新状态快照。
        """

        self._require_checkpointer("get_state")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return self.graph.get_state(
            runtime_config,
            subgraphs=subgraphs,
        )



    async def aget_state(
            self,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
            subgraphs: bool = False,
    ) -> Any:
        """
        异步查询指定线程的最新状态快照。
        """

        self._require_checkpointer("aget_state")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return await self.graph.aget_state(
            runtime_config,
            subgraphs=subgraphs,
        )


    def get_state_history(
            self,
            *,
            thread_id: str,
            limit: int | None = None,
            config: Mapping[str, Any] | None = None,
    ):
        """
        查询指定线程的历史状态快照。
        """

        self._require_checkpointer("get_state_history")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return self.graph.get_state_history(
            runtime_config,
            limit=limit,
        )


    async def aget_state_history(
            self,
            *,
            thread_id: str,
            limit: int | None = None,
            config: Mapping[str, Any] | None = None,
    ):
        """
        异步查询指定线程的历史状态快照。
        """

        self._require_checkpointer("aget_state_history")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        async for snapshot in self.graph.aget_state_history(
            runtime_config,
            limit=limit,
        ):
            yield snapshot


    def stream(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
            stream_mode: Any = "values",
    ):
        """
        流式执行 Agent 请求。
        """

        payload = self._build_payload(
            user_input,
            messages,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return self.graph.stream(
            payload,
            config=runtime_config,
            stream_mode=stream_mode,
        )


    async def astream(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
            stream_mode: Any = "values",
    ):
        """
        异步流式执行 Agent 请求。
        """

        payload = self._build_payload(
            user_input,
            messages,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        async for chunk in self.graph.astream(
            payload,
            config=runtime_config,
            stream_mode=stream_mode,
        ):
            yield chunk