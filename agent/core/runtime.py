from __future__ import annotations

from collections.abc import Iterable,AsyncIterable

import uuid
from typing import Any,Mapping

from langgraph.types import Command

from agent.core.graph import create_research_graph
from agent.core.profile import AgentProfile
from tools.registry import ToolRegistry
from agent.core.events import RuntimeEvent
from tools.builtin import create_default_registry



def _normalize_event_types(
        event_types: Iterable[str] | str | None,
) -> set[str] | None:
    """
    标准化事件类型过滤条件。

    字符串视为单个事件类型，
    其他可迭代对象转换为字符串集合。
    """

    if event_types is None:
        return None

    if isinstance(event_types,str):
        normalized = event_types.strip()

        return (
            {normalized}
            if normalized
            else set()
        )

    try:
        values = iter(event_types)
    except TypeError as exc:
        raise TypeError(
            "event_types 必须是字符串或可迭代对象"
        ) from exc

    return {
        str(item).strip()
        for item in values
        if str(item).strip()
    }


def _debug_event_to_raw_event(
        raw_event: Any,
) -> dict[str,Any]:
    """
    将 LangGraph debug 流事件转换为 RuntimeEvent 使用的原始结构。
    """

    if not isinstance(raw_event,Mapping):
        return {
            "event": "unknown",
            "data": raw_event,
        }

    raw_type = str(
        raw_event.get(
            "type",
            "",
        )
    ).strip()

    payload = raw_event.get("payload")

    if not isinstance(payload,Mapping):
        payload = {}

    name = str(
        payload.get(
            "name",
            "",
        )
    ).strip()

    raw_run_id = payload.get("id")

    raw_result: dict[str,Any] = {
        "event": "unknown",
        "name": name,
        "data": {},
    }

    if raw_run_id is not None:
        raw_result["run_id"] = str(raw_run_id)

    if raw_type == "task":
        raw_result.update(
            {
                "event": "on_chain_start",
                "data": {
                    "input": payload.get("input"),
                },
            }
        )

    elif raw_type == "task_result":
        error = payload.get("error")

        if error is None:
            raw_result.update(
                {
                    "event": "on_chain_end",
                    "data": {
                        "output": payload.get("result"),
                    },
                }
            )

        else:
            raw_result.update(
                {
                    "event": "on_chain_error",
                    "data": {
                        "output": payload.get("result"),
                        "error": str(error),
                    },
                }
            )

    else:
        raw_result["data"] = {
            "debug": dict[payload],
        }

    return raw_result




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
            store: Any = None,
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
        self._store = store
        self._profile = profile
        self._registry = registry

        self.graph = create_research_graph(
            checkpointer=checkpointer,
            store=store,
        )


    @staticmethod
    def _resolve_cancel_token(cancel_token_id: str | None) -> str:
        if cancel_token_id is not None:
            normalized_token = cancel_token_id.strip()

            if not normalized_token:
                raise ValueError(
                    "cancel_token_id 不能为空"
                )

            return normalized_token

        return uuid.uuid4().hex





    def _require_checkpointer(
            self,
            operation: str,
    ) -> None:
        if self._checkpointer is None:
            raise ValueError(
                f"{operation} 需要配置 checkpointer"
            )



    @staticmethod
    def _normalize_thread_id(
            thread_id: str,
    ) -> str:

        if not isinstance(thread_id,str):
            raise TypeError(
                "thread_id 必须是字符串。"
            )

        normalized_thread_id = thread_id.strip()

        if not normalized_thread_id:
            raise ValueError(
                "thread_id 不能为空"
            )

        return normalized_thread_id



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
            configurable["thread_id"] = (
                self._normalize_thread_id(thread_id)
            )

        elif configurable.get("thread_id") is not None:
            configurable["thread_id"] = (
                self._normalize_thread_id(
                    configurable["thread_id"]
                )
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


    @staticmethod
    def _snapshot_values(
            snapshot: Any,
    ) -> dict[str,Any]:
        if isinstance(snapshot,Mapping):
            values = snapshot.get("values") or {}

        else:
            values = getattr(
                snapshot,
                "values",
                {},
            ) or {}

        return dict(values)

    @staticmethod
    def _snap_interrupts(
            snapshot: Any,
    ) -> list[Any]:
        if isinstance(snapshot,Mapping):
            interrupts = snapshot.get("interrupts") or []

        else:
            interrupts = getattr(
                snapshot,
                "interrupts",
                [],
            ) or []

        if interrupts:
            return list(interrupts) if isinstance(interrupts, (list, tuple, set)) else [interrupts]

        tasks = (
            snapshot.get("tasks", [])
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "tasks", [])
        ) or []
        result: list[Any] = []
        for task in tasks:
            task_interrupts = (
                task.get("interrupts", [])
                if isinstance(task, Mapping)
                else getattr(task, "interrupts", [])
            ) or []
            if isinstance(task_interrupts, (list, tuple, set)):
                result.extend(task_interrupts)
            else:
                result.append(task_interrupts)
        return result



    def _build_payload(
            self,
            user_input: str,
            messages: list[Any] | None,
            conversation_history: list[dict[str, str]] | None = None,
    ) -> dict[str, Any]:
        if not isinstance(user_input,str):
            raise ValueError(
                "user_input 必须是字符串。"
            )

        if not user_input.strip():
            raise ValueError(
                "user_input 不能为空。"
            )

        payload = {
            "input": user_input,
            "messages": list(messages or []),
        }

        if conversation_history is not None:
            payload["conversation_history"] = [
                dict(message)
                for message in conversation_history
            ]

        return payload

    @staticmethod
    def _attach_config_metadata(
            payload: dict[str, Any],
            runtime_config: Mapping[str, Any],
    ) -> dict[str, Any]:
        configurable = runtime_config.get(
            "configurable",
            {},
        ) or {}

        if not isinstance(configurable, Mapping):
            return payload

        for key in (
            "thread_id",
            "user_id",
            "agent_id",
            "images",
        ):
            value = configurable.get(key)

            if value is not None:
                payload.setdefault(key, value)

        return payload



    def invoke(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            conversation_history: list[dict[str, str]] | None = None,
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
            conversation_history,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        payload = self._attach_config_metadata(
            payload,
            runtime_config,
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
            conversation_history: list[dict[str, str]] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
    ) -> dict[str,Any]:
        """
        异步执行一次新的 Agent 请求。
        """

        payload = self._build_payload(
            user_input,
            messages,
            conversation_history,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        payload = self._attach_config_metadata(
            payload,
            runtime_config,
        )

        return await self.graph.ainvoke(
            payload,
            config=runtime_config,
        )


    def cancel(
            self,
            *,
            thread_id: str,
            cancel_token_id: str | None = None,
            config: Mapping[str, Any] | None = None,
    ) -> Any:
        """
        持久化取消请求，并返回取消后的状态快照。
        """

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        resolved_token = self._resolve_cancel_token(cancel_token_id)

        self.graph.update_state(
            runtime_config,
            {
                "is_cancelled": True,
                "status": "cancelled",
                "cancel_token_id": resolved_token,
                "pending_tool_calls": [],
            },
        )

        return self.graph.get_state(
            runtime_config,
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



    async def acancel(
            self,
            *,
            thread_id: str,
            cancel_token_id: str | None = None,
            config: Mapping[str, Any] | None = None,
    ) -> Any:
        """
        异步持久化取消请求，并返回取消后的状态快照。
        """

        self._require_checkpointer("acancel")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        resolved_token = self._resolve_cancel_token(cancel_token_id)

        await self.graph.aupdate_state(
            runtime_config,
            {
                "is_cancelled": True,
                "status": "cancelled",
                "cancel_token_id": resolved_token,
                "pending_tool_calls": [],
            },
        )

        return await self.graph.aget_state(
            runtime_config,
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


    def inspect_thread(
            self,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
    ) -> dict[str,Any]:
        """
        返回适合 API 或控制台展示的线程状态摘要。
        """

        normalized_thread_id = (
            self._normalize_thread_id(thread_id)
        )

        snapshot = self.get_state(
            thread_id=normalized_thread_id,
            config=config,
        )

        values = self._snapshot_values(snapshot)
        interrupts = self._snap_interrupts(snapshot)

        return {
            "thread_id": normalized_thread_id,
            "status": values.get(
                "status",
                "pending",
            ),
            "is_complete": bool(
                values.get("is_complete",False)
            ),
            "is_cancelled": bool(
                values.get("is_cancelled",False)
            ),
            "cancel_token_id": values.get(
                "cancel_token_id"
            ),
            "pending_tool_count": len(
                values.get("pending_tool_calls",[])
            ),
            "tool_call_count": int(
                values.get("tool_call_count",0)
            ),
            "interrupt_count": len(interrupts),
            "snapshot": snapshot,
        }




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


    async def aupdate_state(
            self,
            *,
            thread_id: str,
            values: Mapping[str, Any],
            config: Mapping[str, Any] | None = None,
            as_node: str | None = None,
    ) -> Any:
        """异步更新指定线程的检查点状态。"""

        self._require_checkpointer("aupdate_state")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        update_options = (
            {"as_node": as_node}
            if as_node is not None
            else {}
        )

        return await self.graph.aupdate_state(
            runtime_config,
            dict(values),
            **update_options,
        )


    async def ainspect_thread(
            self,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
    ) -> dict[str,Any]:
        """
        异步返回线程状态摘要。
        """

        normalized_thread_id = (
            self._normalize_thread_id(thread_id)
        )

        snapshot = await self.aget_state(
            thread_id=normalized_thread_id,
            config=config,
        )

        values = self._snapshot_values(snapshot)
        interrupts = self._snap_interrupts(snapshot)

        return {
            "thread_id": normalized_thread_id,
            "status": values.get(
                "status",
                "pending",
            ),
            "is_complete": bool(
                values.get("is_complete",False)
            ),
            "is_cancelled": bool(
                values.get("is_cancelled",False)
            ),
            "cancel_token_id": values.get(
                "cancel_token_id"
            ),
            "pending_tool_count": len(
                values.get("pending_tool_calls",[])
            ),
            "tool_call_count": int(
                values.get("tool_call_count",0)
            ),
            "interrupt_count": len(interrupts),
            "snapshot": snapshot,
        }

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
            conversation_history: list[dict[str, str]] | None = None,
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
            conversation_history,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        payload = self._attach_config_metadata(
            payload,
            runtime_config,
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
            conversation_history: list[dict[str, str]] | None = None,
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
            conversation_history,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        payload = self._attach_config_metadata(
            payload,
            runtime_config,
        )

        async for chunk in self.graph.astream(
            payload,
            config=runtime_config,
            stream_mode=stream_mode,
        ):
            yield chunk



    def _stream_sync_events(
            self,
            graph_input: Any,
            runtime_config: Mapping[str,Any],
            *,
            version: str,
            **kwargs: Any,
    ):
        """
        兼容当前 LangGraph 的同步事件流。

        v1/v2 在同步 stream_events 中不可用时，
        使用 debug stream 并转换为旧版事件结构。
        """

        try:
            graph_events = self.graph.stream_events(
                graph_input,
                config=runtime_config,
                version=version,
                **kwargs,
            )

            yield from graph_events
            return

        except NotImplementedError as exc:
            message = str(exc)

            is_unsupported_sync_version = (
                version in {
                    "v1", "v2"
                }
                and "Use astream_events()" in message
            )

            if not is_unsupported_sync_version:
                raise

        if kwargs:
            raise NotImplementedError(
                "当前 LangGraph 同步兼容模式不支持额外事件过滤参数，"
                "请改用异步事件流。"
            )

        debug_events = self.graph.stream(
            graph_input,
            config=runtime_config,
            stream_mode="debug",
        )

        for debug_event in debug_events:
            yield _debug_event_to_raw_event(
                debug_event,
            )




    def stream_events(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
            version: str = "v2",
            **kwargs: Any,
    ):
        """
        同步流式返回 LangGraph 运行事件。

        kwargs 会透传给 LangGraph，
        例如 include_names、exclude_types 等过滤条件。
        """

        payload = self._build_payload(
            user_input,
            messages,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        payload = self._attach_config_metadata(
            payload,
            runtime_config,
        )

        return self._stream_sync_events(
            payload,
            runtime_config,
            version=version,
            **kwargs,
        )


    async def astream_events(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            conversation_history: list[dict[str, str]] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
            version: str = "v2",
            **kwargs: Any,
    ):
        """
        异步流式返回 LangGraph 运行事件。

        kwargs 会透传给 LangGraph，
        例如 include_names、exclude_types 等过滤条件。
        """

        payload = self._build_payload(
            user_input,
            messages,
            conversation_history,
        )

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        payload = self._attach_config_metadata(
            payload,
            runtime_config,
        )

        async for event in self.graph.astream_events(
            payload,
            config=runtime_config,
            version=version,
            **kwargs,
        ):
            yield event


    def resume_events(
            self,
            resume_value: Any,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
            version: str = "v2",
            **kwargs: Any,
    ):
        """
        同步流式恢复一个需要审批或人工介入的线程。
        """

        self._require_checkpointer("resume_events")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        return self._stream_sync_events(
            Command(resume=resume_value),
            runtime_config,
            version=version,
            **kwargs,
        )


    async def aresume_events(
            self,
            resume_value: Any,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
            version: str = "v2",
            **kwargs: Any,
    ):
        """
        异步流式恢复一个需要审批或人工介入的线程。
        """

        self._require_checkpointer("aresume_events")

        runtime_config = self._build_config(
            config=config,
            thread_id=thread_id,
        )

        async for event in self.graph.astream_events(
            Command(resume=resume_value),
            config=runtime_config,
            version=version,
            **kwargs,
        ):
            yield event


    def stream_runtime_events(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
            version: str = "v2",
            event_types: Iterable[str] | str | None = None,
            **kwargs: Any,
    ) -> Iterable[RuntimeEvent]:
        """
        同步返回标准化 RuntimeEvent。
        """

        allowed_events = _normalize_event_types(
            event_types
        )

        for raw_event in self.stream_events(
            user_input,
            messages=messages,
            thread_id=thread_id,
            config=config,
            version=version,
            **kwargs,
        ):
            event = RuntimeEvent.from_raw(raw_event)

            if (
                allowed_events is not None
                and event.event not in allowed_events
            ):
                continue

            yield event

    async def astream_runtime_events(
            self,
            user_input: str,
            *,
            messages: list[Any] | None = None,
            conversation_history: list[dict[str, str]] | None = None,
            thread_id: str | None = None,
            config: Mapping[str, Any] | None = None,
            version: str = "v2",
            event_types: Iterable[str] | str | None = None,
            **kwargs: Any,
    ) -> AsyncIterable[RuntimeEvent]:
        """
        异步返回标准化 RuntimeEvent。
        """

        allowed_events = _normalize_event_types(
            event_types
        )

        async for raw_event in self.astream_events(
            user_input,
            messages=messages,
            conversation_history=conversation_history,
            thread_id=thread_id,
            config=config,
            version=version,
            **kwargs,
        ):
            event = RuntimeEvent.from_raw(raw_event)

            if (
                allowed_events is not None
                and event.event not in allowed_events
            ):
                continue

            yield event

    def resume_runtime_events(
            self,
            resume_value: Any,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
            version: str = "v2",
            event_types: Iterable[str] | str | None = None,
            **kwargs: Any,
    ) -> Iterable[RuntimeEvent]:
        """
        同步恢复线程并返回标准化运行事件。
        """

        allowed_events = _normalize_event_types(
            event_types
        )

        for raw_event in self.resume_events(
            resume_value,
            thread_id=thread_id,
            config=config,
            version=version,
            **kwargs,
        ):
            event = RuntimeEvent.from_raw(raw_event)

            if (
                allowed_events is not None
                and event.event not in allowed_events
            ):
                continue

            yield event

    async def aresume_runtime_events(
            self,
            resume_value: Any,
            *,
            thread_id: str,
            config: Mapping[str, Any] | None = None,
            version: str = "v2",
            event_types: Iterable[str] | str | None = None,
            **kwargs: Any,
    ) -> AsyncIterable[RuntimeEvent]:
        """
        异步恢复线程并返回标准化运行事件。
        """

        allowed_events = _normalize_event_types(
            event_types
        )

        async for raw_event in self.aresume_events(
            resume_value,
            thread_id=thread_id,
            config=config,
            version=version,
            **kwargs,
        ):
            event = RuntimeEvent.from_raw(raw_event)

            if (
                allowed_events is not None
                and event.event not in allowed_events
            ):
                continue

            yield event


    def describe_tools(self) -> list[dict[str,Any]]:
        """
        返回当前运行时可用工具目录。

        如果没有注入自定义注册表，
        则返回默认工具目录。
        """

        registry = (
            self._registry
            if self._registry is not None
            else create_default_registry()
        )

        catalog = registry.describe_tools()

        if self._profile is None:
            return [
                {
                    **item,
                    "enabled": True
                }
                for item in catalog
            ]

        enabled_tools = self._profile.enabled_tools
        blocked_tools = (
            self._profile.blocked_tools or frozenset()
        )

        result: list[dict[str,Any]] = []

        for item in catalog:
            tool_name = item["name"]

            is_enabled = (
                enabled_tools is None
                or tool_name in enabled_tools
            )

            if tool_name in blocked_tools:
                is_enabled = False

            result.append(
                {
                    **item,
                    "enabled": is_enabled,
                }
            )

        return result
