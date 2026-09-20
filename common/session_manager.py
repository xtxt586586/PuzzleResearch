from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.types import interrupt

from agent.core.runtime import AgentRuntime


class SessionManagerError(RuntimeError):
    pass

class SessionNotFoundError(RuntimeError):
    pass

class SessionManager:
    def __init__(
            self,
            runtime: AgentRuntime,
            checkpointer: Any,
    ) -> None:
        self.runtime = runtime
        self.checkpointer = checkpointer

    @staticmethod
    def _thread_config(thread_id) -> dict[str,Any]:
        return {
            "configurable": {
                "thread_id": thread_id,
            }
        }

    @staticmethod
    def _serialize(
            value: Any,
            *,
            depth: int = 0,
    ) -> Any:
        if depth > 5:
            return str(value)

        if value is None or isinstance(value,(str,int,float,bool)):
            return value

        if isinstance(value,BaseMessage):
            return {
                "type": getattr(value,"type","message"),
                "content": SessionManager._serialize(
                    getattr(value,"content",""),
                    depth=depth + 1,
                )
            }

        if isinstance(value,Mapping):
            return {
                str(key): SessionManager._serialize(item,depth=depth + 1)
                for key,item in value.items()
            }

        if isinstance(value,(list,tuple,set)):
            return [
                SessionManager._serialize(item,depth=depth + 1)
                for item in value
            ]

        try:
            json.dumps(value)
            return value
        except (TypeError,ValueError):
            return str(value)

    @staticmethod
    def _snapshot_values(snapshot: Any) -> dict[str,Any]:
        values = getattr(snapshot,"values",{}) or {}

        return dict(values)

    @staticmethod
    def _snapshot_interrupts(snapshot: Any) -> list[Any]:
        interrupts = getattr(snapshot,"interrupts",[]) or []

        return list(interrupts)

    @staticmethod
    def _status(values: Mapping[str,Any]) -> str:
        if values.get("is_cancelled"):
            return "cancelled"

        if values.get("is_complete"):
            return "completed"

        return str(values.get("status","pending"))

    @classmethod
    def _summary(
            cls,
            thread_id: str,
            values: Mapping[str,Any],
    ) -> dict[str,Any]:
        messages = values.get("messages") or []

        return {
            "thread_id": thread_id,
            "status": cls._status(values),
            "topic": str(values.get("input",""))[:100],
            "created_at": str(values.get("created_at","")),
            "updated_at": str(values.get("updated_at","")),
            "route": str(values.get("route","")),
            "has_report": bool(values.get("final_report")),
            "revision_count": int(values.get("revision_count",0) or 0),
            "message_count": (
                len(messages) if isinstance(messages,list) else 0
            ),
            "agent_id": values.get("agent_id","default"),
            "user_id": values.get("user_id","anonymous")
        }

    def _checkpoint(
            self,
            thread_id: str,
    ) -> Any:
        checkpoint = self.checkpointer.get_tuple(
            self._thread_config(thread_id)
        )

        if checkpoint is None:
            raise SessionNotFoundError(
                f"线程不存在：{thread_id}"
            )
        return checkpoint

    def list_sessions(
            self,
            *,
            limit: int = 50,
            status: str | None = None,
    ) -> list[dict[str,Any]]:
        limit = max(min(int(limit),200),1)
        sessions: list[dict[str,Any]] = []
        seen: set[str] = set()

        for checkpoint in self.checkpointer.list(None,limit = limit *5):
            config = getattr(checkpoint,"config",{}) or {}
            configurable = config.get("configurable",{})
            thread_id = str(configurable.get("thread_id","")).strip()

            if not thread_id or thread_id in seen:
                continue

            seen.add(thread_id)

            raw_checkpoint = getattr(checkpoint,"checkpoint",{}) or {}

            values = raw_checkpoint.get("channel_values",{}) or {}

            summary = self._summary(thread_id,values)

            if status and summary["status"] != status:
                continue

            sessions.append(summary)

            if len(sessions) >= limit:
                break

        return sessions


    def get_session(
            self,
            thread_id: str,
    ) -> dict[str,Any]:
        checkpoint = self._checkpoint(thread_id)

        raw_checkpoint = getattr(
            checkpoint,
            "checkpoint",
            {}
        ) or {}

        values = raw_checkpoint.get("channel_values",{},) or {}

        return self._summary(
            thread_id,
            values,
        )

    def get_state(self,thread_id: str,) -> dict[str,Any]:
        self._checkpoint(thread_id)

        snapshot = self.runtime.get_state(
            thread_id=thread_id,
        )

        values = self._snapshot_values(
            snapshot,
        )

        interrupts = self._snapshot_interrupts(
            snapshot,
        )

        snapshot_config = getattr(
            snapshot,
            "config",
            {},
        ) or {}

        if isinstance(snapshot_config,Mapping):
            configurable = snapshot_config.get(
                "configurable",
                snapshot_config,
            ) or {}

        else:
            configurable = {}

        return {
            "thread_id": thread_id,
            "status": self._status(values),
            "state": self._serialize(values),
            "interrupts": self._serialize(interrupts),
            "checkpoint_id": configurable.get("checkpoint_id"),
        }


    def delete_session(self,thread_id: str,) -> None:
        self._checkpoint(thread_id)

        delete_thread = getattr(
            self.checkpointer,
            "delete_thread",
            None,
        )

        if not callable(delete_thread):
            raise SessionManagerError(
                "当前 checkpointer 不支持删除线程。"
            )

        delete_thread(thread_id)