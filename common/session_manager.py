from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from langchain_core.messages import BaseMessage

from agent.core.runtime import AgentRuntime


class SessionManagerError(RuntimeError):
    pass


class SessionNotFoundError(SessionManagerError):
    pass


class SessionOwnerError(PermissionError):
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
    def _normalize_owner_id(
        owner_id: str | None,
    ) -> str:
        value = str(owner_id or "").strip()
        return value or "anonymous"

    @staticmethod
    def _thread_config(
        thread_id: str,
    ) -> dict[str, Any]:
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

        if value is None or isinstance(
            value,
            (str, int, float, bool),
        ):
            return value

        if isinstance(value, BaseMessage):
            return {
                "type": getattr(
                    value,
                    "type",
                    "message",
                ),
                "content": SessionManager._serialize(
                    getattr(
                        value,
                        "content",
                        "",
                    ),
                    depth=depth + 1,
                ),
            }

        if isinstance(value, Mapping):
            return {
                str(key): SessionManager._serialize(
                    item,
                    depth=depth + 1,
                )
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple, set)):
            return [
                SessionManager._serialize(
                    item,
                    depth=depth + 1,
                )
                for item in value
            ]

        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

    @staticmethod
    def _snapshot_values(
        snapshot: Any,
    ) -> dict[str, Any]:
        values = getattr(
            snapshot,
            "values",
            {},
        ) or {}

        return dict(values)

    @staticmethod
    def _snapshot_interrupts(
        snapshot: Any,
    ) -> list[Any]:
        interrupts = getattr(
            snapshot,
            "interrupts",
            [],
        ) or []

        return list(interrupts)

    @staticmethod
    def _status(
        values: Mapping[str, Any],
    ) -> str:
        if values.get("is_cancelled"):
            return "cancelled"

        if values.get("is_complete"):
            return "completed"

        return str(
            values.get(
                "status",
                "pending",
            )
        )

    @classmethod
    def _owner_from_values(
        cls,
        values: Mapping[str, Any],
    ) -> str:
        return cls._normalize_owner_id(
            values.get("owner_id")
            or values.get("user_id")
        )

    @classmethod
    def _summary(
        cls,
        thread_id: str,
        values: Mapping[str, Any],
    ) -> dict[str, Any]:
        messages = values.get("messages") or []
        owner_id = cls._owner_from_values(values)

        return {
            "thread_id": thread_id,
            "status": cls._status(values),
            "topic": str(
                values.get(
                    "input",
                    "",
                )
            )[:100],
            "created_at": str(
                values.get(
                    "created_at",
                    "",
                )
            ),
            "updated_at": str(
                values.get(
                    "updated_at",
                    "",
                )
            ),
            "route": str(
                values.get(
                    "route",
                    "",
                )
            ),
            "has_report": bool(
                values.get("final_report")
            ),
            "revision_count": int(
                values.get(
                    "revision_count",
                    0,
                )
                or 0
            ),
            "message_count": (
                len(messages)
                if isinstance(messages, list)
                else 0
            ),
            "agent_id": values.get(
                "agent_id",
                "default",
            ),
            "user_id": values.get(
                "user_id",
                owner_id,
            ),
            "owner_id": owner_id,
        }

    @staticmethod
    def _checkpoint_values(
        checkpoint: Any,
    ) -> dict[str, Any]:
        raw_checkpoint = getattr(
            checkpoint,
            "checkpoint",
            {},
        ) or {}

        values = raw_checkpoint.get(
            "channel_values",
            {},
        ) or {}

        return dict(values)

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


    def create_session(
            self,
            thread_id: str,
            *,
            owner_id: str = "anonymous",
            agent_id: str = "default",
            user_input: str = "",
    ) -> dict[str,Any]:
        normalized_thread_id = str(
            thread_id or ""
        ).strip()

        if not normalized_thread_id:
            raise ValueError(
                "thread_id 不能为空。"
            )

        try:
            self._checkpoint(
                normalized_thread_id
            )
        except SessionNotFoundError:
            pass
        else:
            raise SessionManagerError(
                f"线程已存在：{normalized_thread_id}"
            )

        normalized_owner_id = self._normalize_owner_id(
            owner_id
        )

        graph = getattr(
            self.runtime,
            "graph",
            None,
        )

        update_state = getattr(
            graph,
            "update_state",
            None,
        )

        if not callable(update_state):
            raise SessionManagerError(
                "当前 runtime 不支持创建会话。"
            )

        initial_state = {
            "thread_id": normalized_thread_id,
            "user_id": normalized_owner_id,
            "agent_id": str(
                agent_id or "default"
            ).strip() or "default",
            "input": str(user_input or ""),
            "status": "pending",
            "is_complete": False,
            "is_cancelled": False,
            "messages": [],
            "revision_count": 0,
            "pending_tool_calls": [],
            "tool_results": [],
            "tool_events": [],
            "errors": [],
        }

        try:
            update_state(
                self._thread_config(
                    normalized_thread_id
                ),
                initial_state,
            )

        except Exception as exc:
            raise SessionManagerError(
                f"创建会话时出错：{exc}"
            ) from exc

        return self.get_session(
            normalized_thread_id,
            owner_id=normalized_owner_id,
        )


    def assert_owner(
        self,
        thread_id: str,
        owner_id: str,
    ) -> None:
        checkpoint = self._checkpoint(thread_id)
        values = self._checkpoint_values(checkpoint)

        expected_owner = self._normalize_owner_id(
            owner_id
        )
        actual_owner = self._owner_from_values(values)

        if actual_owner != expected_owner:
            raise SessionOwnerError(
                "当前用户无权访问该会话。"
            )

    def list_sessions(
        self,
        *,
        limit: int = 50,
        status: str | None = None,
        owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(
            min(int(limit), 200),
            1,
        )

        owner_filter = None

        if owner_id is not None:
            owner_filter = self._normalize_owner_id(
                owner_id
            )

        sessions: list[dict[str, Any]] = []
        seen: set[str] = set()

        for checkpoint in self.checkpointer.list(
            None,
            limit=limit * 5,
        ):
            config = getattr(
                checkpoint,
                "config",
                {},
            ) or {}

            configurable = config.get(
                "configurable",
                {},
            ) or {}

            thread_id = str(
                configurable.get(
                    "thread_id",
                    "",
                )
            ).strip()

            if not thread_id or thread_id in seen:
                continue

            seen.add(thread_id)

            values = self._checkpoint_values(
                checkpoint
            )

            summary = self._summary(
                thread_id,
                values,
            )

            if (
                owner_filter is not None
                and summary["owner_id"] != owner_filter
            ):
                continue

            if (
                status
                and summary["status"] != status
            ):
                continue

            sessions.append(summary)

            if len(sessions) >= limit:
                break

        return sessions

    def get_session(
        self,
        thread_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        checkpoint = self._checkpoint(thread_id)
        values = self._checkpoint_values(checkpoint)

        summary = self._summary(
            thread_id,
            values,
        )

        if owner_id is not None:
            expected_owner = self._normalize_owner_id(
                owner_id
            )

            if summary["owner_id"] != expected_owner:
                raise SessionOwnerError(
                    "当前用户无权访问该会话。"
                )

        return summary

    def get_state(
        self,
        thread_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        self._checkpoint(thread_id)

        if owner_id is not None:
            self.assert_owner(
                thread_id,
                owner_id,
            )

        snapshot = self.runtime.get_state(
            thread_id=thread_id,
        )

        values = self._snapshot_values(
            snapshot
        )

        interrupts = self._snapshot_interrupts(
            snapshot
        )

        snapshot_config = getattr(
            snapshot,
            "config",
            {},
        ) or {}

        if isinstance(
            snapshot_config,
            Mapping,
        ):
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
            "interrupts": self._serialize(
                interrupts
            ),
            "checkpoint_id": configurable.get(
                "checkpoint_id"
            ),
        }

    def delete_session(
        self,
        thread_id: str,
        *,
        owner_id: str | None = None,
    ) -> None:
        self._checkpoint(thread_id)

        if owner_id is not None:
            self.assert_owner(
                thread_id,
                owner_id,
            )

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
