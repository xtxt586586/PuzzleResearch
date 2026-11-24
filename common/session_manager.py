from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
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

        if hasattr(value, "value") and not isinstance(
            value,
            (str, bytes),
        ):
            return SessionManager._serialize(
                getattr(value, "value"),
                depth=depth + 1,
            )


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
        if isinstance(snapshot, Mapping):
            if "values" in snapshot:
                values = snapshot.get("values") or {}
            elif "state" in snapshot:
                values = snapshot.get("state") or {}
            else:
                values = snapshot
        else:
            values = getattr(
                snapshot,
                "values",
                {},
            ) or {}

        return (
            dict(values)
            if isinstance(values, Mapping)
            else {}
        )

    @staticmethod
    def _snapshot_interrupts(
            snapshot: Any,
    ) -> list[Any]:
        def normalize(value: Any) -> list[Any]:
            if value is None:
                return []
            if isinstance(value, (list, tuple, set)):
                return list(value)
            return [value]

        if isinstance(snapshot, Mapping):
            interrupts = snapshot.get(
                "interrupts"
            )

            if interrupts is None:
                interrupts = snapshot.get(
                    "__interrupt__"
                )

            if interrupts is None:
                values = snapshot.get(
                    "values",
                    {},
                ) or {}

                if isinstance(values, Mapping):
                    interrupts = values.get(
                        "__interrupt__"
                    )
        else:
            interrupts = getattr(
                snapshot,
                "interrupts",
                None,
            )

        normalized = normalize(interrupts)
        if normalized:
            return normalized

        tasks = (
            snapshot.get("tasks", [])
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "tasks", [])
        ) or []
        task_interrupts: list[Any] = []
        for task in tasks:
            task_values = (
                task.get("interrupts", [])
                if isinstance(task, Mapping)
                else getattr(task, "interrupts", [])
            ) or []
            task_interrupts.extend(normalize(task_values))

        if task_interrupts:
            return task_interrupts

        values = (
            snapshot.get("values", {})
            if isinstance(snapshot, Mapping)
            else getattr(snapshot, "values", {})
        ) or {}
        if (
            isinstance(values, Mapping)
            and str(values.get("status", "")).lower() == "paused"
        ):
            pending_calls = values.get("pending_tool_calls", [])
            if isinstance(pending_calls, list):
                return [
                    {
                        "type": "tool_approval",
                        "tool_name": str(
                            call.get("tool_name") or call.get("name") or ""
                        ),
                        "arguments": call.get("arguments", call.get("args", {})),
                        "message": "等待工具执行审批。",
                        "allowed_decisions": ["approve", "edit", "reject"],
                    }
                    for call in pending_calls
                    if isinstance(call, Mapping)
                ]

        return []

    @staticmethod
    def _status(
            values: Mapping[str, Any],
            interrupts: list[Any] | None = None,
    ) -> str:
        raw_status = str(
            values.get(
                "status",
                "pending",
            )
            or "pending"
        ).strip().lower()

        if (
                values.get("is_cancelled")
                or raw_status == "cancelled"
        ):
            return "cancelled"

        if raw_status == "failed":
            return "failed"

        if interrupts:
            return "paused"

        if (
                raw_status == "paused"
                or values.get("needs_clarification")
        ):
            return "paused"

        if (
                values.get("is_complete")
                or raw_status == "completed"
        ):
            return "completed"

        return raw_status

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
        if isinstance(checkpoint, Mapping):
            raw_checkpoint = checkpoint.get(
                "checkpoint",
                checkpoint,
            ) or {}
        else:
            raw_checkpoint = getattr(
                checkpoint,
                "checkpoint",
                {},
            ) or {}

        if not isinstance(
                raw_checkpoint,
                Mapping,
        ):
            return {}

        if "channel_values" in raw_checkpoint:
            values = raw_checkpoint.get(
                "channel_values"
            ) or {}
        elif "values" in raw_checkpoint:
            values = raw_checkpoint.get(
                "values"
            ) or {}
        else:
            values = raw_checkpoint

        return (
            dict(values)
            if isinstance(values, Mapping)
            else {}
        )


    @staticmethod
    def _configurable(
        value: Any,
    ) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            return {}

        configurable = value.get(
            "configurable",
            value,
        )

        return (
            dict(configurable)
            if isinstance(
                configurable,
                Mapping,
            )
            else {}
        )


    @classmethod
    def _snapshot_config(
        cls,
        snapshot: Any,
    ) -> dict[str, Any]:
        if isinstance(snapshot, Mapping):
            config = snapshot.get(
                "config",
                {},
            ) or {}
        else:
            config = getattr(
                snapshot,
                "config",
                {},
            ) or {}

        return cls._configurable(config)


    @classmethod
    def _checkpoint_config(
        cls,
        checkpoint: Any,
    ) -> dict[str, Any]:
        if isinstance(checkpoint, Mapping):
            config = checkpoint.get(
                "config",
                {},
            ) or {}
        else:
            config = getattr(
                checkpoint,
                "config",
                {},
            ) or {}

        return cls._configurable(config)


    @staticmethod
    def _checkpoint_metadata(
        checkpoint: Any,
    ) -> dict[str, Any]:
        if isinstance(checkpoint, Mapping):
            metadata = checkpoint.get(
                "metadata",
                {},
            ) or {}
        else:
            metadata = getattr(
                checkpoint,
                "metadata",
                {},
            ) or {}

        return (
            dict(metadata)
            if isinstance(
                metadata,
                Mapping,
            )
            else {}
        )


    @classmethod
    def _parent_checkpoint_id(
        cls,
        checkpoint: Any,
    ) -> str | None:
        if isinstance(checkpoint, Mapping):
            parent_config = checkpoint.get(
                "parent_config",
                {},
            ) or {}
        else:
            parent_config = getattr(
                checkpoint,
                "parent_config",
                {},
            ) or {}

        configurable = cls._configurable(
            parent_config
        )

        parent_id = configurable.get(
            "checkpoint_id"
        )

        return (
            str(parent_id)
            if parent_id
            else None
        )



    @classmethod
    def _extract_deepsearch_artifacts(
        cls,
        state: Mapping[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(state, Mapping):
            return {}

        raw_artifacts = state.get(
            "deepsearch_artifacts",
            {},
        )

        artifacts = (
            dict(raw_artifacts)
            if isinstance(
                raw_artifacts,
                Mapping,
            )
            else {}
        )

        research_plan = state.get(
            "research_plan"
        )

        query_history = state.get(
            "deepsearch_query_history"
        )

        if "queries" not in artifacts:
            if isinstance(
                query_history,
                list,
            ) and query_history:
                artifacts["queries"] = list(
                    query_history
                )
            elif isinstance(
                research_plan,
                list,
            ) and research_plan:
                artifacts["queries"] = list(
                    research_plan
                )

        if (
            "research_tree" not in artifacts
            and isinstance(
                state.get("research_tree"),
                Mapping,
            )
        ):
            artifacts["research_tree"] = dict(
                state["research_tree"]
            )

        if (
            "quality_summary" not in artifacts
            and isinstance(
                state.get("quality_summary"),
                Mapping,
            )
        ):
            artifacts["quality_summary"] = dict(
                state["quality_summary"]
            )

        for field_name in (
            "query_coverage",
            "freshness_summary",
        ):
            if (
                field_name not in artifacts
                and isinstance(
                    state.get(field_name),
                    Mapping,
                )
            ):
                artifacts[field_name] = dict(
                    state[field_name]
                )

        for field_name in (
            "sources",
            "claims",
            "fetched_pages",
            "passages",
        ):
            if (
                field_name not in artifacts
                and isinstance(
                    state.get(field_name),
                    list,
                )
            ):
                artifacts[field_name] = list(
                    state[field_name]
                )

        return artifacts



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
        checkpoint = self._checkpoint(
            thread_id
        )

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

        snapshot_config = self._snapshot_config(
            snapshot
        )

        checkpoint_config = (
            self._checkpoint_config(
                checkpoint
            )
        )

        metadata = self._checkpoint_metadata(
            checkpoint
        )

        artifacts = (
            self._extract_deepsearch_artifacts(
                values
            )
        )

        checkpoint_id = (
                snapshot_config.get(
                    "checkpoint_id"
                )
                or checkpoint_config.get(
            "checkpoint_id"
        )
        )

        checkpoint_ts = str(
            metadata.get(
                "created_at",
                "",
            )
            or ""
        )

        return {
            "thread_id": thread_id,
            "status": self._status(
                values,
                interrupts,
            ),
            "state": self._serialize(
                values
            ),
            "interrupts": self._serialize(
                interrupts
            ),
            "checkpoint_id": checkpoint_id,
            "checkpoint_ts": checkpoint_ts,
            "parent_checkpoint_id": (
                self._parent_checkpoint_id(
                    checkpoint
                )
            ),
            "deepsearch_artifacts": (
                self._serialize(artifacts)
            ),
        }


    def get_session_state(
        self,
        thread_id: str,
        *,
        owner_id: str | None = None,
    ) -> dict[str, Any]:
        checkpoint = self._checkpoint(
            thread_id
        )

        if owner_id is not None:
            self.assert_owner(
                thread_id,
                owner_id,
            )

        values = self._checkpoint_values(
            checkpoint
        )

        metadata = self._checkpoint_metadata(
            checkpoint
        )

        artifacts = (
            self._extract_deepsearch_artifacts(
                values
            )
        )

        return {
            "thread_id": thread_id,
            "checkpoint_ts": str(
                metadata.get(
                    "created_at",
                    "",
                )
                or ""
            ),
            "parent_checkpoint_id": (
                self._parent_checkpoint_id(
                    checkpoint
                )
            ),
            "state": self._serialize(
                values
            ),
            "deepsearch_artifacts": (
                self._serialize(artifacts)
            ),
        }


    def can_resume(
        self,
        thread_id: str,
        *,
        owner_id: str | None = None,
    ) -> tuple[bool, str]:
        try:
            session = self.get_session(
                thread_id,
                owner_id=owner_id,
            )
        except SessionNotFoundError:
            return False, "Session not found"

        status = session["status"]

        if status == "completed":
            return False, "Session already completed"

        if status == "deleted":
            return False, "Session has been deleted"

        if status == "running":
            return False, "Session is currently running"

        return True, "Session can be resumed"




    def build_resume_state(
        self,
        thread_id: str,
        *,
        additional_input: str | None = None,
        update_state: Mapping[str, Any] | None = None,
        owner_id: str | None = None,
    ) -> dict[str, Any] | None:
        checkpoint = self._checkpoint(
            thread_id
        )

        if owner_id is not None:
            self.assert_owner(
                thread_id,
                owner_id,
            )

        restored = deepcopy(
            self._checkpoint_values(
                checkpoint
            )
        )

        artifacts = (
            self._extract_deepsearch_artifacts(
                restored
            )
        )

        if update_state is not None:
            if not isinstance(
                update_state,
                Mapping,
            ):
                raise TypeError(
                    "update_state 必须是对象。"
                )

            restored.update(
                dict(update_state)
            )

        if additional_input:
            restored["resume_input"] = (
                str(additional_input)
            )

        if artifacts:
            restored[
                "deepsearch_artifacts"
            ] = deepcopy(artifacts)

            queries = artifacts.get(
                "queries"
            )

            if (
                isinstance(queries, list)
                and queries
                and not restored.get(
                    "research_plan"
                )
            ):
                restored["research_plan"] = list(
                    queries
                )

            for field_name in (
                "research_tree",
                "quality_summary",
                "query_coverage",
                "freshness_summary",
            ):
                value = artifacts.get(
                    field_name
                )

                if (
                    value
                    and not restored.get(
                        field_name
                    )
                ):
                    restored[field_name] = deepcopy(
                        value
                    )

        restored[
            "resumed_from_checkpoint"
        ] = True

        restored["resumed_at"] = (
            datetime.now(
                timezone.utc
            ).isoformat()
        )

        return restored



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
