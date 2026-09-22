from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Awaitable, Callable


class TriggerType(str, Enum):
    SCHEDULED = "scheduled"
    WEBHOOK = "webhook"
    EVENT = "event"


class TriggerStatus(str, Enum):
    ACTIVE = "active"
    PAUSED = "paused"
    DISABLED = "disabled"
    ERROR = "error"


@dataclass
class BaseTrigger:
    id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )
    name: str = ""
    description: str = ""
    trigger_type: TriggerType = TriggerType.SCHEDULED
    status: TriggerStatus = TriggerStatus.ACTIVE
    agent_id: str = "default"
    task: str = ""
    task_params: dict[str, Any] = field(
        default_factory=dict
    )
    user_id: str | None = None
    tags: list[str] = field(
        default_factory=list
    )
    created_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )
    updated_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )
    last_executed_at: datetime | None = None
    execution_count: int = 0
    failure_count: int = 0

    def to_dict(
        self,
        *,
        include_secret: bool = False,
    ) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "trigger_type": self.trigger_type.value,
            "status": self.status.value,
            "agent_id": self.agent_id,
            "task": self.task,
            "task_params": self.task_params,
            "user_id": self.user_id,
            "tags": self.tags,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "last_executed_at": (
                self.last_executed_at.isoformat()
                if self.last_executed_at
                else None
            ),
            "execution_count": self.execution_count,
            "failure_count": self.failure_count,
        }


@dataclass
class ScheduledTrigger(BaseTrigger):
    trigger_type: TriggerType = TriggerType.SCHEDULED
    schedule: str = "0 * * * *"
    timezone: str = "Asia/Shanghai"
    run_immediately: bool = False
    next_run_at: datetime | None = None

    def to_dict(
        self,
        *,
        include_secret: bool = False,
    ) -> dict[str, Any]:
        result = super().to_dict(
            include_secret=include_secret
        )
        result.update(
            {
                "schedule": self.schedule,
                "timezone": self.timezone,
                "run_immediately": self.run_immediately,
                "next_run_at": (
                    self.next_run_at.isoformat()
                    if self.next_run_at
                    else None
                ),
            }
        )
        return result


@dataclass
class WebhookTrigger(BaseTrigger):
    trigger_type: TriggerType = TriggerType.WEBHOOK
    endpoint_path: str = ""
    http_methods: list[str] = field(
        default_factory=lambda: ["POST"]
    )
    require_auth: bool = False
    auth_token: str | None = None
    rate_limit: int | None = None
    extract_body: bool = True
    extract_query: bool = True
    extract_headers: list[str] = field(
        default_factory=list
    )

    def to_dict(
        self,
        *,
        include_secret: bool = False,
    ) -> dict[str, Any]:
        result = super().to_dict(
            include_secret=include_secret
        )
        result.update(
            {
                "endpoint_path": (
                    self.endpoint_path
                    or f"/api/webhook/{self.id}"
                ),
                "http_methods": self.http_methods,
                "require_auth": self.require_auth,
                "rate_limit": self.rate_limit,
                "extract_body": self.extract_body,
                "extract_query": self.extract_query,
                "extract_headers": self.extract_headers,
            }
        )

        if include_secret:
            result["auth_token"] = self.auth_token

        return result


@dataclass
class EventTrigger(BaseTrigger):
    trigger_type: TriggerType = TriggerType.EVENT
    event_type: str = ""
    event_source: str | None = None
    event_filters: dict[str, Any] = field(
        default_factory=dict
    )
    debounce_seconds: int = 0

    def to_dict(
        self,
        *,
        include_secret: bool = False,
    ) -> dict[str, Any]:
        result = super().to_dict(
            include_secret=include_secret
        )
        result.update(
            {
                "event_type": self.event_type,
                "event_source": self.event_source,
                "event_filters": self.event_filters,
                "debounce_seconds": self.debounce_seconds,
            }
        )
        return result


@dataclass
class TriggerExecution:
    id: str = field(
        default_factory=lambda: uuid.uuid4().hex
    )
    trigger_id: str = ""
    trigger_name: str = ""
    started_at: datetime = field(
        default_factory=lambda: datetime.now(
            timezone.utc
        )
    )
    completed_at: datetime | None = None
    duration_ms: float | None = None
    status: str = "running"
    error_message: str | None = None
    agent_id: str = ""
    thread_id: str | None = None
    task: str = ""
    task_params: dict[str, Any] = field(
        default_factory=dict
    )
    result: dict[str, Any] | None = None
    output_text: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "trigger_id": self.trigger_id,
            "trigger_name": self.trigger_name,
            "started_at": self.started_at.isoformat(),
            "completed_at": (
                self.completed_at.isoformat()
                if self.completed_at
                else None
            ),
            "duration_ms": self.duration_ms,
            "status": self.status,
            "error_message": self.error_message,
            "agent_id": self.agent_id,
            "thread_id": self.thread_id,
            "task": self.task,
            "task_params": self.task_params,
            "result": self.result,
            "output_text": self.output_text,
        }

    def success(
        self,
        result: dict[str, Any],
        output: str,
    ) -> None:
        self.status = "success"
        self.result = result
        self.output_text = output
        self.completed_at = datetime.now(
            timezone.utc
        )
        self.duration_ms = (
            self.completed_at - self.started_at
        ).total_seconds() * 1000

    def failed(self, message: str) -> None:
        self.status = "failed"
        self.error_message = message
        self.completed_at = datetime.now(
            timezone.utc
        )
        self.duration_ms = (
            self.completed_at - self.started_at
        ).total_seconds() * 1000


def _parse_datetime(
    value: Any,
) -> datetime | None:
    if not value:
        return None

    if isinstance(value, datetime):
        return value

    return datetime.fromisoformat(str(value))


def _trigger_from_dict(
    payload: dict[str, Any],
) -> BaseTrigger:
    data = dict(payload)

    data["trigger_type"] = TriggerType(
        data.get(
            "trigger_type",
            "scheduled",
        )
    )
    data["status"] = TriggerStatus(
        data.get(
            "status",
            "active",
        )
    )

    for field_name in (
        "created_at",
        "updated_at",
        "last_executed_at",
        "next_run_at",
    ):
        if field_name in data:
            data[field_name] = _parse_datetime(
                data[field_name]
            )

    trigger_type = data["trigger_type"]

    if trigger_type == TriggerType.WEBHOOK:
        return WebhookTrigger(**data)

    if trigger_type == TriggerType.EVENT:
        return EventTrigger(**data)

    return ScheduledTrigger(**data)


class TriggerManager:
    def __init__(
        self,
        storage_path: str | Path,
        *,
        poll_seconds: float = 1.0,
        history_limit: int = 100,
    ) -> None:
        self.storage_path = Path(
            storage_path
        ).expanduser()
        self.poll_seconds = max(
            float(poll_seconds),
            0.2,
        )
        self.history_limit = max(
            int(history_limit),
            1,
        )
        self.triggers: dict[str, BaseTrigger] = {}
        self.executions: list[TriggerExecution] = []
        self.execution_callback: (
            Callable[
                [BaseTrigger, dict[str, Any]],
                Awaitable[dict[str, Any]],
            ]
            | None
        ) = None
        self._scheduler_task: asyncio.Task | None = None
        self._rate_limits: dict[
            str,
            list[float],
        ] = {}
        self._load()

    def set_execution_callback(
        self,
        callback: Callable[
            [BaseTrigger, dict[str, Any]],
            Awaitable[dict[str, Any]],
        ],
    ) -> None:
        self.execution_callback = callback

    def _ensure_scheduler(self) -> None:
        if (
            self._scheduler_task is not None
            and not self._scheduler_task.done()
        ):
            return

        self._scheduler_task = None

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return

        self._scheduler_task = asyncio.create_task(
            self._scheduler_loop()
        )

    async def start(self) -> None:
        self._ensure_scheduler()

    async def stop(self) -> None:
        task = self._scheduler_task
        self._scheduler_task = None

        if task is None or task.done():
            return

        task.cancel()

        try:
            await task
        except asyncio.CancelledError:
            pass

    async def _scheduler_loop(self) -> None:
        while True:
            await asyncio.sleep(
                self.poll_seconds
            )

            now = datetime.now(timezone.utc)

            for trigger in list(
                self.triggers.values()
            ):
                if not isinstance(
                    trigger,
                    ScheduledTrigger,
                ):
                    continue

                if trigger.status != TriggerStatus.ACTIVE:
                    continue

                if trigger.next_run_at is None:
                    trigger.next_run_at = (
                        self._next_run(
                            trigger.schedule,
                            now,
                        )
                    )
                    self._save()
                    continue

                if trigger.next_run_at <= now:
                    trigger.next_run_at = (
                        self._next_run(
                            trigger.schedule,
                            now,
                        )
                    )
                    self._save()

                    asyncio.create_task(
                        self.fire(
                            trigger.id,
                            {},
                        )
                    )

    @staticmethod
    def _field_matches(
        value: int,
        expression: str,
    ) -> bool:
        expression = expression.strip()

        if expression == "*":
            return True

        for item in expression.split(","):
            item = item.strip()

            if item.startswith("*/"):
                step = max(
                    int(item[2:]),
                    1,
                )
                if value % step == 0:
                    return True
                continue

            if "-" in item:
                start, end = item.split(
                    "-",
                    1,
                )
                if int(start) <= value <= int(end):
                    return True
                continue

            if int(item) == value:
                return True

        return False

    def _next_run(
        self,
        schedule: str,
        now: datetime,
    ) -> datetime:
        fields = str(schedule).split()

        if len(fields) != 5:
            raise ValueError(
                "schedule 必须是五段 cron 表达式。"
            )

        candidate = now.replace(
            second=0,
            microsecond=0,
        ) + timedelta(minutes=1)

        for _ in range(366 * 24 * 60):
            values = (
                candidate.minute,
                candidate.hour,
                candidate.day,
                candidate.month,
                (candidate.weekday() + 1) % 7,
            )

            if all(
                self._field_matches(
                    value,
                    expression,
                )
                for value, expression in zip(
                    values,
                    fields,
                )
            ):
                return candidate

            candidate += timedelta(minutes=1)

        raise ValueError(
            "无法计算下一次触发时间。"
        )

    async def add(
        self,
        trigger: BaseTrigger,
    ) -> str:
        if not trigger.name.strip():
            raise ValueError(
                "触发器名称不能为空。"
            )

        if not trigger.task.strip():
            raise ValueError(
                "触发器任务不能为空。"
            )

        if isinstance(
            trigger,
            WebhookTrigger,
        ):
            if not trigger.endpoint_path:
                trigger.endpoint_path = (
                    f"/api/webhook/{trigger.id}"
                )

            if trigger.require_auth:
                trigger.auth_token = (
                    trigger.auth_token
                    or secrets.token_urlsafe(32)
                )

        if isinstance(
            trigger,
            ScheduledTrigger,
        ):
            trigger.next_run_at = (
                self._next_run(
                    trigger.schedule,
                    datetime.now(timezone.utc),
                )
            )

        self.triggers[trigger.id] = trigger
        self._save()
        self._ensure_scheduler()

        if (
            isinstance(
                trigger,
                ScheduledTrigger,
            )
            and trigger.run_immediately
        ):
            asyncio.create_task(
                self.fire(
                    trigger.id,
                    {},
                )
            )

        return trigger.id

    async def remove(
        self,
        trigger_id: str,
    ) -> bool:
        if trigger_id not in self.triggers:
            return False

        del self.triggers[trigger_id]
        self._save()
        return True

    async def pause(
        self,
        trigger_id: str,
    ) -> bool:
        trigger = self.triggers.get(trigger_id)

        if trigger is None:
            return False

        trigger.status = TriggerStatus.PAUSED
        trigger.updated_at = datetime.now(
            timezone.utc
        )
        self._save()
        return True

    async def resume(
        self,
        trigger_id: str,
    ) -> bool:
        trigger = self.triggers.get(trigger_id)

        if (
            trigger is None
            or trigger.status != TriggerStatus.PAUSED
        ):
            return False

        trigger.status = TriggerStatus.ACTIVE
        trigger.updated_at = datetime.now(
            timezone.utc
        )
        self._save()
        self._ensure_scheduler()
        return True

    def get(
        self,
        trigger_id: str,
    ) -> BaseTrigger | None:
        return self.triggers.get(trigger_id)

    def list(
        self,
        *,
        trigger_type: TriggerType | None = None,
        status: TriggerStatus | None = None,
        user_id: str | None = None,
    ) -> list[BaseTrigger]:
        result = list(
            self.triggers.values()
        )

        if trigger_type is not None:
            result = [
                item
                for item in result
                if item.trigger_type == trigger_type
            ]

        if status is not None:
            result = [
                item
                for item in result
                if item.status == status
            ]

        if user_id is not None:
            result = [
                item
                for item in result
                if item.user_id == user_id
            ]

        return result

    async def fire(
        self,
        trigger_id: str,
        params: dict[str, Any],
    ) -> TriggerExecution:
        trigger = self.triggers[trigger_id]

        execution = TriggerExecution(
            trigger_id=trigger.id,
            trigger_name=trigger.name,
            agent_id=trigger.agent_id,
            task=trigger.task,
            task_params={
                **trigger.task_params,
                **params,
            },
        )

        self.executions.append(execution)
        self.executions = self.executions[
            -self.history_limit:
        ]

        trigger.last_executed_at = datetime.now(
            timezone.utc
        )
        trigger.execution_count += 1

        try:
            if self.execution_callback is None:
                raise RuntimeError(
                    "触发器执行回调未配置。"
                )

            result = await self.execution_callback(
                trigger,
                execution.task_params,
            )

            output = str(
                result.get("output", "")
            )

            execution.thread_id = result.get(
                "thread_id"
            )
            execution.success(
                result,
                output,
            )

        except Exception as exc:
            trigger.failure_count += 1
            execution.failed(str(exc))

        self._save()
        return execution

    async def handle_webhook(
        self,
        trigger_id: str,
        *,
        method: str,
        body: Any,
        query: dict[str, Any],
        headers: dict[str, str],
        authorization: str | None,
    ) -> dict[str, Any]:
        trigger = self.triggers.get(
            trigger_id
        )

        if not isinstance(
            trigger,
            WebhookTrigger,
        ):
            return {
                "success": False,
                "error": "Webhook not found",
                "status_code": 404,
            }

        if trigger.status != TriggerStatus.ACTIVE:
            return {
                "success": False,
                "error": "Webhook is not active",
                "status_code": 503,
            }

        if method.upper() not in {
            item.upper()
            for item in trigger.http_methods
        }:
            return {
                "success": False,
                "error": "Method not allowed",
                "status_code": 405,
            }

        if trigger.require_auth:
            provided = (
                authorization or ""
            ).strip()

            if provided.lower().startswith(
                "bearer "
            ):
                provided = provided[7:].strip()

            if not trigger.auth_token or not hmac.compare_digest(
                provided,
                trigger.auth_token,
            ):
                return {
                    "success": False,
                    "error": "Authentication failed",
                    "status_code": 401,
                }

        params = dict(trigger.task_params)

        if trigger.extract_body and body is not None:
            params["body"] = body

        if trigger.extract_query:
            params["query"] = query

        if trigger.extract_headers:
            allowed = {
                item.casefold()
                for item in trigger.extract_headers
            }
            params["headers"] = {
                key: value
                for key, value in headers.items()
                if key.casefold() in allowed
            }

        execution = await self.fire(
            trigger_id,
            params,
        )

        return {
            "success": execution.status == "success",
            "trigger_id": trigger_id,
            "execution_id": execution.id,
            "status": execution.status,
            "result": execution.result,
            "status_code": (
                200
                if execution.status == "success"
                else 500
            ),
        }

    async def emit_event(
        self,
        event_type: str,
        data: dict[str, Any],
        *,
        source: str | None = None,
    ) -> list[TriggerExecution]:
        executions = []

        for trigger in self.list(
            trigger_type=TriggerType.EVENT,
            status=TriggerStatus.ACTIVE,
        ):
            if not isinstance(
                trigger,
                EventTrigger,
            ):
                continue

            if trigger.event_type != event_type:
                continue

            if (
                trigger.event_source
                and trigger.event_source != source
            ):
                continue

            if not self._matches(
                data,
                trigger.event_filters,
            ):
                continue

            executions.append(
                await self.fire(
                    trigger.id,
                    {
                        "event_type": event_type,
                        "event_data": data,
                        "source": source,
                    },
                )
            )

        return executions

    @staticmethod
    def _matches(
        data: dict[str, Any],
        filters: dict[str, Any],
    ) -> bool:
        for key, expected in filters.items():
            value: Any = data

            for part in key.split("."):
                if not isinstance(value, dict):
                    return False

                value = value.get(part)

            if value != expected:
                return False

        return True

    def executions_for(
        self,
        trigger_id: str,
        limit: int = 50,
    ) -> list[TriggerExecution]:
        return [
            item
            for item in reversed(self.executions)
            if item.trigger_id == trigger_id
        ][: max(min(int(limit), 200), 1)]

    def _save(self) -> None:
        self.storage_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        payload = {
            "triggers": [
                item.to_dict(
                    include_secret=True
                )
                for item in self.triggers.values()
            ]
        }

        self.storage_path.write_text(
            json.dumps(
                payload,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

    def _load(self) -> None:
        if not self.storage_path.exists():
            return

        try:
            payload = json.loads(
                self.storage_path.read_text(
                    encoding="utf-8"
                )
            )

            for item in payload.get(
                "triggers",
                [],
            ):
                trigger = _trigger_from_dict(item)
                self.triggers[trigger.id] = trigger

        except Exception:
            self.triggers.clear()