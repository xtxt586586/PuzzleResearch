from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agent.core.event_stream import get_emitter_sync
from tools.registry import ToolDefinition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TASK_LIST_ROOT = PROJECT_ROOT / "data" / "task_lists"
MAX_SECTIONS = 30
MAX_TASKS = 100
MAX_TEXT_LENGTH = 1000
MAX_RESULT_LENGTH = 4000


class TaskStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    ALL = frozenset({PENDING, RUNNING, COMPLETED, FAILED, CANCELLED})


_STATUS_ALIASES = {
    "待办": TaskStatus.PENDING,
    "未开始": TaskStatus.PENDING,
    "进行中": TaskStatus.RUNNING,
    "处理中": TaskStatus.RUNNING,
    "in_progress": TaskStatus.RUNNING,
    "in progress": TaskStatus.RUNNING,
    "已完成": TaskStatus.COMPLETED,
    "完成": TaskStatus.COMPLETED,
    "done": TaskStatus.COMPLETED,
    "complete": TaskStatus.COMPLETED,
    "失败": TaskStatus.FAILED,
    "已失败": TaskStatus.FAILED,
    "已取消": TaskStatus.CANCELLED,
}


@dataclass(frozen=True)
class TaskSection:
    id: str
    title: str
    order: int


@dataclass(frozen=True)
class TaskItem:
    id: str
    content: str
    status: str
    section_id: str
    progress: int
    result: str
    error: str
    created_at: str
    updated_at: str


_managers: dict[str, "TaskListManager"] = {}
_managers_lock = threading.Lock()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


class TaskListManager:
    def __init__(
        self,
        thread_id: str,
        *,
        storage_root: Path | str = TASK_LIST_ROOT,
    ) -> None:
        self.thread_id = str(thread_id or "").strip()
        if not self.thread_id:
            raise ValueError("thread_id 不能为空。")

        self.storage_root = Path(storage_root)
        safe_name = hashlib.sha256(
            self.thread_id.encode("utf-8")
        ).hexdigest()
        self.path = self.storage_root / f"{safe_name}.json"
        self._lock = threading.RLock()
        self._sections: list[TaskSection] = []
        self._tasks: list[TaskItem] = []
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if payload.get("thread_id") != self.thread_id:
                raise ValueError("任务清单会话标识不匹配。")
            sections = [
                TaskSection(**item)
                for item in payload.get("sections", [])
            ]
            tasks = [TaskItem(**item) for item in payload.get("tasks", [])]
            if any(task.status not in TaskStatus.ALL for task in tasks):
                raise ValueError("任务清单包含无效状态。")
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("任务清单文件损坏或格式无效。") from exc

        self._sections = sections
        self._tasks = tasks

    def _write(
        self,
        sections: list[TaskSection],
        tasks: list[TaskItem],
    ) -> None:
        self.storage_root.mkdir(parents=True, exist_ok=True)
        payload = {
            "thread_id": self.thread_id,
            "sections": [asdict(section) for section in sections],
            "tasks": [asdict(task) for task in tasks],
            "updated_at": _timestamp(),
        }
        temporary_path: Path | None = None
        try:
            descriptor, name = tempfile.mkstemp(
                prefix=".task-list-",
                suffix=".tmp",
                dir=self.storage_root,
            )
            temporary_path = Path(name)
            with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, self.path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink(missing_ok=True)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            sections = [asdict(section) for section in self._sections]
            tasks = [asdict(task) for task in self._tasks]
            grouped = [
                {
                    "section": section,
                    "tasks": [
                        task for task in tasks
                        if task["section_id"] == section["id"]
                    ],
                }
                for section in sections
            ]
            orphan_tasks = [
                task for task in tasks
                if not any(
                    section["id"] == task["section_id"]
                    for section in sections
                )
            ]
            if orphan_tasks:
                grouped.append({
                    "section": {"id": "default", "title": "其他任务", "order": -1},
                    "tasks": orphan_tasks,
                })

            total = len(tasks)
            completed = sum(task["status"] == TaskStatus.COMPLETED for task in tasks)
            progress = {
                "total": total,
                "completed": completed,
                "running": sum(task["status"] == TaskStatus.RUNNING for task in tasks),
                "failed": sum(task["status"] == TaskStatus.FAILED for task in tasks),
                "cancelled": sum(task["status"] == TaskStatus.CANCELLED for task in tasks),
                "pending": sum(task["status"] == TaskStatus.PENDING for task in tasks),
                "progress_percent": round(completed / total * 100) if total else 0,
            }
            return {"task_list": grouped, "progress": progress}

    def _emit(self, action: str) -> None:
        try:
            get_emitter_sync(self.thread_id).emit_sync(
                "task_update",
                {
                    "thread_id": self.thread_id,
                    "action": action,
                    **self.snapshot(),
                },
            )
        except Exception:
            # Task persistence must not depend on an active SSE subscriber.
            pass

    def replace_tasks(self, sections_input: list[dict[str, Any]]) -> dict[str, Any]:
        if not isinstance(sections_input, list) or not sections_input:
            raise ValueError("sections 必须至少包含一个分组。")
        if len(sections_input) > MAX_SECTIONS:
            raise ValueError(f"分组数量不能超过 {MAX_SECTIONS} 个。")

        created_at = _timestamp()
        sections: list[TaskSection] = []
        tasks: list[TaskItem] = []
        for order, item in enumerate(sections_input):
            if not isinstance(item, dict):
                raise ValueError("每个分组必须是对象。")
            title = item.get("title")
            task_contents = item.get("tasks")
            if not isinstance(title, str) or not title.strip():
                raise ValueError("分组 title 不能为空。")
            if len(title.strip()) > MAX_TEXT_LENGTH:
                raise ValueError("分组名称过长。")
            if not isinstance(task_contents, list):
                raise ValueError("每个分组的 tasks 必须是数组。")

            section = TaskSection(_new_id(), title.strip(), order)
            sections.append(section)
            for content in task_contents:
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("任务内容必须是非空文本。")
                normalized_content = content.strip()
                if len(normalized_content) > MAX_TEXT_LENGTH:
                    raise ValueError("任务内容过长。")
                tasks.append(TaskItem(
                    id=_new_id(),
                    content=normalized_content,
                    status=TaskStatus.PENDING,
                    section_id=section.id,
                    progress=0,
                    result="",
                    error="",
                    created_at=created_at,
                    updated_at=created_at,
                ))

        if not tasks:
            raise ValueError("任务清单至少需要包含一项任务。")
        if len(tasks) > MAX_TASKS:
            raise ValueError(f"任务数量不能超过 {MAX_TASKS} 项。")

        with self._lock:
            self._write(sections, tasks)
            self._sections = sections
            self._tasks = tasks
        self._emit("created")
        return self.snapshot()

    def update_task(
        self,
        task_id: str,
        status: str,
        *,
        progress: int | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        normalized_status = str(status or "").strip().lower()
        normalized_status = _STATUS_ALIASES.get(
            normalized_status,
            normalized_status,
        )
        if normalized_status not in TaskStatus.ALL:
            raise ValueError(
                "status 必须是 pending、running、completed、failed 或 cancelled；"
                "也可使用待办、处理中、已完成、失败或已取消。"
            )
        if progress is not None and (type(progress) is not int or not 0 <= progress <= 100):
            raise ValueError("progress 必须是 0 到 100 之间的整数。")
        for label, value in (("result", result), ("error", error)):
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{label} 必须是文本。")
            if value is not None and len(value) > MAX_RESULT_LENGTH:
                raise ValueError(f"{label} 内容过长。")

        with self._lock:
            target = next((task for task in self._tasks if task.id == task_id), None)
            if target is None:
                raise ValueError(f"未找到任务：{task_id}")
            next_progress = progress
            if next_progress is None:
                next_progress = 100 if normalized_status == TaskStatus.COMPLETED else target.progress
            updated = replace(
                target,
                status=normalized_status,
                progress=next_progress,
                result=target.result if result is None else result,
                error=target.error if error is None else error,
                updated_at=_timestamp(),
            )
            tasks = [updated if task.id == task_id else task for task in self._tasks]
            self._write(self._sections, tasks)
            self._tasks = tasks

        self._emit("updated")
        return asdict(updated)

    def get_next_task(self) -> dict[str, Any]:
        with self._lock:
            task = next(
                (item for item in self._tasks if item.status == TaskStatus.PENDING),
                None,
            )
            if task is None:
                return {"has_next": False, **self.snapshot()}
            running_task = self.update_task(task.id, TaskStatus.RUNNING)
            return {"has_next": True, "task": running_task, **self.snapshot()}


def get_task_manager(thread_id: str) -> TaskListManager:
    normalized = str(thread_id or "").strip()
    if not normalized:
        raise ValueError("thread_id 不能为空。")
    with _managers_lock:
        manager = _managers.get(normalized)
        if manager is None:
            manager = TaskListManager(normalized)
            _managers[normalized] = manager
        return manager


def build_task_list_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    manager = get_task_manager(thread_id)

    def create_tasks(sections: list[dict[str, Any]]) -> dict[str, Any]:
        snapshot = manager.replace_tasks(sections)
        total = snapshot["progress"]["total"]
        return {
            "success": True,
            "output": f"已创建任务清单，共 {len(sections)} 个分组、{total} 项任务。",
            "error": "",
            **snapshot,
        }

    def view_tasks() -> dict[str, Any]:
        manager._emit("snapshot")
        snapshot = manager.snapshot()
        progress = snapshot["progress"]
        return {
            "output": (
                f"共 {progress['total']} 项任务：已完成 {progress['completed']}，"
                f"处理中 {progress['running']}，待办 {progress['pending']}，"
                f"失败 {progress['failed']}。"
            ),
            "error": "",
            **snapshot,
        }

    def update_task(
        task_id: str,
        status: str,
        progress: int | None = None,
        result: str | None = None,
        error: str | None = None,
    ) -> dict[str, Any]:
        task = manager.update_task(
            task_id,
            status,
            progress=progress,
            result=result,
            error=error,
        )
        return {
            "success": True,
            "output": (
                f"任务状态已更新：{task['content']} -> {task['status']}"
                f"（{task['progress']}%）。"
            ),
            "error": "",
            "task": task,
        }

    def get_next_task() -> dict[str, Any]:
        result = manager.get_next_task()
        if not result["has_next"]:
            return {**result, "output": "当前没有待办任务。", "error": ""}
        return {
            **result,
            "output": f"已领取任务：{result['task']['content']}",
            "error": "",
        }

    return [
        ToolDefinition(
            name="create_tasks",
            description=(
                "创建或替换当前会话的分组任务清单。参数 sections 是对象数组，"
                "每项包含 title 字符串和 tasks 字符串数组。"
            ),
            handler=create_tasks,
        ),
        ToolDefinition(
            name="view_tasks",
            description="查看当前会话任务清单及完成进度，并同步完整清单到界面。",
            handler=view_tasks,
        ),
        ToolDefinition(
            name="update_task",
            description=(
                "更新指定任务的状态、进度、结果或错误。status 请使用 pending、"
                "running、completed、failed、cancelled；也接受待办、处理中、"
                "已完成、失败、已取消。progress 为 0 到 100 的整数。"
            ),
            handler=update_task,
        ),
        ToolDefinition(
            name="get_next_task",
            description="领取当前清单中下一项待办任务，并将其标记为处理中。",
            handler=get_next_task,
        ),
    ]
