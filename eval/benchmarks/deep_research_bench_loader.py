"""Validated loader for Deep Research benchmark tasks stored as JSONL."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: str
    query: str
    constraints: dict[str, Any] = field(default_factory=dict)
    expected_fields: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "query": self.query,
            "constraints": dict(self.constraints),
            "expected_fields": list(self.expected_fields),
            "metadata": dict(self.metadata),
        }


def _validate_task(raw: Any, line_number: int) -> BenchmarkTask:
    if not isinstance(raw, dict):
        raise ValueError(f"第 {line_number} 行必须是 JSON 对象。")

    query = raw.get("query")
    if not isinstance(query, str) or not query.strip():
        raise ValueError(f"第 {line_number} 行缺少有效的 query 字符串。")

    raw_fields = raw.get("expected_fields")
    if (
        not isinstance(raw_fields, list)
        or not raw_fields
        or any(not isinstance(value, str) or not value.strip() for value in raw_fields)
    ):
        raise ValueError(
            f"第 {line_number} 行的 expected_fields 必须是非空字符串数组。"
        )

    constraints = raw.get("constraints") or {}
    if not isinstance(constraints, dict):
        raise ValueError(f"第 {line_number} 行的 constraints 必须是 JSON 对象。")

    metadata = raw.get("metadata") or {}
    if not isinstance(metadata, dict):
        raise ValueError(f"第 {line_number} 行的 metadata 必须是 JSON 对象。")

    task_id = raw.get("id", f"task_{line_number:03d}")
    if not isinstance(task_id, str) or not task_id.strip():
        raise ValueError(f"第 {line_number} 行的 id 必须是非空字符串。")

    fields = list(dict.fromkeys(value.strip() for value in raw_fields))
    return BenchmarkTask(
        task_id=task_id.strip(),
        query=query.strip(),
        constraints=dict(constraints),
        expected_fields=fields,
        metadata=dict(metadata),
    )


def load_benchmark_tasks(
    path: str | Path,
    max_cases: int | None = None,
) -> list[BenchmarkTask]:
    """Read and validate tasks, reporting malformed input with its line number."""

    input_path = Path(path)
    if not input_path.is_file():
        raise FileNotFoundError(f"找不到基准任务文件：{input_path}")
    if max_cases is not None and max_cases < 1:
        raise ValueError("max_cases 必须大于 0。")

    tasks: list[BenchmarkTask] = []
    seen_ids: set[str] = set()
    with input_path.open("r", encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                raw = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"第 {line_number} 行不是有效 JSON：{exc.msg}。"
                ) from exc

            task = _validate_task(raw, line_number)
            if task.task_id in seen_ids:
                raise ValueError(
                    f"第 {line_number} 行的任务 id 重复：{task.task_id}。"
                )
            seen_ids.add(task.task_id)
            tasks.append(task)
            if max_cases is not None and len(tasks) >= max_cases:
                break
    return tasks
