from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import MagicMock, patch

from tools.registry import ToolRegistry
from tools.task_list import (
    TaskListManager,
    TaskStatus,
    build_task_list_tool_definitions,
)


class TaskListManagerTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_task_lists_are_persisted_and_isolated_by_thread(self):
        first = TaskListManager("thread/one", storage_root=self.root)
        second = TaskListManager("thread-two", storage_root=self.root)
        first.replace_tasks([{"title": "研究", "tasks": ["搜集资料"]}])

        restored = TaskListManager("thread/one", storage_root=self.root)

        self.assertEqual(restored.snapshot()["progress"]["total"], 1)
        self.assertEqual(second.snapshot()["progress"]["total"], 0)
        self.assertNotIn("thread/one", str(first.path))

    def test_updates_persist_progress_and_emit_full_snapshot(self):
        manager = TaskListManager("thread-events", storage_root=self.root)
        emitter = MagicMock()
        with patch("tools.task_list.get_emitter_sync", return_value=emitter):
            created = manager.replace_tasks([
                {"title": "阶段一", "tasks": ["阅读来源", "整理结论"]}
            ])
            task_id = created["task_list"][0]["tasks"][0]["id"]
            manager.update_task(task_id, TaskStatus.COMPLETED, result="完成")

        event_types = [call.args[0] for call in emitter.emit_sync.call_args_list]
        latest_event = emitter.emit_sync.call_args_list[-1].args[1]
        self.assertEqual(event_types, ["task_update", "task_update"])
        self.assertEqual(latest_event["progress"]["completed"], 1)
        self.assertEqual(latest_event["task_list"][0]["tasks"][0]["result"], "完成")

        restored = TaskListManager("thread-events", storage_root=self.root)
        self.assertEqual(restored.snapshot()["progress"]["progress_percent"], 50)

    def test_get_next_marks_first_pending_task_running(self):
        manager = TaskListManager("thread-next", storage_root=self.root)
        manager.replace_tasks([{"title": "执行", "tasks": ["第一项", "第二项"]}])

        result = manager.get_next_task()

        self.assertTrue(result["has_next"])
        self.assertEqual(result["task"]["status"], TaskStatus.RUNNING)
        self.assertEqual(result["progress"]["running"], 1)

        next_result = manager.get_next_task()
        self.assertNotEqual(result["task"]["id"], next_result["task"]["id"])

    def test_concurrent_claims_do_not_return_the_same_task(self):
        manager = TaskListManager("thread-concurrent", storage_root=self.root)
        manager.replace_tasks([{"title": "执行", "tasks": ["第一项", "第二项"]}])

        with ThreadPoolExecutor(max_workers=2) as executor:
            results = list(executor.map(lambda _: manager.get_next_task(), range(2)))

        claimed_ids = {result["task"]["id"] for result in results}
        self.assertEqual(len(claimed_ids), 2)

    def test_registered_tools_create_view_update_and_claim(self):
        manager = TaskListManager("thread-registry", storage_root=self.root)
        with patch("tools.task_list.get_task_manager", return_value=manager):
            registry = ToolRegistry(
                build_task_list_tool_definitions("thread-registry")
            )
        created = registry.invoke(
            "create_tasks",
            {"sections": [{"title": "验证", "tasks": ["运行测试"]}]},
        )
        task_id = created["task_list"][0]["tasks"][0]["id"]
        viewed = registry.invoke("view_tasks", {})
        self.assertEqual(viewed["progress"]["pending"], 1)
        self.assertIn("共 1 项任务", viewed["output"])

        claimed = registry.invoke("get_next_task", {})
        self.assertEqual(claimed["task"]["id"], task_id)
        updated = registry.invoke(
            "update_task",
            {"task_id": task_id, "status": "completed"},
        )
        self.assertEqual(updated["task"]["progress"], 100)
        self.assertIn("状态已更新", updated["output"])

    def test_invalid_status_and_progress_are_rejected(self):
        manager = TaskListManager("thread-validation", storage_root=self.root)
        created = manager.replace_tasks([{"title": "检查", "tasks": ["验证输入"]}])
        task_id = created["task_list"][0]["tasks"][0]["id"]

        with self.assertRaises(ValueError):
            manager.update_task(task_id, "unknown")
        with self.assertRaises(ValueError):
            manager.update_task(task_id, TaskStatus.RUNNING, progress=101)

    def test_localized_status_aliases_are_normalized(self):
        manager = TaskListManager("thread-localized-status", storage_root=self.root)
        created = manager.replace_tasks([{"title": "检查", "tasks": ["验证输入"]}])
        task_id = created["task_list"][0]["tasks"][0]["id"]

        updated = manager.update_task(task_id, "处理中", progress=40)

        self.assertEqual(updated["status"], TaskStatus.RUNNING)
        self.assertEqual(updated["progress"], 40)

    def test_failed_write_does_not_replace_in_memory_list(self):
        manager = TaskListManager("thread-atomic", storage_root=self.root)
        original = manager.replace_tasks([{"title": "原清单", "tasks": ["保留"]}])

        with patch.object(manager, "_write", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                manager.replace_tasks([{"title": "新清单", "tasks": ["不能覆盖"]}])

        self.assertEqual(manager.snapshot(), original)


if __name__ == "__main__":
    unittest.main()
