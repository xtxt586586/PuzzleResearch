import asyncio
import unittest

from common.cancellation import (
    CancellationManager,
    TaskStatus,
)


class CancellationManagerTests(
    unittest.IsolatedAsyncioTestCase
):
    async def test_token_history_is_bounded_without_evicting_active_tasks(self):
        manager = CancellationManager(max_tokens=2)
        active = await manager.create_token("active")
        active.mark_running()
        completed_old = await manager.create_token("completed-old")
        completed_old.mark_completed()
        completed_new = await manager.create_token("completed-new")
        completed_new.mark_completed()

        self.assertIs(manager.get_token("active"), active)
        self.assertIsNone(manager.get_token("completed-old"))
        self.assertIs(manager.get_token("completed-new"), completed_new)
        self.assertEqual(manager.get_stats()["total"], 2)

    async def test_manager_rejects_nonpositive_token_limit(self):
        with self.assertRaises(ValueError):
            CancellationManager(max_tokens=0)

    async def test_terminal_transition_prunes_history_after_active_overflow(self):
        manager = CancellationManager(max_tokens=2)
        tokens = [
            await manager.create_token(f"active-{index}")
            for index in range(3)
        ]
        for token in tokens:
            token.mark_running()

        self.assertEqual(manager.get_stats()["total"], 3)

        tokens[0].mark_completed()

        self.assertEqual(manager.get_stats()["total"], 2)
        self.assertIsNone(manager.get_token("active-0"))
        self.assertIs(manager.get_token("active-1"), tokens[1])
        self.assertIs(manager.get_token("active-2"), tokens[2])

    async def test_failed_transition_prunes_overflowed_history(self):
        manager = CancellationManager(max_tokens=1)
        active = await manager.create_token("active")
        active.mark_running()
        failed = await manager.create_token("failed")
        failed.mark_running()

        failed.mark_failed("test failure")

        self.assertEqual(manager.get_stats()["total"], 1)
        self.assertIs(manager.get_token("active"), active)
        self.assertIsNone(manager.get_token("failed"))

    async def test_cancel_all_prunes_without_mutating_active_iteration(self):
        manager = CancellationManager(max_tokens=1)
        tokens = [
            await manager.create_token(f"active-{index}")
            for index in range(3)
        ]
        for token in tokens:
            token.mark_running()

        cancelled_count = await manager.cancel_all()

        self.assertEqual(cancelled_count, 3)
        self.assertTrue(all(token.is_cancelled for token in tokens))
        self.assertEqual(manager.get_stats()["total"], 1)

    async def test_create_and_cancel_token(self):
        manager = CancellationManager()

        token = await manager.create_token(
            "cancel-test-001",
            metadata={
                "user_id": "user-001",
            },
        )

        token.mark_running()

        self.assertEqual(
            token.status,
            TaskStatus.RUNNING,
        )

        self.assertIn(
            "cancel-test-001",
            manager.get_active_tasks(),
        )

        cancelled = await manager.cancel(
            "cancel-test-001",
            "测试取消",
        )

        self.assertTrue(cancelled)
        self.assertTrue(token.is_cancelled)
        self.assertEqual(
            token.status,
            TaskStatus.CANCELLED,
        )
        self.assertEqual(
            token.metadata["cancel_reason"],
            "测试取消",
        )

        self.assertNotIn(
            "cancel-test-001",
            manager.get_active_tasks(),
        )

    async def test_cancelled_token_raises(self):
        manager = CancellationManager()

        token = await manager.create_token(
            "cancel-test-002"
        )

        await manager.cancel(
            "cancel-test-002"
        )

        with self.assertRaises(
            asyncio.CancelledError
        ):
            token.check("before_tool_call")

    async def test_cancel_all_only_cancels_active_tokens(self):
        manager = CancellationManager()

        active = await manager.create_token(
            "cancel-test-003"
        )
        active.mark_running()

        completed = await manager.create_token(
            "cancel-test-004"
        )
        completed.mark_completed()

        count = await manager.cancel_all(
            "批量测试取消"
        )

        self.assertEqual(count, 1)
        self.assertEqual(
            active.status,
            TaskStatus.CANCELLED,
        )
        self.assertEqual(
            completed.status,
            TaskStatus.COMPLETED,
        )


if __name__ == "__main__":
    unittest.main()
