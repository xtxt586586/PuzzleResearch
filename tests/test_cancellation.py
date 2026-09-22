import asyncio
import unittest

from common.cancellation import (
    CancellationManager,
    TaskStatus,
)


class CancellationManagerTests(
    unittest.IsolatedAsyncioTestCase
):
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