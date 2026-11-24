import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

from common.triggers import (
    EventTrigger,
    TriggerManager,
    TriggerStatus,
    WebhookTrigger,
)


class TriggerManagerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.manager = TriggerManager(
            Path(self.directory.name)
            / "triggers.json"
        )

        async def callback(trigger, params):
            return {
                "status": "completed",
                "thread_id": "thread-trigger-001",
                "output": trigger.task,
                "params": params,
            }

        self.manager.set_execution_callback(
            callback
        )

    async def asyncTearDown(self):
        await self.manager.stop()
        self.directory.cleanup()

    async def test_webhook_requires_auth_token(self):
        trigger = WebhookTrigger(
            name="protected",
            task="执行保护任务",
            require_auth=True,
        )

        await self.manager.add(trigger)

        denied = await self.manager.handle_webhook(
            trigger.id,
            method="POST",
            body={},
            query={},
            headers={},
            authorization="Bearer wrong",
        )

        self.assertEqual(
            denied["status_code"],
            401,
        )

        accepted = await self.manager.handle_webhook(
            trigger.id,
            method="POST",
            body={"value": 1},
            query={},
            headers={},
            authorization=(
                f"Bearer {trigger.auth_token}"
            ),
        )

        self.assertTrue(
            accepted["success"]
        )

    async def test_event_filter_triggers_execution(self):
        trigger = EventTrigger(
            name="event-trigger",
            event_type="research.completed",
            event_filters={
                "status": "completed",
            },
            task="处理研究完成事件",
        )

        await self.manager.add(trigger)

        executions = await self.manager.emit_event(
            "research.completed",
            {
                "status": "completed",
            },
        )

        self.assertEqual(
            len(executions),
            1,
        )
        self.assertEqual(
            executions[0].status,
            "success",
        )

    async def test_pause_and_resume(self):
        trigger = EventTrigger(
            name="pausable",
            event_type="test",
            task="测试任务",
        )

        await self.manager.add(trigger)

        self.assertTrue(
            await self.manager.pause(
                trigger.id
            )
        )
        self.assertEqual(
            trigger.status,
            TriggerStatus.PAUSED,
        )

        self.assertTrue(
            await self.manager.resume(
                trigger.id
            )
        )
        self.assertEqual(
            trigger.status,
            TriggerStatus.ACTIVE,
        )

    def test_scheduled_time_uses_configured_timezone(self):
        now = datetime(2026, 1, 1, 0, 30, tzinfo=timezone.utc)

        next_run = self.manager._next_run(
            "0 9 * * *",
            now,
            "Asia/Shanghai",
        )

        self.assertEqual(
            next_run,
            datetime(2026, 1, 1, 1, 0, tzinfo=timezone.utc),
        )

    def test_scheduled_time_skips_nonexistent_dst_minute(self):
        now = datetime(2026, 3, 8, 5, 0, tzinfo=timezone.utc)

        next_run = self.manager._next_run(
            "30 2 * * *",
            now,
            "America/New_York",
        )

        self.assertEqual(
            next_run,
            datetime(2026, 3, 9, 6, 30, tzinfo=timezone.utc),
        )

    def test_scheduled_time_handles_repeated_dst_minute(self):
        now = datetime(2026, 11, 1, 5, 45, tzinfo=timezone.utc)

        next_run = self.manager._next_run(
            "30 1 * * *",
            now,
            "America/New_York",
        )

        self.assertEqual(
            next_run,
            datetime(2026, 11, 1, 6, 30, tzinfo=timezone.utc),
        )

    def test_scheduled_time_rejects_unknown_timezone(self):
        with self.assertRaisesRegex(ValueError, "无效时区"):
            self.manager._next_run(
                "0 9 * * *",
                datetime(2026, 1, 1, tzinfo=timezone.utc),
                "Invalid/Timezone",
            )


if __name__ == "__main__":
    unittest.main()
