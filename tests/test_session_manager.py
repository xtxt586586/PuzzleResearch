import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from langchain_core.messages import HumanMessage

from common.session_manager import (
    SessionManager,
    SessionManagerError,
    SessionNotFoundError,
)


def make_checkpoint(thread_id, values):
    return SimpleNamespace(
        config={
            "configurable": {
                "thread_id": thread_id,
            }
        },
        checkpoint={
            "channel_values": values,
        },
    )


class FakeCheckpointer:
    def __init__(self, checkpoints):
        self.checkpoints = {
            checkpoint.config["configurable"]["thread_id"]: checkpoint
            for checkpoint in checkpoints
        }
        self.deleted_threads = []

    def get_tuple(self, config):
        thread_id = config["configurable"]["thread_id"]
        return self.checkpoints.get(thread_id)

    def list(self, config=None, *, limit=None):
        checkpoints = list(self.checkpoints.values())

        if limit is not None:
            return checkpoints[:limit]

        return checkpoints

    def delete_thread(self, thread_id):
        self.deleted_threads.append(thread_id)
        self.checkpoints.pop(thread_id, None)


class FakeCheckpointerWithoutDelete(FakeCheckpointer):
    delete_thread = None


class SessionManagerTests(unittest.TestCase):
    def setUp(self):
        self.runtime = Mock()

        self.checkpointer = FakeCheckpointer(
            [
                make_checkpoint(
                    "thread-001",
                    {
                        "input": "研究人工智能",
                        "status": "paused",
                        "created_at": "2026-01-01T00:00:00+00:00",
                        "updated_at": "2026-01-01T00:01:00+00:00",
                        "route": "agent",
                        "final_report": "",
                        "revision_count": 1,
                        "messages": [
                            HumanMessage(
                                content="研究人工智能"
                            ),
                        ],
                        "agent_id": "default",
                        "user_id": "user-001",
                    },
                ),
                make_checkpoint(
                    "thread-002",
                    {
                        "input": "研究量子计算",
                        "status": "running",
                        "created_at": "2026-01-02T00:00:00+00:00",
                        "updated_at": "2026-01-02T00:01:00+00:00",
                        "route": "research",
                        "final_report": "",
                        "revision_count": 0,
                        "messages": [],
                        "agent_id": "research-agent",
                        "user_id": "user-002",
                    },
                ),
                make_checkpoint(
                    "thread-003",
                    {
                        "input": "研究已完成主题",
                        "status": "running",
                        "is_complete": True,
                        "created_at": "2026-01-03T00:00:00+00:00",
                        "updated_at": "2026-01-03T00:01:00+00:00",
                        "route": "direct",
                        "final_report": "研究完成",
                        "revision_count": 2,
                        "messages": [],
                        "agent_id": "default",
                        "user_id": "user-003",
                    },
                ),
            ]
        )

        self.manager = SessionManager(
            runtime=self.runtime,
            checkpointer=self.checkpointer,
        )

    def test_constructor_keeps_runtime_instance(self):
        self.assertIs(
            self.manager.runtime,
            self.runtime,
        )

        self.assertIs(
            self.manager.checkpointer,
            self.checkpointer,
        )

    def test_list_sessions_reads_thread_id(self):
        sessions = self.manager.list_sessions(
            limit=10,
        )

        self.assertEqual(
            len(sessions),
            3,
        )

        thread_ids = {
            session["thread_id"]
            for session in sessions
        }

        self.assertEqual(
            thread_ids,
            {
                "thread-001",
                "thread-002",
                "thread-003",
            },
        )

    def test_list_sessions_filters_by_status(self):
        completed_sessions = self.manager.list_sessions(
            limit=10,
            status="completed",
        )

        self.assertEqual(
            len(completed_sessions),
            1,
        )

        self.assertEqual(
            completed_sessions[0]["thread_id"],
            "thread-003",
        )

        running_sessions = self.manager.list_sessions(
            limit=10,
            status="running",
        )

        self.assertEqual(
            len(running_sessions),
            1,
        )

        self.assertEqual(
            running_sessions[0]["thread_id"],
            "thread-002",
        )

    def test_list_sessions_respects_limit(self):
        sessions = self.manager.list_sessions(
            limit=1,
        )

        self.assertEqual(
            len(sessions),
            1,
        )

    def test_get_session_returns_checkpoint_summary(self):
        session = self.manager.get_session(
            "thread-001",
        )

        self.assertEqual(
            session["thread_id"],
            "thread-001",
        )
        self.assertEqual(
            session["status"],
            "paused",
        )
        self.assertEqual(
            session["topic"],
            "研究人工智能",
        )
        self.assertEqual(
            session["route"],
            "agent",
        )
        self.assertEqual(
            session["revision_count"],
            1,
        )
        self.assertEqual(
            session["message_count"],
            1,
        )
        self.assertEqual(
            session["agent_id"],
            "default",
        )
        self.assertEqual(
            session["user_id"],
            "user-001",
        )
        self.assertFalse(
            session["has_report"],
        )

    def test_completed_state_is_reported_as_completed(self):
        session = self.manager.get_session(
            "thread-003",
        )

        self.assertEqual(
            session["status"],
            "completed",
        )
        self.assertTrue(
            session["has_report"],
        )

    def test_get_state_delegates_to_runtime(self):
        snapshot = SimpleNamespace(
            values={
                "status": "paused",
                "messages": [
                    HumanMessage(
                        content="等待工具审批"
                    ),
                ],
                "nested": {
                    "key": "value",
                },
            },
            interrupts=[
                {
                    "type": "tool_approval",
                    "tool_name": "python_executor",
                }
            ],
            config={
                "configurable": {
                    "thread_id": "thread-001",
                    "checkpoint_id": "checkpoint-001",
                }
            },
        )

        self.runtime.get_state.return_value = snapshot

        result = self.manager.get_state(
            "thread-001",
        )

        self.runtime.get_state.assert_called_once_with(
            thread_id="thread-001",
        )

        self.assertEqual(
            result["thread_id"],
            "thread-001",
        )
        self.assertEqual(
            result["status"],
            "paused",
        )
        self.assertEqual(
            result["state"]["status"],
            "paused",
        )
        self.assertEqual(
            result["state"]["messages"][0]["type"],
            "human",
        )
        self.assertEqual(
            result["state"]["messages"][0]["content"],
            "等待工具审批",
        )
        self.assertEqual(
            result["state"]["nested"]["key"],
            "value",
        )
        self.assertEqual(
            result["interrupts"][0]["type"],
            "tool_approval",
        )
        self.assertEqual(
            result["interrupts"][0]["tool_name"],
            "python_executor",
        )
        self.assertEqual(
            result["checkpoint_id"],
            "checkpoint-001",
        )

    def test_missing_session_raises_session_not_found(self):
        with self.assertRaises(SessionNotFoundError):
            self.manager.get_session(
                "missing-thread",
            )

        with self.assertRaises(SessionNotFoundError):
            self.manager.get_state(
                "missing-thread",
            )

    def test_delete_session_removes_checkpoint(self):
        self.manager.delete_session(
            "thread-001",
        )

        self.assertEqual(
            self.checkpointer.deleted_threads,
            ["thread-001"],
        )

        with self.assertRaises(SessionNotFoundError):
            self.manager.get_session(
                "thread-001",
            )

    def test_delete_session_requires_checkpointer_support(self):
        checkpointer = FakeCheckpointerWithoutDelete(
            [
                make_checkpoint(
                    "thread-001",
                    {
                        "status": "paused",
                    },
                ),
            ]
        )

        manager = SessionManager(
            runtime=self.runtime,
            checkpointer=checkpointer,
        )

        with self.assertRaises(SessionManagerError):
            manager.delete_session(
                "thread-001",
            )


if __name__ == "__main__":
    unittest.main()