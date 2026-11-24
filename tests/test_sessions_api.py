import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

import main


class FakeSessionManager:
    def __init__(self):
        self.sessions = {
            "thread-001": {
                "thread_id": "thread-001",
                "status": "paused",
                "topic": "测试主题",
                "created_at": "2026-01-01T00:00:00+00:00",
                "updated_at": "2026-01-01T00:01:00+00:00",
                "route": "agent",
                "has_report": False,
                "revision_count": 0,
                "message_count": 2,
                "agent_id": "default",
                "user_id": "anonymous",
            }
        }

    def list_sessions(
        self,
        *,
        limit=50,
        status=None,
    ):
        values = list(self.sessions.values())

        if status:
            values = [
                item
                for item in values
                if item["status"] == status
            ]

        return values[:limit]

    def get_session(self, thread_id):
        if thread_id not in self.sessions:
            from common.session_manager import (
                SessionNotFoundError,
            )

            raise SessionNotFoundError(
                f"线程不存在：{thread_id}"
            )

        return self.sessions[thread_id]

    def can_resume(self, thread_id, *, owner_id=None):
        session = self.get_session(thread_id)
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
        thread_id,
        *,
        additional_input=None,
        update_state=None,
        owner_id=None,
    ):
        state = dict(self.get_state(thread_id)["state"])
        if update_state:
            state.update(update_state)
        if additional_input:
            state["resume_input"] = additional_input
        state["resumed_from_checkpoint"] = True
        return state

    def get_state(self, thread_id):
        self.get_session(thread_id)

        return {
            "thread_id": thread_id,
            "status": "paused",
            "state": {
                "status": "paused",
                "route": "agent",
            },
            "interrupts": [
                {
                    "type": "tool_approval",
                    "tool_name": "python_executor",
                    "allowed_decisions": [
                        "approve",
                        "edit",
                        "reject",
                    ],
                    "allowed_tools": [
                        "python_executor",
                        "web_search",
                    ],
                }
            ],
            "checkpoint_id": "checkpoint-001",
        }

    def delete_session(self, thread_id):
        self.get_session(thread_id)
        del self.sessions[thread_id]


class SessionApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)
        self.manager = FakeSessionManager()
        self.manager_patch = patch.object(
            main,
            "session_manager",
            self.manager,
        )
        self.manager_patch.start()

    def tearDown(self):
        self.manager_patch.stop()

    def test_list_sessions(self):
        response = self.client.get(
            "/api/sessions"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["count"],
            1,
        )
        self.assertEqual(
            payload["sessions"][0]["thread_id"],
            "thread-001",
        )

    def test_get_session_state(self):
        response = self.client.get(
            "/api/sessions/thread-001/state"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["status"],
            "paused",
        )
        self.assertEqual(
            payload["state"]["route"],
            "agent",
        )
        self.assertEqual(
            payload["checkpoint_id"],
            "checkpoint-001",
        )

    def test_interrupt_status(self):
        response = self.client.get(
            "/api/interrupt/thread-001/status"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertTrue(
            payload["is_interrupted"]
        )
        self.assertIn(
            "approve",
            payload["available_actions"],
        )
        self.assertEqual(
            payload["available_actions"],
            ["approve", "edit", "reject"],
        )
        self.assertEqual(
            payload["allowed_tools"],
            ["python_executor", "web_search"],
        )
        self.assertEqual(payload["checkpoint_name"], "tool_approval")
        self.assertEqual(payload["interrupt_node"], "")
        self.assertEqual(payload["checkpoint_info"], {})
        self.assertEqual(
            payload["prompts"][0]["type"],
            "tool_approval",
        )

    def test_interrupt_status_uses_generic_actions_for_wrapped_interrupt(self):
        self.manager.get_state = lambda thread_id: {
            "thread_id": thread_id,
            "interrupts": [
                {
                    "id": "interrupt-001",
                    "value": {
                        "checkpoint": "final",
                        "instruction": "请审核报告",
                    },
                }
            ],
        }

        response = self.client.get(
            "/api/interrupt/thread-001/status"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["available_actions"],
            ["approve", "edit"],
        )
        self.assertEqual(payload["allowed_tools"], [])
        self.assertEqual(payload["checkpoint_name"], "final")
        self.assertEqual(
            payload["prompts"],
            [
                {
                    "checkpoint": "final",
                    "instruction": "请审核报告",
                }
            ],
        )

    def test_interrupt_status_merges_puzzleresearch_review_capabilities(self):
        self.manager.get_state = lambda thread_id: {
            "thread_id": thread_id,
            "interrupts": [
                {
                    "value": {
                        "action_requests": [
                            {"name": "python_executor"}
                        ],
                        "review_configs": [
                            {
                                "action_name": "python_executor",
                                "allowed_decisions": [
                                    "approve",
                                    "edit",
                                ],
                            }
                        ],
                    }
                },
                {
                    "available_actions": ["reject"],
                },
            ],
        }

        response = self.client.get(
            "/api/interrupt/thread-001/status"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(
            payload["available_actions"],
            ["approve", "edit", "reject"],
        )
        self.assertEqual(
            payload["allowed_tools"],
            ["python_executor"],
        )

    def test_interrupt_status_returns_empty_capabilities_without_interrupts(self):
        self.manager.get_state = lambda thread_id: {
            "thread_id": thread_id,
            "interrupts": [],
        }

        response = self.client.get(
            "/api/interrupt/thread-001/status"
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertFalse(payload["is_interrupted"])
        self.assertEqual(payload["available_actions"], [])
        self.assertEqual(payload["allowed_tools"], [])
        self.assertEqual(payload["prompts"], [])
        self.assertIsNone(payload["checkpoint_name"])
        self.assertEqual(payload["interrupt_node"], "")
        self.assertEqual(payload["checkpoint_info"], {})

    def test_resume_interrupt_normalizes_native_edit_decision(self):
        edited_action = {
            "name": "python_executor",
            "args": {"code": "print(6 * 7)"},
        }
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(
                return_value={
                    "status": "completed",
                    "final_report": "恢复完成",
                }
            ),
        ) as resume, patch.object(
            main.chat_runtime,
            "aget_state",
            return_value={
                "values": {
                    "conversation_history": [
                        {"role": "user", "content": "继续研究"},
                    ],
                },
            },
        ) as get_state, patch.object(
            main.chat_runtime,
            "aupdate_state",
            new=AsyncMock(),
        ) as update_state:
            response = self.client.post(
                "/api/interrupt/thread-001/resume",
                json={
                    "resume_value": {
                        "decisions": [
                            {
                                "type": "edit",
                                "edited_action": edited_action,
                            }
                        ]
                    }
                },
            )

        self.assertEqual(response.status_code, 200)
        resume.assert_awaited_once()
        self.assertEqual(
            resume.await_args.args[0],
            {
                "type": "edit",
                "edited_action": edited_action,
            },
        )
        get_state.assert_awaited_once_with(
            thread_id="thread-001",
        )
        update_state.assert_awaited_once_with(
            thread_id="thread-001",
            values={
                "conversation_history": [
                    {"role": "user", "content": "继续研究"},
                    {"role": "assistant", "content": "恢复完成"},
                ],
            },
        )

    def test_concurrent_duplicate_resume_only_runs_once(self):
        calls = 0
        entered = asyncio.Event()
        release = asyncio.Event()

        async def slow_resume(*args, **kwargs):
            nonlocal calls
            calls += 1
            entered.set()
            await release.wait()
            return {
                "status": "completed",
                "final_report": "审批后执行完成",
            }

        async def exercise():
            first = asyncio.create_task(
                main._aresume_once(
                    True,
                    thread_id="thread-001",
                    owner_id="anonymous",
                )
            )
            await entered.wait()
            second = asyncio.create_task(
                main._aresume_once(
                    True,
                    thread_id="thread-001",
                    owner_id="anonymous",
                )
            )
            await asyncio.sleep(0)
            release.set()
            return await asyncio.gather(first, second)

        main._resume_request_results.pop(
            "thread-001",
            None,
        )
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=slow_resume,
        ):
            results = asyncio.run(exercise())

        self.assertEqual(calls, 1)
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(
            sorted(replayed for _, replayed in results),
            [False, True],
        )

    def test_resume_interrupt_applies_report_modifications(self):
        revised_report = "修订后的报告正文"
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(
                return_value={
                    "status": "completed",
                    "final_report": revised_report,
                }
            ),
        ) as resume, patch.object(
            main.chat_runtime,
            "aget_state",
            return_value={"values": {"conversation_history": []}},
        ), patch.object(
            main.chat_runtime,
            "aupdate_state",
            new=AsyncMock(),
        ):
            response = self.client.post(
                "/api/interrupt/thread-001/resume",
                json={
                    "action": "modify",
                    "feedback": "请保留证据链",
                    "modifications": {"content": revised_report},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            resume.await_args.args[0],
            {"type": "edit", "content": revised_report},
        )

    def test_resume_interrupt_applies_edited_tool_action(self):
        edited_action = {
            "name": "python_executor",
            "args": {"code": "print(2 + 2)"},
        }
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(
                return_value={
                    "status": "completed",
                    "final_report": "工具执行完成",
                }
            ),
        ) as resume, patch.object(
            main.chat_runtime,
            "aget_state",
            return_value={"values": {"conversation_history": []}},
        ), patch.object(
            main.chat_runtime,
            "aupdate_state",
            new=AsyncMock(),
        ):
            response = self.client.post(
                "/api/interrupt/thread-001/resume",
                json={
                    "action": "modify",
                    "modifications": {"edited_action": edited_action},
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            resume.await_args.args[0],
            {"type": "edit", "edited_action": edited_action},
        )

    def test_resume_interrupt_rejects_unusable_modifications(self):
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(),
        ) as resume:
            response = self.client.post(
                "/api/interrupt/thread-001/resume",
                json={
                    "action": "modify",
                    "feedback": "请补充更多背景",
                },
            )

        self.assertEqual(response.status_code, 400)
        resume.assert_not_awaited()

    def test_resume_interrupt_rejects_multiple_native_decisions(self):
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(),
        ) as resume:
            response = self.client.post(
                "/api/interrupt/thread-001/resume",
                json={
                    "resume_value": {
                        "decisions": [
                            {"type": "approve"},
                            {"type": "reject"},
                        ]
                    }
                },
            )

        self.assertEqual(response.status_code, 400)
        resume.assert_not_awaited()

    def test_missing_session_returns_404(self):
        response = self.client.get(
            "/api/sessions/missing-thread"
        )

        self.assertEqual(
            response.status_code,
            404,
        )

    def test_delete_session(self):
        response = self.client.delete(
            "/api/sessions/thread-001"
        )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["status"],
            "deleted",
        )

    def test_resume_session(self):
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(
                return_value={
                    "status": "completed",
                    "final_report": "恢复完成",
                }
            ),
        ) as resume, patch.object(
            main.chat_runtime,
            "aget_state",
            return_value={"values": {"conversation_history": []}},
        ), patch.object(
            main.chat_runtime,
            "aupdate_state",
            new=AsyncMock(),
        ) as update_state:
            response = self.client.post(
                "/api/sessions/thread-001/resume",
                json={
                    "resume_value": True,
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertEqual(
            response.json()["status"],
            "resumed",
        )
        resume.assert_awaited_once()
        update_state.assert_awaited_once_with(
            thread_id="thread-001",
            values={
                "conversation_history": [
                    {"role": "assistant", "content": "恢复完成"},
                ],
            },
        )

    def test_resume_session_prepares_checkpoint_context(self):
        restored_state = {
            "route": "deep",
            "revision_count": 2,
            "research_plan": ["查询一", "查询二"],
            "resumed_from_checkpoint": True,
            "deepsearch_artifacts": {
                "mode": "tree",
                "queries": ["查询一", "查询二"],
                "quality_summary": {
                    "query_coverage_score": 0.75,
                    "freshness_warning": "部分来源较旧",
                },
                "freshness_summary": {"fresh_ratio": 0.5},
            },
        }
        with patch.object(
            self.manager,
            "build_resume_state",
            return_value=restored_state,
        ) as build_resume_state, patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(),
        ) as resume:
            response = self.client.post(
                "/api/sessions/thread-001/resume",
                json={
                    "additional_input": "继续研究来源时效性",
                    "update_state": {"revision_count": 2},
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertTrue(payload["success"])
        self.assertEqual(payload["status"], "ready_to_resume")
        self.assertEqual(
            payload["current_state"]["route"],
            "agent",
        )
        self.assertEqual(
            payload["deepsearch_resume"]["mode"],
            "tree",
        )
        self.assertEqual(
            payload["deepsearch_resume"]["query_coverage_score"],
            0.75,
        )
        self.assertEqual(
            payload["deepsearch_resume"]["freshness_warning"],
            "部分来源较旧",
        )
        self.assertEqual(
            payload["resume_state"]["research_plan_count"],
            2,
        )
        build_resume_state.assert_called_once_with(
            "thread-001",
            additional_input="继续研究来源时效性",
            update_state={"revision_count": 2},
            owner_id="anonymous",
        )
        resume.assert_not_awaited()

    def test_resume_session_does_not_persist_while_still_interrupted(self):
        with patch.object(
            main.chat_runtime,
            "aresume",
            new=AsyncMock(
                return_value={
                    "__interrupt__": [{"value": "等待审批"}],
                    "final_report": "尚未完成的报告",
                }
            ),
        ), patch.object(
            main.chat_runtime,
            "aget_state",
            new_callable=AsyncMock,
        ) as get_state, patch.object(
            main.chat_runtime,
            "aupdate_state",
            new_callable=AsyncMock,
        ) as update_state:
            response = self.client.post(
                "/api/sessions/thread-001/resume",
                json={"resume_value": True},
            )

        self.assertEqual(response.status_code, 200)
        get_state.assert_not_awaited()
        update_state.assert_not_awaited()

    def test_resume_session_rejects_ineligible_statuses(self):
        for status, reason in (
            ("completed", "Session already completed"),
            ("running", "Session is currently running"),
        ):
            with self.subTest(status=status):
                self.manager.sessions["thread-001"]["status"] = status
                with patch.object(
                    main.chat_runtime,
                    "aresume",
                    new=AsyncMock(),
                ) as resume:
                    response = self.client.post(
                        "/api/sessions/thread-001/resume",
                        json={"resume_value": True},
                    )

                self.assertEqual(response.status_code, 400)
                self.assertEqual(response.json()["detail"], reason)
                resume.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
