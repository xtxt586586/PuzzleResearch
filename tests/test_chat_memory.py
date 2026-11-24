import tempfile
import asyncio
import threading
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient
from langchain_core.messages import SystemMessage

import main
from agent.core.events import RuntimeEvent
from common.long_term_store import create_long_term_store
from common.support_memory import SupportMemoryStore


class ChatMemoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def setUp(self):
        self.langgraph_store_patch = patch.object(
            main,
            "langgraph_store",
            create_long_term_store("memory"),
        )
        self.langgraph_store_patch.start()

    def tearDown(self):
        self.langgraph_store_patch.stop()

    def test_non_stream_chat_retrieves_and_persists_memory_by_principal(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SupportMemoryStore(Path(directory) / "chat.json")
            store.add(
                "owner-001",
                "我更喜欢简体中文回答",
                "请始终用简体中文回复我。",
            )

            with patch.object(
                main,
                "_chat_memory_store",
                return_value=store,
            ), patch.object(
                main.settings,
                "enable_chat_memory",
                True,
            ), patch.object(
                main,
                "configured_api_key",
                "secret-key",
            ), patch.object(
                main.chat_runtime,
                "ainvoke",
                new=AsyncMock(
                    return_value={
                        "final_report": "已用中文回答。",
                        "status": "completed",
                    }
                ),
            ) as invoke:
                response = self.client.post(
                    "/api/chat",
                    headers={
                        "Authorization": "Bearer secret-key",
                        "X-Weaver-User": "owner-001",
                    },
                    json={
                        "messages": [
                            {"role": "user", "content": "介绍一下你的功能"}
                        ],
                        "stream": False,
                        "user_id": "attacker-001",
                    },
                )

            self.assertEqual(response.status_code, 200)
            runtime_messages = invoke.await_args.kwargs["messages"]
            self.assertIsInstance(runtime_messages[0], SystemMessage)
            self.assertIn("请始终用简体中文回复我。", runtime_messages[0].content)
            self.assertEqual(
                invoke.await_args.kwargs["config"]["configurable"]["user_id"],
                "owner-001",
            )
            self.assertEqual(
                invoke.await_args.kwargs["conversation_history"],
                [{"role": "user", "content": "介绍一下你的功能"}],
            )
            self.assertEqual(
                len(store.search("owner-001", limit=10)),
                2,
            )
            self.assertEqual(store.search("attacker-001", limit=10), [])

    def test_research_stream_retrieves_and_saves_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SupportMemoryStore(Path(directory) / "chat.json")
            store.add(
                "research-user",
                "我关注本地优先的软件设计",
                "后续回答时优先考虑本地部署方案。",
            )
            runtime_messages = []

            async def fake_events(*args, **kwargs):
                runtime_messages.extend(kwargs["messages"])
                yield RuntimeEvent(
                    event="on_chain_end",
                    name="finalize",
                    data={
                        "output": {
                            "final_report": "本地部署研究结论。",
                            "status": "completed",
                        }
                    },
                )

            with patch.object(
                main,
                "_chat_memory_store",
                return_value=store,
            ), patch.object(
                main.settings,
                "enable_chat_memory",
                True,
            ), patch.object(
                main.chat_runtime,
                "aget_state",
                new=AsyncMock(return_value={}),
            ), patch.object(
                main.chat_runtime,
                "astream_runtime_events",
                new=fake_events,
            ):
                response = self.client.post(
                    "/api/research/sse",
                    headers={"X-Weaver-User": "research-user"},
                    json={"query": "比较本地部署与云端部署"},
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("event: done", response.text)
            self.assertIsInstance(runtime_messages[0], SystemMessage)
            self.assertIn(
                "后续回答时优先考虑本地部署方案。",
                runtime_messages[0].content,
            )
            saved = store.search(
                "research-user",
                "比较本地部署与云端部署",
                limit=10,
            )
            self.assertTrue(
                any(
                    item["response"] == "本地部署研究结论。"
                    for item in saved
                )
            )

    def test_failed_stream_does_not_save_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SupportMemoryStore(Path(directory) / "chat.json")

            async def failing_events(*args, **kwargs):
                if False:
                    yield RuntimeEvent(event="unused", data={})
                raise RuntimeError("模拟运行失败")

            with patch.object(
                main,
                "_chat_memory_store",
                return_value=store,
            ), patch.object(
                main.settings,
                "enable_chat_memory",
                True,
            ), patch.object(
                main.chat_runtime,
                "aget_state",
                new=AsyncMock(return_value={}),
            ), patch.object(
                main.chat_runtime,
                "astream_runtime_events",
                new=failing_events,
            ):
                response = self.client.post(
                    "/api/chat/sse",
                    headers={"X-Weaver-User": "failure-user"},
                    json={
                        "messages": [
                            {"role": "user", "content": "这次会失败"}
                        ]
                    },
                )

            self.assertEqual(response.status_code, 200)
            self.assertIn("event: error", response.text)
            self.assertEqual(store.search("failure-user", limit=10), [])

    def test_chinese_memory_search_matches_shared_phrases(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SupportMemoryStore(Path(directory) / "chat.json")
            store.add(
                "user-001",
                "我喜欢简体中文交流",
                "之后请使用中文解释技术问题。",
            )

            matches = store.search("user-001", "中文解释")

            self.assertEqual(len(matches), 1)

    def test_memory_status_reports_chat_memory_separately(self):
        with tempfile.TemporaryDirectory() as directory:
            chat_path = Path(directory) / "chat.json"
            support_path = Path(directory) / "support.json"
            chat_store = SupportMemoryStore(chat_path)
            chat_store.add("user-001", "聊天问题", "聊天回答")

            with patch.object(
                main.settings,
                "chat_memory_store_path",
                str(chat_path),
            ), patch.object(
                main.settings,
                "chat_memory_max_entries",
                50,
            ), patch.object(
                main.settings,
                "memory_store_path",
                str(support_path),
            ), patch.object(
                main.settings,
                "enable_chat_memory",
                True,
            ):
                response = self.client.get("/api/memory/status")

            self.assertEqual(response.status_code, 200)
            payload = response.json()
            self.assertIn("backend", payload)
            self.assertIn("checkpointer", payload)
            self.assertEqual(
                payload["chat_memory"],
                {
                    "enabled": True,
                    "backend": "local_json",
                    "local_store": {
                        "path": str(chat_path),
                        "users": 1,
                        "entries": 1,
                    },
                },
            )
            self.assertIn("langgraph_store", payload)
            self.assertIn(
                payload["langgraph_store"]["active_backend"],
                {"memory", "postgres", "redis"},
            )
            self.assertIn("persistent", payload["langgraph_store"])


class AsyncChatMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_async_memory_lookup_runs_off_event_loop_thread(self):
        event_loop_thread = threading.get_ident()
        lookup_threads = []
        expected = [SystemMessage(content="memory")]

        def lookup(history, user_id, query):
            lookup_threads.append(threading.get_ident())
            self.assertEqual(history, [])
            self.assertEqual(user_id, "user-001")
            self.assertEqual(query, "question")
            return expected

        with patch.object(main, "_with_chat_memory", side_effect=lookup):
            result = await main._with_chat_memory_async(
                [],
                "user-001",
                "question",
            )

        self.assertEqual(result, expected)
        self.assertEqual(len(lookup_threads), 1)
        self.assertNotEqual(lookup_threads[0], event_loop_thread)


if __name__ == "__main__":
    unittest.main()
