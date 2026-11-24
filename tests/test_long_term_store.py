import unittest
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import main
from agent.core.graph import create_research_graph
from agent.core.runtime import AgentRuntime
from common.long_term_store import (
    close_long_term_store,
    create_long_term_store,
)


class LongTermStoreTests(unittest.TestCase):
    def test_memory_backend_creates_a_langgraph_store(self):
        store = create_long_term_store("memory")

        store.put(
            ("user-001", "memories"),
            "memory-001",
            {"query": "问题", "content": "回答"},
        )
        results = store.search(
            ("user-001", "memories"),
            query="",
            limit=1,
        )

        self.assertEqual(len(results), 1)
        self.assertEqual(results[0].value["content"], "回答")

    def test_remote_backends_require_a_connection_url(self):
        for backend in ("postgres", "redis"):
            with self.subTest(backend=backend):
                with self.assertRaisesRegex(ValueError, "memory_store_url"):
                    create_long_term_store(backend)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "memory_store_backend"):
            create_long_term_store("unknown")

    def test_postgres_store_context_is_kept_until_close(self):
        store = MagicMock()
        context_manager = MagicMock()
        context_manager.__enter__.return_value = store
        postgres_store = MagicMock()
        postgres_store.from_conn_string.return_value = context_manager

        with patch.dict(
            sys.modules,
            {"langgraph.store.postgres": SimpleNamespace(PostgresStore=postgres_store)},
        ):
            created = create_long_term_store(
                "postgres",
                "postgresql://database",
            )

        self.assertIs(created, store)
        store.setup.assert_called_once_with()
        context_manager.__exit__.assert_not_called()

        close_long_term_store(created)

        context_manager.__exit__.assert_called_once_with(None, None, None)

    def test_redis_store_context_is_kept_until_close(self):
        store = MagicMock()
        context_manager = MagicMock()
        context_manager.__enter__.return_value = store
        redis_store = MagicMock()
        redis_store.from_conn_string.return_value = context_manager

        with patch.dict(
            sys.modules,
            {"langgraph.store.redis": SimpleNamespace(RedisStore=redis_store)},
        ):
            created = create_long_term_store(
                "redis",
                "redis://cache",
            )

        self.assertIs(created, store)
        store.setup.assert_called_once_with()
        context_manager.__exit__.assert_not_called()

        close_long_term_store(created)

        context_manager.__exit__.assert_called_once_with(None, None, None)

    def test_store_context_closes_when_setup_fails(self):
        store = MagicMock()
        store.setup.side_effect = RuntimeError("setup failed")
        context_manager = MagicMock()
        context_manager.__enter__.return_value = store
        postgres_store = MagicMock()
        postgres_store.from_conn_string.return_value = context_manager

        with patch.dict(
            sys.modules,
            {"langgraph.store.postgres": SimpleNamespace(PostgresStore=postgres_store)},
        ), self.assertRaisesRegex(RuntimeError, "setup failed"):
            create_long_term_store("postgres", "postgresql://database")

        context_manager.__exit__.assert_called_once()

    def test_unavailable_remote_backend_falls_back_to_memory_store(self):
        with patch.object(
            main.settings,
            "memory_store_backend",
            "postgres",
        ), patch.object(
            main.settings,
            "memory_store_url",
            "",
        ):
            store, available = main._initialize_langgraph_store()

        self.assertFalse(available)
        self.assertEqual(type(store).__name__, "InMemoryStore")

    def test_unavailable_remote_backend_fails_when_storage_fallback_is_disabled(self):
        with patch.object(
            main.settings,
            "memory_store_backend",
            "redis",
        ), patch.object(
            main.settings,
            "memory_store_url",
            "",
        ), patch.object(
            main.settings,
            "storage_fallback_to_memory",
            False,
        ):
            with self.assertRaisesRegex(
                RuntimeError,
                "LangGraph Store backend redis initialization failed",
            ):
                main._initialize_langgraph_store()

    def test_checkpointer_uses_configured_storage_fallback_policy(self):
        with patch.object(
            main.settings,
            "database_url",
            "postgresql://database",
        ), patch.object(
            main.settings,
            "storage_fallback_to_memory",
            False,
        ), patch.object(
            main,
            "create_checkpointer",
            return_value=object(),
        ) as factory:
            checkpointer = main._initialize_chat_checkpointer()

        self.assertIs(checkpointer, factory.return_value)
        factory.assert_called_once_with(
            "postgresql://database",
            fallback_to_memory=False,
        )

    def test_store_is_passed_through_runtime_into_compiled_graph(self):
        store = create_long_term_store("memory")

        graph = create_research_graph(store=store)
        runtime = AgentRuntime(store=store)

        self.assertIs(graph.store, store)
        self.assertIs(runtime.graph.store, store)

    def test_chat_memory_reads_and_writes_user_scoped_store_entries(self):
        langgraph_store = MagicMock()
        langgraph_store.search.return_value = [
            SimpleNamespace(
                value={
                    "query": "用户偏好",
                    "content": "使用简体中文回答。",
                }
            )
        ]
        local_store = MagicMock()
        local_store.search.return_value = []

        with patch.object(main, "langgraph_store", langgraph_store), patch.object(
            main,
            "_chat_memory_store",
            return_value=local_store,
        ), patch.object(main.settings, "enable_chat_memory", True):
            message = main._chat_memory_system_message(
                "owner-001",
                "介绍一下 Python",
            )
            main._remember_chat_turn(
                "owner-001",
                "介绍一下 Python",
                "Python 是一种编程语言。",
            )

        self.assertIn("使用简体中文回答。", message.content)
        langgraph_store.search.assert_called_with(
            ("owner-001", "memories"),
            query="介绍一下 Python",
            limit=3,
        )
        langgraph_store.put.assert_called_once()
        self.assertEqual(
            langgraph_store.put.call_args.args[0],
            ("owner-001", "memories"),
        )
        self.assertEqual(
            langgraph_store.put.call_args.args[2]["response"],
            "Python 是一种编程语言。",
        )


if __name__ == "__main__":
    unittest.main()
