import asyncio
import unittest
from unittest.mock import MagicMock, patch

from langgraph.checkpoint.memory import InMemorySaver

from agent.core.checkpointer import (
    _create_postgres_checkpointer,
    close_checkpointer,
    create_checkpointer,
)


class CheckpointerFactoryTests(unittest.TestCase):
    def test_empty_database_url_uses_memory_saver(self):
        checkpointer = create_checkpointer("")

        self.assertIsInstance(
            checkpointer,
            InMemorySaver,
        )

    def test_close_checkpointer_closes_its_connection(self):
        checkpointer = MagicMock()

        close_checkpointer(checkpointer)

        checkpointer.conn.close.assert_called_once_with()

    def test_postgres_connection_enables_autocommit_for_migrations(self):
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver

        connection = MagicMock()
        with patch.object(
            psycopg,
            "connect",
            return_value=connection,
        ) as connect, patch.object(
            PostgresSaver,
            "__init__",
            return_value=None,
        ), patch.object(
            PostgresSaver,
            "setup",
        ) as setup:
            created = _create_postgres_checkpointer("postgresql://database")

        connect.assert_called_once_with(
            "postgresql://database",
            autocommit=True,
            prepare_threshold=0,
        )
        self.assertIsInstance(created, PostgresSaver)
        setup.assert_called_once_with()

    def test_postgres_checkpointer_async_methods_delegate_to_sync_saver(self):
        import psycopg
        from langgraph.checkpoint.postgres import PostgresSaver

        connection = MagicMock()
        config = {"configurable": {"thread_id": "thread-1"}}
        checkpoint = MagicMock()
        metadata = {"source": "input"}
        versions = {"messages": "1"}
        writes = [("messages", "value")]

        with patch.object(
            psycopg,
            "connect",
            return_value=connection,
        ), patch.object(
            PostgresSaver,
            "setup",
        ), patch.object(
            PostgresSaver,
            "get_tuple",
            return_value=checkpoint,
        ) as get_tuple, patch.object(
            PostgresSaver,
            "list",
            return_value=iter([checkpoint]),
        ) as list_checkpoints, patch.object(
            PostgresSaver,
            "put",
            return_value=config,
        ) as put, patch.object(
            PostgresSaver,
            "put_writes",
        ) as put_writes, patch.object(
            PostgresSaver,
            "delete_thread",
        ) as delete_thread:
            checkpointer = _create_postgres_checkpointer(
                "postgresql://database"
            )

            async def exercise_async_methods():
                result = await checkpointer.aget_tuple(config)
                listed = [
                    item
                    async for item in checkpointer.alist(
                        config,
                        limit=1,
                    )
                ]
                updated_config = await checkpointer.aput(
                    config,
                    checkpoint,
                    metadata,
                    versions,
                )
                await checkpointer.aput_writes(
                    config,
                    writes,
                    "task-1",
                )
                await checkpointer.adelete_thread("thread-1")
                return result, listed, updated_config

            result, listed, updated_config = asyncio.run(
                exercise_async_methods()
            )

        self.assertIs(result, checkpoint)
        self.assertEqual(listed, [checkpoint])
        self.assertEqual(updated_config, config)
        get_tuple.assert_called_once_with(config)
        list_checkpoints.assert_called_once_with(
            config,
            filter=None,
            before=None,
            limit=1,
        )
        put.assert_called_once_with(
            config,
            checkpoint,
            metadata,
            versions,
        )
        put_writes.assert_called_once_with(
            config,
            writes,
            "task-1",
            "",
        )
        delete_thread.assert_called_once_with("thread-1")

    def test_postgres_failure_falls_back_to_memory(self):
        with patch(
            "agent.core.checkpointer._create_postgres_checkpointer",
            side_effect=RuntimeError("database unavailable"),
        ):
            checkpointer = create_checkpointer(
                "postgresql://invalid",
            )

        self.assertIsInstance(
            checkpointer,
            InMemorySaver,
        )

    def test_postgres_failure_can_be_strict(self):
        with patch(
            "agent.core.checkpointer._create_postgres_checkpointer",
            side_effect=RuntimeError("database unavailable"),
        ):
            with self.assertRaises(RuntimeError):
                create_checkpointer(
                    "postgresql://invalid",
                    fallback_to_memory=False,
                )


if __name__ == "__main__":
    unittest.main()
