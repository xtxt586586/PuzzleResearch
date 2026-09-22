import unittest
from unittest.mock import patch

from langgraph.checkpoint.memory import InMemorySaver

from agent.core.checkpointer import create_checkpointer


class CheckpointerFactoryTests(unittest.TestCase):
    def test_empty_database_url_uses_memory_saver(self):
        checkpointer = create_checkpointer("")

        self.assertIsInstance(
            checkpointer,
            InMemorySaver,
        )

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