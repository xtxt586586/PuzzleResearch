import tempfile
import unittest
from pathlib import Path

from common.support_memory import SupportMemoryStore


class SupportMemoryStoreTests(unittest.TestCase):
    def test_add_search_and_persist(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "memory.json"

            store = SupportMemoryStore(
                path,
                max_entries=2,
            )

            store.add(
                "user-001",
                "如何使用 LangGraph",
                "可以通过 StateGraph 构建工作流。",
            )
            store.add(
                "user-001",
                "如何配置 DeepSeek",
                "通过环境变量配置 API Key。",
            )

            result = store.search(
                "user-001",
                "LangGraph",
            )

            self.assertEqual(
                len(result),
                1,
            )
            self.assertIn(
                "LangGraph",
                result[0]["query"],
            )

            restored = SupportMemoryStore(
                path,
                max_entries=2,
            )

            self.assertEqual(
                restored.status()["entries"],
                2,
            )

    def test_max_entries_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SupportMemoryStore(
                Path(directory) / "memory.json",
                max_entries=2,
            )

            store.add("user-001", "问题一", "回答一")
            store.add("user-001", "问题二", "回答二")
            store.add("user-001", "问题三", "回答三")

            result = store.search(
                "user-001",
                limit=10,
            )

            self.assertEqual(
                len(result),
                2,
            )
            self.assertEqual(
                result[0]["query"],
                "问题三",
            )

    def test_users_are_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SupportMemoryStore(
                Path(directory) / "memory.json",
            )

            store.add(
                "user-001",
                "私人问题",
                "私人回答",
            )

            self.assertEqual(
                store.search(
                    "user-002",
                    limit=10,
                ),
                [],
            )

    def test_clear_removes_only_target_user(self):
        with tempfile.TemporaryDirectory() as directory:
            store = SupportMemoryStore(
                Path(directory) / "memory.json",
            )

            store.add("user-001", "问题一", "回答一")
            store.add("user-002", "问题二", "回答二")

            self.assertEqual(
                store.clear("user-001"),
                1,
            )
            self.assertEqual(
                store.search("user-001"),
                [],
            )
            self.assertEqual(
                len(store.search("user-002")),
                1,
            )


if __name__ == "__main__":
    unittest.main()