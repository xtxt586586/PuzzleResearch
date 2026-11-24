import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.workflows.nodes import _get_agent_tool_registry
from tools.document_search import (
    DocumentSearchError,
    build_document_search_tool_definition,
    rag_search,
)
from tools.document_store import LocalDocumentStore


class DocumentSearchToolTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def _settings(
        self,
        *,
        enabled=True,
        authenticated=False,
        internal_authenticated=False,
    ):
        return patch.multiple(
            "tools.document_search.settings",
            rag_enabled=enabled,
            rag_store_path=self.temp_dir.name,
            rag_collection_name="documents",
            api_key="configured" if authenticated else "",
            internal_api_key="configured" if internal_authenticated else "",
        )

    def test_search_returns_source_and_content(self):
        LocalDocumentStore(str(self.root), "documents").add_document(
            "notes.txt",
            (
                "PuzzleResearch uses a local document knowledge base.\n"
                "The agent can search uploaded research notes."
            ).encode("utf-8"),
        )

        with self._settings():
            result = rag_search("knowledge base")

        self.assertIn("来源：notes.txt", result)
        self.assertIn("local document knowledge base", result)

    def test_search_uses_authenticated_owner_collection(self):
        owner_id = "owner-001"
        suffix = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:12]
        LocalDocumentStore(str(self.root), f"documents__u_{suffix}").add_document(
            "private.txt",
            "这是 owner-001 的私有研究资料。".encode("utf-8"),
        )

        with self._settings(authenticated=True):
            result = rag_search("私有研究", owner_id=owner_id)
            other = rag_search("私有研究", owner_id="owner-002")

        self.assertIn("来源：private.txt", result)
        self.assertIn("未找到", other)

    def test_agent_tool_uses_owner_collection_with_internal_api_key(self):
        owner_id = "owner-001"
        suffix = hashlib.sha256(owner_id.encode("utf-8")).hexdigest()[:12]
        LocalDocumentStore(
            str(self.root),
            f"documents__u_{suffix}",
        ).add_document(
            "internal-private.txt",
            "这是仅供 owner-001 使用的私有研究资料。".encode("utf-8"),
        )

        with self._settings(internal_authenticated=True):
            owner_registry = _get_agent_tool_registry(
                {
                    "configurable": {
                        "thread_id": "thread-owner-001",
                        "user_id": owner_id,
                    }
                }
            )
            other_registry = _get_agent_tool_registry(
                {
                    "configurable": {
                        "thread_id": "thread-owner-002",
                        "user_id": "owner-002",
                    }
                }
            )
            owner_result = owner_registry.get("rag_search").invoke(
                {"query": "私有研究资料"}
            )
            other_result = other_registry.get("rag_search").invoke(
                {"query": "私有研究资料"}
            )

        self.assertIn("来源：internal-private.txt", owner_result)
        self.assertIn("未找到", other_result)

    def test_disabled_and_invalid_queries_follow_contract(self):
        with self._settings(enabled=False):
            self.assertIn("RAG 未启用", rag_search("anything"))

        with self._settings():
            with self.assertRaises(DocumentSearchError):
                rag_search("   ")
            with self.assertRaises(DocumentSearchError):
                rag_search("anything", n_results=0)
            with self.assertRaises(DocumentSearchError):
                rag_search("anything", n_results=51)

    def test_tool_definition_exposes_machine_readable_contract(self):
        definition = build_document_search_tool_definition(owner_id="owner-001")

        self.assertEqual(definition.name, "rag_search")
        self.assertEqual(
            definition.format_argument_contract(),
            "query (必填), n_results (可选，默认=5)",
        )
        self.assertEqual(
            definition.validate_arguments({"query": "runtime"}),
            {"query": "runtime"},
        )

    def test_search_ranks_rare_multi_term_match_above_common_term_match(self):
        store = LocalDocumentStore(str(self.root), "documents")
        store.add_document(
            "common.txt",
            "研究系统包含证据记录、研究流程和研究工具。".encode("utf-8"),
        )
        store.add_document(
            "specific.txt",
            "研究系统的证据链需要来源注册和质量评估。".encode("utf-8"),
        )

        results = store.search("证据链 质量评估", n_results=2)

        self.assertEqual(results[0]["source"], "specific.txt")
        self.assertGreater(results[0]["score"], results[1]["score"])


if __name__ == "__main__":
    unittest.main()
