import json
import tempfile
import unittest
from pathlib import Path

from tools.document_embeddings import (
    EmbeddingError,
    LocalHashEmbedding,
    build_embedding_provider,
    describe_embedding_provider,
)
from tools.document_store import LocalDocumentStore


class _Settings:
    rag_embedding_provider = "local"
    rag_embedding_dimensions = 64
    rag_embedding_model = ""
    rag_embedding_api_key = ""
    rag_embedding_base_url = ""
    embedding_model = "text-embedding-3-small"


class _FakeEmbedding:
    name = "fake"

    def embed_documents(self, texts):
        return [
            [1.0, 0.0] if "preferred" in text else [0.0, 1.0]
            for text in texts
        ]

    def embed_query(self, text):
        return [1.0, 0.0]


class DocumentEmbeddingTests(unittest.TestCase):
    def test_local_embedding_is_deterministic_and_normalized(self):
        provider = LocalHashEmbedding(64)
        first = provider.embed_query("深度研究 evidence")
        second = provider.embed_query("深度研究 evidence")

        self.assertEqual(first, second)
        self.assertAlmostEqual(
            sum(value * value for value in first),
            1.0,
            places=6,
        )

    def test_local_provider_is_selected_by_default(self):
        provider = build_embedding_provider(_Settings())

        self.assertIsInstance(provider, LocalHashEmbedding)
        self.assertEqual(len(provider.embed_query("local")), 64)

    def test_remote_provider_requires_api_key(self):
        settings = _Settings()
        settings.rag_embedding_provider = "openai"

        with self.assertRaisesRegex(EmbeddingError, "API Key"):
            build_embedding_provider(settings)

    def test_local_provider_status_is_available_without_secrets(self):
        status = describe_embedding_provider(_Settings())

        self.assertEqual(status["provider"], "local-hash")
        self.assertTrue(status["available"])
        self.assertEqual(status["dimensions"], 64)
        self.assertEqual(status["reason"], "")

    def test_remote_provider_status_does_not_expose_api_key(self):
        settings = _Settings()
        settings.rag_embedding_provider = "openai"
        settings.rag_embedding_api_key = "secret-embedding-key"

        status = describe_embedding_provider(settings)
        serialized = str(status)

        self.assertNotIn("secret-embedding-key", serialized)
        self.assertIn("provider", status)
        self.assertIn("available", status)

    def test_store_persists_embeddings_and_uses_vector_similarity(self):
        with tempfile.TemporaryDirectory() as directory:
            store = LocalDocumentStore(
                Path(directory),
                "documents",
                embedding_provider=_FakeEmbedding(),
            )
            store.add_document(
                "preferred.txt",
                "完全不同的文本 preferred".encode("utf-8"),
            )
            store.add_document(
                "other.txt",
                "另一个完全不同的文本".encode("utf-8"),
            )

            state = json.loads(
                (Path(directory) / "documents" / "documents.json").read_text(
                    encoding="utf-8",
                )
            )
            self.assertEqual(state["chunks"][0]["embedding_provider"], "fake")
            self.assertEqual(len(state["chunks"][0]["embedding"]), 2)

            results = store.search("没有共同关键词的查询", n_results=2)

        self.assertEqual(results[0]["source"], "preferred.txt")


if __name__ == "__main__":
    unittest.main()
