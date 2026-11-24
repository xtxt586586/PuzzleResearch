from __future__ import annotations

import hashlib
from typing import Any

from common.config import settings
from tools.document_embeddings import build_embedding_provider
from tools.document_store import DocumentStoreError, LocalDocumentStore
from tools.registry import ToolDefinition


class DocumentSearchError(ValueError):
    """知识库搜索工具错误。"""


def _collection_name(owner_id: str | None = None) -> str:
    base = str(
        getattr(settings, "rag_collection_name", "puzzleresearch_documents")
        or "puzzleresearch_documents"
    ).strip()
    configured_api_key = (
        str(getattr(settings, "internal_api_key", "") or "").strip()
        or str(getattr(settings, "api_key", "") or "").strip()
    )
    owner = str(owner_id or "").strip()
    if configured_api_key and owner:
        suffix = hashlib.sha256(owner.encode("utf-8")).hexdigest()[:12]
        return f"{base}__u_{suffix}"
    return base


def _store(owner_id: str | None = None) -> LocalDocumentStore:
    return LocalDocumentStore(
        root=str(getattr(settings, "rag_store_path", "./data/chroma") or "./data/chroma"),
        collection_name=_collection_name(owner_id),
        embedding_provider=build_embedding_provider(settings),
    )


def rag_search(
    query: str,
    n_results: int = 5,
    *,
    owner_id: str | None = None,
) -> str:
    """搜索当前用户知识库中的文档片段并返回带来源的摘要。"""

    if not bool(getattr(settings, "rag_enabled", False)):
        return "RAG 未启用。请先在配置中启用文档知识库。"

    normalized_query = str(query or "").strip()
    if not normalized_query:
        raise DocumentSearchError("搜索内容不能为空。")
    if type(n_results) is not int or not 1 <= n_results <= 50:
        raise DocumentSearchError("n_results 必须是 1 到 50 之间的整数。")

    try:
        results = _store(owner_id).search(normalized_query, n_results)
    except DocumentStoreError as exc:
        raise DocumentSearchError(str(exc)) from exc

    if not results:
        return "未找到与查询相关的知识库文档。"

    parts: list[str] = []
    for index, result in enumerate(results, 1):
        source = str(result.get("source") or "unknown")
        score = float(result.get("score") or 0.0)
        content = str(result.get("text") or "").strip()
        parts.append(
            f"[{index}] 来源：{source}（相关度：{score:.2f}）\n{content[:1000]}"
        )
    return "\n\n---\n\n".join(parts)


def build_document_search_tool_definition(
    *,
    owner_id: str | None = None,
) -> ToolDefinition:
    def search_documents(query: str, n_results: int = 5) -> str:
        return rag_search(query, n_results, owner_id=owner_id)

    return ToolDefinition(
        "rag_search",
        "搜索当前用户文档知识库中的相关片段，并返回来源和内容摘要。",
        search_documents,
    )


__all__ = [
    "DocumentSearchError",
    "build_document_search_tool_definition",
    "rag_search",
]
