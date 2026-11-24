from __future__ import annotations

from hashlib import blake2b
import math
import re
from typing import Any, Protocol, Sequence


class EmbeddingError(RuntimeError):
    """文档嵌入服务配置或调用错误。"""


class EmbeddingProvider(Protocol):
    name: str

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        ...

    def embed_query(self, text: str) -> list[float]:
        ...


class LocalHashEmbedding:
    """无需外部服务的确定性词项向量，用作本地 RAG 降级方案。"""

    name = "local-hash"

    def __init__(self, dimensions: int = 256) -> None:
        if type(dimensions) is not int or dimensions < 32:
            raise ValueError("本地嵌入维度必须是不小于 32 的整数。")
        self.dimensions = dimensions

    @staticmethod
    def _terms(text: str) -> list[str]:
        terms: list[str] = []
        for fragment in re.findall(
            r"[a-z0-9_]+|[\u4e00-\u9fff]+",
            str(text or "").lower(),
        ):
            if re.fullmatch(r"[\u4e00-\u9fff]+", fragment):
                terms.append(fragment)
                terms.extend(
                    fragment[index:index + 2]
                    for index in range(max(len(fragment) - 1, 0))
                )
            else:
                terms.append(fragment)
        return terms

    def _embed(self, text: str) -> list[float]:
        values = [0.0] * self.dimensions
        for term in self._terms(text):
            digest = blake2b(term.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest, "big") % self.dimensions
            sign = 1.0 if digest[0] & 1 else -1.0
            values[bucket] += sign * (1.0 + math.log1p(len(term)))

        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)


class OpenAIEmbedding:
    name = "openai-compatible"

    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "",
        dimensions: int | None = None,
    ) -> None:
        if not api_key.strip():
            raise EmbeddingError("远程嵌入服务缺少 API Key。")
        if not model.strip():
            raise EmbeddingError("远程嵌入服务缺少模型名称。")

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise EmbeddingError(
                "远程嵌入需要 openai 依赖，请先安装。"
            ) from exc

        client_options: dict[str, Any] = {"api_key": api_key.strip()}
        if base_url.strip():
            client_options["base_url"] = base_url.strip()
        self.client = OpenAI(**client_options)
        self.model = model.strip()
        self.dimensions = dimensions if dimensions and dimensions > 0 else None

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        options: dict[str, Any] = {
            "model": self.model,
            "input": list(texts),
        }
        if self.dimensions is not None:
            options["dimensions"] = self.dimensions

        try:
            response = self.client.embeddings.create(**options)
            data = sorted(response.data, key=lambda item: int(item.index))
            return [list(map(float, item.embedding)) for item in data]
        except Exception as exc:
            raise EmbeddingError(f"远程嵌入调用失败：{exc}") from exc

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        values = self._embed([text])
        return values[0] if values else []


def build_embedding_provider(settings: Any) -> EmbeddingProvider:
    provider = str(
        getattr(settings, "rag_embedding_provider", "local") or "local"
    ).strip().lower()

    if provider in {"", "local", "hash", "local-hash"}:
        dimensions = int(
            getattr(settings, "rag_embedding_dimensions", 256) or 256
        )
        return LocalHashEmbedding(dimensions)

    if provider in {"openai", "openai-compatible", "remote"}:
        model = str(
            getattr(settings, "rag_embedding_model", "")
            or getattr(settings, "embedding_model", "text-embedding-3-small")
        ).strip()
        dimensions = int(
            getattr(settings, "rag_embedding_dimensions", 0) or 0
        )
        return OpenAIEmbedding(
            api_key=str(getattr(settings, "rag_embedding_api_key", "") or ""),
            model=model,
            base_url=str(getattr(settings, "rag_embedding_base_url", "") or ""),
            dimensions=dimensions or None,
        )

    raise EmbeddingError(f"不支持的文档嵌入提供商：{provider}")


def describe_embedding_provider(settings: Any) -> dict[str, Any]:
    """返回嵌入配置的可运维摘要，不实例化远程客户端，也不暴露密钥。"""

    provider = str(
        getattr(settings, "rag_embedding_provider", "local") or "local"
    ).strip().lower()

    if provider in {"", "local", "hash", "local-hash"}:
        try:
            dimensions = int(
                getattr(settings, "rag_embedding_dimensions", 256) or 256
            )
            if dimensions < 32:
                raise ValueError
        except (TypeError, ValueError):
            return {
                "provider": "local-hash",
                "available": False,
                "dimensions": None,
                "reason": "本地嵌入维度必须是不小于 32 的整数。",
            }
        return {
            "provider": "local-hash",
            "available": True,
            "dimensions": dimensions,
            "reason": "",
        }

    if provider in {"openai", "openai-compatible", "remote"}:
        model = str(
            getattr(settings, "rag_embedding_model", "")
            or getattr(settings, "embedding_model", "text-embedding-3-small")
        ).strip()
        api_key = str(
            getattr(settings, "rag_embedding_api_key", "") or ""
        ).strip()
        dimensions_value = getattr(settings, "rag_embedding_dimensions", 0) or 0
        try:
            dimensions = int(dimensions_value) if int(dimensions_value) > 0 else None
        except (TypeError, ValueError):
            dimensions = None

        if not api_key:
            return {
                "provider": "openai-compatible",
                "available": False,
                "dimensions": dimensions,
                "reason": "远程嵌入服务缺少 API Key。",
            }
        if not model:
            return {
                "provider": "openai-compatible",
                "available": False,
                "dimensions": dimensions,
                "reason": "远程嵌入服务缺少模型名称。",
            }
        try:
            import openai  # noqa: F401
        except ImportError:
            return {
                "provider": "openai-compatible",
                "available": False,
                "dimensions": dimensions,
                "reason": "远程嵌入需要 openai 依赖，请先安装。",
            }
        return {
            "provider": "openai-compatible",
            "available": True,
            "dimensions": dimensions,
            "reason": "",
        }

    return {
        "provider": provider or "unknown",
        "available": False,
        "dimensions": None,
        "reason": f"不支持的文档嵌入提供商：{provider}。",
    }


__all__ = [
    "EmbeddingError",
    "EmbeddingProvider",
    "LocalHashEmbedding",
    "OpenAIEmbedding",
    "build_embedding_provider",
    "describe_embedding_provider",
]
