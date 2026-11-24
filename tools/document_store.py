from __future__ import annotations

from io import BytesIO
from collections import Counter
import json
import math
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tools.document_embeddings import (
    EmbeddingProvider,
    LocalHashEmbedding,
)


class DocumentStoreError(RuntimeError):
    pass


class LocalDocumentStore:
    ALLOWED_EXTENSIONS = {
        "txt",
        "md",
        "csv",
        "pdf",
        "doc",
        "docx",
    }

    def __init__(
        self,
        root: str | Path,
        collection_name: str,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        collection = re.sub(
            r"[^a-zA-Z0-9_.-]+",
            "_",
            collection_name,
        ).strip("._")

        self.directory = (
            Path(root).expanduser()
            / (collection or "documents")
        )
        self.state_path = (
            self.directory / "documents.json"
        )
        self.embedding_provider = (
            embedding_provider
            if embedding_provider is not None
            else LocalHashEmbedding()
        )

    def _read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {
                "documents": {},
                "chunks": [],
            }

        try:
            value = json.loads(
                self.state_path.read_text(
                    encoding="utf-8",
                )
            )
        except (
            OSError,
            json.JSONDecodeError,
        ) as exc:
            raise DocumentStoreError(
                "文档索引文件无法读取。"
            ) from exc

        if not isinstance(value, dict):
            raise DocumentStoreError(
                "文档索引格式无效。"
            )

        value.setdefault("documents", {})
        value.setdefault("chunks", [])
        return value

    def _write_state(
        self,
        state: dict[str, Any],
    ) -> None:
        self.directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=".documents-",
            suffix=".tmp",
            dir=self.directory,
        )

        temporary_path = Path(
            temporary_name
        )

        try:
            with open(
                descriptor,
                "w",
                encoding="utf-8",
                newline="\n",
            ) as handle:
                json.dump(
                    state,
                    handle,
                    ensure_ascii=False,
                    indent=2,
                )
                handle.write("\n")
                handle.flush()

            temporary_path.replace(
                self.state_path
            )
        finally:
            if temporary_path.exists():
                temporary_path.unlink()

    @staticmethod
    def _normalize_filename(
        filename: str,
    ) -> str:
        value = Path(
            str(filename or "")
        ).name.strip()

        if not value:
            raise DocumentStoreError(
                "文件名不能为空。"
            )

        extension = (
            value.rsplit(".", 1)[-1].lower()
            if "." in value
            else ""
        )

        if extension not in (
            LocalDocumentStore.ALLOWED_EXTENSIONS
        ):
            raise DocumentStoreError(
                f"不支持的文件类型：.{extension}"
            )

        return value

    @staticmethod
    def _decode_content(
        filename: str,
        content: bytes,
    ) -> str:
        extension = (
            filename.rsplit(".", 1)[-1].lower()
        )

        if extension == "pdf":
            try:
                import fitz

                with fitz.open(stream=content, filetype="pdf") as document:
                    if document.is_encrypted and not document.authenticate(""):
                        raise DocumentStoreError(
                            "PDF 已加密，无法提取文本。"
                        )
                    pages = [
                        f"[Page {page_number}]\n{text}"
                        for page_number, page in enumerate(document, 1)
                        if (text := page.get_text("text").strip())
                    ]
                return "\n\n".join(pages).strip()
            except DocumentStoreError:
                raise
            except ImportError as exc:
                raise DocumentStoreError(
                    "PDF 解析依赖缺失，请安装 PyMuPDF。"
                ) from exc
            except Exception as exc:
                raise DocumentStoreError(
                    "PDF 文件损坏或无法解析。"
                ) from exc

        if extension == "docx":
            try:
                from docx import Document

                document = Document(BytesIO(content))
                body: list[str] = []
                for block in document.iter_inner_content():
                    if hasattr(block, "text"):
                        text = str(block.text).strip()
                        if text:
                            body.append(text)
                    elif hasattr(block, "rows"):
                        for row in block.rows:
                            cells = [cell.text.strip() for cell in row.cells]
                            if any(cells):
                                body.append(" | ".join(cells))
                return "\n\n".join(body).strip()
            except ImportError as exc:
                raise DocumentStoreError(
                    "DOCX 解析依赖缺失，请安装 python-docx。"
                ) from exc
            except Exception as exc:
                raise DocumentStoreError(
                    "DOCX 文件损坏或无法解析。"
                ) from exc

        if extension == "doc":
            raise DocumentStoreError(
                "暂不支持旧版 .doc 文件，请另存为 .docx 后上传。"
            )

        try:
            return content.decode(
                "utf-8",
                errors="replace",
            ).strip()
        except Exception as exc:
            raise DocumentStoreError(
                "文档内容无法解析。"
            ) from exc

    @staticmethod
    def _chunks(
        text: str,
        chunk_size: int = 1200,
        overlap: int = 120,
    ) -> list[str]:
        normalized = re.sub(
            r"\r\n?",
            "\n",
            text,
        ).strip()

        if not normalized:
            return []

        result: list[str] = []
        start = 0
        step = max(
            chunk_size - overlap,
            1,
        )

        while start < len(normalized):
            chunk = normalized[
                start:start + chunk_size
            ].strip()

            if chunk:
                result.append(chunk)

            if start + chunk_size >= len(
                normalized
            ):
                break

            start += step

        return result

    def add_document(
        self,
        filename: str,
        content: bytes,
    ) -> dict[str, Any]:
        normalized_filename = (
            self._normalize_filename(filename)
        )
        text = self._decode_content(
            normalized_filename,
            content,
        )

        chunks = self._chunks(text)

        if not chunks:
            raise DocumentStoreError(
                "文档内容不能为空。"
            )

        try:
            embeddings = self.embedding_provider.embed_documents(chunks)
        except Exception as exc:
            raise DocumentStoreError(
                f"文档向量索引失败：{exc}"
            ) from exc

        if len(embeddings) != len(chunks):
            raise DocumentStoreError(
                "文档向量索引返回数量与文本块数量不一致。"
            )

        state = self._read_state()
        documents = state["documents"]
        raw_chunks = state["chunks"]

        raw_chunks = [
            item
            for item in raw_chunks
            if not isinstance(item, dict)
            or item.get("source")
            != normalized_filename
        ]

        document_id = uuid.uuid4().hex
        uploaded_at = datetime.now(
            timezone.utc
        ).isoformat()

        for index, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            raw_chunks.append(
                {
                    "id": f"{document_id}-{index}",
                    "source": normalized_filename,
                    "chunk_index": index,
                    "text": chunk,
                    "embedding": embedding,
                    "embedding_provider": self.embedding_provider.name,
                }
            )

        documents[normalized_filename] = {
            "source": normalized_filename,
            "filename": normalized_filename,
            "size": len(content),
            "chunks": len(chunks),
            "uploaded_at": uploaded_at,
        }

        state["documents"] = documents
        state["chunks"] = raw_chunks
        self._write_state(state)

        return {
            "success": True,
            "filename": normalized_filename,
            "chunks": len(chunks),
        }

    def list_documents(
        self,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        state = self._read_state()
        values = list(
            state["documents"].values()
        )

        values.sort(
            key=lambda item: str(
                item.get("uploaded_at", "")
            ),
            reverse=True,
        )

        return values[
            :max(min(int(limit), 500), 1)
        ]

    def count(self) -> int:
        state = self._read_state()
        return len(
            state.get("chunks", [])
        )

    def delete_document(
        self,
        source: str,
    ) -> dict[str, Any]:
        normalized_source = Path(
            str(source or "")
        ).name.strip()

        state = self._read_state()
        documents = state["documents"]

        if normalized_source not in documents:
            return {
                "success": False,
                "error": "文档不存在。",
            }

        documents.pop(
            normalized_source,
            None,
        )

        state["chunks"] = [
            item
            for item in state["chunks"]
            if item.get("source")
            != normalized_source
        ]

        state["documents"] = documents
        self._write_state(state)

        return {
            "success": True,
            "source": normalized_source,
        }

    @staticmethod
    def _search_terms(text: str) -> list[str]:
        """Extract Latin words and Chinese phrases for local ranked search."""

        terms: list[str] = []
        for fragment in re.findall(
            r"[a-z0-9_]+|[\u4e00-\u9fff]+",
            str(text or "").lower(),
        ):
            if re.fullmatch(r"[\u4e00-\u9fff]+", fragment):
                terms.append(fragment)
                if len(fragment) > 1:
                    terms.extend(
                        fragment[index:index + 2]
                        for index in range(len(fragment) - 1)
                    )
            else:
                terms.append(fragment)
        return terms

    def search(
        self,
        query: str,
        n_results: int = 5,
    ) -> list[dict[str, Any]]:
        normalized_query = str(
            query or ""
        ).strip().lower()

        if not normalized_query:
            raise DocumentStoreError(
                "搜索内容不能为空。"
            )

        state = self._read_state()
        chunks = [
            item
            for item in state.get("chunks", [])
            if isinstance(item, dict)
        ]
        query_terms = self._search_terms(normalized_query)
        if not query_terms:
            return []

        try:
            query_embedding = self.embedding_provider.embed_query(
                normalized_query
            )
        except Exception as exc:
            raise DocumentStoreError(
                f"查询向量生成失败：{exc}"
            ) from exc

        document_terms = [
            self._search_terms(str(item.get("text", "")))
            for item in chunks
        ]
        document_frequency = Counter(
            term
            for terms in document_terms
            for term in set(terms)
        )
        document_count = max(len(chunks), 1)
        inverse_frequency = {
            term: math.log(
                (1 + document_count) / (1 + frequency)
            ) + 1.0
            for term, frequency in document_frequency.items()
        }
        query_frequency = Counter(query_terms)
        query_weight = sum(
            inverse_frequency.get(term, 1.0) * count
            for term, count in query_frequency.items()
        ) or 1.0
        scored: list[dict[str, Any]] = []

        for item, terms in zip(chunks, document_terms):
            text = str(
                item.get("text", "")
            )
            lowered_text = text.lower()
            term_frequency = Counter(terms)
            matched_weight = sum(
                inverse_frequency.get(term, 1.0)
                * min(term_frequency.get(term, 0), 2)
                for term in query_frequency
                if term_frequency.get(term, 0) > 0
            )
            phrase_hits = lowered_text.count(normalized_query)
            lexical_score = 0.0
            if matched_weight > 0 or phrase_hits > 0:
                lexical_score = (
                    (matched_weight / query_weight) * 3.0
                    + min(phrase_hits, 3) * 0.35
                )

            vector_score = 0.0
            raw_embedding = item.get("embedding")
            if isinstance(raw_embedding, list) and query_embedding:
                try:
                    document_embedding = [float(value) for value in raw_embedding]
                    if len(document_embedding) == len(query_embedding):
                        numerator = sum(
                            left * right
                            for left, right in zip(
                                document_embedding,
                                query_embedding,
                            )
                        )
                        document_norm = math.sqrt(
                            sum(value * value for value in document_embedding)
                        )
                        query_norm = math.sqrt(
                            sum(value * value for value in query_embedding)
                        )
                        if document_norm and query_norm:
                            vector_score = max(
                                numerator / (document_norm * query_norm),
                                0.0,
                            )
                except (TypeError, ValueError):
                    vector_score = 0.0

            score = round(lexical_score + vector_score * 1.5, 6)

            if score <= 0:
                continue

            scored.append(
                {
                    "source": item.get(
                        "source",
                        "",
                    ),
                    "chunk_index": item.get(
                        "chunk_index",
                        0,
                    ),
                    "text": text,
                    "score": score,
                }
            )

        scored.sort(
            key=lambda item: (
                item["score"],
                str(item.get("source", "")),
                int(item.get("chunk_index", 0)),
            ),
            reverse=True,
        )

        return scored[
            :max(min(int(n_results), 50), 1)
        ]
