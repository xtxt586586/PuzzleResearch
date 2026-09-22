from __future__ import annotations

import json
import re
import tempfile
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

        if extension in {
            "pdf",
            "doc",
            "docx",
        }:
            raise DocumentStoreError(
                "PDF/DOC/DOCX 解析需要额外的可选依赖。"
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

        for index, chunk in enumerate(chunks):
            raw_chunks.append(
                {
                    "id": f"{document_id}-{index}",
                    "source": normalized_filename,
                    "chunk_index": index,
                    "text": chunk,
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
        scored: list[dict[str, Any]] = []

        terms = [
            item
            for item in re.findall(
                r"\w+",
                normalized_query,
            )
            if item
        ]

        for item in state.get(
            "chunks",
            [],
        ):
            text = str(
                item.get("text", "")
            )
            lowered_text = text.lower()
            score = 0.0

            if normalized_query in lowered_text:
                score += 2.0
                score += lowered_text.count(
                    normalized_query
                )

            for term in terms:
                if term in lowered_text:
                    score += 1.0

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
            key=lambda item: item["score"],
            reverse=True,
        )

        return scored[
            :max(min(int(n_results), 50), 1)
        ]