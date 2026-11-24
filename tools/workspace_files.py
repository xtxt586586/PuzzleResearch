from __future__ import annotations

import base64
import binascii
import codecs
import hashlib
import os
import re
import tempfile
import threading
from pathlib import Path, PurePosixPath
from typing import Any

from tools.registry import ToolDefinition


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT / "data" / "agent_workspaces"
MAX_FILE_BYTES = 1_000_000
MAX_BINARY_DOWNLOAD_CHUNK_BYTES = 12_000
DEFAULT_READ_CHARS = 4_000
MAX_READ_CHARS = 12_000
MAX_LIST_FILES = 200
MAX_PATH_CHARS = 1_000
_WINDOWS_RESERVED_NAMES = {
    "CON", "PRN", "AUX", "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


class WorkspaceFileError(ValueError):
    pass


class WorkspaceFileNotFoundError(WorkspaceFileError):
    pass


class WorkspaceFileConflictError(WorkspaceFileError):
    pass


def _validate_relative_path(value: str, *, allow_root: bool = False) -> tuple[str, ...]:
    raw = str(value or "").strip().replace("\\", "/")
    if allow_root and raw in {"", "."}:
        return ()
    if not raw or len(raw) > MAX_PATH_CHARS or "\x00" in raw:
        raise WorkspaceFileError("文件路径不能为空且长度不能超过 1000 个字符。")

    raw_parts = raw.split("/")
    path = PurePosixPath(raw)
    parts = path.parts
    if (
        path.is_absolute()
        or not parts
        or any(part in {"", ".", ".."} for part in raw_parts)
        or any(part in {".", ".."} for part in parts)
        or any(":" in part or part.endswith((".", " ")) for part in parts)
        or any(re.search(r'[<>:"|?*\x00-\x1f]', part) for part in parts)
        or any(
            part.split(".", 1)[0].upper() in _WINDOWS_RESERVED_NAMES
            for part in parts
        )
    ):
        raise WorkspaceFileError("只允许使用工作区内的相对文件路径。")
    return parts


class SessionWorkspace:
    def __init__(
        self,
        thread_id: str,
        *,
        storage_root: str | Path = WORKSPACE_ROOT,
    ) -> None:
        normalized_thread_id = str(thread_id or "").strip()
        if not normalized_thread_id:
            raise WorkspaceFileError("thread_id 不能为空。")
        safe_id = hashlib.sha256(normalized_thread_id.encode("utf-8")).hexdigest()
        self.root = Path(storage_root).expanduser().resolve() / safe_id
        self._lock = threading.RLock()

    def _target(self, path: str, *, allow_root: bool = False) -> Path:
        parts = _validate_relative_path(path, allow_root=allow_root)
        self.root.mkdir(parents=True, exist_ok=True)
        root = self.root.resolve()
        target = root.joinpath(*parts)

        current = root
        for part in parts:
            current = current / part
            if current.is_symlink():
                raise WorkspaceFileError("工作区路径不允许包含符号链接。")

        resolved = target.resolve()
        try:
            resolved.relative_to(root)
        except ValueError as exc:
            raise WorkspaceFileError("文件路径不能越出当前会话工作区。") from exc
        return target

    def resolve_directory(self, directory: str = ".") -> Path:
        """解析并校验工作区内的目录，供命令工具作为 cwd 使用。"""

        target = self._target(directory, allow_root=True)
        if not target.is_dir():
            raise WorkspaceFileNotFoundError("工作区目录不存在。")
        return target

    @staticmethod
    def _encoding(value: str) -> str:
        try:
            return codecs.lookup(str(value or "utf-8").strip()).name
        except LookupError as exc:
            raise WorkspaceFileError("文本编码名称无效。") from exc

    def _write_bytes(
        self,
        target: Path,
        content: bytes,
        *,
        overwrite: bool,
    ) -> None:
        with self._lock:
            if target.exists() and not overwrite:
                raise WorkspaceFileConflictError(
                    "文件已存在；如需覆盖请设置 overwrite=true。"
                )
            if target.exists() and not target.is_file():
                raise WorkspaceFileError("目标路径不是普通文件。")
            target.parent.mkdir(parents=True, exist_ok=True)
            descriptor, temporary_name = tempfile.mkstemp(
                prefix=".workspace-",
                suffix=".tmp",
                dir=target.parent,
            )
            temporary = Path(temporary_name)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(content)
                    handle.flush()
                    os.fsync(handle.fileno())
                if target.exists() and not overwrite:
                    raise WorkspaceFileConflictError(
                        "文件已存在；如需覆盖请设置 overwrite=true。"
                    )
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_bytes(target: Path) -> bytes:
        try:
            with target.open("rb") as handle:
                content = handle.read(MAX_FILE_BYTES + 1)
        except OSError as exc:
            raise WorkspaceFileError("工作区文件无法读取。") from exc
        if len(content) > MAX_FILE_BYTES:
            raise WorkspaceFileError(f"文件不能超过 {MAX_FILE_BYTES} 字节。")
        return content

    def write_file(
        self,
        path: str,
        content: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(content, str):
            raise WorkspaceFileError("文件内容必须是文本。")
        if "\x00" in content:
            raise WorkspaceFileError("不支持写入二进制内容。")
        if type(overwrite) is not bool:
            raise WorkspaceFileError("overwrite 必须是布尔值。")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise WorkspaceFileError(f"单个文件不能超过 {MAX_FILE_BYTES} 字节。")

        target = self._target(path)
        self._write_bytes(target, encoded, overwrite=overwrite)

        return {
            "success": True,
            "path": PurePosixPath(*_validate_relative_path(path)).as_posix(),
            "bytes": len(encoded),
            "output": f"已写入工作区文件：{path}",
            "error": "",
        }

    def update_file(self, path: str, content: str) -> dict[str, Any]:
        if not isinstance(content, str):
            raise WorkspaceFileError("文件内容必须是文本。")
        if "\x00" in content:
            raise WorkspaceFileError("不支持写入二进制内容。")
        encoded = content.encode("utf-8")
        if len(encoded) > MAX_FILE_BYTES:
            raise WorkspaceFileError(f"单个文件不能超过 {MAX_FILE_BYTES} 字节。")

        target = self._target(path)
        with self._lock:
            if not target.is_file():
                raise WorkspaceFileNotFoundError("要更新的工作区文件不存在。")
            self._write_bytes(target, encoded, overwrite=True)

        normalized_path = PurePosixPath(*_validate_relative_path(path)).as_posix()
        return {
            "success": True,
            "path": normalized_path,
            "bytes": len(encoded),
            "output": f"已更新工作区文件：{normalized_path}",
            "error": "",
        }

    def upload_text_file(self, filename: str, content: bytes) -> dict[str, Any]:
        if not isinstance(content, bytes):
            raise WorkspaceFileError("上传内容必须是文件字节。")
        try:
            text = content.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise WorkspaceFileError("仅支持 UTF-8 编码的文本文件。") from exc
        if "\x00" in text:
            raise WorkspaceFileError("不支持上传二进制文件。")
        return self.upload_file(filename, content)

    def upload_file(self, filename: str, content: bytes) -> dict[str, Any]:
        if not isinstance(filename, str) or not filename or filename != filename.strip():
            raise WorkspaceFileError("上传文件名无效。")
        parts = _validate_relative_path(filename)
        if len(parts) != 1:
            raise WorkspaceFileError("上传时只能选择单个文件，不能包含目录。")
        if not isinstance(content, bytes):
            raise WorkspaceFileError("上传内容必须是文件字节。")
        if len(content) > MAX_FILE_BYTES:
            raise WorkspaceFileError(f"单个文件不能超过 {MAX_FILE_BYTES} 字节。")
        target = self._target(parts[0])
        self._write_bytes(target, content, overwrite=False)
        normalized_path = PurePosixPath(*parts).as_posix()
        return {
            "success": True,
            "path": normalized_path,
            "bytes": len(content),
            "output": f"已上传工作区文件：{normalized_path}",
            "error": "",
        }

    def upload_binary_file(
        self,
        path: str,
        content_base64: str,
        *,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        if not isinstance(content_base64, str):
            raise WorkspaceFileError("content_base64 必须是 Base64 文本。")
        if type(overwrite) is not bool:
            raise WorkspaceFileError("overwrite 必须是布尔值。")
        if len(content_base64) > 4 * ((MAX_FILE_BYTES + 2) // 3):
            raise WorkspaceFileError(f"单个文件不能超过 {MAX_FILE_BYTES} 字节。")
        try:
            content = base64.b64decode(
                content_base64.encode("ascii"),
                validate=True,
            )
        except (UnicodeEncodeError, binascii.Error, ValueError) as exc:
            raise WorkspaceFileError("content_base64 不是有效的 Base64 内容。") from exc
        if len(content) > MAX_FILE_BYTES:
            raise WorkspaceFileError(f"单个文件不能超过 {MAX_FILE_BYTES} 字节。")

        target = self._target(path)
        self._write_bytes(target, content, overwrite=overwrite)
        normalized_path = PurePosixPath(*_validate_relative_path(path)).as_posix()
        return {
            "success": True,
            "path": normalized_path,
            "size": len(content),
            "output": f"已上传二进制文件：{normalized_path}（{len(content)} 字节）",
            "error": "",
        }

    def read_file(
        self,
        path: str,
        *,
        max_chars: int = DEFAULT_READ_CHARS,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        if type(max_chars) is not int or not 1 <= max_chars <= MAX_READ_CHARS:
            raise WorkspaceFileError(f"max_chars 必须是 1 到 {MAX_READ_CHARS} 之间的整数。")
        target = self._target(path)
        if not target.is_file():
            raise WorkspaceFileNotFoundError("工作区文件不存在。")
        try:
            content = self._read_bytes(target).decode(self._encoding(encoding))
        except UnicodeError as exc:
            raise WorkspaceFileError("文件无法按指定编码读取。") from exc

        truncated = len(content) > max_chars
        visible_content = content[:max_chars]
        return {
            "success": True,
            "path": PurePosixPath(*_validate_relative_path(path)).as_posix(),
            "content": visible_content,
            "truncated": truncated,
            "output": visible_content + ("\n...[内容已截断]" if truncated else ""),
            "error": "",
        }

    def list_files(
        self,
        directory: str = ".",
        *,
        max_files: int = 100,
        recursive: bool = True,
        max_depth: int = 3,
    ) -> dict[str, Any]:
        if type(max_files) is not int or not 1 <= max_files <= MAX_LIST_FILES:
            raise WorkspaceFileError(f"max_files 必须是 1 到 {MAX_LIST_FILES} 之间的整数。")
        if type(recursive) is not bool:
            raise WorkspaceFileError("recursive 必须是布尔值。")
        if type(max_depth) is not int or not 0 <= max_depth <= 20:
            raise WorkspaceFileError("max_depth 必须是 0 到 20 之间的整数。")
        target = self._target(directory, allow_root=True)
        if not target.is_dir():
            raise WorkspaceFileNotFoundError("工作区目录不存在。")

        entries: list[str] = []
        file_details: list[dict[str, Any]] = []
        directories: list[dict[str, str]] = []

        def visit(current: Path, depth: int) -> None:
            try:
                with os.scandir(current) as scan:
                    children = sorted(scan, key=lambda item: item.name)
            except OSError as exc:
                raise WorkspaceFileError("工作区目录无法读取。") from exc

            for child in children:
                child_path = Path(child.path)
                if child.is_symlink():
                    continue
                relative_path = child_path.relative_to(self.root).as_posix()
                if child.is_dir(follow_symlinks=False):
                    directories.append({
                        "path": relative_path,
                        "type": "directory",
                    })
                    if recursive and depth < max_depth:
                        visit(child_path, depth + 1)
                    continue
                if not child.is_file(follow_symlinks=False):
                    continue
                if len(entries) >= max_files:
                    return
                try:
                    size = child.stat(follow_symlinks=False).st_size
                except OSError as exc:
                    raise WorkspaceFileError("工作区文件信息无法读取。") from exc
                entries.append(relative_path)
                file_details.append({
                    "path": relative_path,
                    "type": "file",
                    "size": size,
                })

        visit(target, 0)
        entries.sort()
        file_details.sort(key=lambda entry: entry["path"])
        directories.sort(key=lambda entry: entry["path"])
        output_lines = [
            *(f"目录：{entry['path']}" for entry in directories),
            *(f"文件：{path}" for path in entries),
        ]
        return {
            "success": True,
            "files": entries,
            "file_details": file_details,
            "directories": directories,
            "total_files": len(entries),
            "total_dirs": len(directories),
            "output": "\n".join(output_lines) if output_lines else "工作区为空。",
            "error": "",
        }

    def str_replace(
        self,
        path: str,
        search: str,
        replace: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        if not isinstance(search, str) or not search:
            raise WorkspaceFileError("search 必须是非空文本。")
        if not isinstance(replace, str):
            raise WorkspaceFileError("replace 必须是文本。")
        if len(search.encode("utf-8")) > MAX_FILE_BYTES:
            raise WorkspaceFileError("search 内容过长。")
        if len(replace.encode("utf-8")) > MAX_FILE_BYTES:
            raise WorkspaceFileError("replace 内容过长。")
        target = self._target(path)
        if not target.is_file():
            raise WorkspaceFileError("工作区文件不存在。")
        codec = self._encoding(encoding)

        with self._lock:
            try:
                original = self._read_bytes(target).decode(codec)
            except UnicodeError as exc:
                raise WorkspaceFileError("文件无法按指定编码读取。") from exc
            count = original.count(search)
            if count == 0:
                raise WorkspaceFileError("待替换文本在文件中不存在。")
            if count > 1:
                raise WorkspaceFileError(
                    f"待替换文本出现 {count} 次；请提供更具体的文本，只替换唯一匹配项。"
                )
            updated = original.replace(search, replace, 1)
            encoded = updated.encode(codec)
            if len(encoded) > MAX_FILE_BYTES:
                raise WorkspaceFileError(f"替换后文件不能超过 {MAX_FILE_BYTES} 字节。")
            self._write_bytes(target, encoded, overwrite=True)

        return {
            "success": True,
            "path": PurePosixPath(*_validate_relative_path(path)).as_posix(),
            "replacements": 1,
            "output": f"已在 {path} 中替换 1 处。",
            "error": "",
        }

    def download_file(self, path: str) -> dict[str, Any]:
        target = self._target(path)
        if not target.is_file():
            raise WorkspaceFileNotFoundError("工作区文件不存在。")
        return {
            "path": PurePosixPath(*_validate_relative_path(path)).as_posix(),
            "filename": target.name,
            "content": self._read_bytes(target),
        }

    def download_file_chunk(
        self,
        path: str,
        *,
        offset: int = 0,
        max_bytes: int = MAX_BINARY_DOWNLOAD_CHUNK_BYTES,
    ) -> dict[str, Any]:
        if type(offset) is not int or offset < 0:
            raise WorkspaceFileError("offset 必须是非负整数。")
        if (
            type(max_bytes) is not int
            or not 1 <= max_bytes <= MAX_BINARY_DOWNLOAD_CHUNK_BYTES
        ):
            raise WorkspaceFileError(
                f"max_bytes 必须是 1 到 {MAX_BINARY_DOWNLOAD_CHUNK_BYTES} 之间的整数。"
            )
        target = self._target(path)
        if not target.is_file():
            raise WorkspaceFileNotFoundError("工作区文件不存在。")
        content = self._read_bytes(target)
        if offset > len(content):
            raise WorkspaceFileError("offset 超出文件长度。")

        chunk = content[offset:offset + max_bytes]
        next_offset = offset + len(chunk)
        return {
            "success": True,
            "path": PurePosixPath(*_validate_relative_path(path)).as_posix(),
            "content_base64": base64.b64encode(chunk).decode("ascii"),
            "size": len(content),
            "offset": offset,
            "next_offset": next_offset,
            "eof": next_offset >= len(content),
            "output": (
                f"已读取 {offset} 到 {next_offset} 字节；"
                f"文件共 {len(content)} 字节。"
            ),
            "error": "",
        }

    def delete_file(self, path: str) -> dict[str, Any]:
        target = self._target(path)
        with self._lock:
            if not target.exists():
                raise WorkspaceFileNotFoundError("工作区文件不存在。")
            if not target.is_file():
                raise WorkspaceFileError("只允许删除普通文件，不支持删除目录。")
            try:
                target.unlink()
            except FileNotFoundError as exc:
                raise WorkspaceFileNotFoundError("工作区文件不存在。") from exc
            except OSError as exc:
                raise WorkspaceFileError("工作区文件无法删除。") from exc

        normalized_path = PurePosixPath(*_validate_relative_path(path)).as_posix()
        return {
            "success": True,
            "path": normalized_path,
            "output": f"已删除工作区文件：{normalized_path}",
            "error": "",
        }


_workspaces: dict[tuple[str, str], SessionWorkspace] = {}
_workspaces_lock = threading.Lock()


def get_session_workspace(
    thread_id: str,
    *,
    storage_root: str | Path = WORKSPACE_ROOT,
) -> SessionWorkspace:
    root = str(Path(storage_root).expanduser().resolve())
    normalized_id = str(thread_id or "").strip()
    if not normalized_id:
        raise WorkspaceFileError("thread_id 不能为空。")
    key = (root, normalized_id)
    with _workspaces_lock:
        workspace = _workspaces.get(key)
        if workspace is None:
            workspace = SessionWorkspace(normalized_id, storage_root=root)
            _workspaces[key] = workspace
        return workspace


def build_workspace_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    workspace = get_session_workspace(thread_id)

    def write_file(path: str, content: str, overwrite: bool = False) -> dict[str, Any]:
        return workspace.write_file(path, content, overwrite=overwrite)

    def update_file(path: str, content: str) -> dict[str, Any]:
        return workspace.update_file(path, content)

    def read_file(
        path: str,
        max_chars: int = DEFAULT_READ_CHARS,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        return workspace.read_file(path, max_chars=max_chars, encoding=encoding)

    def list_files(
        directory: str = ".",
        recursive: bool = False,
        max_depth: int = 3,
        max_files: int = 100,
    ) -> dict[str, Any]:
        return workspace.list_files(
            directory,
            max_files=max_files,
            recursive=recursive,
            max_depth=max_depth,
        )

    def str_replace(
        path: str,
        search: str,
        replace: str,
        encoding: str = "utf-8",
    ) -> dict[str, Any]:
        return workspace.str_replace(path, search, replace, encoding)

    def delete_file(path: str) -> dict[str, Any]:
        return workspace.delete_file(path)

    def upload_file(
        path: str,
        content_base64: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return workspace.upload_binary_file(
            path,
            content_base64,
            overwrite=overwrite,
        )

    def download_file(
        path: str,
        offset: int = 0,
        max_bytes: int = MAX_BINARY_DOWNLOAD_CHUNK_BYTES,
    ) -> dict[str, Any]:
        return workspace.download_file_chunk(
            path,
            offset=offset,
            max_bytes=max_bytes,
        )

    return [
        ToolDefinition("write_file", "在当前会话的隔离工作区创建文本文件；默认不覆盖已有文件。", write_file),
        ToolDefinition(
            "update_file",
            "完整替换当前会话工作区中已有文件的内容；文件不存在时不会创建新文件。",
            update_file,
        ),
        ToolDefinition("read_file", "读取当前会话工作区中的文本文件。", read_file),
        ToolDefinition(
            "list_files",
            "列出当前会话工作区中的文件和目录；可选择递归列出并限制递归深度。",
            list_files,
        ),
        ToolDefinition(
            "str_replace",
            "仅当目标文本在当前会话工作区文件中恰好出现一次时替换；只能使用工作区相对路径。",
            str_replace,
        ),
        ToolDefinition(
            "delete_file",
            "删除当前会话工作区中的单个文件；不能删除目录或工作区外的路径。",
            delete_file,
        ),
        ToolDefinition(
            "upload_file",
            "使用 Base64 将二进制内容写入当前会话工作区；文件上限 1 MB，默认不覆盖已有文件。",
            upload_file,
        ),
        ToolDefinition(
            "download_file",
            "以 Base64 分块读取当前会话工作区文件；每块最多 12 KB，使用 next_offset 继续读取。",
            download_file,
        ),
    ]
