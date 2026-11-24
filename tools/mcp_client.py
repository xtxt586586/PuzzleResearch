"""Minimal MCP stdio client used by PuzzleResearch.

The implementation intentionally uses only the Python standard library.  MCP
is optional for PuzzleResearch, so importing this module must not require the
third-party ``mcp`` package or any network client.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import subprocess
import sys
import threading
from queue import Empty, Queue
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import urljoin, urlsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from agent.core.event_stream import get_emitter_sync
from tools.registry import ToolDefinition


class MCPError(RuntimeError):
    """Base error for MCP connection and tool failures."""


class MCPProtocolError(MCPError):
    """The MCP server returned an invalid or unsuccessful JSON-RPC response."""


class MCPToolError(MCPError):
    """An MCP tool reported a failed invocation."""


class _RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, _request, _fp, _code, _message, _headers, _new_url):
        return None


_SSE_OPENER = build_opener(_RejectRedirects())


@dataclass(frozen=True)
class MCPServerSpec:
    server_id: str
    transport: str = "stdio"
    command: str = ""
    args: tuple[str, ...] = ()
    env: Mapping[str, str] | None = None
    cwd: str | None = None
    url: str | None = None

    @classmethod
    def from_config(
        cls,
        server_id: str,
        config: Mapping[str, Any],
    ) -> "MCPServerSpec":
        configured_type = config.get("type")
        inferred_type = (
            "sse"
            if not configured_type and config.get("url") and not config.get("command")
            else "stdio"
        )
        transport = str(configured_type or inferred_type).strip().lower()
        if transport == "sse":
            url = str(config.get("url", "")).strip()
            if not url:
                raise ValueError(f"MCP 服务器「{server_id}」缺少 url。")
            return cls(
                server_id=str(server_id),
                transport="sse",
                url=url,
            )
        if transport != "stdio":
            raise ValueError(
                f"MCP 服务器「{server_id}」不支持连接类型：{transport}。"
            )

        raw_command = config.get("command")
        if isinstance(raw_command, (list, tuple)):
            parts = [str(item) for item in raw_command if str(item)]
            if not parts:
                raise ValueError(f"MCP 服务器「{server_id}」缺少 command。")
            command = parts[0]
            command_args = parts[1:]
        else:
            command = str(raw_command or "").strip()
            if not command:
                raise ValueError(f"MCP 服务器「{server_id}」缺少 command。")
            command_args = []

        raw_args = config.get("args", [])
        if raw_args is None:
            raw_args = []
        if not isinstance(raw_args, (list, tuple)):
            raise ValueError(f"MCP 服务器「{server_id}」的 args 必须是数组。")

        raw_env = config.get("env")
        if raw_env is not None and not isinstance(raw_env, Mapping):
            raise ValueError(f"MCP 服务器「{server_id}」的 env 必须是对象。")

        return cls(
            server_id=str(server_id),
            transport="stdio",
            command=command,
            args=tuple([*command_args, *(str(item) for item in raw_args)]),
            env=(
                {str(key): str(value) for key, value in raw_env.items()}
                if isinstance(raw_env, Mapping)
                else None
            ),
            cwd=(str(config["cwd"]) if config.get("cwd") else None),
        )


_SERVER_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$")
_ENV_NAME_PATTERN = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_BLOCKED_ENV_NAMES = {
    "PATH",
    "PYTHONPATH",
    "PYTHONHOME",
    "COMSPEC",
    "LD_PRELOAD",
    "DYLD_INSERT_LIBRARIES",
}


def _csv_values(value: str | list[str] | tuple[str, ...] | None) -> set[str]:
    if value is None:
        return set()
    values = value.split(",") if isinstance(value, str) else value
    return {str(item).strip() for item in values if str(item).strip()}


def validate_mcp_server_configs(
    servers: Mapping[str, Any],
    *,
    allowed_commands: str | list[str] | tuple[str, ...] | None = None,
    allowed_sse_hosts: str | list[str] | tuple[str, ...] | None = None,
    allowed_cwd_roots: str | list[str] | tuple[str, ...] | None = None,
) -> dict[str, dict[str, Any]]:
    """Validate untrusted MCP configuration before starting processes or requests."""

    if not isinstance(servers, Mapping):
        raise ValueError("MCP servers 配置必须是对象。")
    if len(servers) > 32:
        raise ValueError("MCP 服务器数量不能超过 32 个。")

    allowed_command_values = _csv_values(allowed_commands)
    command_names = {
        command.lower()
        for command in allowed_command_values
        if "/" not in command and "\\" not in command
    }
    command_paths = {
        os.path.normcase(os.path.realpath(command))
        for command in allowed_command_values
        if "/" in command or "\\" in command
    }
    sse_host_allowlist = {
        host.strip().strip("[]").lower()
        for host in _csv_values(allowed_sse_hosts)
    }
    cwd_roots = [
        os.path.realpath(path)
        for path in _csv_values(allowed_cwd_roots)
    ]

    normalized: dict[str, dict[str, Any]] = {}
    for raw_server_id, raw_config in servers.items():
        server_id = str(raw_server_id)
        if not _SERVER_ID_PATTERN.fullmatch(server_id):
            raise ValueError(
                f"MCP 服务器 ID「{server_id}」格式无效。"
            )
        if not isinstance(raw_config, Mapping):
            raise ValueError(f"MCP 服务器「{server_id}」配置必须是对象。")

        config = dict(raw_config)
        configured_type = config.get("type")
        transport = str(
            configured_type
            or ("sse" if config.get("url") and not config.get("command") else "stdio")
        ).strip().lower()

        if transport == "sse":
            unknown = set(config) - {"type", "url"}
            if unknown:
                raise ValueError(
                    f"MCP SSE 服务器「{server_id}」包含不支持的配置项。"
                )
            url = str(config.get("url", "")).strip()
            parts = urlsplit(url)
            host = (parts.hostname or "").lower()
            if (
                parts.scheme not in {"http", "https"}
                or not host
                or parts.username is not None
                or parts.password is not None
            ):
                raise ValueError(
                    f"MCP SSE 服务器「{server_id}」URL 必须是无凭据的 HTTP(S) URL。"
                )
            if host not in sse_host_allowlist:
                raise ValueError(
                    f"MCP SSE 主机「{host}」未列入 MCP_ALLOWED_SSE_HOSTS。"
                )
            config["type"] = "sse"
            normalized[server_id] = config
            continue

        if transport != "stdio":
            raise ValueError(
                f"MCP 服务器「{server_id}」连接类型无效：{transport}。"
            )

        unknown = set(config) - {"type", "command", "args", "env", "cwd"}
        if unknown:
            raise ValueError(
                f"MCP stdio 服务器「{server_id}」包含不支持的配置项。"
            )

        command = config.get("command")
        if not isinstance(command, str) or not command.strip():
            raise ValueError(f"MCP 服务器「{server_id}」缺少有效 command。")
        command_name = command.strip().replace("\\", "/").rsplit("/", 1)[-1].lower()
        requested_command = command.strip()
        has_path = "/" in requested_command or "\\" in requested_command
        command_allowed = (
            os.path.normcase(os.path.realpath(requested_command)) in command_paths
            if has_path
            else requested_command.lower() in command_names
        )
        if not command_allowed:
            raise ValueError(
                f"MCP 命令「{command_name}」未列入 MCP_ALLOWED_COMMANDS。"
            )

        args = config.get("args", [])
        if not isinstance(args, (list, tuple)) or len(args) > 128:
            raise ValueError(
                f"MCP 服务器「{server_id}」args 必须是最多 128 项的数组。"
            )
        if any(
            not isinstance(arg, str)
            or len(arg) > 4096
            or "\x00" in arg
            for arg in args
        ):
            raise ValueError(
                f"MCP 服务器「{server_id}」包含无效启动参数。"
            )

        env = config.get("env", {})
        if not isinstance(env, Mapping) or len(env) > 64:
            raise ValueError(
                f"MCP 服务器「{server_id}」env 必须是最多 64 项的对象。"
            )
        for key, value in env.items():
            env_name = str(key)
            if (
                not _ENV_NAME_PATTERN.fullmatch(env_name)
                or env_name.upper() in _BLOCKED_ENV_NAMES
                or not isinstance(value, str)
                or len(value) > 8192
                or "\x00" in value
            ):
                raise ValueError(
                    f"MCP 服务器「{server_id}」包含无效或禁止的环境变量。"
                )

        cwd = config.get("cwd")
        if cwd:
            if not cwd_roots:
                raise ValueError(
                    "设置 MCP stdio cwd 前，必须配置 MCP_ALLOWED_CWD_ROOTS。"
                )
            resolved_cwd = os.path.realpath(str(cwd))
            contained = False
            for root in cwd_roots:
                try:
                    if os.path.commonpath([resolved_cwd, root]) == root:
                        contained = True
                        break
                except ValueError:
                    continue
            if not os.path.isdir(resolved_cwd) or not contained:
                raise ValueError(
                    f"MCP 服务器「{server_id}」cwd 不在允许目录内。"
                )

        config["type"] = "stdio"
        normalized[server_id] = config

    return normalized


class _MCPConnection:
    """Shared JSON-RPC lifecycle for MCP transports."""

    def __init__(self, spec: MCPServerSpec) -> None:
        self.spec = spec
        self._request_id = 0
        self._lock = threading.RLock()

    def start(self) -> None:
        raise NotImplementedError

    @property
    def is_running(self) -> bool:
        raise NotImplementedError

    def _initialize_session(self) -> None:
        self._request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {
                    "name": "PuzzleResearch",
                    "version": "0.1.0",
                },
            },
        )
        self._notify("notifications/initialized", {})

    def list_tools(self) -> list[dict[str, Any]]:
        tools: list[dict[str, Any]] = []
        cursor: str | None = None

        while True:
            params: dict[str, Any] = {}
            if cursor:
                params["cursor"] = cursor
            result = self._request("tools/list", params)
            page = result.get("tools", [])
            if not isinstance(page, list):
                raise MCPProtocolError(
                    f"MCP 服务器「{self.spec.server_id}」的 tools/list 返回无效。"
                )
            tools.extend(
                item for item in page if isinstance(item, Mapping)
            )
            next_cursor = result.get("nextCursor")
            if not next_cursor:
                break
            cursor = str(next_cursor)

        return tools

    def call_tool(
        self,
        name: str,
        arguments: Mapping[str, Any],
    ) -> Any:
        result = self._request(
            "tools/call",
            {
                "name": name,
                "arguments": dict(arguments),
            },
        )

        if bool(result.get("isError")):
            raise MCPToolError(
                _content_to_text(result.get("content"))
                or f"MCP 工具「{name}」执行失败。"
            )

        if "structuredContent" in result:
            return result["structuredContent"]
        return _content_to_value(result.get("content", []))

    def _notify(self, method: str, params: Mapping[str, Any]) -> None:
        self._send_message(
            {
                "jsonrpc": "2.0",
                "method": method,
                "params": dict(params),
            }
        )

    def _request(self, method: str, params: Mapping[str, Any]) -> dict[str, Any]:
        with self._lock:
            if not self.is_running:
                raise MCPError(
                    f"MCP 服务器「{self.spec.server_id}」未运行。"
                )

            self._request_id += 1
            request_id = self._request_id
            self._send_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": dict(params),
                }
            )

            while True:
                response = self._receive_message()
                if response.get("id") != request_id:
                    # MCP notifications may be interleaved with a response.
                    continue
                if "error" in response:
                    error = response.get("error")
                    raise MCPProtocolError(
                        f"MCP 请求 {method} 失败：{_json_text(error)}"
                    )
                result = response.get("result")
                if not isinstance(result, Mapping):
                    raise MCPProtocolError(
                        f"MCP 请求 {method} 返回了无效结果。"
                    )
                return dict(result)

    def _send_message(self, message: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def _receive_message(self) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError


class MCPStdioConnection(_MCPConnection):
    """Synchronous MCP JSON-RPC connection over a child process stdio pipe."""

    def __init__(self, spec: MCPServerSpec) -> None:
        super().__init__(spec)
        self._process: subprocess.Popen[bytes] | None = None

    @property
    def is_running(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return

            environment = os.environ.copy()
            if self.spec.env:
                environment.update(self.spec.env)

            self._process = subprocess.Popen(
                [self.spec.command, *self.spec.args],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.spec.cwd,
                env=environment,
                bufsize=0,
            )

            try:
                self._initialize_session()
            except Exception:
                self.close()
                raise

    def close(self) -> None:
        with self._lock:
            process = self._process
            self._process = None
            if process is None:
                return

            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=2)

            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except OSError:
                        pass

    def _send_message(self, message: Mapping[str, Any]) -> None:
        process = self._process
        if process is None or process.stdin is None:
            raise MCPError("MCP stdio 写入端不可用。")
        body = json.dumps(
            message,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        try:
            process.stdin.write(header + body)
            process.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise MCPError("MCP stdio 写入失败。") from exc

    def _receive_message(self) -> dict[str, Any]:
        process = self._process
        if process is None or process.stdout is None:
            raise MCPError("MCP stdio 读取端不可用。")

        headers: dict[str, str] = {}
        while True:
            line = process.stdout.readline()
            if not line:
                raise MCPError(
                    f"MCP 服务器「{self.spec.server_id}」提前退出。"
                )
            stripped = line.strip()
            if not stripped:
                break
            if b":" not in stripped:
                continue
            key, value = stripped.split(b":", 1)
            headers[key.decode("ascii", errors="replace").lower()] = (
                value.decode("ascii", errors="replace").strip()
            )

        try:
            content_length = int(headers["content-length"])
        except (KeyError, ValueError) as exc:
            raise MCPProtocolError("MCP 响应缺少有效的 Content-Length。") from exc

        body = process.stdout.read(content_length)
        if len(body) != content_length:
            raise MCPProtocolError("MCP 响应内容长度不足。")
        try:
            payload = json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise MCPProtocolError("MCP 响应不是有效 JSON。") from exc
        if not isinstance(payload, dict):
            raise MCPProtocolError("MCP 响应必须是 JSON 对象。")
        return payload


class MCPSSEConnection(_MCPConnection):
    """Legacy MCP SSE transport used by Weaver-compatible servers."""

    def __init__(self, spec: MCPServerSpec) -> None:
        super().__init__(spec)
        self._stream: Any = None
        self._endpoint: str | None = None
        self._messages: Queue[dict[str, Any] | BaseException] = Queue()
        self._endpoint_ready = threading.Event()
        self._closed = threading.Event()
        self._reader: threading.Thread | None = None

    @property
    def is_running(self) -> bool:
        return (
            not self._closed.is_set()
            and self._stream is not None
            and self._endpoint is not None
        )

    def start(self) -> None:
        if not self.spec.url:
            raise MCPError("MCP SSE 缺少连接 URL。")

        request = Request(
            self.spec.url,
            headers={
                "Accept": "text/event-stream",
                "Cache-Control": "no-cache",
            },
            method="GET",
        )
        try:
            self._stream = _SSE_OPENER.open(request, timeout=10)
        except (HTTPError, URLError, OSError) as exc:
            raise MCPError(f"MCP SSE 连接失败：{exc}") from exc

        raw_socket = getattr(
            getattr(getattr(self._stream, "fp", None), "raw", None),
            "_sock",
            None,
        )
        if raw_socket is not None:
            raw_socket.settimeout(None)

        self._closed.clear()
        self._reader = threading.Thread(
            target=self._read_sse_loop,
            name=f"mcp-sse-{self.spec.server_id}",
            daemon=True,
        )
        self._reader.start()

        if not self._endpoint_ready.wait(timeout=10):
            self.close()
            raise MCPProtocolError("MCP SSE 未在规定时间内返回消息端点。")
        if self._endpoint is None:
            self.close()
            raise MCPProtocolError("MCP SSE 返回的消息端点为空。")

        try:
            self._initialize_session()
        except Exception:
            self.close()
            raise

    def _send_message(self, message: Mapping[str, Any]) -> None:
        endpoint = self._endpoint
        if not endpoint:
            raise MCPError("MCP SSE 消息端点不可用。")
        body = json.dumps(
            message,
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        request = Request(
            endpoint,
            data=body,
            headers={
                "Accept": "application/json, text/event-stream",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        try:
            with _SSE_OPENER.open(request, timeout=10) as response:
                direct_body = response.read()
        except (HTTPError, URLError, OSError) as exc:
            raise MCPError(f"MCP SSE 消息发送失败：{exc}") from exc

        # Some compatible servers return the JSON-RPC response directly from
        # POST instead of publishing it on the SSE stream.
        if direct_body:
            try:
                payload = json.loads(direct_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                payload = None
            if isinstance(payload, dict):
                self._messages.put(payload)

    def _receive_message(self) -> dict[str, Any]:
        try:
            payload = self._messages.get(timeout=30)
        except Empty as exc:
            raise MCPError("等待 MCP SSE 响应超时。") from exc
        if isinstance(payload, BaseException):
            raise MCPError(str(payload)) from payload
        return payload

    def _read_sse_loop(self) -> None:
        event_name = "message"
        data_lines: list[str] = []
        try:
            while not self._closed.is_set() and self._stream is not None:
                raw_line = self._stream.readline()
                if not raw_line:
                    raise MCPError("MCP SSE 服务器提前关闭连接。")
                line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
                if not line:
                    self._handle_sse_event(event_name, data_lines)
                    event_name = "message"
                    data_lines = []
                    continue
                if line.startswith(":"):
                    continue
                field, _, value = line.partition(":")
                if value.startswith(" "):
                    value = value[1:]
                if field == "event":
                    event_name = value
                elif field == "data":
                    data_lines.append(value)
        except BaseException as exc:
            if not self._closed.is_set():
                self._messages.put(exc)
                self._endpoint_ready.set()

    def _handle_sse_event(self, event_name: str, data_lines: list[str]) -> None:
        data = "\n".join(data_lines).strip()
        if not data:
            return
        if event_name == "endpoint" and self._endpoint is None:
            if self.spec.url:
                base = urlsplit(self.spec.url)
                endpoint = urlsplit(urljoin(self.spec.url, data))
                if (
                    endpoint.scheme != base.scheme
                    or endpoint.hostname != base.hostname
                    or endpoint.port != base.port
                    or endpoint.username is not None
                    or endpoint.password is not None
                ):
                    raise MCPProtocolError(
                        "MCP SSE 消息端点必须与 SSE 服务处于同一来源。"
                    )
                self._endpoint = endpoint.geturl()
            else:
                self._endpoint = data
            self._endpoint_ready.set()
            return
        try:
            payload = json.loads(data)
        except json.JSONDecodeError as exc:
            self._messages.put(MCPProtocolError("MCP SSE data 不是有效 JSON。"))
            raise exc
        if isinstance(payload, dict):
            self._messages.put(payload)

    def close(self) -> None:
        self._closed.set()
        stream = self._stream
        self._stream = None
        if stream is not None:
            # Closing HTTPResponse alone may leave a blocking readline alive
            # on Windows.  Close the underlying socket first to wake it up.
            raw = getattr(getattr(stream, "fp", None), "raw", None)
            socket = getattr(raw, "_sock", None)
            socket_closed = False
            if socket is not None:
                try:
                    socket.shutdown(2)
                except OSError:
                    pass
                try:
                    socket.close()
                    socket_closed = True
                except OSError:
                    pass
            if not socket_closed:
                try:
                    stream.close()
                except OSError:
                    pass
        reader = self._reader
        if reader is not None and reader is not threading.current_thread():
            reader.join(timeout=1)
        self._reader = None
        self._endpoint = None


def _json_text(value: Any) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except TypeError:
        return str(value)


def _content_to_text(content: Any) -> str:
    if not isinstance(content, list):
        return str(content) if content is not None else ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, Mapping):
            text = item.get("text")
            if isinstance(text, str):
                parts.append(text)
            elif item.get("type") == "resource":
                parts.append(_json_text(item))
        elif item is not None:
            parts.append(str(item))
    return "\n".join(part for part in parts if part)


def _content_to_value(content: Any) -> Any:
    if not isinstance(content, list):
        return content
    text = _content_to_text(content)
    if text:
        return text
    return content


def _safe_tool_name(server_id: str, tool_name: str) -> str:
    value = f"mcp_{server_id}_{tool_name}"
    return re.sub(r"[^A-Za-z0-9_-]+", "_", value)


class MCPClientManager:
    """Load multiple stdio MCP servers and expose proxy ToolDefinitions."""

    def __init__(self, *, thread_id: str = "default") -> None:
        self.thread_id = str(thread_id or "default")
        self.connections: dict[str, _MCPConnection] = {}
        self.tools: list[ToolDefinition] = []
        self.tool_servers: dict[str, str] = {}
        self.errors: dict[str, str] = {}

    @property
    def loaded_tool_count(self) -> int:
        return len(self.tools)

    def load(self, servers: Mapping[str, Any] | None) -> list[ToolDefinition]:
        self.close()
        self.errors = {}
        for server_id, raw_config in dict(servers or {}).items():
            if str(server_id).startswith("__"):
                continue
            if not isinstance(raw_config, Mapping):
                self.errors[str(server_id)] = "服务器配置必须是对象。"
                continue
            connection: _MCPConnection | None = None
            try:
                spec = MCPServerSpec.from_config(str(server_id), raw_config)
                connection = (
                    MCPSSEConnection(spec)
                    if spec.transport == "sse"
                    else MCPStdioConnection(spec)
                )
                connection.start()
                self.connections[spec.server_id] = connection
                for tool in connection.list_tools():
                    definition = self._build_tool(spec, connection, tool)
                    self.tools.append(definition)
                    self.tool_servers[definition.name] = spec.server_id
            except Exception as exc:
                self.errors[str(server_id)] = str(exc)
                if isinstance(connection, _MCPConnection):
                    connection.close()
                self.connections.pop(str(server_id), None)
                self.tools = [
                    tool
                    for tool in self.tools
                    if self.tool_servers.get(tool.name) != str(server_id)
                ]
                self.tool_servers = {
                    name: owner
                    for name, owner in self.tool_servers.items()
                    if owner != str(server_id)
                }
        return list(self.tools)

    def close(self) -> None:
        for connection in list(self.connections.values()):
            connection.close()
        self.connections.clear()
        self.tools.clear()
        self.tool_servers.clear()

    def tools_for_servers(
        self,
        server_ids: frozenset[str] | set[str] | None = None,
    ) -> list[ToolDefinition]:
        if server_ids is None:
            return list(self.tools)
        selected = {str(server_id) for server_id in server_ids}
        return [
            tool
            for tool in self.tools
            if self.tool_servers.get(tool.name) in selected
        ]

    def _build_tool(
        self,
        spec: MCPServerSpec,
        connection: _MCPConnection,
        tool: Mapping[str, Any],
    ) -> ToolDefinition:
        original_name = str(tool.get("name", "")).strip()
        if not original_name:
            raise MCPProtocolError(
                f"MCP 服务器「{spec.server_id}」返回了无名称工具。"
            )
        description = str(tool.get("description", "MCP 外部工具"))
        schema = tool.get("inputSchema")
        if not isinstance(schema, Mapping):
            schema = {
                "type": "object",
                "properties": {},
            }

        def invoke_mcp_tool(**arguments: Any) -> dict[str, Any]:
            tool_name = _safe_tool_name(spec.server_id, original_name)
            emitter = get_emitter_sync(self.thread_id)
            emitter.emit_sync(
                "tool_start",
                {
                    "tool": tool_name,
                    "server": spec.server_id,
                    "args": dict(arguments),
                },
            )
            try:
                result = connection.call_tool(original_name, arguments)
                emitter.emit_sync(
                    "tool_result",
                    {
                        "tool": tool_name,
                        "server": spec.server_id,
                        "result": result,
                        "success": True,
                    },
                )
                return {
                    "output": result,
                    "error": "",
                }
            except Exception as exc:
                emitter.emit_sync(
                    "tool_error",
                    {
                        "tool": tool_name,
                        "server": spec.server_id,
                        "error": str(exc),
                    },
                )
                raise MCPToolError(str(exc)) from exc

        return ToolDefinition(
            name=_safe_tool_name(spec.server_id, original_name),
            description=description,
            handler=invoke_mcp_tool,
            parameters_schema=dict(schema),
            requires_approval=True,
        )


__all__ = [
    "MCPClientManager",
    "MCPError",
    "MCPProtocolError",
    "MCPServerSpec",
    "MCPSSEConnection",
    "MCPStdioConnection",
    "MCPToolError",
    "validate_mcp_server_configs",
]
