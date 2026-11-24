from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from queue import Empty, Queue
import sys
import threading
import unittest

from tools.mcp_client import (
    MCPClientManager,
    MCPServerSpec,
    MCPSSEConnection,
    MCPToolError,
    validate_mcp_server_configs,
)


FIXTURE = Path(__file__).parent / "fixtures" / "mcp_echo_server.py"


class _SSEHandler(BaseHTTPRequestHandler):
    def log_message(self, *_args):
        return

    def do_GET(self):
        if self.path != "/sse":
            self.send_error(404)
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(b"event: endpoint\ndata: /messages\n\n")
        self.wfile.flush()

        while not self.server.stop_event.is_set():
            try:
                payload = self.server.responses.get(timeout=0.1)
            except Empty:
                continue
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            try:
                self.wfile.write(
                    b"event: message\ndata: "
                    + body
                    + b"\n\n"
                )
                self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                return

    def do_POST(self):
        if self.path != "/messages":
            self.send_error(404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        request = json.loads(self.rfile.read(length).decode("utf-8"))
        self.server.received.append(request)
        method = request.get("method")
        request_id = request.get("id")

        if request_id is None:
            result = None
        elif method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "sse-test", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "SSE 回显",
                        "inputSchema": {
                            "type": "object",
                            "properties": {"text": {"type": "string"}},
                            "required": ["text"],
                        },
                    }
                ]
            }
        elif method == "tools/call":
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": request["params"]["arguments"]["text"],
                    }
                ]
            }
        else:
            result = None

        if request_id is not None:
            self.server.responses.put(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "result": result,
                }
            )

        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()


class _SSETestServer:
    def __init__(self):
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), _SSEHandler)
        self.httpd.daemon_threads = True
        self.httpd.responses = Queue()
        self.httpd.received = []
        self.httpd.stop_event = threading.Event()
        self.thread = threading.Thread(
            target=self.httpd.serve_forever,
            daemon=True,
        )
        self.thread.start()

    @property
    def url(self) -> str:
        port = self.httpd.server_address[1]
        return f"http://127.0.0.1:{port}/sse"

    def close(self):
        self.httpd.stop_event.set()
        self.httpd.shutdown()
        self.httpd.server_close()
        self.thread.join(timeout=2)


class MCPClientTests(unittest.TestCase):
    def create_manager(self) -> MCPClientManager:
        manager = MCPClientManager(thread_id="mcp-test-thread")
        manager.load(
            {
                "demo": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(FIXTURE)],
                }
            }
        )
        self.addCleanup(manager.close)
        return manager

    def test_stdio_server_initializes_and_lists_prefixed_tools(self):
        manager = self.create_manager()

        self.assertEqual(manager.loaded_tool_count, 3)
        self.assertEqual(
            {tool.name for tool in manager.tools},
            {
                "mcp_demo_echo",
                "mcp_demo_add",
                "mcp_demo_fail",
            },
        )
        echo = next(tool for tool in manager.tools if tool.name == "mcp_demo_echo")
        self.assertEqual(
            echo.argument_schema(),
            [
                {
                    "name": "text",
                    "required": True,
                    "annotation": "string",
                    "description": "要回显的文本",
                }
            ],
        )

    def test_dynamic_tool_validates_json_schema_before_server_call(self):
        manager = self.create_manager()
        echo = next(tool for tool in manager.tools if tool.name == "mcp_demo_echo")

        with self.assertRaisesRegex(ValueError, "缺少必填参数"):
            echo.invoke({})
        with self.assertRaisesRegex(ValueError, "类型无效"):
            echo.invoke({"text": 42})

    def test_tool_call_forwards_arguments_and_structured_result(self):
        manager = self.create_manager()
        echo = next(tool for tool in manager.tools if tool.name == "mcp_demo_echo")
        add = next(tool for tool in manager.tools if tool.name == "mcp_demo_add")

        self.assertEqual(
            echo.invoke({"text": "hello"}),
            {"output": "hello", "error": ""},
        )
        self.assertEqual(
            add.invoke({"left": 2, "right": 3}),
            {"output": {"sum": 5}, "error": ""},
        )

    def test_mcp_tool_error_is_propagated(self):
        manager = self.create_manager()
        failing = next(tool for tool in manager.tools if tool.name == "mcp_demo_fail")

        with self.assertRaisesRegex(MCPToolError, "测试 MCP 工具错误"):
            failing.invoke({})

    def test_reload_and_close_remove_old_connections(self):
        manager = self.create_manager()
        self.assertEqual(len(manager.connections), 1)

        manager.load({})

        self.assertEqual(manager.loaded_tool_count, 0)
        self.assertEqual(manager.connections, {})

    def test_invalid_server_does_not_block_other_servers(self):
        manager = MCPClientManager()
        self.addCleanup(manager.close)

        manager.load(
            {
                "broken": {
                    "type": "stdio",
                    "command": "__missing_mcp_command__",
                },
                "demo": {
                    "type": "stdio",
                    "command": sys.executable,
                    "args": [str(FIXTURE)],
                },
            }
        )

        self.assertIn("broken", manager.errors)
        self.assertEqual(manager.loaded_tool_count, 3)

    def test_sse_server_initializes_lists_and_calls_tool(self):
        server = _SSETestServer()
        self.addCleanup(server.close)
        manager = MCPClientManager(thread_id="mcp-sse-test-thread")
        self.addCleanup(manager.close)

        manager.load(
            {
                "remote": {
                    "type": "sse",
                    "url": server.url,
                }
            }
        )

        self.assertEqual(manager.loaded_tool_count, 1)
        connection = manager.connections["remote"]
        stream_socket = connection._stream.fp.raw._sock
        self.assertIsNone(stream_socket.gettimeout())
        tool = manager.tools[0]
        self.assertEqual(tool.name, "mcp_remote_echo")
        self.assertEqual(
            tool.invoke({"text": "来自 SSE"}),
            {"output": "来自 SSE", "error": ""},
        )
        self.assertEqual(
            [request["method"] for request in server.httpd.received],
            [
                "initialize",
                "notifications/initialized",
                "tools/list",
                "tools/call",
            ],
        )

    def test_server_validation_requires_command_allowlist(self):
        config = {
            "demo": {
                "type": "stdio",
                "command": sys.executable,
                "args": [str(FIXTURE)],
            }
        }
        with self.assertRaisesRegex(ValueError, "MCP_ALLOWED_COMMANDS"):
            validate_mcp_server_configs(config)

        validated = validate_mcp_server_configs(
            config,
            allowed_commands=sys.executable,
        )
        self.assertEqual(validated["demo"]["type"], "stdio")

    def test_bare_command_allowlist_does_not_authorize_arbitrary_path(self):
        command_name = Path(sys.executable).name
        untrusted_path = str(Path(sys.executable).parent / "untrusted" / command_name)

        with self.assertRaisesRegex(ValueError, "MCP_ALLOWED_COMMANDS"):
            validate_mcp_server_configs(
                {"demo": {"command": untrusted_path}},
                allowed_commands=command_name,
            )

    def test_server_validation_rejects_untrusted_hosts_and_env(self):
        with self.assertRaisesRegex(ValueError, "未列入 MCP_ALLOWED_SSE_HOSTS"):
            validate_mcp_server_configs(
                {"remote": {"type": "sse", "url": "https://example.com/sse"}}
            )

        with self.assertRaisesRegex(ValueError, "禁止的环境变量"):
            validate_mcp_server_configs(
                {
                    "demo": {
                        "command": sys.executable,
                        "env": {"PYTHONPATH": "C:/untrusted"},
                    }
                },
                allowed_commands=sys.executable,
            )

    def test_sse_endpoint_cannot_change_origin(self):
        spec = MCPServerSpec.from_config(
            "remote",
            {"type": "sse", "url": "https://example.com/sse"},
        )
        connection = MCPSSEConnection(spec)

        with self.assertRaisesRegex(Exception, "同一来源"):
            connection._handle_sse_event(
                "endpoint",
                ["https://attacker.example/messages"],
            )

    def test_manager_can_select_tools_by_server(self):
        manager = MCPClientManager()
        self.addCleanup(manager.close)
        server = {
            "type": "stdio",
            "command": sys.executable,
            "args": [str(FIXTURE)],
        }
        manager.load({"alpha": server, "beta": server})

        self.assertEqual(manager.loaded_tool_count, 6)
        self.assertEqual(len(manager.tools_for_servers({"alpha"})), 3)
        self.assertEqual(len(manager.tools_for_servers(set())), 0)


if __name__ == "__main__":
    unittest.main()
