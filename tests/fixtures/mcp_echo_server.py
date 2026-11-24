"""Small dependency-free MCP stdio server used by integration tests."""

from __future__ import annotations

import json
import sys


def read_message() -> dict:
    headers: dict[str, str] = {}
    while True:
        line = sys.stdin.buffer.readline()
        if not line:
            raise EOFError
        line = line.strip()
        if not line:
            break
        key, value = line.split(b":", 1)
        headers[key.decode().lower()] = value.decode().strip()
    length = int(headers["content-length"])
    payload = sys.stdin.buffer.read(length)
    return json.loads(payload.decode("utf-8"))


def write_message(payload: dict) -> None:
    body = json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    sys.stdout.buffer.write(
        f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
        + body
    )
    sys.stdout.buffer.flush()


def main() -> None:
    while True:
        try:
            request = read_message()
        except EOFError:
            return

        method = request.get("method")
        request_id = request.get("id")
        if request_id is None:
            continue

        if method == "initialize":
            result = {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "mcp-echo-test", "version": "1.0"},
            }
        elif method == "tools/list":
            result = {
                "tools": [
                    {
                        "name": "echo",
                        "description": "回显输入文本",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "text": {
                                    "type": "string",
                                    "description": "要回显的文本",
                                }
                            },
                            "required": ["text"],
                        },
                    },
                    {
                        "name": "add",
                        "description": "计算两个整数之和",
                        "inputSchema": {
                            "type": "object",
                            "properties": {
                                "left": {"type": "integer"},
                                "right": {"type": "integer"},
                            },
                            "required": ["left", "right"],
                        },
                    },
                    {
                        "name": "fail",
                        "description": "返回 MCP 工具错误",
                        "inputSchema": {
                            "type": "object",
                            "properties": {},
                        },
                    },
                ]
            }
        elif method == "tools/call":
            params = request.get("params", {})
            name = params.get("name")
            arguments = params.get("arguments", {})
            if name == "echo":
                result = {
                    "content": [
                        {"type": "text", "text": str(arguments.get("text", ""))}
                    ]
                }
            elif name == "add":
                result = {
                    "structuredContent": {
                        "sum": arguments["left"] + arguments["right"]
                    }
                }
            elif name == "fail":
                result = {
                    "isError": True,
                    "content": [{"type": "text", "text": "测试 MCP 工具错误"}],
                }
            else:
                write_message(
                    {
                        "jsonrpc": "2.0",
                        "id": request_id,
                        "error": {"code": -32602, "message": "unknown tool"},
                    }
                )
                continue
        else:
            write_message(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {"code": -32601, "message": "method not found"},
                }
            )
            continue

        write_message({"jsonrpc": "2.0", "id": request_id, "result": result})


if __name__ == "__main__":
    main()
