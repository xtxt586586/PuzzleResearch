from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import quote

import httpx

from .sse import normalize_sse_event, parse_sse_frame
from .types import (
    McpConfigResponse,
    ResearchTreeResponse,
    SandboxBrowserDiagnoseResponse,
    SessionStateResponse,
    StreamEvent,
)


@dataclass(frozen=True)
class PuzzleResearchApiError(RuntimeError):
    status: int
    path: str
    body_text: str

    def __str__(self) -> str:
        suffix = f": {self.body_text}" if self.body_text else ""
        return (
            f"PuzzleResearch API request failed "
            f"({self.status}) {self.path}{suffix}"
        )


def _normalize_base_url(raw: str) -> str:
    text = str(raw or "").strip()
    return (text or "http://127.0.0.1:8001").rstrip("/")


def _encode_path_param(value: str) -> str:
    return quote(str(value or ""), safe="")


class PuzzleResearchClient:
    def __init__(
        self,
        *,
        base_url: str = "http://127.0.0.1:8001",
        headers: dict[str, str] | None = None,
        timeout_s: float = 60.0,
        http: httpx.Client | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        self.headers = dict(headers or {})
        self.timeout_s = float(timeout_s)
        self._http = http or httpx.Client(timeout=self.timeout_s)
        self.last_thread_id: str | None = None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "PuzzleResearchClient":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    def _url(self, path: str) -> str:
        normalized = path if path.startswith("/") else f"/{path}"
        return f"{self.base_url}{normalized}"

    def _capture_thread_id(self, response: httpx.Response) -> None:
        self.last_thread_id = (
            response.headers.get("X-Thread-ID")
            or response.headers.get("x-thread-id")
            or None
        )

    def _raise_for_status(
        self,
        response: httpx.Response,
        path: str,
    ) -> None:
        if 200 <= response.status_code < 300:
            return
        raise PuzzleResearchApiError(
            status=response.status_code,
            path=path,
            body_text=response.text,
        )

    def request_json(
        self,
        path: str,
        *,
        method: str = "GET",
        json_body: Any = None,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> Any:
        merged_headers = {
            "Accept": "application/json",
            **self.headers,
            **(headers or {}),
        }
        response = self._http.request(
            method=method,
            url=self._url(path),
            headers=merged_headers,
            params=params,
            json=json_body,
        )
        self._raise_for_status(response, path)
        if not response.text.strip():
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def _iter_sse_response(
        self,
        response: httpx.Response,
        path: str,
    ) -> Iterator[StreamEvent]:
        self._raise_for_status(response, path)
        self._capture_thread_id(response)

        buffer = ""
        for chunk in response.iter_bytes():
            buffer += chunk.decode("utf-8", errors="ignore")
            buffer = buffer.replace("\r\n", "\n")
            frames = buffer.split("\n\n")
            buffer = frames.pop() or ""

            for frame in frames:
                parsed = parse_sse_frame(frame)
                if parsed is None:
                    continue
                normalized = normalize_sse_event(parsed)
                if normalized is not None:
                    yield normalized

        parsed = parse_sse_frame(buffer.strip())
        if parsed is not None:
            normalized = normalize_sse_event(parsed)
            if normalized is not None:
                yield normalized

    def chat_sse(
        self,
        payload: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            **self.headers,
        }
        with self._http.stream(
            "POST",
            self._url("/api/chat/sse"),
            headers=headers,
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ) as response:
            yield from self._iter_sse_response(
                response,
                "/api/chat/sse",
            )

    def research_stream(
        self,
        query: str,
    ) -> Iterator[StreamEvent]:
        headers = {
            "Accept": "text/event-stream",
            **self.headers,
        }
        path = "/api/research"
        with self._http.stream(
            "POST",
            self._url(path),
            headers=headers,
            params={"query": str(query or "")},
        ) as response:
            self._raise_for_status(response, path)
            self._capture_thread_id(response)

            buffer = ""
            for chunk in response.iter_bytes():
                buffer += chunk.decode("utf-8", errors="ignore")
                lines = buffer.split("\n")
                buffer = lines.pop() or ""
                for line in lines:
                    if not line.startswith("0:"):
                        continue
                    try:
                        event = json.loads(line[2:])
                    except ValueError:
                        continue
                    if isinstance(event, dict) and {
                        "type",
                        "data",
                    }.issubset(event):
                        yield event

            if buffer.startswith("0:"):
                try:
                    event = json.loads(buffer[2:])
                except ValueError:
                    event = None
                if isinstance(event, dict) and {
                    "type",
                    "data",
                }.issubset(event):
                    yield event

    def research_sse(
        self,
        payload: dict[str, Any],
    ) -> Iterator[StreamEvent]:
        headers = {
            "Accept": "text/event-stream",
            "Content-Type": "application/json",
            **self.headers,
        }
        path = "/api/research/sse"
        with self._http.stream(
            "POST",
            self._url(path),
            headers=headers,
            content=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        ) as response:
            yield from self._iter_sse_response(response, path)

    def stream_events(
        self,
        thread_id: str,
        *,
        last_event_id: int | str | None = None,
    ) -> Iterator[StreamEvent]:
        headers = {
            "Accept": "text/event-stream",
            **self.headers,
        }
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)

        safe_id = _encode_path_param(thread_id)
        path = f"/api/events/{safe_id}"
        with self._http.stream(
            "GET",
            self._url(path),
            headers=headers,
        ) as response:
            yield from self._iter_sse_response(response, path)

    def cancel_chat(self, thread_id: str) -> Any:
        safe_id = _encode_path_param(thread_id)
        return self.request_json(
            f"/api/chat/cancel/{safe_id}",
            method="POST",
        )

    def cancel_all_chats(self) -> Any:
        return self.request_json(
            "/api/chat/cancel-all",
            method="POST",
        )

    def list_sessions(
        self,
        *,
        limit: int | None = None,
        status: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {}
        if limit is not None:
            params["limit"] = int(limit)
        if status is not None:
            params["status"] = status
        return self.request_json(
            "/api/sessions",
            params=params or None,
        )

    def get_session(self, thread_id: str) -> Any:
        safe_id = _encode_path_param(thread_id)
        return self.request_json(f"/api/sessions/{safe_id}")

    def get_state(self, thread_id: str) -> Any:
        safe_id = _encode_path_param(thread_id)
        return self.request_json(f"/api/sessions/{safe_id}/state")

    def get_session_state(
        self,
        thread_id: str,
    ) -> SessionStateResponse:
        safe_id = _encode_path_param(
            thread_id
        )

        return self.request_json(
            f"/api/sessions/{safe_id}/state"
        )

    def get_evidence(self, thread_id: str) -> Any:
        safe_id = _encode_path_param(thread_id)
        return self.request_json(f"/api/sessions/{safe_id}/evidence")

    def get_research_tree(
        self,
        thread_id: str,
    ) -> ResearchTreeResponse:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/sessions/{safe_id}/research-tree"
        )

    def stream_research_events(
        self,
        thread_id: str,
        *,
        last_event_id: int | str | None = None,
    ) -> Iterator[StreamEvent]:
        yield from self.stream_events(
            thread_id,
            last_event_id=last_event_id,
        )

    def resume_session(
        self,
        thread_id: str,
        *,
        resume_value: Any = True,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)
        return self.request_json(
            f"/api/sessions/{safe_id}/resume",
            method="POST",
            json_body={"resume_value": resume_value},
        )

    def resume_interrupt(
        self,
        thread_id: str,
        payload: Any,
        *,
        model: str | None = None,
        search_mode: Any = None,
        agent_id: str | None = None,
    ) -> Any:
        body: dict[str, Any] = {
            "thread_id": thread_id,
            "payload": payload,
        }

        if model is not None:
            body["model"] = model

        if search_mode is not None:
            body["search_mode"] = search_mode

        if agent_id is not None:
            body["agent_id"] = agent_id

        return self.request_json(
            "/api/interrupt/resume",
            method="POST",
            json_body=body,
        )


    def get_mcp_config(
        self,
    ) -> McpConfigResponse:
        return self.request_json(
            "/api/mcp/config"
        )


    def update_mcp_config(
        self,
        *,
        enable: bool | None = None,
        servers: dict[str, Any] | None = None,
    ) -> McpConfigResponse:
        body: dict[str, Any] = {}

        if enable is not None:
            body["enable"] = bool(enable)

        if servers is not None:
            body["servers"] = dict(servers)

        return self.request_json(
            "/api/mcp/config",
            method="POST",
            json_body=body,
        )


    def diagnose_browser(
        self,
        *,
        deep: bool = False,
    ) -> SandboxBrowserDiagnoseResponse:
        return self.request_json(
            "/api/sandbox/browser/diagnose",
            params={
                "deep": str(bool(deep)).lower(),
            },
        )


    def list_tools(self) -> Any:
        return self.request_json("/api/tools/registry")

