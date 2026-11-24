from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterator
from urllib.parse import quote

import httpx

from .sse import normalize_sse_event, parse_sse_frame
from .types import (
    AgentHealthResponse,
    ChatResponse,
    DocumentDeleteResponse,
    DocumentListResponse,
    DocumentSearchResponse,
    DocumentUploadResponse,
    EvidenceResponse,
    McpDiagnoseResponse,
    McpConfigResponse,
    ResearchTreeResponse,
    SandboxBrowserDiagnoseResponse,
    SessionStateResponse,
    StreamEvent,
    WorkspaceDeleteResponse,
    WorkspaceFilesResponse,
    WorkspaceUploadResponse,
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


def _encode_workspace_file_path(value: str) -> str:
    return "/".join(
        quote(part, safe="")
        for part in str(value or "").split("/")
    )


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

        self._capture_thread_id(response)
        self._raise_for_status(response, path)
        if not response.text.strip():
            return None
        try:
            return response.json()
        except ValueError:
            return response.text

    def request_bytes(
        self,
        path: str,
        *,
        method: str = "GET",
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
    ) -> tuple[bytes, str | None]:
        merged_headers = {
            "Accept": "*/*",
            **self.headers,
            **(headers or {}),
        }

        response = self._http.request(
            method=method,
            url=self._url(path),
            headers=merged_headers,
            params=params,
        )

        self._capture_thread_id(response)
        self._raise_for_status(response, path)

        return (
            response.content,
            response.headers.get("content-type"),
        )

    def request_multipart(
        self,
        path: str,
        *,
        filename: str,
        content: bytes,
        content_type: str | None = None,
        params: dict[str, Any] | None = None,
    ) -> Any:
        headers = {
            "Accept": "application/json",
            **self.headers,
        }

        files = {
            "file": (
                filename,
                content,
                content_type,
            ),
        }

        response = self._http.request(
            method="POST",
            url=self._url(path),
            headers=headers,
            params=params,
            files=files,
        )

        self._capture_thread_id(response)
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

    def chat(
        self,
        payload: dict[str, Any],
    ) -> ChatResponse:
        return self.request_json(
            "/api/chat",
            method="POST",
            json_body=payload,
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

    def cancel_chat(
        self,
        thread_id: str,
        *,
        reason: str | None = None,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        body = None

        if reason is not None:
            body = {
                "reason": reason,
            }

        return self.request_json(
            f"/api/chat/cancel/{safe_id}",
            method="POST",
            json_body=body,
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

    def list_workspace_files(
        self,
        thread_id: str,
    ) -> WorkspaceFilesResponse:
        safe_id = _encode_path_param(thread_id)
        return self.request_json(
            f"/api/sessions/{safe_id}/workspace/files"
        )

    def upload_workspace_file(
        self,
        thread_id: str,
        filename: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> WorkspaceUploadResponse:
        safe_id = _encode_path_param(thread_id)
        return self.request_multipart(
            f"/api/sessions/{safe_id}/workspace/files/upload",
            filename=filename,
            content=content,
            content_type=content_type,
        )

    def download_workspace_file(
        self,
        thread_id: str,
        file_path: str,
    ) -> tuple[bytes, str | None]:
        safe_id = _encode_path_param(thread_id)
        safe_path = _encode_workspace_file_path(file_path)
        return self.request_bytes(
            f"/api/sessions/{safe_id}/workspace/files/{safe_path}"
        )

    def delete_workspace_file(
        self,
        thread_id: str,
        file_path: str,
    ) -> WorkspaceDeleteResponse:
        safe_id = _encode_path_param(thread_id)
        safe_path = _encode_workspace_file_path(file_path)
        return self.request_json(
            f"/api/sessions/{safe_id}/workspace/files/{safe_path}",
            method="DELETE",
        )

    def get_evidence(self, thread_id: str) -> EvidenceResponse:
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


    def diagnose_mcp(self) -> McpDiagnoseResponse:
        return self.request_json(
            "/api/mcp/diagnose"
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

    def get_root(self) -> Any:
        return self.request_json("/")

    def get_health(self) -> Any:
        return self.request_json("/health")

    def get_metrics(self) -> str:
        return self.request_json("/metrics")

    def get_memory_status(self) -> Any:
        return self.request_json(
            "/api/memory/status"
        )

    def get_agent_health(self) -> AgentHealthResponse:
        return self.request_json(
            "/api/health/agent"
        )

    def get_public_config(self) -> Any:
        return self.request_json(
            "/api/config/public"
        )

    def refresh_tool_registry(
        self,
        *,
        reset: bool = False,
    ) -> Any:
        return self.request_json(
            "/api/tools/registry/refresh",
            method="POST",
            params={
                "reset": str(bool(reset)).lower(),
            },
        )

    def get_search_providers(self) -> Any:
        return self.request_json(
            "/api/search/providers"
        )

    def reset_search_providers(self) -> Any:
        return self.request_json(
            "/api/search/providers/reset",
            method="POST",
        )

    def get_search_cache_stats(self) -> Any:
        return self.request_json(
            "/api/search/cache/stats"
        )

    def clear_search_cache(self) -> Any:
        return self.request_json(
            "/api/search/cache/clear",
            method="POST",
        )

    def get_active_tasks(self) -> Any:
        return self.request_json(
            "/api/tasks/active"
        )

    def delete_session(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/sessions/{safe_id}",
            method="DELETE",
        )

    def get_interrupt_status(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/interrupt/{safe_id}/status"
        )

    def resume_thread_interrupt(
        self,
        thread_id: str,
        payload: dict[str, Any],
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/interrupt/{safe_id}/resume",
            method="POST",
            json_body=payload,
        )

    def list_runs(self) -> Any:
        return self.request_json(
            "/api/runs"
        )

    def get_run_metrics(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/runs/{safe_id}"
        )

    def get_trace(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/traces/{safe_id}"
        )

    def get_trace_summary(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/traces/{safe_id}/summary"
        )

    def get_all_traces(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/traces/{safe_id}/all"
        )

    def list_agents(self) -> Any:
        return self.request_json(
            "/api/agents"
        )

    def get_agent(
        self,
        agent_id: str,
    ) -> Any:
        safe_id = _encode_path_param(agent_id)

        return self.request_json(
            f"/api/agents/{safe_id}"
        )

    def create_agent(
        self,
        payload: dict[str, Any],
    ) -> Any:
        return self.request_json(
            "/api/agents",
            method="POST",
            json_body=payload,
        )

    def update_agent(
        self,
        agent_id: str,
        payload: dict[str, Any],
    ) -> Any:
        safe_id = _encode_path_param(agent_id)

        return self.request_json(
            f"/api/agents/{safe_id}",
            method="PUT",
            json_body=payload,
        )

    def delete_agent(
        self,
        agent_id: str,
    ) -> Any:
        safe_id = _encode_path_param(agent_id)

        return self.request_json(
            f"/api/agents/{safe_id}",
            method="DELETE",
        )

    def upload_document(
        self,
        filename: str,
        content: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> DocumentUploadResponse:
        return self.request_multipart(
            "/api/documents/upload",
            filename=filename,
            content=content,
            content_type=content_type,
        )

    def list_documents(
        self,
        *,
        limit: int = 100,
    ) -> DocumentListResponse:
        return self.request_json(
            "/api/documents/list",
            params={
                "limit": int(limit),
            },
        )

    def delete_document(
        self,
        source: str,
    ) -> DocumentDeleteResponse:
        safe_source = _encode_path_param(source)

        return self.request_json(
            f"/api/documents/{safe_source}",
            method="DELETE",
        )

    def search_documents(
        self,
        query: str,
        *,
        n_results: int = 5,
    ) -> DocumentSearchResponse:
        return self.request_json(
            "/api/documents/search",
            method="POST",
            params={
                "query": query,
                "n_results": int(n_results),
            },
        )

    def list_export_templates(self) -> Any:
        return self.request_json(
            "/api/export/templates"
        )

    def get_export_templates(self) -> Any:
        return self.list_export_templates()

    def export_report(
        self,
        thread_id: str,
        *,
        format: str = "html",
        template: str = "default",
        title: str | None = None,
    ) -> tuple[bytes, str | None]:
        safe_id = _encode_path_param(thread_id)

        params: dict[str, Any] = {
            "format": format,
            "template": template,
        }

        if title is not None:
            params["title"] = title

        return self.request_bytes(
            f"/api/export/{safe_id}",
            params=params,
        )

    def get_screenshot(
        self,
        filename: str,
    ) -> tuple[bytes, str | None]:
        safe_filename = _encode_path_param(filename)

        return self.request_bytes(
            f"/api/screenshots/{safe_filename}",
        )

    def list_screenshots(
        self,
        *,
        thread_id: str | None = None,
        limit: int = 50,
    ) -> Any:
        params: dict[str, Any] = {
            "limit": int(limit),
        }

        if thread_id is not None:
            params["thread_id"] = thread_id

        return self.request_json(
            "/api/screenshots",
            params=params,
        )

    def cleanup_screenshots(self) -> Any:
        return self.request_json(
            "/api/screenshots/cleanup",
            method="POST",
        )

    def get_browser_session_info(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/browser/{safe_id}/info"
        )

    def trigger_browser_screenshot(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/browser/{safe_id}/screenshot",
            method="POST",
        )

    def recognize_speech(
        self,
        payload: dict[str, Any],
    ) -> Any:
        return self.request_json(
            "/api/asr/recognize",
            method="POST",
            json_body=payload,
        )

    def recognize_speech_upload(
        self,
        filename: str,
        content: bytes,
        *,
        sample_rate: int = 16000,
        content_type: str = "audio/wav",
    ) -> Any:
        return self.request_multipart(
            "/api/asr/upload",
            filename=filename,
            content=content,
            content_type=content_type,
            params={
                "sample_rate": int(sample_rate),
            },
        )

    def get_asr_status(self) -> Any:
        return self.request_json(
            "/api/asr/status"
        )

    def synthesize_speech(
        self,
        payload: dict[str, Any],
    ) -> Any:
        return self.request_json(
            "/api/tts/synthesize",
            method="POST",
            json_body=payload,
        )

    def get_tts_voices(self) -> Any:
        return self.request_json(
            "/api/tts/voices"
        )

    def get_tts_status(self) -> Any:
        return self.request_json(
            "/api/tts/status"
        )

    def support_chat(
        self,
        payload: dict[str, Any],
    ) -> Any:
        return self.request_json(
            "/api/support/chat",
            method="POST",
            json_body=payload,
        )

    def create_scheduled_trigger(
        self,
        payload: dict[str, Any],
    ) -> Any:
        return self.request_json(
            "/api/triggers/scheduled",
            method="POST",
            json_body=payload,
        )

    def create_webhook_trigger(
        self,
        payload: dict[str, Any],
    ) -> Any:
        return self.request_json(
            "/api/triggers/webhook",
            method="POST",
            json_body=payload,
        )

    def create_event_trigger(
        self,
        payload: dict[str, Any],
    ) -> Any:
        return self.request_json(
            "/api/triggers/event",
            method="POST",
            json_body=payload,
        )

    def list_triggers(
        self,
        *,
        trigger_type: str | None = None,
        status: str | None = None,
    ) -> Any:
        params: dict[str, Any] = {}

        if trigger_type is not None:
            params["trigger_type"] = trigger_type

        if status is not None:
            params["status"] = status

        return self.request_json(
            "/api/triggers",
            params=params or None,
        )

    def get_trigger(
        self,
        trigger_id: str,
    ) -> Any:
        safe_id = _encode_path_param(trigger_id)

        return self.request_json(
            f"/api/triggers/{safe_id}"
        )

    def delete_trigger(
        self,
        trigger_id: str,
    ) -> Any:
        safe_id = _encode_path_param(trigger_id)

        return self.request_json(
            f"/api/triggers/{safe_id}",
            method="DELETE",
        )

    def pause_trigger(
        self,
        trigger_id: str,
    ) -> Any:
        safe_id = _encode_path_param(trigger_id)

        return self.request_json(
            f"/api/triggers/{safe_id}/pause",
            method="POST",
        )

    def resume_trigger(
        self,
        trigger_id: str,
    ) -> Any:
        safe_id = _encode_path_param(trigger_id)

        return self.request_json(
            f"/api/triggers/{safe_id}/resume",
            method="POST",
        )

    def get_trigger_executions(
        self,
        trigger_id: str,
        *,
        limit: int = 50,
    ) -> Any:
        safe_id = _encode_path_param(trigger_id)

        return self.request_json(
            f"/api/triggers/{safe_id}/executions",
            params={
                "limit": int(limit),
            },
        )

    def invoke_webhook(
        self,
        trigger_id: str,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        safe_id = _encode_path_param(trigger_id)

        return self.request_json(
            f"/api/webhook/{safe_id}",
            method="POST",
            json_body=payload or {},
        )

    def create_share(
        self,
        thread_id: str,
        *,
        permissions: str = "view",
        expires_hours: int | None = 72,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/sessions/{safe_id}/share",
            method="POST",
            json_body={
                "permissions": permissions,
                "expires_hours": expires_hours,
            },
        )

    def get_share(
        self,
        share_id: str,
    ) -> Any:
        safe_id = _encode_path_param(share_id)

        return self.request_json(
            f"/api/share/{safe_id}"
        )

    def delete_share(
        self,
        share_id: str,
    ) -> Any:
        safe_id = _encode_path_param(share_id)

        return self.request_json(
            f"/api/share/{safe_id}",
            method="DELETE",
        )

    def add_session_comment(
        self,
        thread_id: str,
        content: str,
        *,
        message_id: str | None = None,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        body: dict[str, Any] = {
            "content": content,
        }

        if message_id is not None:
            body["message_id"] = message_id

        return self.request_json(
            f"/api/sessions/{safe_id}/comments",
            method="POST",
            json_body=body,
        )

    def get_session_comments(
        self,
        thread_id: str,
        *,
        message_id: str | None = None,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        params: dict[str, Any] | None = None

        if message_id is not None:
            params = {
                "message_id": message_id,
            }

        return self.request_json(
            f"/api/sessions/{safe_id}/comments",
            params=params,
        )

    def get_session_versions(
        self,
        thread_id: str,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        return self.request_json(
            f"/api/sessions/{safe_id}/versions"
        )

    def create_session_version(
        self,
        thread_id: str,
        *,
        label: str | None = None,
    ) -> Any:
        safe_id = _encode_path_param(thread_id)

        params: dict[str, Any] | None = None

        if label is not None:
            params = {
                "label": label,
            }

        return self.request_json(
            f"/api/sessions/{safe_id}/versions",
            method="POST",
            params=params,
        )

    def restore_session_version(
        self,
        thread_id: str,
        version_id: str,
    ) -> Any:
        safe_thread_id = _encode_path_param(
            thread_id
        )
        safe_version_id = _encode_path_param(
            version_id
        )

        return self.request_json(
            (
                f"/api/sessions/{safe_thread_id}"
                f"/restore/{safe_version_id}"
            ),
            method="POST",
        )
