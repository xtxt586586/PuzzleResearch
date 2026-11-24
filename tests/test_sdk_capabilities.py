import unittest

import httpx
from urllib.parse import urlsplit

from sdk.python.puzzleresearch_sdk import (
    PuzzleResearchClient,
)


class PuzzleResearchCapabilitySdkTests(
    unittest.TestCase
):
    def test_control_plane_and_agent_routes(self):
        calls: list[tuple[str, str, str]] = []

        def handler(
            request: httpx.Request,
        ) -> httpx.Response:
            parsed_url = urlsplit(str(request.url))

            calls.append(
                (
                    request.method,
                    parsed_url.path,
                    request.url.query.decode(),
                )
            )

            return httpx.Response(
                200,
                json={"ok": True},
                request=request,
            )

        http = httpx.Client(
            transport=httpx.MockTransport(handler)
        )

        client = PuzzleResearchClient(
            base_url="http://example.test",
            http=http,
        )

        client.get_root()
        client.get_health()
        client.get_memory_status()
        client.get_agent_health()
        client.get_public_config()
        client.list_tools()
        client.refresh_tool_registry(reset=True)
        client.get_search_providers()
        client.reset_search_providers()
        client.get_search_cache_stats()
        client.clear_search_cache()
        client.get_active_tasks()

        client.delete_session("thread/one")
        client.get_interrupt_status("thread/one")
        client.resume_thread_interrupt(
            "thread/one",
            {"approved": True},
        )

        client.list_runs()
        client.get_run_metrics("thread/one")
        client.get_trace("thread/one")
        client.get_trace_summary("thread/one")
        client.get_all_traces("thread/one")

        client.list_agents()
        client.get_agent("agent/one")
        client.create_agent(
            {
                "name": "Agent One",
            }
        )
        client.update_agent(
            "agent/one",
            {
                "name": "Updated Agent",
            },
        )
        client.delete_agent("agent/one")

        self.assertEqual(
            calls,
            [
                ("GET", "/", ""),
                ("GET", "/health", ""),
                (
                    "GET",
                    "/api/memory/status",
                    "",
                ),
                (
                    "GET",
                    "/api/health/agent",
                    "",
                ),
                (
                    "GET",
                    "/api/config/public",
                    "",
                ),
                (
                    "GET",
                    "/api/tools/registry",
                    "",
                ),
                (
                    "POST",
                    "/api/tools/registry/refresh",
                    "reset=true",
                ),
                (
                    "GET",
                    "/api/search/providers",
                    "",
                ),
                (
                    "POST",
                    "/api/search/providers/reset",
                    "",
                ),
                (
                    "GET",
                    "/api/search/cache/stats",
                    "",
                ),
                (
                    "POST",
                    "/api/search/cache/clear",
                    "",
                ),
                (
                    "GET",
                    "/api/tasks/active",
                    "",
                ),
                (
                    "DELETE",
                    "/api/sessions/thread%2Fone",
                    "",
                ),
                (
                    "GET",
                    "/api/interrupt/thread%2Fone/status",
                    "",
                ),
                (
                    "POST",
                    "/api/interrupt/thread%2Fone/resume",
                    "",
                ),
                (
                    "GET",
                    "/api/runs",
                    "",
                ),
                (
                    "GET",
                    "/api/runs/thread%2Fone",
                    "",
                ),
                (
                    "GET",
                    "/api/traces/thread%2Fone",
                    "",
                ),
                (
                    "GET",
                    "/api/traces/thread%2Fone/summary",
                    "",
                ),
                (
                    "GET",
                    "/api/traces/thread%2Fone/all",
                    "",
                ),
                (
                    "GET",
                    "/api/agents",
                    "",
                ),
                (
                    "GET",
                    "/api/agents/agent%2Fone",
                    "",
                ),
                (
                    "POST",
                    "/api/agents",
                    "",
                ),
                (
                    "PUT",
                    "/api/agents/agent%2Fone",
                    "",
                ),
                (
                    "DELETE",
                    "/api/agents/agent%2Fone",
                    "",
                ),
            ],
        )

        client.close()


if __name__ == "__main__":
    unittest.main()