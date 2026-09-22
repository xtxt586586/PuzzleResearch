import asyncio
import json
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from agent.core.events import RuntimeEvent
from agent.core.event_stream import (
    remove_emitter_sync,
)


class ResearchObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(main.app)

    def test_research_tree_endpoint_returns_tree_and_quality(self):
        snapshot = {
            "thread_id": "research-tree-001",
            "state": {
                "research_tree": {
                    "topic": "LangGraph",
                    "nodes": [],
                },
                "quality_summary": {
                    "overall_score": 0.85,
                    "citation_coverage": 1.0,
                },
            },
            "interrupts": [],
            "checkpoint_id": "checkpoint-001",
        }

        with patch.object(
            main.session_manager,
            "get_state",
            return_value=snapshot,
        ):
            response = self.client.get(
                "/api/sessions/research-tree-001/research-tree"
            )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["thread_id"],
            "research-tree-001",
        )
        self.assertEqual(
            payload["tree"]["topic"],
            "LangGraph",
        )
        self.assertEqual(
            payload["quality"]["overall_score"],
            0.85,
        )

    def test_research_tree_endpoint_enforces_owner(self):
        with patch.object(
            main.session_manager,
            "get_state",
            side_effect=main.SessionOwnerError(
                "当前用户无权访问该会话。"
            ),
        ):
            response = self.client.get(
                "/api/sessions/research-tree-002/research-tree"
            )

        self.assertEqual(
            response.status_code,
            403,
        )

    def test_openapi_exposes_research_tree_contract(self):
        spec = main.app.openapi()

        self.assertIn(
            "/api/sessions/{thread_id}/research-tree",
            spec["paths"],
        )

        schema = spec["components"]["schemas"][
            "ResearchTreeResponse"
        ]

        self.assertEqual(
            set(schema["properties"]),
            {
                "thread_id",
                "tree",
                "quality",
            },
        )

    def test_chat_sse_emits_research_observability_events(self):
        async def fake_events(*args, **kwargs):
            yield RuntimeEvent(
                event="on_chain_end",
                name="deep_writer",
                data={
                    "output": {
                        "research_tree": {
                            "topic": "AI",
                            "nodes": [],
                        },
                        "quality_summary": {
                            "overall_score": 0.8,
                        },
                        "final_report": "研究报告",
                    },
                },
            )

        with patch.object(
            main.chat_runtime,
            "astream_runtime_events",
            new=fake_events,
        ):
            response = self.client.post(
                "/api/chat/sse",
                json={
                    "messages": [
                        {
                            "role": "user",
                            "content": "研究 AI",
                        }
                    ],
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertIn(
            "event: research_tree_update",
            response.text,
        )
        self.assertIn(
            "event: quality_update",
            response.text,
        )
        self.assertIn(
            '"topic": "AI"',
            response.text,
        )

        thread_id = response.headers.get(
            "x-thread-id"
        )

        if thread_id:
            remove_emitter_sync(thread_id)

    def test_runtime_event_stream_replays_observability_events(
        self,
    ):
        thread_id = (
            "research-observability-stream-001"
        )

        async def fake_events(*args, **kwargs):
            yield RuntimeEvent(
                event="on_chain_end",
                name="deep_writer",
                data={
                    "output": {
                        "research_tree": {
                            "topic": "Python",
                            "nodes": [],
                        },
                        "quality_summary": {
                            "overall_score": 0.9,
                        },
                        "final_report": "完成",
                    },
                },
            )

        async def collect_frames():
            with patch.object(
                main.chat_runtime,
                "astream_runtime_events",
                new=fake_events,
            ):
                return [
                    frame
                    async for frame
                    in main._iter_chat_events(
                        user_input="研究 Python",
                        history=[],
                        thread_id=thread_id,
                        config={
                            "configurable": {
                                "thread_id": thread_id,
                                "user_id": "anonymous",
                                "agent_id": "default",
                            },
                        },
                    )
                ]

        try:
            asyncio.run(collect_frames())

            response = self.client.get(
                f"/api/events/{thread_id}"
            )

            self.assertEqual(
                response.status_code,
                200,
            )

            events = [
                json.loads(
                    line.removeprefix("data: ")
                )
                for line in response.text.splitlines()
                if line.startswith("data: ")
            ]

            event_types = [
                item["type"]
                for item in events
            ]

            self.assertIn(
                "research_tree_update",
                event_types,
            )
            self.assertIn(
                "quality_update",
                event_types,
            )

        finally:
            remove_emitter_sync(thread_id)


if __name__ == "__main__":
    unittest.main()