import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
from agent.core.principal import Principal


class ExportApiTests(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(main.app)

        self.snapshot = {
            "thread_id": "thread-export-001",
            "status": "completed",
            "state": {
                "final_report": "这是测试报告。",
                "sources": [
                    {
                        "title": "Example",
                        "url": "https://example.com",
                    }
                ],
                "deepsearch_artifacts": {
                    "claims": [
                        {
                            "claim": "测试结论",
                            "status": "supported",
                        }
                    ],
                    "quality_summary": {
                        "score": 0.9,
                    },
                },
            },
            "interrupts": [],
            "checkpoint_id": "checkpoint-export-001",
        }

    def _owned_session(self):
        return patch.object(
            main,
            "_require_owned_session",
            return_value=Principal(
                user_id="owner-001",
                authenticated=False,
            ),
        )

    def _state(self):
        return patch.object(
            main,
            "_session_manager_with_owner",
            return_value=self.snapshot,
        )

    def test_templates_are_listed(self):
        response = self.client.get(
            "/api/export/templates"
        )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            len(payload["templates"]),
            4,
        )
        self.assertIn(
            "default",
            [
                item["id"]
                for item in payload["templates"]
            ],
        )

    def test_json_report_export(self):
        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={
                    "format": "json",
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )

        payload = response.json()

        self.assertEqual(
            payload["thread_id"],
            "thread-export-001",
        )
        self.assertEqual(
            payload["report"],
            "这是测试报告。",
        )
        self.assertEqual(
            payload["sources"][0]["url"],
            "https://example.com",
        )

    def test_markdown_report_export(self):
        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={
                    "format": "markdown",
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertIn(
            "# Research Report",
            response.text,
        )
        self.assertIn(
            "这是测试报告。",
            response.text,
        )
        self.assertIn(
            "https://example.com",
            response.text,
        )

    def test_html_report_export(self):
        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={
                    "format": "html",
                    "title": "自定义标题",
                    "template": "academic",
                },
            )

        self.assertEqual(
            response.status_code,
            200,
        )
        self.assertIn(
            "<!doctype html>",
            response.text,
        )
        self.assertIn(
            "自定义标题",
            response.text,
        )
        self.assertIn(
            "academic",
            response.text,
        )

    def test_missing_report_returns_404(self):
        snapshot = {
            "thread_id": "thread-export-002",
            "state": {},
        }

        with patch.object(
            main,
            "_require_owned_session",
            return_value=Principal(
                user_id="owner-001",
            ),
        ), patch.object(
            main,
            "_session_manager_with_owner",
            return_value=snapshot,
        ):
            response = self.client.get(
                "/api/export/thread-export-002",
            )

        self.assertEqual(
            response.status_code,
            404,
        )

    def test_optional_formats_return_not_implemented(self):
        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={
                    "format": "pdf",
                },
            )

        self.assertEqual(
            response.status_code,
            501,
        )

    def test_unknown_format_returns_bad_request(self):
        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={
                    "format": "xml",
                },
            )

        self.assertEqual(
            response.status_code,
            400,
        )


if __name__ == "__main__":
    unittest.main()