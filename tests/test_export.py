from io import BytesIO
import unittest
from unittest.mock import AsyncMock, patch

import fitz
from fastapi.testclient import TestClient
from docx import Document

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
                    "fetched_pages": [
                        {
                            "url": "https://example.com",
                            "raw_url": "https://example.com",
                            "method": "direct",
                            "http_status": 200,
                        }
                    ],
                    "passages": [
                        {
                            "url": "https://example.com",
                            "text": "这是关键证据片段。",
                            "start_char": 0,
                            "end_char": 10,
                            "heading": "关键证据",
                            "snippet_hash": "passage-001",
                        }
                    ],
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

    def test_conversation_archive_helpers_append_and_merge_turns(self):
        archived = [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
        ]
        incoming = [
            {"role": "user", "content": "第二问"},
        ]

        merged = main._merge_conversation_history(
            archived,
            incoming,
        )
        completed = main._append_assistant_message(
            merged,
            "第二答",
        )

        self.assertEqual(
            completed,
            [
                *archived,
                {"role": "user", "content": "第二问"},
                {"role": "assistant", "content": "第二答"},
            ],
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

        self.assertEqual(
            payload["passages"][0]["snippet_hash"],
            "passage-001",
        )
        self.assertEqual(
            payload["fetched_pages"][0]["http_status"],
            200,
        )

    def test_evidence_is_rendered_in_markdown_and_html_exports(self):
        for format_name in ("markdown", "html"):
            with self.subTest(format=format_name):
                with self._owned_session(), self._state():
                    response = self.client.get(
                        "/api/export/thread-export-001",
                        params={"format": format_name},
                    )

                self.assertEqual(response.status_code, 200)
                self.assertIn("测试结论", response.text)
                self.assertIn("关键证据", response.text)
                self.assertIn("这是关键证据片段。", response.text)
                self.assertIn("https://example.com", response.text)

                if format_name == "markdown":
                    self.assertIn("## Claim Verification", response.text)
                    self.assertIn("## Evidence Passages", response.text)
                else:
                    self.assertIn("Claim Verification", response.text)
                    self.assertIn("Evidence Passages", response.text)

    def test_json_export_includes_full_conversation(self):
        self.snapshot["state"]["messages"] = [
            {
                "type": "human",
                "content": "第一轮问题",
            },
            {
                "type": "ai",
                "content": "第一轮回答",
            },
            {
                "type": "tool",
                "content": "工具内部结果",
            },
            {
                "type": "human",
                "content": "第二轮问题",
            },
            {
                "type": "ai",
                "content": [
                    {"type": "text", "text": "第二轮"},
                    {"type": "text", "text": "回答"},
                ],
            },
        ]

        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={"format": "json"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["conversation"],
            [
                {"role": "user", "content": "第一轮问题"},
                {"role": "assistant", "content": "第一轮回答"},
                {"role": "user", "content": "第二轮问题"},
                {"role": "assistant", "content": "第二轮回答"},
            ],
        )

    def test_json_export_prefers_uncapped_conversation_archive(self):
        self.snapshot["state"]["messages"] = [
            {"type": "human", "content": "最近的问题"},
            {"type": "ai", "content": "这是测试报告。"},
        ]
        self.snapshot["state"]["conversation_history"] = [
            {"role": "user", "content": "很早以前的问题"},
            {"role": "assistant", "content": "很早以前的回答"},
            {"role": "user", "content": "最近的问题"},
            {"role": "assistant", "content": "这是测试报告。"},
        ]

        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={"format": "json"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["conversation"][0],
            {"role": "user", "content": "很早以前的问题"},
        )
        self.assertEqual(
            len(response.json()["conversation"]),
            4,
        )

    def test_post_export_syncs_visible_history_for_legacy_sessions(self):
        self.snapshot["state"]["messages"] = [
            {"type": "human", "content": "最近的问题"},
            {"type": "ai", "content": "这是测试报告。"},
        ]

        async def save_archive(*, values, **kwargs):
            self.snapshot["state"].update(values)

        with self._owned_session(), self._state(), patch.object(
            main.chat_runtime,
            "aupdate_state",
            new_callable=AsyncMock,
            side_effect=save_archive,
        ):
            response = self.client.post(
                "/api/export/thread-export-001",
                params={"format": "json"},
                json={
                    "messages": [
                        {"role": "user", "content": "很早以前的问题"},
                        {"role": "assistant", "content": "很早以前的回答"},
                        {"role": "user", "content": "最近的问题"},
                        {"role": "assistant", "content": "这是测试报告。"},
                    ],
                },
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json()["conversation"][0],
            {"role": "user", "content": "很早以前的问题"},
        )
        self.assertEqual(len(response.json()["conversation"]), 4)

    def test_markdown_export_includes_all_turns_without_duplicate_final(self):
        self.snapshot["state"]["messages"] = [
            {"type": "human", "content": "第一轮问题"},
            {"type": "ai", "content": "第一轮回答"},
            {"type": "human", "content": "第二轮问题"},
            {"type": "ai", "content": "这是测试报告。"},
        ]

        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={"format": "markdown"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertLess(
            response.text.index("第一轮问题"),
            response.text.index("第一轮回答"),
        )
        self.assertLess(
            response.text.index("第一轮回答"),
            response.text.index("第二轮问题"),
        )
        self.assertLess(
            response.text.index("第二轮问题"),
            response.text.index("这是测试报告。"),
        )
        self.assertEqual(
            response.text.count("这是测试报告。"),
            1,
        )

    def test_html_export_includes_all_turns(self):
        self.snapshot["state"]["messages"] = [
            {"type": "human", "content": "第一轮问题"},
            {"type": "ai", "content": "第一轮回答"},
            {"type": "human", "content": "第二轮问题"},
            {"type": "ai", "content": "这是测试报告。"},
        ]

        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={"format": "html"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertIn("第一轮问题", response.text)
        self.assertIn("第一轮回答", response.text)
        self.assertIn("第二轮问题", response.text)
        self.assertEqual(
            response.text.count("这是测试报告。"),
            1,
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

    def test_pdf_export_returns_a_pdf_document(self):
        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={"format": "pdf"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content[:4], b"%PDF")
        self.assertIn("application/pdf", response.headers["content-type"])
        self.assertIn("attachment", response.headers["content-disposition"])
        with fitz.open(stream=response.content, filetype="pdf") as document:
            pdf_text = "\n".join(page.get_text() for page in document)
        self.assertIn("这是测试报告。", pdf_text)

    def test_docx_export_returns_a_word_document(self):
        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={"format": "docx"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content[:2], b"PK")
        self.assertIn(
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            response.headers["content-type"],
        )
        document = Document(BytesIO(response.content))
        docx_text = "\n".join(paragraph.text for paragraph in document.paragraphs)
        self.assertIn("这是测试报告。", docx_text)

    def test_doc_alias_uses_docx_output(self):
        with self._owned_session(), self._state():
            response = self.client.get(
                "/api/export/thread-export-001",
                params={"format": "doc"},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.content[:2], b"PK")

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
