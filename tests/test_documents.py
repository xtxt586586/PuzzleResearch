from io import BytesIO
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import main
import fitz
from docx import Document


class DocumentApiTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.client = TestClient(main.app)

        self.enabled_patch = patch.object(
            main.settings,
            "rag_enabled",
            True,
        )
        self.path_patch = patch.object(
            main.settings,
            "rag_store_path",
            self.temp_dir.name,
        )

        self.enabled_patch.start()
        self.path_patch.start()

    def tearDown(self):
        self.path_patch.stop()
        self.enabled_patch.stop()
        self.temp_dir.cleanup()

    def test_rag_disabled_returns_400(self):
        with patch.object(
            main.settings,
            "rag_enabled",
            False,
        ):
            response = self.client.get(
                "/api/documents/list"
            )

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_upload_list_search_and_delete(self):
        upload = self.client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "notes.txt",
                    (
                        "Python Agent research notes. "
                        "LangGraph runtime and tools."
                    ).encode("utf-8"),
                    "text/plain",
                )
            },
        )

        self.assertEqual(
            upload.status_code,
            200,
        )
        self.assertTrue(
            upload.json()["success"]
        )

        listed = self.client.get(
            "/api/documents/list"
        )

        self.assertEqual(
            listed.status_code,
            200,
        )
        self.assertEqual(
            listed.json()["documents"][0]["filename"],
            "notes.txt",
        )

        searched = self.client.post(
            "/api/documents/search",
            params={
                "query": "LangGraph",
                "n_results": 3,
            },
        )

        self.assertEqual(
            searched.status_code,
            200,
        )
        self.assertGreaterEqual(
            len(searched.json()["results"]),
            1,
        )
        self.assertEqual(
            searched.json()["results"][0]["source"],
            "notes.txt",
        )

        deleted = self.client.delete(
            "/api/documents/notes.txt"
        )

        self.assertEqual(
            deleted.status_code,
            200,
        )
        self.assertTrue(
            deleted.json()["success"]
        )

    def test_invalid_extension_is_rejected(self):
        response = self.client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "payload.exe",
                    b"invalid",
                    "application/octet-stream",
                )
            },
        )

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_pdf_upload_extracts_page_text_for_search(self):
        pdf = fitz.open()
        page = pdf.new_page()
        page.insert_text((72, 72), "PuzzleResearch PDF ingestion marker")
        content = pdf.tobytes()
        pdf.close()

        upload = self.client.post(
            "/api/documents/upload",
            files={"file": ("research.pdf", content, "application/pdf")},
        )

        self.assertEqual(upload.status_code, 200, upload.text)
        self.assertEqual(upload.json()["chunks"], 1)
        searched = self.client.post(
            "/api/documents/search",
            params={"query": "ingestion marker", "n_results": 3},
        )
        self.assertEqual(searched.status_code, 200)
        self.assertIn("[Page 1]", searched.json()["results"][0]["text"])

    def test_docx_upload_extracts_paragraphs_and_tables(self):
        document = Document()
        document.add_paragraph("PuzzleResearch DOCX ingestion marker")
        table = document.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "key"
        table.cell(0, 1).text = "table-value"
        content = BytesIO()
        document.save(content)

        upload = self.client.post(
            "/api/documents/upload",
            files={
                "file": (
                    "research.docx",
                    content.getvalue(),
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                )
            },
        )

        self.assertEqual(upload.status_code, 200, upload.text)
        searched = self.client.post(
            "/api/documents/search",
            params={"query": "table-value", "n_results": 3},
        )
        self.assertEqual(searched.status_code, 200)
        self.assertEqual(searched.json()["results"][0]["source"], "research.docx")

    def test_malformed_pdf_and_legacy_doc_return_clear_errors(self):
        malformed = self.client.post(
            "/api/documents/upload",
            files={"file": ("broken.pdf", b"not a PDF", "application/pdf")},
        )
        legacy_doc = self.client.post(
            "/api/documents/upload",
            files={"file": ("legacy.doc", b"binary document", "application/msword")},
        )

        self.assertEqual(malformed.status_code, 400)
        self.assertIn("PDF 文件损坏", malformed.json()["detail"])
        self.assertEqual(legacy_doc.status_code, 400)
        self.assertIn("另存为 .docx", legacy_doc.json()["detail"])

    def test_empty_search_is_rejected(self):
        response = self.client.post(
            "/api/documents/search",
            params={
                "query": "   ",
            },
        )

        self.assertEqual(
            response.status_code,
            400,
        )

    def test_missing_document_returns_404(self):
        response = self.client.delete(
            "/api/documents/missing.txt"
        )

        self.assertEqual(
            response.status_code,
            404,
        )


if __name__ == "__main__":
    unittest.main()
