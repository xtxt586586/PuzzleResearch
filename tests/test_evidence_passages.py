import unittest

from agent.workflows.evidence_extractor import extract_message_sources
from agent.workflows.evidence_passages import (
    passages_from_fetched_pages,
    split_into_passages,
)


class EvidencePassageTests(unittest.TestCase):
    def test_split_preserves_offsets(self):
        text = "A" * 2000
        passages = split_into_passages(text, max_chars=500)
        self.assertTrue(passages)
        self.assertEqual(passages[0]["start_char"], 0)
        self.assertGreater(passages[0]["end_char"], passages[0]["start_char"])
        self.assertEqual("".join(item["text"] for item in passages), text)

    def test_split_tracks_markdown_heading_path(self):
        text = "# Intro\n\nAlpha.\n\n## Details\n\nBeta.\n\nGamma.\n"
        passages = split_into_passages(text, max_chars=40)
        alpha = next(item for item in passages if "Alpha" in item["text"])
        beta = next(item for item in passages if "Beta" in item["text"])
        self.assertEqual(alpha["heading"], "Intro")
        self.assertEqual(alpha["heading_path"], ["Intro"])
        self.assertEqual(beta["heading"], "Details")
        self.assertEqual(beta["heading_path"], ["Intro", "Details"])

    def test_page_passages_keep_page_metadata_and_dedupe(self):
        pages = [
            {
                "url": "https://example.com/",
                "title": "Example",
                "method": "direct_http",
                "retrieved_at": "2026-09-25T00:00:00+00:00",
                "text": "First paragraph.\n\nSecond paragraph.",
            },
            {
                "url": "https://example.com/",
                "title": "Example",
                "method": "direct_http",
                "text": "First paragraph.\n\nSecond paragraph.",
            },
        ]
        passages = passages_from_fetched_pages(pages, max_chars=100)
        self.assertEqual(len(passages), 1)
        self.assertEqual(passages[0]["url"], "https://example.com/")
        self.assertEqual(passages[0]["page_title"], "Example")

    def test_extract_message_sources_canonicalizes_and_merges_duplicates(self):
        sources = extract_message_sources(
            [
                {
                    "results": [
                        {
                            "title": "A",
                            "url": "https://example.com/?utm_source=x",
                            "provider": "fake",
                        },
                        {
                            "title": "A updated",
                            "url": "https://example.com/",
                            "provider": "fake",
                        },
                    ]
                }
            ]
        )
        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["url"], "https://example.com/")
        self.assertEqual(sources[0]["title"], "A")


if __name__ == "__main__":
    unittest.main()
