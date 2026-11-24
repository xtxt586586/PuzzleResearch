import unittest
from urllib.parse import urlsplit

import httpx

from sdk.python.puzzleresearch_sdk import PuzzleResearchClient


class PuzzleResearchQualitySdkTests(unittest.TestCase):
    def test_get_evidence_preserves_quality_and_passage_contract(self):
        def handler(request: httpx.Request) -> httpx.Response:
            self.assertEqual(
                urlsplit(str(request.url)).path,
                "/api/sessions/thread%2Fquality/evidence",
            )
            return httpx.Response(
                200,
                json={
                    "sources": [{"url": "https://example.com"}],
                    "claims": [
                        {
                            "claim": "收入增长",
                            "status": "supported",
                            "evidence_urls": ["https://example.com"],
                            "evidence_passages": [
                                {
                                    "url": "https://example.com",
                                    "snippet_hash": "passage-001",
                                }
                            ],
                            "score": 0.9,
                            "notes": "有直接证据",
                        }
                    ],
                    "quality_summary": {
                        "overall_score": 0.9,
                        "citation_coverage": 1.0,
                    },
                    "fetched_pages": [],
                    "passages": [
                        {
                            "url": "https://example.com",
                            "text": "收入增长",
                            "start_char": 0,
                            "end_char": 4,
                            "snippet_hash": "passage-001",
                        }
                    ],
                },
                request=request,
            )

        http = httpx.Client(transport=httpx.MockTransport(handler))
        client = PuzzleResearchClient(
            base_url="http://example.test",
            http=http,
        )

        result = client.get_evidence("thread/quality")

        self.assertEqual(result["quality_summary"]["overall_score"], 0.9)
        self.assertEqual(
            result["claims"][0]["evidence_passages"][0]["snippet_hash"],
            "passage-001",
        )
        self.assertEqual(result["passages"][0]["start_char"], 0)
        client.close()


if __name__ == "__main__":
    unittest.main()
