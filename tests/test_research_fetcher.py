import types
import unittest
from unittest.mock import patch

from tools.research.content_fetcher import ContentFetcher
from tools.research.models import FetchedPage, truncate_bytes
from tools.research.reader_client import ReaderClient
from tools.research.source_registry import SourceRegistry


class _Response:
    def __init__(self, status_code=200, content_type="text/html", content=b""):
        self.status_code = status_code
        self.headers = {"content-type": content_type}
        self.content = content
        self.text = content.decode("utf-8", errors="replace")

    def iter_content(self, chunk_size=65536):
        yield self.content

    def close(self):
        return None


class ResearchFetcherTests(unittest.TestCase):
    def test_reader_client_builds_and_validates_reader_urls(self):
        client = ReaderClient("public", "https://r.jina.ai/", "")
        self.assertEqual(
            client.build_reader_url("https://example.com/article"),
            "https://r.jina.ai/https://example.com/article",
        )
        with self.assertRaises(ValueError):
            ReaderClient("self_hosted", "https://r.jina.ai", "").build_reader_url(
                "https://example.com/article"
            )

    def test_source_registry_removes_tracking_parameters(self):
        url = SourceRegistry().canonicalize_url(
            "https://Example.com/path/?utm_source=x&b=2&a=1"
        )
        self.assertEqual(url, "https://example.com/path/?a=1&b=2")

    def test_fetched_page_serializes_json_safe_fields(self):
        page = FetchedPage(
            url="https://example.com/",
            raw_url="https://example.com/?utm_source=x",
            method="direct_http",
            text="hello",
        )
        self.assertEqual(page.to_dict()["method"], "direct_http")
        self.assertEqual(truncate_bytes(b"abcdef", max_bytes=3), b"abc")

    def test_direct_fetch_extracts_text_title_and_canonical_url(self):
        html = b"<html><head><title>Example</title></head><body><p>Hello</p><script>x()</script></body></html>"
        with patch(
            "tools.research.content_fetcher.requests",
            types.SimpleNamespace(
                get=lambda *args, **kwargs: _Response(content_type="text/html", content=html)
            ),
        ):
            page = ContentFetcher(reader_mode="off").fetch(
                "https://Example.com/?utm_source=ignored"
            )

        self.assertEqual(page.url, "https://example.com/")
        self.assertEqual(page.title, "Example")
        self.assertEqual(page.text, "Hello")
        self.assertEqual(page.method, "direct_http")

    def test_direct_html_fetch_also_exposes_markdown(self):
        html = b"<h1>Title</h1><p>Hello <strong>world</strong></p><ul><li>One</li></ul>"
        with patch(
            "tools.research.content_fetcher.requests",
            types.SimpleNamespace(
                get=lambda *args, **kwargs: _Response(content_type="text/html", content=html)
            ),
        ):
            page = ContentFetcher(reader_mode="off").fetch("https://example.com/")

        self.assertIn("Title", page.markdown or "")
        self.assertIn("Hello", page.markdown or "")

    def test_auto_render_handles_javascript_interstitial(self):
        html = (
            b"<html><body><p>Checking your browser before accessing</p>"
            b"<p>Checking your browser before accessing</p></body></html>"
        )
        with patch(
            "tools.research.content_fetcher.settings.research_fetch_render_mode",
            "auto",
        ), patch(
            "tools.research.content_fetcher.settings.research_fetch_render_min_chars",
            200,
        ), patch(
            "tools.research.content_fetcher.requests",
            types.SimpleNamespace(
                get=lambda *args, **kwargs: _Response(
                    content_type="text/html", content=html
                )
            ),
        ), patch(
            "tools.crawl.crawler.crawl_urls",
            return_value=[{"url": "https://example.com/", "content": "Rendered content"}],
        ):
            page = ContentFetcher(reader_mode="off").fetch("https://example.com/")

        self.assertEqual(page.method, "render_crawler")
        self.assertEqual(page.text, "Rendered content")

    def test_direct_fetch_caps_streamed_response(self):
        response = _Response(content_type="text/plain", content=b"abcdef")
        with patch(
            "tools.research.content_fetcher.requests",
            types.SimpleNamespace(get=lambda *args, **kwargs: response),
        ), patch("tools.research.content_fetcher.settings.research_fetch_max_bytes", 3):
            page = ContentFetcher(reader_mode="off").fetch("https://example.com/")

        self.assertEqual(page.text, "abc")

    def test_local_targets_are_blocked_before_request(self):
        called = []
        with patch(
            "tools.research.content_fetcher.requests",
            types.SimpleNamespace(get=lambda *args, **kwargs: called.append(args)),
        ):
            page = ContentFetcher(reader_mode="off").fetch("http://localhost:8000/")

        self.assertIn("blocked", page.error.lower())
        self.assertEqual(called, [])

    def test_reader_fallback_returns_reader_content(self):
        calls = []

        def get(url, **kwargs):
            calls.append(url)
            if "r.jina.ai" in url:
                return _Response(content_type="text/plain", content=b"Reader text")
            return _Response(status_code=403, content_type="text/html", content=b"")

        with patch(
            "tools.research.content_fetcher.requests",
            types.SimpleNamespace(get=get),
        ):
            page = ContentFetcher(
                reader_mode="public",
                reader_public_base="https://r.jina.ai",
            ).fetch("https://example.com/")

        self.assertEqual(page.method, "reader_public")
        self.assertEqual(page.text, "Reader text")
        self.assertTrue(any("r.jina.ai" in url for url in calls))

    def test_fetch_many_deduplicates_canonical_urls(self):
        calls = []

        def get(url, **kwargs):
            calls.append(url)
            return _Response(content_type="text/plain", content=b"ok")

        with patch(
            "tools.research.content_fetcher.requests",
            types.SimpleNamespace(get=get),
        ):
            pages = ContentFetcher(reader_mode="off").fetch_many(
                [
                    "https://example.com/?utm_source=a",
                    "https://example.com/?utm_source=b",
                ]
            )

        self.assertEqual(len(pages), 1)
        self.assertEqual(calls, ["https://example.com/"])


if __name__ == "__main__":
    unittest.main()
