import unittest
from unittest.mock import patch

from common.config import settings
from tools.search.brave_provider import BraveProvider
from tools.search.manager import SearchManager
from tools.search.multi_search import MultiSearchOrchestrator
from tools.search.serper_provider import SerperProvider


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self._payload


class SearchProviderTests(unittest.TestCase):
    def test_missing_keys_disable_paid_providers(self):
        with patch.object(settings, "brave_api_key", ""), patch.object(
            settings, "serper_api_key", ""
        ):
            self.assertFalse(BraveProvider().is_available())
            self.assertFalse(SerperProvider().is_available())

    def test_brave_normalizes_web_results(self):
        with patch.object(settings, "brave_api_key", "brave-secret"), patch(
            "tools.search.brave_provider.requests.get",
            return_value=FakeResponse(
                {
                    "web": {
                        "results": [
                            {
                                "title": "Brave result",
                                "url": "https://example.com/brave",
                                "description": "A Brave snippet",
                                "extra_snippets": ["More context"],
                            }
                        ]
                    }
                }
            ),
        ) as request:
            result = BraveProvider().search("query", max_results=3)

        self.assertEqual(result[0]["provider"], "brave")
        self.assertEqual(result[0]["url"], "https://example.com/brave")
        self.assertIn("More context", result[0]["content"])
        request.assert_called_once()
        self.assertEqual(request.call_args.kwargs["params"], {"q": "query", "count": 3})
        self.assertNotIn("brave-secret", str(result))

    def test_serper_normalizes_organic_results(self):
        with patch.object(settings, "serper_api_key", "serper-secret"), patch(
            "tools.search.serper_provider.requests.post",
            return_value=FakeResponse(
                {
                    "organic": [
                        {
                            "title": "Serper result",
                            "link": "https://example.com/serper",
                            "snippet": "A Serper snippet",
                        }
                    ]
                }
            ),
        ) as request:
            result = SerperProvider().search("query", max_results=4)

        self.assertEqual(result[0]["provider"], "serper")
        self.assertEqual(result[0]["url"], "https://example.com/serper")
        self.assertEqual(result[0]["content"], "A Serper snippet")
        self.assertEqual(request.call_args.kwargs["json"], {"q": "query", "num": 4})
        self.assertNotIn("serper-secret", str(result))

    def test_http_failures_do_not_expose_api_keys(self):
        with patch.object(settings, "brave_api_key", "brave-secret"), patch(
            "tools.search.brave_provider.requests.get",
            return_value=FakeResponse({}, status_code=401),
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 401") as error:
                BraveProvider().search("query")
        self.assertNotIn("brave-secret", str(error.exception))

        with patch.object(settings, "serper_api_key", "serper-secret"), patch(
            "tools.search.serper_provider.requests.post",
            return_value=FakeResponse({}, status_code=403),
        ):
            with self.assertRaisesRegex(RuntimeError, "HTTP 403") as error:
                SerperProvider().search("query")
        self.assertNotIn("serper-secret", str(error.exception))

    def test_default_search_paths_register_paid_providers(self):
        with patch.object(settings, "brave_api_key", ""), patch.object(
            settings, "serper_api_key", ""
        ):
            manager = SearchManager()
            orchestrator = MultiSearchOrchestrator([])

        self.assertIn("brave", manager.providers)
        self.assertIn("serper", manager.providers)
        self.assertEqual(
            {provider.name for provider in orchestrator.providers},
            set(),
        )


if __name__ == "__main__":
    unittest.main()
