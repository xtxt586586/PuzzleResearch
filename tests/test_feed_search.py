import unittest
from unittest.mock import patch

from common.config import settings
from tools.search.feeds.hackernews_provider import HackerNewsProvider
from tools.search.feeds.reddit_provider import RedditProvider
from tools.search.feeds.twitter_provider import TwitterProvider
from tools.search.manager import SearchManager


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.ok = status_code < 400

    def json(self):
        return self._payload


class FeedSearchTests(unittest.TestCase):
    def test_availability_respects_configuration(self):
        with patch.object(settings, "hackernews_enabled", False), patch.object(
            settings, "reddit_client_id", ""
        ), patch.object(settings, "reddit_client_secret", ""), patch.object(
            settings, "twitter_bearer_token", ""
        ):
            self.assertFalse(HackerNewsProvider().is_available())
            self.assertFalse(RedditProvider().is_available())
            self.assertFalse(TwitterProvider().is_available())

    def test_hackernews_normalizes_algolia_results(self):
        with patch.object(settings, "hackernews_enabled", True), patch(
            "tools.search.feeds.hackernews_provider.requests.get",
            return_value=FakeResponse(
                {
                    "hits": [
                        {
                            "objectID": "1",
                            "_tags": ["story"],
                            "title": "HN story",
                            "url": "https://example.com/story",
                            "points": 42,
                        }
                    ]
                }
            ),
        ) as request:
            result = HackerNewsProvider().search("python", max_results=2)

        self.assertEqual(result[0]["provider"], "hackernews")
        self.assertEqual(result[0]["url"], "https://example.com/story")
        self.assertIn("积分：42", result[0]["content"])
        self.assertEqual(request.call_args.kwargs["params"]["hitsPerPage"], 2)

    def test_reddit_uses_oauth_then_normalizes_posts(self):
        with patch.object(settings, "reddit_client_id", "client-id"), patch.object(
            settings, "reddit_client_secret", "client-secret"
        ), patch(
            "tools.search.feeds.reddit_provider.requests.post",
            return_value=FakeResponse({"access_token": "access-token"}),
        ) as token_request, patch(
            "tools.search.feeds.reddit_provider.requests.get",
            return_value=FakeResponse(
                {
                    "data": {
                        "children": [
                            {
                                "data": {
                                    "title": "Reddit post",
                                    "permalink": "/r/python/comments/1/post",
                                    "selftext": "Post body",
                                    "subreddit": "python",
                                    "score": 10,
                                }
                            }
                        ]
                    }
                }
            ),
        ):
            result = RedditProvider().search("python", max_results=1)

        self.assertEqual(result[0]["provider"], "reddit")
        self.assertEqual(
            result[0]["url"],
            "https://www.reddit.com/r/python/comments/1/post",
        )
        self.assertEqual(token_request.call_count, 1)

    def test_twitter_normalizes_recent_tweets(self):
        with patch.object(settings, "twitter_bearer_token", "bearer-secret"), patch(
            "tools.search.feeds.twitter_provider.requests.get",
            return_value=FakeResponse(
                {
                    "data": [{"id": "123", "author_id": "u1", "text": "A tweet"}],
                    "includes": {"users": [{"id": "u1", "username": "researcher"}]},
                }
            ),
        ) as request:
            result = TwitterProvider().search("research", max_results=10)

        self.assertEqual(result[0]["provider"], "twitter")
        self.assertEqual(result[0]["url"], "https://x.com/i/web/status/123")
        self.assertIn("@researcher", result[0]["content"])
        self.assertNotIn("bearer-secret", str(result))
        self.assertEqual(request.call_args.kwargs["params"]["max_results"], 10)

    def test_manager_registers_feed_providers(self):
        manager = SearchManager()
        self.assertTrue(
            {"hackernews", "reddit", "twitter"}.issubset(manager.providers)
        )


if __name__ == "__main__":
    unittest.main()
