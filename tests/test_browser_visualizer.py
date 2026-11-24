import unittest
from unittest.mock import call, patch

from agent.workflows import browser_visualizer
from agent.workflows.nodes import deep_linear_search_node
from common.config import settings


class BrowserVisualizerTests(unittest.TestCase):
    def setUp(self):
        with browser_visualizer._LOCK:
            browser_visualizer._ACTIVE_THREADS.clear()
            browser_visualizer._LAST_STARTED.clear()
            browser_visualizer._VISITED_URLS.clear()

    def _visualize(self, results, *, config=None, state=None):
        browser_visualizer.visualize_urls_from_results(
            state=state or {"thread_id": "visualizer-test"},
            config=config or {},
            results=results,
            reason="test",
        )

    def test_previews_source_and_scrolls_in_background_worker(self):
        with (
            patch.object(settings, "enable_browser_use", True),
            patch.object(settings, "deepsearch_visualize_browser", True),
            patch.object(browser_visualizer, "browser_use") as browser_use,
            patch.object(browser_visualizer, "_spawn", side_effect=lambda fn: fn()),
        ):
            browser_use.return_value = {"status": "ok"}
            self._visualize([
                {"url": "https://Example.com/article#section"},
                {"url": "https://example.com/other"},
            ])

        self.assertEqual(
            browser_use.call_args_list,
            [
                call("visualizer-test", "go_to_url", url="https://example.com/article"),
                call("visualizer-test", "scroll_down", scroll_amount=520),
            ],
        )

    def test_browser_error_does_not_attempt_scroll(self):
        with (
            patch.object(settings, "enable_browser_use", True),
            patch.object(settings, "deepsearch_visualize_browser", True),
            patch.object(browser_visualizer, "browser_use", return_value={"error": "failed"}) as browser_use,
            patch.object(browser_visualizer, "_spawn", side_effect=lambda fn: fn()),
        ):
            self._visualize([{"url": "https://example.com"}])

        browser_use.assert_called_once_with(
            "visualizer-test",
            "go_to_url",
            url="https://example.com/",
        )

    def test_disabled_cancelled_and_video_urls_are_ignored(self):
        with (
            patch.object(settings, "enable_browser_use", False),
            patch.object(settings, "deepsearch_visualize_browser", True),
            patch.object(browser_visualizer, "browser_use") as browser_use,
            patch.object(browser_visualizer, "_spawn") as spawn,
        ):
            self._visualize([{"url": "https://example.com"}])

        self._visualize(
            [{"url": "https://example.com"}],
            state={"thread_id": "cancelled", "is_cancelled": True},
        )
        with (
            patch.object(settings, "enable_browser_use", True),
            patch.object(browser_visualizer, "browser_use") as browser_use_enabled,
            patch.object(browser_visualizer, "_spawn") as spawn_enabled,
        ):
            self._visualize([{"url": "https://www.youtube.com/watch?v=1"}])

        browser_use.assert_not_called()
        spawn.assert_not_called()
        browser_use_enabled.assert_not_called()
        spawn_enabled.assert_not_called()

    def test_private_and_credentialed_urls_are_not_opened(self):
        with (
            patch.object(settings, "enable_browser_use", True),
            patch.object(settings, "deepsearch_visualize_browser", True),
            patch.object(browser_visualizer, "browser_use") as browser_use,
            patch.object(browser_visualizer, "_spawn") as spawn,
        ):
            self._visualize([
                {"url": "http://127.0.0.1/admin"},
                {"url": "http://user:pass@example.com/"},
            ])

        browser_use.assert_not_called()
        spawn.assert_not_called()

    def test_public_ipv6_urls_preserve_brackets_and_port(self):
        self.assertEqual(
            browser_visualizer._normalize_url(
                "https://[2606:4700:4700::1111]:8443/path"
            ),
            "https://[2606:4700:4700::1111]:8443/path",
        )

    def test_thread_is_deduplicated_and_throttled(self):
        with (
            patch.object(settings, "enable_browser_use", True),
            patch.object(settings, "deepsearch_visualize_browser", True),
            patch.object(browser_visualizer, "browser_use") as browser_use,
            patch.object(browser_visualizer, "_spawn", side_effect=lambda fn: fn()),
        ):
            browser_use.return_value = {"status": "ok"}
            self._visualize([{"url": "https://example.com"}])
            self._visualize([{"url": "https://example.com"}])

        self.assertEqual(browser_use.call_count, 2)

    def test_linear_deep_search_passes_results_to_visualizer(self):
        state = {
            "input": "research topic",
            "research_plan": ["first query"],
        }
        config = {"configurable": {"thread_id": "integration-test"}}
        results = [{"url": "https://example.com", "title": "Example"}]

        with (
            patch(
                "agent.workflows.nodes.search_node",
                return_value={"search_results": results, "sources": []},
            ),
            patch.object(browser_visualizer, "visualize_urls_from_results") as visualize,
        ):
            deep_linear_search_node(state, config)

        visualize.assert_called_once_with(
            state=state,
            config=config,
            results=results,
            max_urls=1,
            reason="deepsearch:linear",
        )


if __name__ == "__main__":
    unittest.main()
