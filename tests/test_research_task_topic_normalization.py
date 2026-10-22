import unittest
from unittest.mock import patch

from agent.workflows.nodes import deep_tree_search_node
from common.config import settings


class _Response:
    def __init__(self, content):
        self.content = content


class _Model:
    def invoke(self, messages, config=None):
        return _Response("branch summary")


class ResearchTaskTopicNormalizationTests(unittest.TestCase):
    def test_assigned_task_matches_tree_topic_with_repeated_whitespace(self):
        searches = []

        def fake_search(state):
            searches.append(state["query"])
            return {
                "search_results": [{
                    "title": "Evidence",
                    "url": "https://example.test/evidence",
                    "snippet": "Evidence for normalized topic",
                }],
                "sources": [],
                "fetched_pages": [],
                "passages": [],
            }

        topic = " cost   analysis "
        task = {
            "task_id": "task-cost-001",
            "topic": topic,
            "objective": "Find cost evidence",
            "allowed_tools": ["web_search"],
        }

        with (
            patch.object(settings, "deepsearch_tree_parallel_branches", 0),
            patch("agent.workflows.nodes.ModelFactory.for_task", return_value=_Model()),
            patch("agent.workflows.nodes.search_node", side_effect=fake_search),
        ):
            result = deep_tree_search_node({
                "input": "Compare costs",
                "research_plan": [topic],
                "research_tasks": [task],
                "deepsearch_epoch": 1,
            }, {})

        self.assertEqual(searches, ["cost analysis"])
        self.assertEqual(len(result["research_agent_results"]), 1)
        self.assertEqual(result["research_agent_results"][0]["task_id"], task["task_id"])


if __name__ == "__main__":
    unittest.main()
