import unittest

from agent.workflows.research_tree import ResearchTreeNode, TreeExplorer


class _Model:
    def invoke(self, *args, **kwargs):
        raise AssertionError("model calls are not part of this test")


class ResearchSearchInvocationTests(unittest.TestCase):
    def test_provider_type_error_does_not_repeat_search_call(self):
        calls = []

        def search(payload, config=None):
            calls.append(payload["query"])
            raise TypeError("provider returned an invalid response")

        explorer = TreeExplorer(
            planner_llm=_Model(),
            researcher_llm=_Model(),
            writer_llm=_Model(),
            search_func=search,
            max_depth=0,
            queries_per_branch=1,
        )

        with self.assertRaisesRegex(TypeError, "invalid response"):
            explorer._search(
                ResearchTreeNode(topic="single attempt"),
                "single attempt",
                {},
                1,
            )

        self.assertEqual(calls, ["single attempt"])


if __name__ == "__main__":
    unittest.main()
