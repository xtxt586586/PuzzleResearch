from __future__ import annotations

import asyncio
import threading
import unittest
from unittest.mock import patch

from agent.workflows.research_tree import ResearchTree, ResearchTreeNode, TreeExplorer
from common.config import settings


class _Response:
    def __init__(self, content, usage_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata


class _Planner:
    def __init__(self, calls):
        self.calls = calls

    def invoke(self, message, config=None):
        self.calls.append("llm")
        return _Response('["extra query"]')


class _Noop:
    def invoke(self, message, config=None):
        return _Response("")


class _ManyPlanner(_Noop):
    def invoke(self, message, config=None):
        return _Response('["q1", "q2"]')


class _UsagePlanner(_Noop):
    def invoke(self, message, config=None):
        return _Response('["q1", "q2"]', {"total_tokens": 20})


class _SummaryModel:
    def __init__(self, content):
        self.content = content
        self.calls = []

    def invoke(self, message, config=None):
        self.calls.append(message)
        return _Response(self.content)


class ResearchTreeExplorerTests(unittest.TestCase):
    def test_search_results_are_trimmed_to_remaining_branch_token_budget(self):
        explorer = TreeExplorer(
            _Noop(),
            _Noop(),
            _Noop(),
            lambda payload: [
                {
                    "title": f"来源 {index}",
                    "url": f"https://example.test/{index}",
                    "snippet": "证据内容" * 500,
                }
                for index in range(payload["max_results"])
            ],
            max_tokens=60,
            count_topic_tokens=False,
        )
        node = ResearchTreeNode(topic="主题")

        results = explorer._search(node, "查询", {}, max_results=5)

        self.assertTrue(results)
        self.assertLessEqual(explorer.get_budget_stats()["tokens_used"], 60)
        self.assertGreater(
            explorer.get_budget_stats()["token_usage_by_stage"]["search_context"],
            0,
        )
        self.assertLess(len("".join(item["snippet"] for item in results)), 300)

    def test_node_queries_round_trip_through_dict(self):
        node = ResearchTreeNode(
            topic="Topic",
            queries=["Topic", "supporting query"],
            agent_result={"task_id": "branch-1", "status": "completed"},
        )

        restored = ResearchTreeNode.from_dict(node.to_dict())

        self.assertEqual(restored.queries, ["Topic", "supporting query"])
        self.assertEqual(
            restored.agent_result,
            {"task_id": "branch-1", "status": "completed"},
        )

    def test_node_findings_round_trip_for_followup_research(self):
        tree = ResearchTree(topic="Topic")
        node = tree.create_root("Topic")
        node.mark_complete()
        node.findings = [
            {
                "query": "supporting query",
                "result": {
                    "title": "Source title",
                    "url": "https://example.test/source",
                    "snippet": "Evidence excerpt",
                    "provider": "test-search",
                    "content": "Large page body should not be checkpointed",
                },
            }
        ]

        checkpoint = tree.to_dict()
        restored = ResearchTree.from_dict(checkpoint)

        self.assertEqual(restored.get_all_findings(), [{
            "query": "supporting query",
            "result": {
                "title": "Source title",
                "url": "https://example.test/source",
                "snippet": "Evidence excerpt",
                "provider": "test-search",
            },
        }])
        self.assertNotIn("content", checkpoint["nodes"][0]["findings"][0])

    def test_compacted_checkpoint_preserves_total_counts_and_sources(self):
        node = ResearchTreeNode(topic="Topic")
        node.result_count = 12
        node.findings = [
            {
                "query": "Topic",
                "result": {
                    "title": f"Source {index}",
                    "url": f"https://example.test/{index}",
                    "snippet": f"Finding {index}",
                },
            }
            for index in range(12)
        ]
        node.sources = [f"https://example.test/{index}" for index in range(7)]
        node.source_urls = list(node.sources)

        checkpoint = node.to_dict()
        restored = ResearchTreeNode.from_dict(checkpoint)
        second_checkpoint = restored.to_dict()

        self.assertEqual(len(checkpoint["findings"]), 5)
        self.assertEqual(checkpoint["findings_count"], 12)
        self.assertEqual(restored.result_count, 12)
        self.assertEqual(len(restored.sources), 7)
        self.assertEqual(second_checkpoint["findings_count"], 12)
        self.assertEqual(second_checkpoint["sources_count"], 7)

    def test_failed_branch_keeps_partial_findings_in_tree_results(self):
        tree = ResearchTree(topic="Topic")
        node = tree.create_root("Topic")
        node.mark_failed("summary generation failed")
        node.findings = [{
            "query": "Topic",
            "result": {
                "title": "Partial source",
                "url": "https://example.test/partial",
                "snippet": "Useful evidence found before failure",
            },
        }]

        self.assertEqual(tree.get_all_findings(), node.findings)

    def test_branch_searches_raw_topic_before_query_generation(self):
        calls = []

        def search(payload, config=None):
            calls.append(f"search:{payload['query']}")
            return []

        explorer = TreeExplorer(
            planner_llm=_Planner(calls),
            researcher_llm=_Noop(),
            writer_llm=_Noop(),
            search_func=search,
            max_depth=0,
            max_branches=1,
            queries_per_branch=2,
        )
        explorer.explore_branch(ResearchTreeNode(topic="Test Topic"), {})
        self.assertEqual(calls, ["search:Test Topic", "llm", "search:extra query"])

    def test_branch_and_tree_summaries_use_research_and_writer_models(self):
        researcher = _SummaryModel("分支关键发现")
        writer = _SummaryModel("合并后的研究摘要")
        explorer = TreeExplorer(
            planner_llm=_Noop(),
            researcher_llm=researcher,
            writer_llm=writer,
            search_func=lambda payload, config=None: [{
                "title": "来源",
                "url": "https://example.test/source",
                "snippet": "证据摘要",
            }],
            max_depth=0,
            queries_per_branch=1,
        )

        explorer.run("Topic", {}, decompose_root=False)

        self.assertEqual(explorer.tree.get_root().summary, "分支关键发现")
        self.assertEqual(explorer.get_final_summary(), "合并后的研究摘要")
        self.assertEqual(len(researcher.calls), 1)
        self.assertEqual(len(writer.calls), 1)

    def test_async_tree_does_not_deadlock_when_branch_limit_is_full(self):
        explorer = TreeExplorer(
            planner_llm=_Noop(),
            researcher_llm=_Noop(),
            writer_llm=_Noop(),
            search_func=lambda *args, **kwargs: [],
            max_depth=2,
            max_branches=2,
            queries_per_branch=1,
        )
        def decompose(topic, existing_knowledge="", num_subtopics=4):
            if topic == "Root Topic":
                return [("Child A", 1.0), ("Child B", 1.0)]
            return [("Grandchild", 1.0)]
        explorer.decompose_topic = decompose

        async def run():
            return await asyncio.wait_for(
                explorer.run_async("Root Topic", {}, decompose_root=True),
                timeout=1,
            )
        tree = asyncio.run(run())
        self.assertIsNotNone(tree.root_id)
        self.assertGreaterEqual(len(tree.nodes), 1)

    def test_async_branches_keep_findings_and_sources_isolated(self):
        branch_searches_ready = threading.Barrier(2)

        def search(payload, config=None):
            query = payload["query"]
            if query in {"Child A", "Child B"}:
                branch_searches_ready.wait(timeout=1)
            return [{
                "url": f"https://example.test/{query.replace(' ', '-').lower()}",
                "snippet": f"finding for {query}",
            }]

        explorer = TreeExplorer(
            planner_llm=_Noop(),
            researcher_llm=_Noop(),
            writer_llm=_Noop(),
            search_func=search,
            max_depth=1,
            max_branches=2,
            queries_per_branch=1,
        )
        explorer.decompose_topic = lambda *args, **kwargs: [
            ("Child A", 1.0),
            ("Child B", 1.0),
        ]

        tree = asyncio.run(explorer.run_async("Root Topic", {}, decompose_root=True))
        children = {node.topic: node for node in tree.get_children(tree.root_id)}

        self.assertEqual(children["Child A"].findings[0]["query"], "Child A")
        self.assertEqual(children["Child B"].findings[0]["query"], "Child B")
        self.assertEqual(children["Child A"].sources, ["https://example.test/child-a"])
        self.assertEqual(children["Child B"].sources, ["https://example.test/child-b"])

    def test_search_events_publish_cumulative_progress_per_branch(self):
        events = []

        def search(payload, config=None):
            return [
                {"url": "https://example.test/a", "snippet": "A"},
                {"url": "https://example.test/b", "snippet": "B"},
            ]

        explorer = TreeExplorer(
            planner_llm=_Noop(),
            researcher_llm=_Noop(),
            writer_llm=_Noop(),
            search_func=search,
            max_depth=0,
            queries_per_branch=1,
            event_callback=lambda event_type, data: events.append((event_type, data)),
        )
        node = ResearchTreeNode(
            id="branch-a",
            topic="子主题 A",
            parent_id="research-root",
            depth=1,
        )

        explorer.explore_branch(node, {})

        progress = next(
            data for event_type, data in events
            if event_type == "search" and data.get("node_id") == "branch-a"
        )
        self.assertEqual(progress["status"], "running")
        self.assertEqual(progress["result_count"], 2)
        self.assertEqual(progress["sources_count"], 2)
        self.assertEqual(progress["topic"], "子主题 A")
        self.assertEqual(progress["parent_id"], "research-root")
        self.assertEqual(node.result_count, 2)
        self.assertEqual(node.source_urls, [
            "https://example.test/a",
            "https://example.test/b",
        ])

    def test_search_budget_stops_additional_searches(self):
        calls = []
        with patch.object(settings, "deepsearch_tree_max_searches", 2, create=True):
            explorer = TreeExplorer(
                planner_llm=_ManyPlanner(),
                researcher_llm=_Noop(),
                writer_llm=_Noop(),
                search_func=lambda payload, config=None: calls.append(payload["query"]) or [],
                max_depth=0,
                queries_per_branch=3,
            )
            with self.assertRaisesRegex(Exception, "search_budget_exceeded"):
                explorer.explore_branch(ResearchTreeNode(topic="Topic"), {})
        self.assertEqual(calls, ["Topic", "q1"])

    def test_search_budget_can_be_overridden_per_explorer(self):
        calls = []
        explorer = TreeExplorer(
            planner_llm=_ManyPlanner(),
            researcher_llm=_Noop(),
            writer_llm=_Noop(),
            search_func=lambda payload, config=None: calls.append(payload["query"]) or [],
            max_depth=0,
            queries_per_branch=3,
            max_searches=2,
        )

        with self.assertRaisesRegex(Exception, "search_budget_exceeded"):
            explorer.explore_branch(ResearchTreeNode(topic="Topic"), {})

        self.assertEqual(calls, ["Topic", "q1"])

    def test_token_budget_skips_model_call_when_prompt_cannot_fit(self):
        searches = []
        explorer = TreeExplorer(
            planner_llm=_UsagePlanner(),
            researcher_llm=_Noop(),
            writer_llm=_Noop(),
            search_func=lambda payload, config=None: searches.append(payload["query"]) or [],
            max_depth=0,
            queries_per_branch=3,
            max_tokens=10,
        )

        with self.assertRaisesRegex(Exception, "token_budget_exceeded"):
            explorer.run("Topic", {}, decompose_root=False)

        self.assertEqual(searches, ["Topic"])
        stats = explorer.get_budget_stats()
        self.assertEqual(stats["stop_reason"], "token_budget_exceeded")
        self.assertLess(stats["tokens_used"], 10)

    def test_model_output_limit_uses_remaining_shared_token_budget(self):
        captured = {}

        class BindableModel:
            def bind(self, **kwargs):
                captured.update(kwargs)
                return self

            def invoke(self, messages, config=None):
                return _Response("bounded", {"total_tokens": 5})

        explorer = TreeExplorer(
            planner_llm=_Noop(),
            researcher_llm=_Noop(),
            writer_llm=_Noop(),
            search_func=lambda payload: [],
            max_seconds=30,
            max_tokens=100,
            count_topic_tokens=False,
        )

        explorer._invoke(
            BindableModel(),
            [{"role": "user", "content": "small prompt"}],
        )

        self.assertGreater(captured["max_tokens"], 0)
        self.assertLess(captured["max_tokens"], 100)
        self.assertEqual(explorer.get_budget_stats()["tokens_used"], 5)

    def test_time_budget_includes_elapsed_time_from_prior_epoch(self):
        searches = []
        explorer = TreeExplorer(
            planner_llm=_Noop(),
            researcher_llm=_Noop(),
            writer_llm=_Noop(),
            search_func=lambda payload, config=None: searches.append(payload["query"]) or [],
            max_depth=0,
            queries_per_branch=1,
            max_seconds=1,
            initial_elapsed_seconds=1,
        )

        with self.assertRaisesRegex(Exception, "time_budget_exceeded"):
            explorer.run("Topic", {}, decompose_root=False)

        self.assertEqual(searches, [])
        self.assertEqual(explorer.get_budget_stats()["stop_reason"], "time_budget_exceeded")


if __name__ == "__main__":
    unittest.main()
