import unittest
from types import SimpleNamespace
from unittest.mock import patch

from agent.core.graph import evaluation_decision
from agent.workflows.nodes import (
    deep_linear_search_node,
    deep_followup_plan_node,
    deep_tree_search_node,
    deep_writer_node,
    evaluator_node,
    revise_node,
)
from agent.workflows.research_tree import ResearchTree
from agent.workflows.research_budget import (
    DEFAULT_DEEPSEARCH_MAX_SECONDS,
    DEFAULT_DEEPSEARCH_MAX_TOKENS,
    add_budget_tokens,
    estimate_tokens,
    model_call_tokens,
    new_research_budget,
)
from common.config import settings


class LinearResearchBudgetTests(unittest.TestCase):
    def test_zero_limits_use_bounded_defaults(self):
        budget = new_research_budget(
            {},
            max_seconds=0,
            max_tokens=0,
            now=1_000.0,
        )

        self.assertEqual(
            budget["max_seconds"],
            DEFAULT_DEEPSEARCH_MAX_SECONDS,
        )
        self.assertEqual(
            budget["max_tokens"],
            DEFAULT_DEEPSEARCH_MAX_TOKENS,
        )

    def test_model_usage_falls_back_to_text_estimates_without_metadata(self):
        response = SimpleNamespace(content="y" * 20)
        self.assertEqual(model_call_tokens("x" * 40, response), 15)

    def test_token_estimate_counts_cjk_characters_conservatively(self):
        self.assertEqual(estimate_tokens("中国新能源出口"), 7)
        self.assertEqual(estimate_tokens("China 出口 2026"), 5)

    def test_reported_model_usage_takes_precedence_over_estimate(self):
        response = SimpleNamespace(
            content="y" * 200,
            usage_metadata={"total_tokens": 7},
        )
        self.assertEqual(model_call_tokens("x" * 400, response), 7)

    def test_token_usage_is_accumulated_by_stage(self):
        budget = add_budget_tokens(
            {"tokens_used": 10},
            7,
            stage="planner",
            now=1_000.0,
        )
        budget = add_budget_tokens(
            budget,
            13,
            stage="writer",
            now=1_001.0,
        )

        self.assertEqual(budget["tokens_used"], 30)
        self.assertEqual(
            budget["token_usage_by_stage"],
            {"planner": 7, "writer": 13},
        )

    def test_token_budget_stops_before_starting_the_next_query(self):
        calls = []

        def fake_search(state):
            query = state["query"]
            calls.append(query)
            return {
                "search_results": [{
                    "query": query,
                    "title": "Evidence",
                    "url": "https://example.test/evidence",
                    "snippet": "x" * 100,
                }],
                "sources": [{
                    "title": "Evidence",
                    "url": "https://example.test/evidence",
                    "query": query,
                }],
                "fetched_pages": [],
                "passages": [],
            }

        with patch("agent.workflows.nodes.search_node", side_effect=fake_search):
            result = deep_linear_search_node(
                {
                    "input": "research topic",
                    "research_plan": ["first query", "second query"],
                },
                {"configurable": {"deepsearch_max_tokens": 20}},
            )

        self.assertEqual(calls, ["first query"])
        self.assertEqual(len(result["search_results"]), 1)
        budget = result["deepsearch_artifacts"]["linear_budget"]
        self.assertEqual(budget["stop_reason"], "token_budget_exceeded")
        self.assertEqual(budget["searches_used"], 1)
        self.assertEqual(result["budget_stop_reason"], "token_budget_exceeded")
        self.assertGreaterEqual(
            result["deepsearch_tokens_used"],
            budget["tokens_used"],
        )
        self.assertGreaterEqual(
            result["deepsearch_elapsed_seconds"],
            budget["elapsed_seconds"],
        )
        self.assertEqual(
            result["quality_summary"]["budget_stop_reason"],
            "token_budget_exceeded",
        )
        self.assertEqual(result["quality_summary"]["max_tokens"], 20)
        self.assertEqual(result["quality_summary"]["searches_used"], 1)

    def test_existing_budget_is_carried_across_research_epochs(self):
        with patch(
            "agent.workflows.nodes.search_node",
            side_effect=AssertionError("an exhausted budget must block search"),
        ):
            result = deep_linear_search_node(
                {
                    "input": "topic",
                    "research_plan": ["new query"],
                    "deepsearch_artifacts": {
                        "linear_budget": {
                            "tokens_used": 40,
                            "elapsed_seconds": 2.5,
                            "searches_used": 3,
                        }
                    },
                },
                {"configurable": {"deepsearch_max_tokens": 20}},
            )

        budget = result["deepsearch_artifacts"]["linear_budget"]
        self.assertEqual(budget["tokens_used"], 40)
        self.assertEqual(budget["searches_used"], 3)
        self.assertGreaterEqual(budget["elapsed_seconds"], 2.5)
        self.assertEqual(budget["stop_reason"], "token_budget_exceeded")

    def test_time_budget_includes_elapsed_time_from_previous_epoch(self):
        with patch(
            "agent.workflows.nodes.search_node",
            side_effect=AssertionError("an expired time budget must block search"),
        ):
            result = deep_linear_search_node(
                {
                    "input": "topic",
                    "research_plan": ["new query"],
                    "deepsearch_artifacts": {
                        "linear_budget": {
                            "tokens_used": 3,
                            "elapsed_seconds": 5.0,
                            "searches_used": 1,
                        }
                    },
                },
                {"configurable": {"deepsearch_max_seconds": 2.0}},
            )

        self.assertEqual(
            result["deepsearch_artifacts"]["linear_budget"]["stop_reason"],
            "time_budget_exceeded",
        )

    def test_exhausted_budget_skips_followup_planner_model_call(self):
        class FakePlanner:
            called = False

            def invoke(self, messages, config=None):
                self.called = True
                raise AssertionError("planner invocation is over budget")

        planner = FakePlanner()
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=planner,
        ):
            result = deep_followup_plan_node(
                {
                    "input": "research topic",
                    "deepsearch_epoch": 1,
                    "deepsearch_max_epochs": 3,
                    "missing_topics": ["missing evidence"],
                    "deepsearch_artifacts": {
                        "research_budget": {
                            "tokens_used": 8,
                            "elapsed_seconds": 0.5,
                            "searches_used": 1,
                        }
                    },
                },
                {"configurable": {"deepsearch_max_tokens": 1}},
            )

        self.assertFalse(planner.called)
        self.assertEqual(
            result["budget_stop_reason"],
            "token_budget_exceeded",
        )
        self.assertTrue(result["research_plan"])

    def test_tree_search_honors_budget_before_model_initialization(self):
        with (
            patch.object(settings, "deepsearch_max_tokens", 1),
            patch(
                "agent.workflows.nodes.ModelFactory.for_task",
                side_effect=AssertionError("an exhausted budget skips tree setup"),
            ),
            patch(
                "agent.workflows.nodes.search_node",
                side_effect=AssertionError("fallback exceeded its token budget"),
            ),
        ):
            result = deep_tree_search_node(
                {
                    "input": "research topic",
                    "research_plan": ["first query", "second query"],
                },
                {},
            )

        self.assertEqual(result["deepsearch_mode"], "tree")
        self.assertEqual(
            result["deepsearch_artifacts"]["tree_budget"]["stop_reason"],
            "token_budget_exceeded",
        )
        self.assertEqual(result["search_results"], [])

    def test_tree_search_preserves_exhausted_legacy_budget(self):
        with (
            patch(
                "agent.workflows.nodes.ModelFactory.for_task",
                side_effect=AssertionError("an exhausted budget skips tree setup"),
            ),
            patch(
                "agent.workflows.nodes.search_node",
                side_effect=AssertionError("prior tree budget was already exhausted"),
            ),
        ):
            result = deep_tree_search_node(
                {
                    "input": "research topic",
                    "research_plan": ["next query"],
                    "deepsearch_artifacts": {
                        "tree_budget": {
                            "tokens_used": 60,
                            "elapsed_seconds": 4.0,
                            "searches_used": 2,
                            "stop_reason": "",
                        }
                    },
                },
                {"configurable": {"deepsearch_max_tokens": 30}},
            )

        budget = result["deepsearch_artifacts"]["tree_budget"]
        self.assertEqual(budget["tokens_used"], 60)
        self.assertGreaterEqual(budget["elapsed_seconds"], 4.0)
        self.assertEqual(budget["searches_used"], 2)
        self.assertEqual(budget["stop_reason"], "token_budget_exceeded")

    def test_tree_usage_is_merged_into_the_shared_budget(self):
        node = SimpleNamespace(
            id="root",
            query="research topic",
            queries=["research topic"],
        )

        class FakeTree:
            topic = "research topic"
            root_id = "root"
            nodes = {"root": node}

            def get_all_findings(self):
                return [{
                    "query": "research topic",
                    "branch_id": "root",
                    "result": {
                        "title": "Evidence",
                        "url": "https://example.test/evidence",
                        "snippet": "Supported finding",
                    },
                }]

            def to_dict(self):
                return {"topic": self.topic, "nodes": []}

        class FakeExplorer:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self.tree = FakeTree()

            def run(self, topic, state, decompose_root=True):
                return self.tree

            def get_all_findings(self):
                return self.tree.get_all_findings()

            def get_final_summary(self):
                return "Summary"

            def get_budget_stats(self):
                return {
                    "max_seconds": self.kwargs["max_seconds"],
                    "max_tokens": self.kwargs["max_tokens"],
                    "elapsed_seconds": self.kwargs["initial_elapsed_seconds"] + 1,
                    "tokens_used": self.kwargs["initial_tokens_used"] + 20,
                    "searches_used": 2,
                    "stop_reason": "",
                }

        constructor_kwargs = {}

        def create_explorer(**kwargs):
            constructor_kwargs.update(kwargs)
            return FakeExplorer(**kwargs)

        with (
            patch(
                "agent.workflows.nodes.ModelFactory.for_task",
                return_value=object(),
            ),
            patch(
                "agent.workflows.nodes.TreeExplorer",
                side_effect=create_explorer,
            ),
        ):
            result = deep_tree_search_node(
                {
                    "input": "research topic",
                    "research_plan": ["research topic"],
                    "deepsearch_epoch": 1,
                    "deepsearch_artifacts": {
                        "research_budget": {
                            "tokens_used": 50,
                            "elapsed_seconds": 3.0,
                            "searches_used": 4,
                        }
                    },
                },
                {
                    "configurable": {
                        "deepsearch_max_seconds": 60,
                        "deepsearch_max_tokens": 1000,
                        "deepsearch_tree_parallel_branches": 0,
                    }
                },
            )

        budget = result["deepsearch_artifacts"]["research_budget"]
        self.assertEqual(budget["tokens_used"], 70)
        self.assertEqual(budget["searches_used"], 6)
        self.assertGreaterEqual(budget["elapsed_seconds"], 4.0)
        self.assertFalse(constructor_kwargs["count_topic_tokens"])

    def test_empty_budgeted_search_returns_an_explicit_report(self):
        result = deep_writer_node(
            {
                "input": "research topic",
                "research_plan": ["research query"],
                "search_results": [],
                "deepsearch_artifacts": {
                    "budget_stop_reason": "token_budget_exceeded",
                    "linear_budget": {
                        "tokens_used": 5,
                        "elapsed_seconds": 0.2,
                        "stop_reason": "token_budget_exceeded",
                    },
                    "knowledge_gap_analysis": {"overall_coverage": 0.4},
                },
            },
            {},
        )

        self.assertIn("无法生成有来源支持", result["final_report"])
        self.assertIn("token_budget_exceeded", result["final_report"])
        self.assertEqual(
            result["deepsearch_artifacts"]["knowledge_gap_analysis"]["overall_coverage"],
            0.4,
        )

    def test_writer_preserves_prior_budget_and_gap_diagnostics(self):
        class FakeWriter:
            def invoke(self, messages, config=None):
                raise AssertionError("budget exhaustion must skip model writing")

        gap_analysis = {"overall_coverage": 0.65, "gaps": []}
        linear_budget = {"tokens_used": 25, "stop_reason": ""}
        state = {
            "input": "research topic",
            "research_plan": ["research topic official evidence"],
            "search_results": [{
                "query": "research topic official evidence",
                "title": "Evidence",
                "url": "https://example.test/evidence",
                "snippet": "A supported finding.",
                "content": "A supported finding with evidence.",
            }],
            "research_tree": ResearchTree.from_queries(
                "research topic",
                ["research topic official evidence"],
            ).to_dict(),
            "deepsearch_artifacts": {
                "knowledge_gap_analysis": gap_analysis,
                "linear_budget": linear_budget,
                "research_budget": {
                    "max_tokens": 10,
                    "tokens_used": 25,
                    "stop_reason": "token_budget_exceeded",
                },
            },
        }

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=FakeWriter(),
        ):
            result = deep_writer_node(state, config={})

        self.assertEqual(
            result["deepsearch_artifacts"]["knowledge_gap_analysis"],
            gap_analysis,
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["linear_budget"]["tokens_used"],
            linear_budget["tokens_used"],
        )
        self.assertEqual(
            result["deepsearch_artifacts"]["research_budget"]["tokens_used"],
            25,
        )
        self.assertIn("预算限制提前停止", result["final_report"])
        self.assertIn("[来源1]", result["final_report"])
        self.assertEqual(result["writer_agent_result"]["status"], "skipped")
        self.assertEqual(result["writer_agent_result"]["tools_used"], [])

    def test_writer_skips_model_when_prompt_would_exceed_remaining_budget(self):
        class FakeWriter:
            def invoke(self, messages, config=None):
                raise AssertionError("oversized writer prompt must not be sent")

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=FakeWriter(),
        ):
            result = deep_writer_node(
                {
                    "input": "research topic",
                    "route": "deep",
                    "research_plan": ["research topic evidence"],
                    "search_results": [{
                        "query": "research topic evidence",
                        "title": "Evidence",
                        "url": "https://example.test/evidence",
                        "snippet": "A supported finding.",
                    }],
                    "deepsearch_artifacts": {
                        "research_budget": {
                            "max_tokens": 500,
                            "tokens_used": 499,
                            "stop_reason": "",
                        },
                    },
                },
                config={"configurable": {"deepsearch_max_tokens": 500}},
            )

        budget = result["deepsearch_artifacts"]["research_budget"]
        self.assertEqual(budget["stop_reason"], "token_budget_exceeded")
        self.assertEqual(result["writer_agent_result"]["status"], "skipped")
        self.assertIn("预算限制提前停止", result["final_report"])

    def test_writer_reserves_tokens_for_evaluator(self):
        class FakeWriter:
            def invoke(self, messages, config=None):
                raise AssertionError("writer must not consume the evaluator reserve")

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=FakeWriter(),
        ):
            result = deep_writer_node(
                {
                    "input": "research topic",
                    "route": "deep",
                    "research_plan": ["research topic evidence"],
                    "search_results": [{
                        "query": "research topic evidence",
                        "title": "Evidence",
                        "url": "https://example.test/evidence",
                        "snippet": "A supported finding.",
                    }],
                    "deepsearch_artifacts": {
                        "research_budget": {
                            "max_tokens": 10000,
                            "tokens_used": 9500,
                            "max_seconds": 120,
                            "elapsed_seconds": 20,
                            "stop_reason": "",
                        },
                    },
                },
                config={"configurable": {
                    "deepsearch_max_tokens": 10000,
                    "deepsearch_max_seconds": 120,
                }},
            )

        budget = result["deepsearch_artifacts"]["research_budget"]
        self.assertEqual(budget["stop_reason"], "")
        self.assertEqual(result["writer_agent_result"]["status"], "skipped")
        self.assertEqual(
            result["writer_agent_result"]["error"],
            "evaluator_budget_reserved",
        )

    def test_writer_budget_leaves_room_for_evaluator_model_call(self):
        writer_messages = []
        evaluator_model = None

        class FakeWriter:
            def invoke(self, messages, config=None):
                writer_messages.extend(messages)
                return SimpleNamespace(
                    content="# 研究结论\n\n基于证据的简短结论。[来源1]",
                    usage_metadata={"total_tokens": 1200},
                )

            def bind(self, **kwargs):
                return self

        class FakeEvaluator:
            output_limit = None

            def bind(self, **kwargs):
                self.output_limit = kwargs["max_tokens"]
                return self

            def invoke(self, messages, config=None):
                return SimpleNamespace(
                    content=(
                        '{"verdict":"pass","evaluation":"证据充分",'
                        '"eval_dimensions":{"accuracy":0.9,"completeness":0.8},'
                        '"missing_topics":[]}'
                    ),
                    usage_metadata={"total_tokens": 300},
                )

        evaluator_model = FakeEvaluator()

        def model_for_task(task, **kwargs):
            return evaluator_model if task == "evaluation" else FakeWriter()

        evidence = [
            {
                "query": f"topic {index}",
                "title": f"Evidence {index}",
                "url": f"https://example.test/{index}",
                "snippet": "支持材料。" * 1000,
            }
            for index in range(12)
        ]
        state = {
            "input": "Compare the evidence and state the limitations.",
            "route": "deep",
            "deepsearch_mode": "tree",
            "research_plan": ["official docs", "limitations"],
            "search_results": evidence,
            "sources": evidence,
            "deepsearch_artifacts": {
                "tree_summary": "分支摘要。" * 2000,
                "research_budget": {
                    "max_tokens": 10000,
                    "tokens_used": 3000,
                    "max_seconds": 120,
                    "elapsed_seconds": 20,
                    "stop_reason": "",
                },
            },
        }

        with (
            patch("agent.workflows.nodes.ModelFactory.for_task", side_effect=model_for_task),
            patch.object(settings, "deepsearch_use_gap_analysis", False),
        ):
            written = deep_writer_node(
                state,
                config={"configurable": {
                    "deepsearch_max_tokens": 10000,
                    "deepsearch_max_seconds": 120,
                }},
            )
            evaluated = evaluator_node(
                {**state, **written},
                config={"configurable": {
                    "deepsearch_max_tokens": 10000,
                    "deepsearch_max_seconds": 120,
                }},
            )

        self.assertLess(len(str(writer_messages)), 12000)
        self.assertEqual(written["writer_agent_result"]["status"], "completed")
        self.assertEqual(evaluated["evaluator_agent_result"]["status"], "completed")
        self.assertNotEqual(evaluated["verdict"], "budget_exhausted")
        self.assertIsNotNone(evaluator_model.output_limit)
        self.assertFalse(evaluated["quality_summary"].get("evaluation_skipped", False))
        self.assertLessEqual(
            evaluated["deepsearch_artifacts"]["research_budget"]["tokens_used"],
            10000,
        )

    def test_evaluator_skips_model_when_research_budget_is_exhausted(self):
        state = {
            "input": "research topic",
            "route": "deep",
            "deepsearch_mode": "tree",
            "draft_report": "Evidence summary [来源1]。",
            "search_results": [{
                "title": "Evidence",
                "url": "https://example.test/evidence",
            }],
            "deepsearch_artifacts": {
                "research_budget": {
                    "max_tokens": 10,
                    "tokens_used": 10,
                    "stop_reason": "token_budget_exceeded",
                },
            },
        }
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            side_effect=AssertionError("exhausted budget must skip evaluation"),
        ):
            result = evaluator_node(state, config={})

        self.assertEqual(result["verdict"], "budget_exhausted")
        self.assertTrue(result["quality_summary"]["evaluation_skipped"])
        self.assertEqual(
            result["deepsearch_artifacts"]["research_budget"]["tokens_used"],
            10,
        )

    def test_evaluator_caps_completion_tokens_to_remaining_budget(self):
        class FakeModel:
            output_limit = None

            def bind(self, **kwargs):
                self.output_limit = kwargs["max_tokens"]
                return self

            def invoke(self, messages, config=None):
                return SimpleNamespace(
                    content='{"verdict":"pass","evaluation":"ok","eval_dimensions":{},"missing_topics":[]}',
                    usage_metadata={"total_tokens": 200},
                )

        model = FakeModel()
        state = {
            "input": "research topic",
            "route": "deep",
            "deepsearch_mode": "tree",
            "draft_report": "A short evidence-based report. [来源1]",
            "search_results": [{
                "title": "Evidence",
                "url": "https://example.test/evidence",
                "snippet": "A short supporting excerpt.",
            }],
            "sources": [{
                "title": "Evidence",
                "url": "https://example.test/evidence",
            }],
        }
        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model), patch.object(
            settings, "deepsearch_use_gap_analysis", False
        ):
            result = evaluator_node(
                state,
                config={"configurable": {"deepsearch_max_tokens": 10000}},
            )

        self.assertIsNotNone(model.output_limit)
        self.assertGreater(model.output_limit, 0)
        self.assertEqual(model.output_limit, 1536)
        self.assertLessEqual(
            result["deepsearch_artifacts"]["research_budget"]["tokens_used"],
            10000,
        )

    def test_evaluator_skips_when_prompt_leaves_no_completion_budget(self):
        class FakeModel:
            def invoke(self, messages, config=None):
                raise AssertionError("evaluation must not exceed the remaining budget")

        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=FakeModel()):
            result = evaluator_node(
                {
                    "input": "research topic",
                    "route": "deep",
                    "deepsearch_mode": "tree",
                    "draft_report": "A report.",
                    "search_results": [],
                },
                config={"configurable": {"deepsearch_max_tokens": 100}},
            )

        self.assertEqual(result["verdict"], "budget_exhausted")
        self.assertTrue(result["quality_summary"]["evaluation_skipped"])
        self.assertEqual(
            result["deepsearch_artifacts"]["research_budget"]["stop_reason"],
            "token_budget_exceeded",
        )

    def test_revision_is_output_bounded_and_preserves_re_evaluation_budget(self):
        class FakeModel:
            output_limit = None

            def bind(self, **kwargs):
                self.output_limit = kwargs["max_tokens"]
                return self

            def invoke(self, messages, config=None):
                return SimpleNamespace(
                    content="修订后的报告 [来源1]",
                    usage_metadata={"total_tokens": 300},
                )

        model = FakeModel()
        state = {
            "input": "研究主题",
            "route": "deep",
            "deepsearch_mode": "tree",
            "draft_report": "原始报告 [来源1]",
            "search_results": [{
                "title": "来源",
                "url": "https://example.test/source",
                "snippet": "证据内容",
            }],
            "missing_topics": ["尚缺的内容"],
            "revision_count": 0,
            "deepsearch_artifacts": {
                "research_budget": {
                    "max_tokens": 10000,
                    "tokens_used": 1000,
                    "max_seconds": 120,
                    "elapsed_seconds": 1,
                    "stop_reason": "",
                },
            },
        }

        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model):
            result = revise_node(
                state,
                config={"configurable": {
                    "deepsearch_max_tokens": 10000,
                    "deepsearch_max_seconds": 120,
                }},
            )

        self.assertGreater(model.output_limit, 0)
        self.assertLessEqual(model.output_limit, 1024)
        self.assertEqual(result["revision_count"], 1)
        self.assertTrue(result["draft_report"].startswith("修订后的报告 [来源1]"))
        self.assertEqual(
            result["deepsearch_artifacts"]["research_budget"]["tokens_used"],
            1300,
        )

    def test_revision_skips_model_when_it_would_consume_re_evaluation_reserve(self):
        model = SimpleNamespace(
            invoke=lambda *_args, **_kwargs: self.fail(
                "revision must not consume the reserved re-evaluation budget"
            )
        )
        state = {
            "input": "研究主题",
            "route": "deep",
            "deepsearch_mode": "tree",
            "draft_report": "原始报告 [来源1]",
            "search_results": [{
                "title": "来源",
                "url": "https://example.test/source",
                "snippet": "证据内容",
            }],
            "missing_topics": ["尚缺的内容"],
            "revision_count": 0,
            "deepsearch_artifacts": {
                "research_budget": {
                    "max_tokens": 10000,
                    "tokens_used": 9900,
                    "max_seconds": 120,
                    "elapsed_seconds": 1,
                    "stop_reason": "",
                },
            },
        }

        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model):
            result = revise_node(
                state,
                config={"configurable": {
                    "deepsearch_max_tokens": 10000,
                    "deepsearch_max_seconds": 120,
                }},
            )

        self.assertEqual(result["revision_agent_result"]["status"], "skipped")
        self.assertEqual(result["revision_count"], 0)
        self.assertEqual(
            result["deepsearch_artifacts"]["research_budget"]["stop_reason"],
            "token_budget_exceeded",
        )

    def test_budget_exhaustion_prevents_another_research_epoch(self):
        self.assertEqual(
            evaluation_decision({
                "route": "deep",
                "verdict": "revise",
                "deepsearch_epoch": 1,
                "deepsearch_max_epochs": 3,
                "revision_count": 0,
                "max_revisions": 0,
                "search_results": [],
                "deepsearch_artifacts": {
                    "linear_budget": {
                        "stop_reason": "token_budget_exceeded",
                    }
                },
            }),
            "human_review",
        )

    def test_budget_exhaustion_prevents_revision_even_with_existing_results(self):
        self.assertEqual(
            evaluation_decision({
                "route": "deep",
                "verdict": "revise",
                "revision_count": 0,
                "max_revisions": 2,
                "search_results": [{"title": "evidence"}],
                "deepsearch_artifacts": {
                    "research_budget": {
                        "stop_reason": "time_budget_exceeded",
                    },
                },
            }),
            "human_review",
        )


if __name__ == "__main__":
    unittest.main()
