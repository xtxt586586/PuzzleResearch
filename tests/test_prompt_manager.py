import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.prompts import PromptManager, get_prompt_manager, reset_prompt_manager
from tools.registry import ToolDefinition, ToolRegistry


class PromptManagerTests(unittest.TestCase):
    def tearDown(self):
        reset_prompt_manager()

    def test_styles_return_distinct_agent_planner_guidance(self):
        simple = PromptManager("simple").get_agent_planner_prompt()
        enhanced = PromptManager("enhanced").get_agent_planner_prompt()

        self.assertNotEqual(simple, enhanced)
        self.assertIn("所必需的下一步", simple)
        self.assertIn("不得编造", enhanced)

    def test_prompt_catalog_covers_research_and_answer_nodes(self):
        manager = PromptManager("enhanced")
        for prompt_type in (
            "web_planner",
            "deep_research_planner",
            "web_writer",
            "deep_writer",
            "direct_answer",
            "report_evaluator",
            "report_reviser",
            "deep_followup_planner",
            "clarifier",
            "tool_selector",
            "research_coordinator",
            "task_router",
            "domain_classifier",
            "conversation_summarizer",
            "tree_branch_summarizer",
            "tree_merge_summarizer",
            "task_planner",
            "presentation_outline",
            "presentation_slide_expander",
            "search_result_summarizer",
            "multimodal_image_analyzer",
            "research_compressor",
        ):
            with self.subTest(prompt_type=prompt_type):
                self.assertTrue(manager.get_prompt(prompt_type))

    def test_custom_prompt_overrides_selected_style(self):
        manager = PromptManager(
            "simple",
            custom_prompts={"agent_planner": "Use the custom planner rules."},
        )

        self.assertEqual(
            manager.get_agent_planner_prompt(),
            "Use the custom planner rules.",
        )

    def test_custom_prompt_can_be_loaded_from_utf8_file(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "planner.txt"
            prompt_path.write_text("自定义规划规则", encoding="utf-8")
            manager = PromptManager("enhanced")
            manager.load_custom_prompt("agent_planner", prompt_path)

        self.assertEqual(manager.get_agent_planner_prompt(), "自定义规划规则")

    def test_prompt_directory_loads_named_overrides(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "deep_writer.txt"
            prompt_path.write_text("Custom deep writer", encoding="utf-8")
            manager = PromptManager("simple")
            manager.load_prompt_directory(directory)

        self.assertEqual(manager.get_prompt("deep_writer"), "Custom deep writer")
        self.assertIn("直接问答", manager.get_prompt("direct_answer"))

    def test_invalid_style_and_empty_custom_prompt_are_rejected(self):
        with self.assertRaises(ValueError):
            PromptManager("unknown")
        with self.assertRaises(ValueError):
            PromptManager().set_custom_prompt("agent_planner", "  ")

    def test_runtime_config_can_supply_prompt_manager(self):
        manager = PromptManager("simple")
        self.assertIs(
            get_prompt_manager({"configurable": {"prompt_manager": manager}}),
            manager,
        )

    def test_standalone_task_planner_uses_managed_prompt_and_json_contract(self):
        from tools.planning_tool import plan_steps

        class Model:
            messages = None

            def invoke(self, messages):
                self.messages = messages
                return type("Response", (), {"content": '{"steps": ["第一步"]}'})()

        model = Model()
        manager = PromptManager(
            "simple",
            custom_prompts={"task_planner": "CUSTOM TASK PLANNER"},
        )
        with patch("tools.planning_tool.get_prompt_manager", return_value=manager), patch(
            "tools.planning_tool.ModelFactory.for_task", return_value=model
        ):
            result = plan_steps("完成目标")

        self.assertEqual(result["steps"], ["第一步"])
        self.assertIn("CUSTOM TASK PLANNER", model.messages[0].content)
        self.assertIn('{"steps":', model.messages[0].content)

    def test_search_summary_uses_managed_prompt_without_live_api(self):
        from tools.search.search import _summarize_content

        class Model:
            messages = None

            def invoke(self, messages):
                self.messages = messages
                return type("Response", (), {"content": "摘要"})()

        model = Model()
        manager = PromptManager(
            "simple",
            custom_prompts={
                "search_result_summarizer": "CUSTOM SEARCH SUMMARY"
            },
        )
        with patch("tools.search.search.settings.deepseek_api_key", "test-key"), patch(
            "tools.search.search.get_prompt_manager", return_value=manager
        ), patch("langchain_openai.ChatOpenAI", return_value=model):
            result = _summarize_content("正文")

        self.assertEqual(result, "摘要")
        self.assertIn("CUSTOM SEARCH SUMMARY", model.messages[0].content)

    def test_agent_planner_uses_managed_prompt_and_keeps_tool_contract(self):
        from agent.workflows.nodes import agent_plan_node

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {"content": '{"action": "finish", "final_report": "done"}'})()

        model = CapturingModel()
        registry = ToolRegistry(
            [ToolDefinition("echo", "Returns input", lambda text: text)]
        )
        manager = PromptManager(
            "simple",
            custom_prompts={"agent_planner": "CUSTOM PLANNER GUIDANCE"},
        )

        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model), patch(
            "agent.workflows.nodes._select_agent_tools",
            return_value=registry.list_tools(),
        ):
            agent_plan_node(
                {"input": "Answer this request"},
                config={
                    "configurable": {
                        "tool_registry": registry,
                        "prompt_manager": manager,
                    }
                },
            )

        system_prompt = model.messages[0].content
        self.assertIn("CUSTOM PLANNER GUIDANCE", system_prompt)
        self.assertIn('"name": "echo"', system_prompt)
        self.assertIn('"action": "tool"', system_prompt)

    def test_web_planner_uses_managed_prompt_and_json_contract(self):
        from agent.workflows.nodes import web_plan_node

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {"content": '{"queries": ["q1"]}'} )()

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"web_planner": "CUSTOM WEB PLANNER"},
        )
        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model):
            result = web_plan_node(
                {"input": "Find sources", "suggested_queries": []},
                config={"configurable": {"prompt_manager": manager}},
            )

        self.assertEqual(result["research_plan"], ["q1"])
        prompt = model.messages[0].content
        self.assertIn("CUSTOM WEB PLANNER", prompt)
        self.assertIn('"queries"', prompt)

    def test_direct_answer_uses_managed_prompt(self):
        from agent.workflows.nodes import direct_answer_node

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {"content": "Answer"})()

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"direct_answer": "CUSTOM DIRECT ANSWER"},
        )
        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model):
            direct_answer_node(
                {"input": "Question"},
                config={"configurable": {"prompt_manager": manager}},
            )

        self.assertIn("CUSTOM DIRECT ANSWER", model.messages[0].content)
        self.assertIn("不联网", model.messages[0].content)

    def test_web_writer_uses_managed_prompt_and_keeps_citation_guidance(self):
        from agent.workflows.nodes import writer_node

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {"content": "Supported answer [来源1]"})()

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"web_writer": "CUSTOM WEB WRITER"},
        )
        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model):
            writer_node(
                {
                    "input": "Question",
                    "search_results": [{
                        "title": "Evidence",
                        "url": "https://example.test/source",
                        "snippet": "Evidence excerpt",
                    }],
                },
                config={"configurable": {"prompt_manager": manager}},
            )

        prompt = model.messages[0].content
        self.assertIn("CUSTOM WEB WRITER", prompt)
        self.assertIn("[来源1]", prompt)

    def test_evaluator_uses_managed_prompt_and_keeps_json_contract(self):
        import json

        from agent.workflows.nodes import evaluator_node
        from agent.workflows.quality_assessor import QualityReport

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {
                    "content": json.dumps({
                        "verdict": "pass",
                        "evaluation": "Supported",
                        "eval_dimensions": {"accuracy": 0.9},
                        "missing_topics": [],
                    })
                })()

        class FakeAssessor:
            def __init__(self, llm=None, config=None):
                pass

            def assess(self, report, scraped_content, sources=None):
                return QualityReport(
                    overall_score=0.9,
                    citation_coverage_score=1.0,
                )

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"report_evaluator": "CUSTOM EVALUATOR GUIDANCE"},
        )
        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model), patch(
            "agent.workflows.quality_assessor.QualityAssessor", FakeAssessor
        ):
            result = evaluator_node(
                {
                    "route": "web",
                    "input": "Summarize the source",
                    "draft_report": "A supported finding [来源1].",
                    "sources": [{"title": "Source", "url": "https://example.test/source"}],
                },
                config={"configurable": {"prompt_manager": manager}},
            )

        self.assertEqual(result["verdict"], "pass")
        prompt = model.messages[0].content
        self.assertIn("CUSTOM EVALUATOR GUIDANCE", prompt)
        self.assertIn('"verdict"', prompt)

    def test_reviser_uses_managed_prompt_and_keeps_evidence_rules(self):
        from agent.workflows.nodes import revise_node

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {"content": "Revised answer [来源1]."})()

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"report_reviser": "CUSTOM REVISION GUIDANCE"},
        )
        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model):
            revise_node(
                {
                    "input": "Question",
                    "draft_report": "Original answer.",
                    "missing_topics": ["Missing detail"],
                    "search_results": [{
                        "title": "Evidence",
                        "url": "https://example.test/evidence",
                        "snippet": "Evidence excerpt",
                    }],
                },
                config={"configurable": {"prompt_manager": manager}},
            )

        prompt = model.messages[0].content
        self.assertIn("CUSTOM REVISION GUIDANCE", prompt)
        self.assertIn("不能编造事实", prompt)
        self.assertIn("不要单独添加参考来源列表", prompt)

    def test_followup_planner_uses_managed_prompt_and_query_contract(self):
        from agent.workflows.nodes import deep_followup_plan_node

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {
                    "content": '{"queries": ["supplement evidence"], "reasoning": "gap"}'
                })()

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"deep_followup_planner": "CUSTOM FOLLOW-UP PLANNER"},
        )
        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model):
            result = deep_followup_plan_node(
                {
                    "input": "research topic",
                    "deepsearch_epoch": 1,
                    "deepsearch_max_epochs": 3,
                    "missing_topics": ["missing evidence"],
                    "deepsearch_query_history": ["first query"],
                },
                config={"configurable": {"prompt_manager": manager}},
            )

        self.assertTrue(result["research_plan"])
        self.assertIn("CUSTOM FOLLOW-UP PLANNER", model.messages[0].content)
        self.assertIn('"queries"', model.messages[0].content)

    def test_clarifier_uses_managed_prompt_and_single_question_rule(self):
        from agent.workflows.nodes import clarify_node

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {"content": "What is the target audience?"})()

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"clarifier": "CUSTOM CLARIFIER GUIDANCE"},
        )
        with patch("agent.workflows.nodes.ModelFactory.for_task", return_value=model):
            clarify_node(
                {"input": "Make it better"},
                config={"configurable": {"prompt_manager": manager}},
            )

        prompt = model.messages[0].content
        self.assertIn("CUSTOM CLARIFIER GUIDANCE", prompt)
        self.assertIn("只输出一个问题", prompt)

    def test_coordinator_uses_managed_prompt_and_action_contract(self):
        from agent.workflows.coordinator import coordinator_node

        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {
                    "content": '{"action":"research","reasoning":"fill gap",'
                    '"priority_topics":["cost"]}'
                })()

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"research_coordinator": "CUSTOM COORDINATOR GUIDANCE"},
        )
        state = {
            "input": "Compare options",
            "route": "deep",
            "research_plan": ["option comparison"],
            "search_results": [{"url": "https://example.test/source"}],
            "draft_report": "Draft report",
            "final_report": "Draft report",
            "verdict": "revise",
            "evaluation": "Needs cost evidence",
            "missing_topics": ["cost"],
            "deepsearch_epoch": 1,
            "deepsearch_max_epochs": 3,
            "revision_count": 0,
            "max_revisions": 2,
        }
        with patch("agent.workflows.coordinator.ModelFactory.for_task", return_value=model):
            result = coordinator_node(
                state,
                {"configurable": {"prompt_manager": manager}},
            )

        self.assertEqual(result["coordinator_action"], "research")
        prompt = model.messages[0].content
        self.assertIn("CUSTOM COORDINATOR GUIDANCE", prompt)
        self.assertIn('"priority_topics"', prompt)

    def test_tool_selector_uses_managed_prompt(self):
        from agent.workflows.nodes import _select_agent_tools

        class SelectionModel:
            messages = None

            def with_structured_output(self, schema, method=None):
                return self

            def invoke(self, messages):
                self.messages = messages
                return {"tools": ["search"]}

        model = SelectionModel()
        tools = [
            ToolDefinition(name, f"Description for {name}", lambda: "ok")
            for name in ("search", "python", "browser")
        ]
        manager = PromptManager(
            "simple",
            custom_prompts={"tool_selector": "CUSTOM TOOL SELECTOR GUIDANCE"},
        )
        with patch("agent.workflows.nodes.settings.agent_tool_selector_enabled", True), patch(
            "agent.workflows.nodes.settings.agent_tool_selector_max_tools", 1
        ), patch("agent.workflows.nodes.settings.agent_tool_selector_always_include", ""), patch(
            "agent.workflows.nodes.settings.enable_todo_middleware", False
        ):
            selected = _select_agent_tools(
                model,
                tools,
                user_input="Search for current evidence",
                tool_history="None",
                config={"configurable": {"prompt_manager": manager}},
            )

        self.assertEqual([tool.name for tool in selected], ["search"])
        self.assertIn("CUSTOM TOOL SELECTOR GUIDANCE", model.messages[0].content)

    def test_task_router_uses_managed_prompt(self):
        from agent.core.smart_router import SmartRouter

        class CapturingModel:
            messages = None

            def with_structured_output(self, schema):
                return self

            def invoke(self, messages, config=None):
                self.messages = messages
                return {"route": "direct", "reasoning": "simple question"}

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"task_router": "CUSTOM ROUTER GUIDANCE"},
        )
        with patch("agent.core.smart_router.ModelFactory.for_task", return_value=model):
            result = SmartRouter().route(
                "Explain a simple concept",
                config={"configurable": {"prompt_manager": manager}},
            )

        self.assertEqual(result.route, "direct")
        self.assertIn("CUSTOM ROUTER GUIDANCE", model.messages[0].content)

    def test_domain_classifier_uses_managed_prompt(self):
        from agent.workflows.domain_router import DomainClassifier

        class CapturingModel:
            prompt = None

            def with_structured_output(self, schema):
                return self

            def invoke(self, prompt, config=None):
                self.prompt = prompt
                return {
                    "domain": "technical",
                    "confidence": 0.9,
                    "reasoning": "software topic",
                    "search_hints": ["technical docs"],
                }

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"domain_classifier": "CUSTOM DOMAIN CLASSIFIER"},
        )
        result = DomainClassifier(
            llm=model,
            config={"configurable": {"prompt_manager": manager}},
        ).classify("software architecture")

        self.assertEqual(result.domain.value, "technical")
        self.assertIn("CUSTOM DOMAIN CLASSIFIER", model.prompt)

    def test_conversation_summary_uses_default_managed_prompt(self):
        from langchain_core.messages import HumanMessage

        from agent.core.message_utils import summarize_messages
        from agent.prompts import set_prompt_manager

        class CapturingModel:
            messages = None

            def invoke(self, messages):
                self.messages = messages
                return type("Response", (), {"content": "Summary"})()

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"conversation_summarizer": "CUSTOM SUMMARY GUIDANCE"},
        )
        set_prompt_manager(manager)
        with patch("agent.core.message_utils.ModelFactory.create", return_value=model):
            summarize_messages([HumanMessage(content="Remember this goal")])

        self.assertIn("CUSTOM SUMMARY GUIDANCE", model.messages[0].content)

    def test_tree_summaries_use_managed_prompts(self):
        from agent.workflows.research_tree import (
            ResearchNodeStatus,
            ResearchTreeNode,
            TreeExplorer,
        )

        class CapturingModel:
            def __init__(self, response):
                self.response = response
                self.messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return type("Response", (), {"content": self.response})()

        branch_model = CapturingModel("Branch summary")
        merge_model = CapturingModel("Merged summary")
        manager = PromptManager(
            "simple",
            custom_prompts={
                "tree_branch_summarizer": "CUSTOM BRANCH SUMMARY",
                "tree_merge_summarizer": "CUSTOM TREE MERGE",
            },
        )
        explorer = TreeExplorer(
            planner_llm=branch_model,
            researcher_llm=branch_model,
            writer_llm=merge_model,
            search_func=lambda query: [],
            config={"configurable": {"prompt_manager": manager}},
        )
        branch = ResearchTreeNode(
            topic="Research topic",
            findings=[{"result": {"title": "Source", "content": "Finding", "url": "https://example.test"}}],
        )
        completed = ResearchTreeNode(
            topic="Branch A",
            summary="Branch A finding",
            status=ResearchNodeStatus.COMPLETED,
        )

        self.assertEqual(explorer._summarize_branch(branch), "Branch summary")
        self.assertEqual(explorer.merge_branches([completed]), "Merged summary")
        self.assertIn("CUSTOM BRANCH SUMMARY", branch_model.messages[0]["content"])
        self.assertIn("CUSTOM TREE MERGE", merge_model.messages[0]["content"])

    def test_set_prompt_manager_remains_the_default_until_settings_change(self):
        manager = PromptManager("simple")
        from agent.prompts import set_prompt_manager

        set_prompt_manager(manager)
        self.assertIs(get_prompt_manager(), manager)

    def test_settings_select_style_and_custom_file(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_path = Path(directory) / "planner.txt"
            prompt_path.write_text("configured prompt", encoding="utf-8")
            with patch("agent.prompts.prompt_manager.settings.prompt_style", "simple"), patch(
                "agent.prompts.prompt_manager.settings.agent_planner_prompt_file",
                str(prompt_path),
            ):
                manager = get_prompt_manager()

        self.assertEqual(manager.prompt_style, "simple")
        self.assertEqual(manager.get_agent_planner_prompt(), "configured prompt")


if __name__ == "__main__":
    unittest.main()
