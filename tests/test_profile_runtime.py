import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agent.core.model_factory import ModelFactory
from agent.core.profile import AgentProfile
from agent.workflows.nodes import direct_answer_node


class ProfileRuntimeTests(unittest.TestCase):
    def test_task_specific_model_override_precedes_request_and_profile_models(self):
        profile = AgentProfile(
            name="profile-agent",
            model="deepseek-chat",
        )
        config = {
            "configurable": {
                "agent_profile": profile,
                "model_override": "deepseek-reasoner",
                "planning_model": "deepseek-planner-free",
            }
        }

        with patch.object(
            ModelFactory,
            "create",
            return_value=MagicMock(),
        ) as create:
            ModelFactory.for_task("planning", config=config)
            ModelFactory.for_task("writing", config=config)

        self.assertEqual(
            create.call_args_list[0].kwargs["model"],
            "deepseek-planner-free",
        )
        self.assertEqual(
            create.call_args_list[1].kwargs["model"],
            "deepseek-reasoner",
        )

    def test_reasoning_model_override_applies_only_to_reasoning_tasks(self):
        config = {
            "configurable": {
                "reasoning_model": "deepseek-reasoner-free",
                "model_override": "deepseek-chat",
            }
        }

        with patch.object(
            ModelFactory,
            "create",
            return_value=MagicMock(),
        ) as create:
            ModelFactory.for_task("evaluation", config=config)
            ModelFactory.for_task("research", config=config)

        self.assertEqual(
            create.call_args_list[0].kwargs["model"],
            "deepseek-reasoner-free",
        )
        self.assertEqual(
            create.call_args_list[1].kwargs["model"],
            "deepseek-chat",
        )

    def test_weaver_task_aliases_accept_task_specific_overrides(self):
        config = {
            "configurable": {
                "query_gen_model": "deepseek-query-free",
                "critique_model": "deepseek-critic-free",
                "synthesis_model": "deepseek-synthesis-free",
                "reflection_model": "deepseek-reflection-free",
                "gap_analysis_model": "deepseek-gap-free",
            }
        }

        with patch.object(
            ModelFactory,
            "create",
            return_value=MagicMock(),
        ) as create:
            for task in (
                "query_gen",
                "critique",
                "synthesis",
                "reflection",
                "gap_analysis",
            ):
                ModelFactory.for_task(task, config=config)

        self.assertEqual(
            [call.kwargs["model"] for call in create.call_args_list],
            [
                "deepseek-query-free",
                "deepseek-critic-free",
                "deepseek-synthesis-free",
                "deepseek-reflection-free",
                "deepseek-gap-free",
            ],
        )

    def test_request_model_overrides_profile_model(self):
        profile = AgentProfile(
            name="profile-agent",
            model="deepseek-chat",
        )

        config = {
            "configurable": {
                "agent_profile": profile,
                "model_override": "deepseek-reasoner",
            }
        }

        with patch.object(
            ModelFactory,
            "create",
            return_value=MagicMock(),
        ) as create:
            ModelFactory.for_task(
                "writing",
                config=config,
            )

        self.assertEqual(
            create.call_args.kwargs["model"],
            "deepseek-reasoner",
        )

    def test_profile_model_is_used_without_request_override(self):
        profile = AgentProfile(
            name="profile-agent",
            model="deepseek-reasoner",
        )

        config = {
            "configurable": {
                "agent_profile": profile,
            }
        }

        with patch.object(
            ModelFactory,
            "create",
            return_value=MagicMock(),
        ) as create:
            ModelFactory.for_task(
                "writing",
                config=config,
            )

        self.assertEqual(
            create.call_args.kwargs["model"],
            "deepseek-reasoner",
        )

    def test_system_prompt_is_added_to_direct_answer(self):
        model = MagicMock()
        model.stream = None
        model.invoke.return_value = AIMessage(
            content="测试回答"
        )

        profile = AgentProfile(
            name="prompt-agent",
            system_prompt="必须使用简洁中文回答。",
        )

        config = {
            "configurable": {
                "agent_profile": profile,
            }
        }

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            result = direct_answer_node(
                {
                    "input": "测试问题",
                },
                config,
            )

        self.assertEqual(
            result["final_report"],
            "测试回答",
        )

        messages = model.invoke.call_args.args[0]

        self.assertIn(
            "必须使用简洁中文回答。",
            messages[0].content,
        )


if __name__ == "__main__":
    unittest.main()
