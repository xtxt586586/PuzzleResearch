import unittest
from unittest.mock import MagicMock, patch

from langchain_core.messages import AIMessage

from agent.core.model_factory import ModelFactory
from agent.core.profile import AgentProfile
from agent.workflows.nodes import direct_answer_node


class ProfileRuntimeTests(unittest.TestCase):
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