import unittest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from unittest.mock import MagicMock, patch

from agent.core.runtime import AgentRuntime
from agent.workflows.nodes import (
    _build_model_user_content,
    direct_answer_node,
)
from agent.prompts import PromptManager


class MultimodalInputTests(unittest.TestCase):
    def test_text_only_content_remains_a_plain_string(self):
        self.assertEqual(
            _build_model_user_content("请总结这段内容。", []),
            "请总结这段内容。",
        )

    def test_image_payload_is_converted_to_openai_compatible_content(self):
        content = _build_model_user_content(
            "请分析图片。",
            [
                {
                    "name": "diagram.png",
                    "mime": "image/png",
                    "data": "YWJj",
                }
            ],
        )

        self.assertEqual(
            content,
            [
                {"type": "text", "text": "请分析图片。"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "data:image/png;base64,YWJj",
                    },
                },
            ],
        )

    def test_data_url_is_preserved_and_non_image_mime_is_ignored(self):
        content = _build_model_user_content(
            "看图",
            [
                {
                    "mime": "image/jpeg",
                    "data": "data:image/jpeg;base64,AA==",
                },
                {
                    "mime": "application/pdf",
                    "data": "JVBERi0x",
                },
            ],
        )

        self.assertEqual(len(content), 2)
        self.assertEqual(
            content[1]["image_url"]["url"],
            "data:image/jpeg;base64,AA==",
        )

    def test_runtime_carries_configured_images_into_graph_state(self):
        payload = {"input": "请分析图片。"}
        config = {
            "configurable": {
                "images": [
                    {"name": "diagram.png", "mime": "image/png", "data": "YWJj"}
                ]
            }
        }

        result = AgentRuntime._attach_config_metadata(payload, config)

        self.assertEqual(result["images"], config["configurable"]["images"])

    def test_direct_answer_passes_images_to_the_model(self):
        model = MagicMock()
        model.stream = None
        model.invoke.return_value.content = "图中是一张流程图。"
        state = {
            "input": "请描述这张图。",
            "images": [
                {
                    "name": "diagram.png",
                    "mime": "image/png",
                    "data": "YWJj",
                }
            ],
        }

        prompt_manager = PromptManager(
            "simple",
            custom_prompts={
                "multimodal_image_analyzer": "CUSTOM IMAGE ANALYSIS RULES"
            },
        )
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ), patch(
            "agent.workflows.nodes.get_prompt_manager",
            return_value=prompt_manager,
        ):
            direct_answer_node(state, {})

        sent_messages = model.invoke.call_args.args[0]
        self.assertIn("CUSTOM IMAGE ANALYSIS RULES", sent_messages[0].content)
        self.assertEqual(
            sent_messages[-1].content[1]["image_url"]["url"],
            "data:image/png;base64,YWJj",
        )

    def test_direct_answer_does_not_add_image_role_for_text_only_input(self):
        model = MagicMock()
        model.stream = None
        model.invoke.return_value.content = "普通回答"
        manager = PromptManager(
            "simple",
            custom_prompts={
                "multimodal_image_analyzer": "CUSTOM IMAGE ANALYSIS RULES"
            },
        )
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ), patch(
            "agent.workflows.nodes.get_prompt_manager",
            return_value=manager,
        ):
            direct_answer_node({"input": "解释一下概念"}, {})

        system_prompt = model.invoke.call_args.args[0][0].content
        self.assertNotIn("CUSTOM IMAGE ANALYSIS RULES", system_prompt)

    def test_direct_answer_streams_chat_model_chunks(self):
        class StreamingModel(BaseChatModel):
            @property
            def _llm_type(self):
                return "test-streaming-model"

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                return ChatResult(
                    generations=[
                        ChatGeneration(message=AIMessage(content="流式回答"))
                    ]
                )

            def _stream(self, messages, stop=None, run_manager=None, **kwargs):
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="流式")
                )
                yield ChatGenerationChunk(
                    message=AIMessageChunk(content="回答")
                )

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=StreamingModel(),
        ), patch.object(
            StreamingModel,
            "_generate",
            side_effect=AssertionError("direct answer should stream"),
        ):
            result = direct_answer_node({"input": "测试流式输出"}, {})

        self.assertEqual(result["final_report"], "流式回答")
        self.assertGreaterEqual(
            result["agent_performance"]["generation_duration_ms"],
            0,
        )
        self.assertIsNotNone(
            result["agent_performance"]["generation_first_token_ms"]
        )

    def test_direct_answer_does_not_retry_when_stream_fails_before_first_chunk(self):
        class FailingStreamingModel:
            def __init__(self):
                self.invoke_calls = 0

            def stream(self, messages, config=None):
                raise RuntimeError("upstream stream failed")

            def invoke(self, messages, config=None):
                self.invoke_calls += 1
                return AIMessage(content="fallback response")

        model = FailingStreamingModel()
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            with self.assertRaisesRegex(RuntimeError, "upstream stream failed"):
                direct_answer_node({"input": "测试"}, {})

        self.assertEqual(model.invoke_calls, 0)

    def test_direct_answer_does_not_retry_when_stream_is_empty(self):
        class EmptyStreamingModel:
            def __init__(self):
                self.invoke_calls = 0

            def stream(self, messages, config=None):
                return iter(())

            def invoke(self, messages, config=None):
                self.invoke_calls += 1
                return AIMessage(content="fallback response")

        model = EmptyStreamingModel()
        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            with self.assertRaisesRegex(ValueError, "流式接口未返回文本"):
                direct_answer_node({"input": "测试"}, {})

        self.assertEqual(model.invoke_calls, 0)

    def test_direct_answer_includes_checkpointed_conversation_history(self):
        model = MagicMock()
        model.stream = None
        model.invoke.return_value = AIMessage(content="12")

        with patch(
            "agent.workflows.nodes.ModelFactory.for_task",
            return_value=model,
        ):
            direct_answer_node(
                {
                    "input": "把刚才的答案乘以3是多少？",
                    "messages": [
                        HumanMessage(content="2+2等于几？"),
                        AIMessage(content="4"),
                        HumanMessage(content="把刚才的答案乘以3是多少？"),
                    ],
                },
                {},
            )

        model_messages = model.invoke.call_args.args[0]
        self.assertEqual(
            [(message.type, message.content) for message in model_messages[1:]],
            [
                ("human", "2+2等于几？"),
                ("ai", "4"),
                ("human", "把刚才的答案乘以3是多少？"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
