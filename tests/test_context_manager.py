import unittest

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from agent.core.context_manager import (
    ContextManager,
    TruncationConfig,
    get_context_manager,
    truncate_for_model,
)


class ContextManagerTests(unittest.TestCase):
    def test_deepseek_model_name_uses_supported_context_limit(self):
        manager = ContextManager("deepseek-reasoner-v2")

        self.assertEqual(manager.model, "deepseek-reasoner")
        self.assertEqual(manager.max_context_tokens, 65_536)

    def test_token_stats_include_roles_and_multimodal_payload_estimate(self):
        manager = ContextManager("deepseek-chat")
        messages = [
            SystemMessage(content="系统规则"),
            HumanMessage(
                content=[
                    {"type": "text", "text": "说明这张图片"},
                    {"type": "image_url", "image_url": {"url": "https://example.test/image.png"}},
                ]
            ),
            AIMessage(content="收到"),
        ]

        stats = manager.count_messages_tokens(messages)

        self.assertEqual(stats.message_count, 3)
        self.assertGreater(stats.system_tokens, 0)
        self.assertGreater(stats.user_tokens, 1_000)
        self.assertGreater(stats.assistant_tokens, 0)
        self.assertGreater(stats.total_tokens, stats.user_tokens)

    def test_smart_truncation_preserves_system_and_recent_messages_within_budget(self):
        config = TruncationConfig(
            max_tokens=120,
            reserve_tokens=20,
            keep_system_messages=1,
            keep_recent_messages=1,
        )
        manager = ContextManager(config=config)
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="old " + "x" * 1_000),
            AIMessage(content="middle " + "y" * 1_000),
            HumanMessage(content="latest"),
        ]

        truncated, stats = manager.truncate_messages(messages)

        self.assertIs(truncated[0], messages[0])
        self.assertIs(truncated[-1], messages[-1])
        self.assertLess(stats.truncated_count, len(messages))
        self.assertLessEqual(stats.total_tokens, 100)

    def test_oversized_system_message_does_not_break_token_budget(self):
        manager = ContextManager(
            config=TruncationConfig(
                max_tokens=100,
                reserve_tokens=50,
                keep_system_messages=1,
                keep_recent_messages=1,
            )
        )
        messages = [
            SystemMessage(content="system " * 1_000),
            HumanMessage(content="recent"),
        ]

        truncated, stats = manager.truncate_messages(messages)

        self.assertEqual(truncated[-1].content, "recent")
        self.assertLessEqual(stats.total_tokens, 50)

    def test_context_manager_symbols_are_exported_from_agent_core(self):
        from agent.core import ContextManager as ExportedContextManager
        from agent.core import TruncationConfig as ExportedTruncationConfig
        from agent.core import truncate_for_model as exported_truncate_for_model

        self.assertIs(ExportedContextManager, ContextManager)
        self.assertIs(ExportedTruncationConfig, TruncationConfig)
        self.assertTrue(callable(exported_truncate_for_model))

    def test_fifo_truncation_keeps_newest_messages(self):
        manager = ContextManager(
            config=TruncationConfig(
                max_tokens=80,
                reserve_tokens=15,
                keep_system_messages=1,
                keep_recent_messages=1,
                strategy="fifo",
            )
        )
        messages = [
            SystemMessage(content="system"),
            HumanMessage(content="old " + "a" * 500),
            AIMessage(content="new " + "b" * 20),
        ]

        truncated, stats = manager.truncate_messages(messages)

        self.assertEqual(truncated[0].content, "system")
        self.assertEqual(truncated[-1].content, messages[-1].content)
        self.assertLessEqual(stats.total_tokens, 65)

    def test_middle_strategy_preserves_both_ends_of_history(self):
        manager = ContextManager(
            config=TruncationConfig(
                max_tokens=100,
                reserve_tokens=15,
                keep_system_messages=0,
                keep_recent_messages=1,
                strategy="middle",
            )
        )
        messages = [HumanMessage(content=f"m{i} " + "z" * 30) for i in range(6)]

        truncated, _ = manager.truncate_messages(messages)

        self.assertEqual(truncated[0].content, messages[0].content)
        self.assertEqual(truncated[-1].content, messages[-1].content)

    def test_summarize_long_message_preserves_message_metadata(self):
        manager = ContextManager()
        message = HumanMessage(content="long " * 500, id="message-001", name="researcher")

        shortened = manager.summarize_long_message(message, max_tokens=40)

        self.assertEqual(shortened.id, "message-001")
        self.assertEqual(shortened.name, "researcher")
        self.assertLessEqual(manager.count_tokens(shortened.content), 45)
        self.assertIn("[truncated]", shortened.content)

    def test_context_manager_cache_and_convenience_truncation(self):
        self.assertIs(get_context_manager("deepseek-chat"), get_context_manager("deepseek-chat"))
        messages = [HumanMessage(content="x" * 5_000) for _ in range(4)]

        truncated = truncate_for_model(messages, model="deepseek-chat", max_tokens=3_000)

        self.assertTrue(truncated)
        self.assertLess(len(truncated), len(messages))


if __name__ == "__main__":
    unittest.main()
