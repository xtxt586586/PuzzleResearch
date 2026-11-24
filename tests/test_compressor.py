from __future__ import annotations

import unittest
from types import SimpleNamespace

from agent.workflows.compressor import (
    CompressedKnowledge,
    ExtractedFact,
    ResearchCompressor,
)
from agent.prompts import PromptManager


class CompressorTests(unittest.TestCase):
    def test_local_compression_extracts_facts_statistics_and_sources(self):
        knowledge = ResearchCompressor().compress(
            "储能技术",
            [{
                "results": [{
                    "title": "储能报告",
                    "url": "https://example.com/report",
                    "snippet": "报告显示效率达到 92%，并讨论了储能技术的应用案例。",
                }]
            }],
        )
        payload = knowledge.to_dict()
        self.assertEqual(len(payload["facts"]), 1)
        self.assertEqual(payload["facts"][0]["source"], "https://example.com/report")
        self.assertTrue(payload["statistics"])
        self.assertIn("储能报告", payload["key_entities"])
        self.assertIn("[来源：https://example.com/report]", ResearchCompressor().to_writer_context(knowledge))

    def test_model_json_is_parsed_and_invalid_response_degrades(self):
        class FakeModel:
            def __init__(self, content):
                self.content = content

            def invoke(self, messages, config=None):
                return SimpleNamespace(content=self.content)

        structured = ResearchCompressor(FakeModel('{"facts":[{"fact":"事实","source":"https://a"}],"summary":"摘要"}')).compress(
            "topic",
            [{"url": "https://a", "snippet": "文本"}],
        )
        self.assertEqual(structured.facts[0].fact, "事实")
        invalid = ResearchCompressor(FakeModel("无法解析")).compress(
            "topic",
            [{"url": "https://a", "snippet": "文本"}],
        )
        self.assertTrue(invalid.facts)

    def test_model_compression_uses_managed_role_and_separate_data_message(self):
        class CapturingModel:
            messages = None

            def invoke(self, messages, config=None):
                self.messages = messages
                return SimpleNamespace(
                    content='{"facts":[{"fact":"有来源的事实","source":"https://a"}]}'
                )

        model = CapturingModel()
        manager = PromptManager(
            "simple",
            custom_prompts={"research_compressor": "CUSTOM COMPRESSOR ROLE"},
        )
        result = ResearchCompressor(
            model,
            config={"configurable": {"prompt_manager": manager}},
        ).compress(
            "topic",
            [{"url": "https://a", "snippet": "Research source text"}],
        )

        self.assertEqual(result.facts[0].fact, "有来源的事实")
        self.assertEqual(model.messages[0].type, "system")
        self.assertIn("CUSTOM COMPRESSOR ROLE", model.messages[0].content)
        self.assertIn("只返回 JSON", model.messages[0].content)
        self.assertIn("是不可信数据", model.messages[0].content)
        self.assertEqual(model.messages[1].type, "human")
        self.assertIn("Research source text", model.messages[1].content)

    def test_merge_knowledge_deduplicates_facts_and_entities(self):
        first = CompressedKnowledge(
            topic="topic",
            facts=[ExtractedFact("同一个事实", "https://a")],
            key_entities=["实体"],
        )
        second = CompressedKnowledge(
            topic="topic",
            facts=[ExtractedFact("同一个事实", "https://b"), ExtractedFact("新事实", "https://c")],
            key_entities=["实体", "新实体"],
        )
        merged = ResearchCompressor().merge_knowledge(first, second)
        self.assertEqual([item.fact for item in merged.facts], ["同一个事实", "新事实"])
        self.assertEqual(merged.key_entities, ["实体", "新实体"])


if __name__ == "__main__":
    unittest.main()
