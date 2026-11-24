from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from pptx import Presentation

from tools.builtin import create_default_registry
from tools.workspace_files import SessionWorkspace
from tools.workspace_presentation_outline import (
    PresentationOutline,
    SlideOutline,
    expand_slide_content,
    generate_presentation_outline,
    outline_to_presentation,
    refine_presentation_outline,
)


def _outline_payload() -> dict:
    return {
        "title": "PuzzleResearch 研究报告",
        "subtitle": "阶段性成果",
        "author": "研究团队",
        "total_slides": 2,
        "estimated_duration_minutes": 10,
        "target_audience": "技术团队",
        "key_takeaways": ["完成基础能力复现"],
        "slides": [
            {
                "slide_number": 1,
                "title": "研究目标",
                "layout": "title_content",
                "content": ["明确复现范围", "建立本地工具链"],
                "speaker_notes": "介绍本阶段目标。",
                "has_image": False,
                "image_suggestion": "",
            },
            {
                "slide_number": 2,
                "title": "完成情况",
                "layout": "title_content",
                "content": ["表格工具", "演示文稿工具"],
                "speaker_notes": "总结已经完成的能力。",
                "has_image": False,
                "image_suggestion": "",
            },
        ],
    }


class FakeStructuredModel:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def with_structured_output(self, schema):
        return self

    def invoke(self, messages):
        self.messages = messages
        return self.payload


class FakeJsonModel:
    def __init__(self, payload):
        self.payload = payload
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return AIMessage(content=f"```json\n{json.dumps(self.payload, ensure_ascii=False)}\n```")


class WorkspacePresentationOutlineTests(unittest.TestCase):
    def test_generate_uses_structured_model_output(self):
        from agent.prompts import PromptManager

        model = FakeStructuredModel(_outline_payload())
        manager = PromptManager(
            "simple",
            custom_prompts={"presentation_outline": "CUSTOM OUTLINE ROLE"},
        )
        with patch(
            "tools.workspace_presentation_outline.ModelFactory.for_task",
            return_value=model,
        ) as factory, patch(
            "tools.workspace_presentation_outline.get_prompt_manager",
            return_value=manager,
        ):
            result = generate_presentation_outline(
                "PuzzleResearch 复现进度",
                num_slides=5,
                target_audience="开发者",
            )
        self.assertTrue(result["success"])
        self.assertEqual(result["outline"]["total_slides"], 2)
        self.assertIn("CUSTOM OUTLINE ROLE", model.messages[0].content)
        self.assertIn("layout 只能使用", model.messages[0].content)
        self.assertIn("每页最多 5 个", model.messages[0].content)
        factory.assert_called_once_with("writing", temperature=0.7)

    def test_slide_expander_uses_its_own_managed_prompt(self):
        from agent.prompts import PromptManager

        model = FakeStructuredModel(_outline_payload()["slides"][0])
        manager = PromptManager(
            "simple",
            custom_prompts={
                "presentation_slide_expander": "CUSTOM SLIDE ROLE"
            },
        )
        with patch(
            "tools.workspace_presentation_outline.ModelFactory.for_task",
            return_value=model,
        ), patch(
            "tools.workspace_presentation_outline.get_prompt_manager",
            return_value=manager,
        ):
            expand_slide_content("研究风险")

        self.assertIn("CUSTOM SLIDE ROLE", model.messages[0].content)
        self.assertIn("只返回符合给定结构的 JSON", model.messages[0].content)

    def test_slide_expander_fallback_keeps_role_and_schema_constraints(self):
        from agent.prompts import PromptManager

        payload = _outline_payload()["slides"][0]
        model = FakeJsonModel(payload)
        manager = PromptManager(
            "simple",
            custom_prompts={
                "presentation_slide_expander": "CUSTOM FALLBACK SLIDE ROLE"
            },
        )
        with patch(
            "tools.workspace_presentation_outline.ModelFactory.for_task",
            return_value=model,
        ), patch(
            "tools.workspace_presentation_outline.get_prompt_manager",
            return_value=manager,
        ):
            result = expand_slide_content("研究风险")

        self.assertEqual(result["slide"]["title"], "研究目标")
        self.assertIn("CUSTOM FALLBACK SLIDE ROLE", model.messages[0].content)
        self.assertIn("每页最多 5 个", model.messages[0].content)

    def test_slide_schema_rejects_invalid_layout_and_too_many_bullets(self):
        with self.assertRaises(ValueError):
            SlideOutline(slide_number=1, title="x", layout="unsupported")
        with self.assertRaises(ValueError):
            SlideOutline(
                slide_number=1,
                title="x",
                content=["1", "2", "3", "4", "5", "6"],
            )

        payload = _outline_payload()
        payload["slides"][0]["layout"] = "unsupported"
        with self.assertRaises(ValueError):
            PresentationOutline.model_validate(payload)

    def test_generate_falls_back_to_plain_json(self):
        with patch(
            "tools.workspace_presentation_outline.ModelFactory.for_task",
            return_value=FakeJsonModel(_outline_payload()),
        ):
            result = generate_presentation_outline("本地工具链", num_slides=3)
        self.assertTrue(result["success"])
        self.assertEqual(result["outline"]["title"], "PuzzleResearch 研究报告")

    def test_incomplete_plain_json_is_completed_before_validation(self):
        incomplete = {
            "title": "PuzzleResearch 路线图",
            "slides": [
                {"layout": "title_content", "title": "研究目标"},
            ],
        }
        with patch(
            "tools.workspace_presentation_outline.ModelFactory.for_task",
            return_value=FakeJsonModel(incomplete),
        ):
            result = generate_presentation_outline("复现路线", num_slides=3)
        outline = result["outline"]
        self.assertEqual(outline["estimated_duration_minutes"], 15)
        self.assertEqual(outline["target_audience"], "general")
        self.assertEqual(outline["slides"][0]["slide_number"], 1)
        self.assertEqual(outline["slides"][0]["content"], [])

    def test_refine_and_expand_validate_model_contract(self):
        payload = _outline_payload()
        slide_payload = payload["slides"][0]
        with patch(
            "tools.workspace_presentation_outline.ModelFactory.for_task",
            side_effect=[
                FakeStructuredModel(payload),
                FakeStructuredModel(slide_payload),
            ],
        ):
            refined = refine_presentation_outline(payload, "增加风险与下一步")
            expanded = expand_slide_content("研究风险", style="technical")
        self.assertTrue(refined["success"])
        self.assertEqual(len(refined["outline"]["slides"]), 2)
        self.assertEqual(expanded["slide"]["title"], "研究目标")

    def test_outline_to_presentation_writes_pptx_without_model(self):
        with tempfile.TemporaryDirectory() as root:
            workspace = SessionWorkspace("outline-thread", storage_root=Path(root))
            with patch(
                "tools.workspace_presentations.get_session_workspace",
                return_value=workspace,
            ):
                result = outline_to_presentation(
                    "outline-thread",
                    _outline_payload(),
                    "reports/outline.pptx",
                )
            self.assertEqual(result["slide_count"], 3)
            presentation = Presentation(str(next(Path(root).rglob("outline.pptx"))))
            self.assertEqual(len(presentation.slides), 3)

    def test_outline_tools_are_registered(self):
        names = {tool.name for tool in create_default_registry().list_tools()}
        self.assertTrue(
            {
                "generate_presentation_outline",
                "outline_to_presentation",
                "refine_presentation_outline",
                "expand_slide_content",
            }
            <= names
        )

    def test_invalid_generation_arguments_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "topic"):
            generate_presentation_outline("", num_slides=3)
        with self.assertRaisesRegex(ValueError, "num_slides"):
            generate_presentation_outline("主题", num_slides=2)
        with self.assertRaisesRegex(ValueError, "feedback"):
            refine_presentation_outline(_outline_payload(), "")


if __name__ == "__main__":
    unittest.main()
