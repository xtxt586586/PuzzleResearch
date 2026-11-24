from __future__ import annotations

import json
import re
from typing import Any, Literal, Mapping

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from agent.core.model_factory import ModelFactory
from agent.prompts import get_prompt_manager
from tools.registry import ToolDefinition
from tools.workspace_presentations import add_slide, create_presentation


PresentationStyle = Literal[
    "business",
    "educational",
    "technical",
    "creative",
    "minimal",
]
SlideLayout = Literal["title", "title_content", "section", "two_content", "blank"]


class SlideOutline(BaseModel):
    slide_number: int = Field(ge=1)
    title: str
    layout: SlideLayout = "title_content"
    content: list[str] = Field(default_factory=list, max_length=5)
    speaker_notes: str = ""
    has_image: bool = False
    image_suggestion: str = ""


class PresentationOutline(BaseModel):
    title: str
    subtitle: str = ""
    author: str = ""
    total_slides: int = Field(ge=1)
    estimated_duration_minutes: int = Field(ge=1)
    target_audience: str
    key_takeaways: list[str] = Field(default_factory=list)
    slides: list[SlideOutline] = Field(default_factory=list)


_OUTLINE_OUTPUT_CONTRACT = """输出约束：每页最多 5 个简洁要点；layout 只能使用 title、title_content、section、two_content、blank；
为每页提供演讲者备注，需要视觉材料时提供 image_suggestion。只返回符合给定结构的 JSON 对象，不要输出 Markdown 或解释文字。"""


def _system_prompt(prompt_type: str = "presentation_outline") -> str:
    return f"{get_prompt_manager().get_prompt(prompt_type)}\n\n{_OUTLINE_OUTPUT_CONTRACT}"


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                text = item.get("text", "")
                if text:
                    parts.append(str(text))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content)


def _parse_json_object(response: Any) -> dict[str, Any]:
    content = _response_text(response).strip()
    fenced = re.search(
        r"```(?:json)?\s*(\{.*?\})\s*```",
        content,
        flags=re.IGNORECASE | re.DOTALL,
    )
    json_text = fenced.group(1) if fenced else content[content.find("{") : content.rfind("}") + 1]
    if not json_text or not json_text.startswith("{"):
        raise ValueError("模型返回中没有找到有效 JSON 对象。")
    payload = json.loads(json_text)
    if not isinstance(payload, dict):
        raise ValueError("模型返回的 JSON 必须是对象。")
    return payload


def _model_payload(response: Any) -> dict[str, Any]:
    if hasattr(response, "model_dump"):
        return response.model_dump()
    if isinstance(response, Mapping):
        return dict(response)
    return _parse_json_object(response)


def _normalize_payload(payload: Mapping[str, Any], schema: type[BaseModel]) -> dict[str, Any]:
    """补齐模型常省略的字段，保持 DeepSeek 普通 JSON 可被消费。"""

    normalized = dict(payload)
    if schema is SlideOutline:
        normalized.setdefault("slide_number", 1)
        normalized.setdefault("title", "未命名页面")
        normalized.setdefault("layout", "title_content")
        content = normalized.get("content", [])
        normalized["content"] = (
            [str(item) for item in content]
            if isinstance(content, list)
            else ([str(content)] if content else [])
        )
        normalized.setdefault("speaker_notes", "")
        normalized.setdefault("has_image", False)
        normalized.setdefault("image_suggestion", "")
        return normalized

    if schema is PresentationOutline:
        normalized.setdefault("title", "未命名演示文稿")
        normalized.setdefault("subtitle", "")
        normalized.setdefault("author", "")
        normalized.setdefault("estimated_duration_minutes", 15)
        normalized.setdefault("target_audience", "general")
        normalized.setdefault("key_takeaways", [])
        raw_slides = normalized.get("slides", [])
        if not isinstance(raw_slides, list):
            raw_slides = []
        slides: list[dict[str, Any]] = []
        for index, item in enumerate(raw_slides, start=1):
            if not isinstance(item, Mapping):
                continue
            slide = _normalize_payload(item, SlideOutline)
            if type(slide.get("slide_number")) is not int or slide["slide_number"] < 1:
                slide["slide_number"] = index
            slides.append(slide)
        normalized["slides"] = slides
        normalized["total_slides"] = max(1, len(slides))
    return normalized


def _invoke_structured(model: Any, schema: type[BaseModel], messages: list[Any]) -> BaseModel:
    structured_error: Exception | None = None
    try:
        structured = model.with_structured_output(schema)
        response = structured.invoke(messages)
        return schema.model_validate(
            _normalize_payload(_model_payload(response), schema)
        )
    except Exception as exc:
        structured_error = exc

    fallback_messages = [
        SystemMessage(
            content=(
                f"{messages[0].content if hasattr(messages[0], 'content') else _system_prompt()}\n"
                "请严格返回 JSON，并确保字段完整。"
            )
        ),
        messages[-1],
    ]
    try:
        response = model.invoke(fallback_messages)
        return schema.model_validate(
            _normalize_payload(_parse_json_object(response), schema)
        )
    except Exception as fallback_error:
        raise ValueError(
            f"结构化输出和普通 JSON 输出都失败：{structured_error}; {fallback_error}"
        ) from fallback_error


def _validate_generation_args(
    topic: str,
    num_slides: int,
    duration_minutes: int,
    target_audience: str,
) -> tuple[str, int, int, str]:
    normalized_topic = str(topic or "").strip()
    if not normalized_topic:
        raise ValueError("topic 不能为空。")
    if type(num_slides) is not int or not 3 <= num_slides <= 30:
        raise ValueError("num_slides 必须是 3 到 30 之间的整数。")
    if type(duration_minutes) is not int or not 1 <= duration_minutes <= 240:
        raise ValueError("duration_minutes 必须是 1 到 240 之间的整数。")
    normalized_audience = str(target_audience or "").strip()
    if not normalized_audience:
        raise ValueError("target_audience 不能为空。")
    return normalized_topic, num_slides, duration_minutes, normalized_audience


def generate_presentation_outline(
    topic: str,
    num_slides: int = 10,
    style: PresentationStyle = "business",
    target_audience: str = "general",
    duration_minutes: int = 15,
    include_images: bool = True,
    additional_context: str = "",
) -> dict[str, Any]:
    topic, num_slides, duration_minutes, target_audience = _validate_generation_args(
        topic, num_slides, duration_minutes, target_audience
    )
    if style not in {"business", "educational", "technical", "creative", "minimal"}:
        raise ValueError("style 参数无效。")
    if type(include_images) is not bool:
        raise ValueError("include_images 必须是布尔值。")
    model = ModelFactory.for_task("writing", temperature=0.7)
    prompt = f"""请为以下主题生成演示文稿大纲：

主题：{topic}
目标页数：约 {num_slides} 页（不含标题页）
风格：{style}
目标听众：{target_audience}
预计时长：{duration_minutes} 分钟
是否需要图片建议：{include_images}
补充要求：{additional_context.strip()}

返回 PresentationOutline 对应的 JSON。"""
    outline = _invoke_structured(
        model,
        PresentationOutline,
        [SystemMessage(content=_system_prompt()), HumanMessage(content=prompt)],
    )
    payload = outline.model_dump()
    payload["total_slides"] = len(payload["slides"])
    return {
        "success": True,
        "message": f"已生成“{topic}”的演示文稿大纲。",
        "outline": payload,
    }


def refine_presentation_outline(
    outline: dict[str, Any],
    feedback: str,
) -> dict[str, Any]:
    current = PresentationOutline.model_validate(
        _normalize_payload(outline, PresentationOutline)
    )
    normalized_feedback = str(feedback or "").strip()
    if not normalized_feedback:
        raise ValueError("feedback 不能为空。")
    model = ModelFactory.for_task("writing", temperature=0.5)
    prompt = (
        "请根据以下修改意见调整演示文稿大纲，并返回完整 PresentationOutline JSON。\n"
        f"当前大纲：{json.dumps(current.model_dump(), ensure_ascii=False)}\n"
        f"修改意见：{normalized_feedback}"
    )
    refined = _invoke_structured(
        model,
        PresentationOutline,
        [SystemMessage(content=_system_prompt()), HumanMessage(content=prompt)],
    )
    payload = refined.model_dump()
    payload["total_slides"] = len(payload["slides"])
    return {"success": True, "message": "演示文稿大纲已更新。", "outline": payload}


def expand_slide_content(
    topic: str,
    context: str = "",
    style: PresentationStyle = "business",
) -> dict[str, Any]:
    normalized_topic = str(topic or "").strip()
    if not normalized_topic:
        raise ValueError("topic 不能为空。")
    if style not in {"business", "educational", "technical", "creative", "minimal"}:
        raise ValueError("style 参数无效。")
    model = ModelFactory.for_task("writing", temperature=0.7)
    prompt = f"""请扩展以下单页演示文稿内容，并返回 SlideOutline JSON：
主题：{normalized_topic}
风格：{style}
上下文：{str(context or '').strip()}
要求：提供有吸引力的标题、最多 5 条要点、演讲者备注和必要的图片建议。"""
    slide = _invoke_structured(
        model,
        SlideOutline,
        [
            SystemMessage(content=_system_prompt("presentation_slide_expander")),
            HumanMessage(content=prompt),
        ],
    )
    return {"success": True, "slide": slide.model_dump()}


def outline_to_presentation(
    thread_id: str,
    outline: dict[str, Any],
    file_path: str,
    overwrite: bool = False,
) -> dict[str, Any]:
    validated = PresentationOutline.model_validate(outline)
    created = create_presentation(
        thread_id,
        file_path,
        validated.title,
        "\n".join(item for item in [validated.subtitle, validated.author] if item),
        overwrite,
    )
    for slide in validated.slides:
        add_slide(
            thread_id,
            file_path,
            layout=slide.layout,
            title=slide.title,
            content="\n".join(slide.content),
            notes=slide.speaker_notes,
        )
    return {
        "success": True,
        "message": f"已根据大纲创建演示文稿，共 {created['slide_count'] + len(validated.slides)} 张幻灯片。",
        "path": file_path,
        "slide_count": created["slide_count"] + len(validated.slides),
    }


def build_workspace_presentation_outline_tool_definitions(
    thread_id: str,
) -> list[ToolDefinition]:
    def generate_tool(
        topic: str,
        num_slides: int = 10,
        style: PresentationStyle = "business",
        target_audience: str = "general",
        duration_minutes: int = 15,
        include_images: bool = True,
        additional_context: str = "",
    ) -> dict[str, Any]:
        return generate_presentation_outline(
            topic, num_slides, style, target_audience,
            duration_minutes, include_images, additional_context,
        )

    def outline_to_presentation_tool(
        outline: dict[str, Any],
        file_path: str,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return outline_to_presentation(thread_id, outline, file_path, overwrite)

    def refine_tool(outline: dict[str, Any], feedback: str) -> dict[str, Any]:
        return refine_presentation_outline(outline, feedback)

    def expand_tool(
        topic: str,
        context: str = "",
        style: PresentationStyle = "business",
    ) -> dict[str, Any]:
        return expand_slide_content(topic, context, style)

    return [
        ToolDefinition(
            "generate_presentation_outline",
            "使用当前配置的模型生成结构化演示文稿大纲。",
            generate_tool,
        ),
        ToolDefinition(
            "outline_to_presentation",
            "将已确认的演示文稿大纲转换为当前会话工作区中的 .pptx 文件。",
            outline_to_presentation_tool,
        ),
        ToolDefinition(
            "refine_presentation_outline",
            "根据修改意见调整已有演示文稿大纲。",
            refine_tool,
        ),
        ToolDefinition(
            "expand_slide_content",
            "使用当前配置的模型扩展单页演示文稿内容。",
            expand_tool,
        ),
    ]
