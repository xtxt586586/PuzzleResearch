from __future__ import annotations

import re
import tempfile
from pathlib import Path
from typing import Any

from tools.registry import ToolDefinition
from tools.workspace_files import (
    MAX_FILE_BYTES,
    SessionWorkspace,
    WorkspaceFileConflictError,
    WorkspaceFileError,
    WorkspaceFileNotFoundError,
    get_session_workspace,
)


MAX_SLIDES = 200
MAX_TABLE_ROWS = 100
MAX_TABLE_COLUMNS = 50
_COLOR_RE = re.compile(r"^(?:[0-9A-Fa-f]{6}|[0-9A-Fa-f]{8})$")
_SLIDE_LAYOUTS = {
    "title": 0,
    "title_content": 1,
    "section": 2,
    "two_content": 3,
    "comparison": 4,
    "title_only": 5,
    "blank": 6,
    "content_caption": 7,
    "picture_caption": 8,
}
_SHAPE_NAMES = {
    "rectangle": "RECTANGLE",
    "oval": "OVAL",
    "rounded_rectangle": "ROUNDED_RECTANGLE",
    "arrow_right": "RIGHT_ARROW",
    "arrow_left": "LEFT_ARROW",
    "triangle": "ISOSCELES_TRIANGLE",
    "diamond": "DIAMOND",
    "pentagon": "PENTAGON",
    "hexagon": "HEXAGON",
    "star": "STAR_5_POINT",
}


class WorkspacePresentationError(WorkspaceFileError):
    """工作区演示文稿操作错误。"""


def _pptx() -> Any:
    try:
        from pptx import Presentation
    except ImportError as exc:
        raise WorkspacePresentationError(
            "演示文稿工具需要 python-pptx；请先安装项目 requirements.txt 中的依赖。"
        ) from exc
    return Presentation


def _validate_path(path: str) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized.lower().endswith(".pptx"):
        raise WorkspacePresentationError("演示文稿文件只支持 .pptx 格式。")
    return normalized


def _validate_slide_number(slide_number: int, count: int) -> int:
    if type(slide_number) is not int or not 1 <= slide_number <= count:
        raise WorkspacePresentationError(
            f"slide_number 必须是 1 到 {count} 之间的整数。"
        )
    return slide_number - 1


def _validate_dimension(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 100:
        raise WorkspacePresentationError(f"{name} 必须是 0 到 100 之间的数字。")
    return float(value)


def _validate_color(value: str | None) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip().lstrip("#")
    if not _COLOR_RE.fullmatch(normalized):
        raise WorkspacePresentationError("颜色必须是 6 位或 8 位十六进制值。")
    return normalized[-6:].upper()


def _load(workspace: SessionWorkspace, file_path: str) -> tuple[Path, Any]:
    normalized = _validate_path(file_path)
    target = workspace._target(normalized)
    if not target.is_file():
        raise WorkspaceFileNotFoundError("工作区演示文稿不存在。")
    Presentation = _pptx()
    try:
        return target, Presentation(str(target))
    except Exception as exc:
        raise WorkspacePresentationError("演示文稿无法读取或格式无效。") from exc


def _save(presentation: Any, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".presentation-",
        suffix=".pptx",
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        import os

        os.close(descriptor)
        presentation.save(str(temporary))
        if temporary.stat().st_size > MAX_FILE_BYTES:
            raise WorkspacePresentationError(
                f"演示文稿不能超过 {MAX_FILE_BYTES} 字节。"
            )
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def _content_shape(slide: Any) -> Any | None:
    title_shape = slide.shapes.title
    title_id = title_shape.shape_id if title_shape is not None else None
    for shape in slide.placeholders:
        if shape.shape_id != title_id and getattr(shape, "has_text_frame", False):
            return shape
    for shape in slide.shapes:
        if shape.shape_id != title_id and getattr(shape, "has_text_frame", False):
            return shape
    return None


def _set_content(slide: Any, content: str | None) -> None:
    if content is None:
        return
    shape = _content_shape(slide)
    if shape is None:
        return
    text_frame = shape.text_frame
    text_frame.clear()
    lines = str(content).splitlines() or [""]
    text_frame.paragraphs[0].text = lines[0].strip()
    for line in lines[1:]:
        paragraph = text_frame.add_paragraph()
        paragraph.text = line.strip()
        paragraph.level = 0


def _set_notes(slide: Any, notes: str | None) -> None:
    if notes is None:
        return
    slide.notes_slide.notes_text_frame.text = str(notes)


def create_presentation(
    thread_id: str,
    file_path: str,
    title: str = "",
    subtitle: str = "",
    overwrite: bool = False,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    normalized = _validate_path(file_path)
    if type(overwrite) is not bool:
        raise WorkspacePresentationError("overwrite 必须是布尔值。")
    target = workspace._target(normalized)
    if target.exists() and not overwrite:
        raise WorkspaceFileConflictError("文件已存在；如需覆盖请设置 overwrite=true。")
    Presentation = _pptx()
    presentation = Presentation()
    has_title_slide = bool(str(title or "").strip())
    if has_title_slide:
        slide = presentation.slides.add_slide(presentation.slide_layouts[0])
        slide.shapes.title.text = str(title)
        if subtitle and len(slide.placeholders) > 1:
            slide.placeholders[1].text = str(subtitle)
    _save(presentation, target)
    return {
        "success": True,
        "path": normalized,
        "slide_count": len(presentation.slides),
        "has_title_slide": has_title_slide,
        "output": f"已创建工作区演示文稿：{normalized}",
    }


def add_slide(
    thread_id: str,
    file_path: str,
    layout: str = "title_content",
    title: str = "",
    content: str | None = None,
    notes: str | None = None,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    target, presentation = _load(workspace, file_path)
    selected_layout = str(layout or "title_content").strip().lower()
    if selected_layout not in _SLIDE_LAYOUTS:
        raise WorkspacePresentationError(
            f"layout 必须是以下选项之一：{', '.join(_SLIDE_LAYOUTS)}。"
        )
    if len(presentation.slides) >= MAX_SLIDES:
        raise WorkspacePresentationError(f"幻灯片数量不能超过 {MAX_SLIDES}。")
    slide = presentation.slides.add_slide(
        presentation.slide_layouts[_SLIDE_LAYOUTS[selected_layout]]
    )
    if slide.shapes.title is not None and title:
        slide.shapes.title.text = str(title)
    _set_content(slide, content)
    _set_notes(slide, notes)
    _save(presentation, target)
    return {
        "success": True,
        "path": _validate_path(file_path),
        "slide_number": len(presentation.slides),
        "layout": selected_layout,
        "output": f"已新增第 {len(presentation.slides)} 张幻灯片。",
    }


def add_image_to_slide(
    thread_id: str,
    file_path: str,
    slide_number: int,
    image_path: str,
    left: float = 1.0,
    top: float = 2.0,
    width: float | None = None,
    height: float | None = None,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    target, presentation = _load(workspace, file_path)
    image_target = workspace._target(str(image_path).strip().replace("\\", "/"))
    if not image_target.is_file():
        raise WorkspaceFileNotFoundError("工作区图片不存在。")
    slide_index = _validate_slide_number(slide_number, len(presentation.slides))
    left = _validate_dimension(left, "left")
    top = _validate_dimension(top, "top")
    if width is not None:
        width = _validate_dimension(width, "width")
    if height is not None:
        height = _validate_dimension(height, "height")
    from pptx.util import Inches

    presentation.slides[slide_index].shapes.add_picture(
        str(image_target),
        Inches(left),
        Inches(top),
        width=Inches(width) if width is not None else None,
        height=Inches(height) if height is not None else None,
    )
    _save(presentation, target)
    return {
        "success": True,
        "path": _validate_path(file_path),
        "slide_number": slide_number,
        "image_path": str(image_path).replace("\\", "/"),
        "output": f"已向第 {slide_number} 张幻灯片添加图片。",
    }


def add_table_to_slide(
    thread_id: str,
    file_path: str,
    slide_number: int,
    data: list[list[Any]],
    left: float = 1.0,
    top: float = 2.0,
    width: float = 8.0,
    height: float = 3.0,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    target, presentation = _load(workspace, file_path)
    slide_index = _validate_slide_number(slide_number, len(presentation.slides))
    if not isinstance(data, list) or not data or not all(isinstance(row, list) for row in data):
        raise WorkspacePresentationError("data 必须是非空二维数组。")
    columns = len(data[0])
    if not columns or len(data) > MAX_TABLE_ROWS or columns > MAX_TABLE_COLUMNS:
        raise WorkspacePresentationError("表格行列超出允许范围。")
    if any(len(row) != columns for row in data):
        raise WorkspacePresentationError("表格每一行必须包含相同数量的列。")
    left = _validate_dimension(left, "left")
    top = _validate_dimension(top, "top")
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    from pptx.util import Inches

    table = presentation.slides[slide_index].shapes.add_table(
        len(data),
        columns,
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    ).table
    for row_index, row in enumerate(data):
        for column_index, value in enumerate(row):
            cell = table.cell(row_index, column_index)
            cell.text = "" if value is None else str(value)
            if row_index == 0:
                for paragraph in cell.text_frame.paragraphs:
                    for run in paragraph.runs:
                        run.font.bold = True
    _save(presentation, target)
    return {
        "success": True,
        "path": _validate_path(file_path),
        "slide_number": slide_number,
        "rows": len(data),
        "columns": columns,
        "output": f"已向第 {slide_number} 张幻灯片添加表格。",
    }


def add_shape_to_slide(
    thread_id: str,
    file_path: str,
    slide_number: int,
    shape_type: str,
    left: float,
    top: float,
    width: float,
    height: float,
    text: str | None = None,
    fill_color: str | None = None,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    target, presentation = _load(workspace, file_path)
    slide_index = _validate_slide_number(slide_number, len(presentation.slides))
    selected_shape = str(shape_type or "").strip().lower()
    if selected_shape not in _SHAPE_NAMES:
        raise WorkspacePresentationError(
            f"shape_type 必须是以下选项之一：{', '.join(_SHAPE_NAMES)}。"
        )
    left = _validate_dimension(left, "left")
    top = _validate_dimension(top, "top")
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    color = _validate_color(fill_color)
    from pptx.dml.color import RGBColor
    from pptx.enum.shapes import MSO_SHAPE
    from pptx.util import Inches

    shape = presentation.slides[slide_index].shapes.add_shape(
        getattr(MSO_SHAPE, _SHAPE_NAMES[selected_shape]),
        Inches(left),
        Inches(top),
        Inches(width),
        Inches(height),
    )
    if color:
        shape.fill.solid()
        shape.fill.fore_color.rgb = RGBColor.from_string(color)
    if text is not None:
        shape.text = str(text)
    _save(presentation, target)
    return {
        "success": True,
        "path": _validate_path(file_path),
        "slide_number": slide_number,
        "shape_type": selected_shape,
        "output": f"已向第 {slide_number} 张幻灯片添加形状。",
    }


def get_presentation_info(thread_id: str, file_path: str) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    _, presentation = _load(workspace, file_path)
    slides: list[dict[str, Any]] = []
    for index, slide in enumerate(presentation.slides, start=1):
        slides.append(
            {
                "number": index,
                "title": slide.shapes.title.text if slide.shapes.title else "",
                "shape_count": len(slide.shapes),
            }
        )
    return {
        "success": True,
        "path": _validate_path(file_path),
        "slide_count": len(presentation.slides),
        "width_inches": round(presentation.slide_width.inches, 3),
        "height_inches": round(presentation.slide_height.inches, 3),
        "slides": slides,
        "output": f"已读取演示文稿信息：{file_path}",
    }


def update_slide(
    thread_id: str,
    file_path: str,
    slide_number: int,
    title: str | None = None,
    content: str | None = None,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    target, presentation = _load(workspace, file_path)
    slide_index = _validate_slide_number(slide_number, len(presentation.slides))
    slide = presentation.slides[slide_index]
    if title is not None and slide.shapes.title is not None:
        slide.shapes.title.text = str(title)
    _set_content(slide, content)
    _save(presentation, target)
    return {
        "success": True,
        "path": _validate_path(file_path),
        "slide_number": slide_number,
        "output": f"已更新第 {slide_number} 张幻灯片。",
    }


def delete_slide(thread_id: str, file_path: str, slide_number: int) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    target, presentation = _load(workspace, file_path)
    slide_index = _validate_slide_number(slide_number, len(presentation.slides))
    slide_id = presentation.slides._sldIdLst[slide_index].rId
    presentation.part.drop_rel(slide_id)
    del presentation.slides._sldIdLst[slide_index]
    _save(presentation, target)
    return {
        "success": True,
        "path": _validate_path(file_path),
        "deleted_slide": slide_number,
        "slide_count": len(presentation.slides),
        "output": f"已删除第 {slide_number} 张幻灯片。",
    }


def build_workspace_presentation_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    def create_presentation_tool(
        file_path: str,
        title: str = "",
        subtitle: str = "",
        overwrite: bool = False,
    ) -> dict[str, Any]:
        return create_presentation(thread_id, file_path, title, subtitle, overwrite)

    def add_slide_tool(
        file_path: str,
        layout: str = "title_content",
        title: str = "",
        content: str | None = None,
        notes: str | None = None,
    ) -> dict[str, Any]:
        return add_slide(thread_id, file_path, layout, title, content, notes)

    def add_image_to_slide_tool(
        file_path: str,
        slide_number: int,
        image_path: str,
        left: float = 1.0,
        top: float = 2.0,
        width: float | None = None,
        height: float | None = None,
    ) -> dict[str, Any]:
        return add_image_to_slide(
            thread_id, file_path, slide_number, image_path,
            left, top, width, height,
        )

    def add_table_to_slide_tool(
        file_path: str,
        slide_number: int,
        data: list[list[Any]],
        left: float = 1.0,
        top: float = 2.0,
        width: float = 8.0,
        height: float = 3.0,
    ) -> dict[str, Any]:
        return add_table_to_slide(
            thread_id, file_path, slide_number, data,
            left, top, width, height,
        )

    def add_shape_to_slide_tool(
        file_path: str,
        slide_number: int,
        shape_type: str,
        left: float,
        top: float,
        width: float,
        height: float,
        text: str | None = None,
        fill_color: str | None = None,
    ) -> dict[str, Any]:
        return add_shape_to_slide(
            thread_id, file_path, slide_number, shape_type,
            left, top, width, height, text, fill_color,
        )

    def get_presentation_info_tool(file_path: str) -> dict[str, Any]:
        return get_presentation_info(thread_id, file_path)

    def update_slide_tool(
        file_path: str,
        slide_number: int,
        title: str | None = None,
        content: str | None = None,
    ) -> dict[str, Any]:
        return update_slide(thread_id, file_path, slide_number, title, content)

    def delete_slide_tool(file_path: str, slide_number: int) -> dict[str, Any]:
        return delete_slide(thread_id, file_path, slide_number)

    return [
        ToolDefinition(
            "create_presentation",
            "在当前会话工作区创建 PowerPoint 演示文稿，可选创建标题页。",
            create_presentation_tool,
        ),
        ToolDefinition(
            "add_slide",
            "向工作区 PowerPoint 演示文稿新增指定布局的幻灯片。",
            add_slide_tool,
        ),
        ToolDefinition(
            "add_image_to_slide",
            "向工作区 PowerPoint 幻灯片添加图片。",
            add_image_to_slide_tool,
        ),
        ToolDefinition(
            "add_table_to_slide",
            "向工作区 PowerPoint 幻灯片添加二维数据表格。",
            add_table_to_slide_tool,
        ),
        ToolDefinition(
            "add_shape_to_slide",
            "向工作区 PowerPoint 幻灯片添加形状、文字和填充色。",
            add_shape_to_slide_tool,
        ),
        ToolDefinition(
            "get_presentation_info",
            "读取工作区 PowerPoint 的幻灯片数量、标题和尺寸信息。",
            get_presentation_info_tool,
        ),
        ToolDefinition(
            "update_slide",
            "更新工作区 PowerPoint 既有幻灯片的标题和正文。",
            update_slide_tool,
        ),
        ToolDefinition(
            "delete_slide",
            "按 1 起始的序号删除工作区 PowerPoint 幻灯片。",
            delete_slide_tool,
        ),
    ]
