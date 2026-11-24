from __future__ import annotations

from typing import Any, Literal

from tools.registry import ToolDefinition
from tools.workspace_files import SessionWorkspace, WorkspaceFileError
from tools.workspace_files import get_session_workspace
from tools.workspace_vision import (
    _image_target,
    _pillow,
    _save_image,
    _validate_image_path,
)


FilterType = Literal[
    "blur",
    "gaussian_blur",
    "sharpen",
    "edge_enhance",
    "emboss",
    "contour",
    "detail",
    "smooth",
]
EffectType = Literal[
    "grayscale",
    "sepia",
    "invert",
    "posterize",
    "solarize",
    "auto_contrast",
    "equalize",
]


class WorkspaceImageEditError(WorkspaceFileError):
    """工作区图像编辑错误。"""


def _validate_factor(value: float, name: str, minimum: float = 0.0) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not minimum <= value <= 3.0:
        raise WorkspaceImageEditError(f"{name} 必须是 {minimum} 到 3.0 之间的数字。")
    return float(value)


def _validate_hex_color(value: str) -> str:
    normalized = str(value or "").strip().lstrip("#")
    if len(normalized) not in {6, 8} or any(char not in "0123456789abcdefABCDEF" for char in normalized):
        raise WorkspaceImageEditError("颜色必须是 6 位或 8 位十六进制值。")
    return normalized[:6].upper()


def _load_for_edit(workspace: SessionWorkspace, image_path: str, output_path: str) -> tuple[Any, Any, str, str]:
    source_path = _image_target(workspace, image_path)
    normalized_output = _validate_image_path(output_path, output=True)
    target_path = workspace._target(normalized_output)
    Image = _pillow()
    try:
        with Image.open(source_path) as source:
            image = source.copy()
    except OSError as exc:
        raise WorkspaceImageEditError("图片无法读取或格式无效。") from exc
    return image, target_path, str(image_path).replace("\\", "/"), normalized_output


def apply_image_filter(
    thread_id: str,
    image_path: str,
    output_path: str,
    filter_type: FilterType,
    intensity: float = 1.0,
) -> dict[str, Any]:
    selected = str(filter_type or "").strip().lower()
    filters = {
        "blur": "BLUR",
        "gaussian_blur": "GAUSSIAN",
        "sharpen": "SHARPEN",
        "edge_enhance": "EDGE_ENHANCE",
        "emboss": "EMBOSS",
        "contour": "CONTOUR",
        "detail": "DETAIL",
        "smooth": "SMOOTH",
    }
    if selected not in filters:
        raise WorkspaceImageEditError("filter_type 参数无效。")
    intensity = _validate_factor(intensity, "intensity", 0.1)
    workspace = get_session_workspace(thread_id)
    image, target, normalized_input, normalized_output = _load_for_edit(
        workspace, image_path, output_path
    )
    from PIL import ImageFilter

    if selected == "gaussian_blur":
        filter_object = ImageFilter.GaussianBlur(radius=max(1, int(intensity * 2)))
    else:
        filter_object = getattr(ImageFilter, filters[selected])
    result = image
    for _ in range(max(1, round(intensity))):
        result = result.filter(filter_object)
    size = _save_image(result, target)
    return {
        "success": True,
        "input_path": normalized_input,
        "output_path": normalized_output,
        "filter": selected,
        "width": result.width,
        "height": result.height,
        "size": size,
        "output": f"已应用图片滤镜：{selected}",
    }


def apply_image_effect(
    thread_id: str,
    image_path: str,
    output_path: str,
    effect_type: EffectType,
) -> dict[str, Any]:
    selected = str(effect_type or "").strip().lower()
    if selected not in {
        "grayscale", "sepia", "invert", "posterize", "solarize", "auto_contrast", "equalize",
    }:
        raise WorkspaceImageEditError("effect_type 参数无效。")
    workspace = get_session_workspace(thread_id)
    image, target, normalized_input, normalized_output = _load_for_edit(
        workspace, image_path, output_path
    )
    from PIL import ImageOps

    if selected in {"invert", "posterize", "solarize", "auto_contrast", "equalize"}:
        image = image.convert("RGB")
    if selected == "grayscale":
        result = ImageOps.grayscale(image)
    elif selected == "sepia":
        gray = ImageOps.grayscale(image.convert("RGB"))
        result = ImageOps.colorize(gray, black="#1B0D05", white="#F3D7A4")
    elif selected == "invert":
        result = ImageOps.invert(image)
    elif selected == "posterize":
        result = ImageOps.posterize(image, 4)
    elif selected == "solarize":
        result = ImageOps.solarize(image, threshold=128)
    elif selected == "auto_contrast":
        result = ImageOps.autocontrast(image)
    else:
        result = ImageOps.equalize(image)
    size = _save_image(result, target)
    return {
        "success": True,
        "input_path": normalized_input,
        "output_path": normalized_output,
        "effect": selected,
        "width": result.width,
        "height": result.height,
        "mode": result.mode,
        "size": size,
        "output": f"已应用图片效果：{selected}",
    }


def adjust_image(
    thread_id: str,
    image_path: str,
    output_path: str,
    brightness: float = 1.0,
    contrast: float = 1.0,
    saturation: float = 1.0,
    sharpness: float = 1.0,
) -> dict[str, Any]:
    factors = {
        "brightness": _validate_factor(brightness, "brightness"),
        "contrast": _validate_factor(contrast, "contrast"),
        "saturation": _validate_factor(saturation, "saturation"),
        "sharpness": _validate_factor(sharpness, "sharpness"),
    }
    workspace = get_session_workspace(thread_id)
    image, target, normalized_input, normalized_output = _load_for_edit(
        workspace, image_path, output_path
    )
    from PIL import ImageEnhance

    result = image
    for key, enhancer_class in (
        ("brightness", ImageEnhance.Brightness),
        ("contrast", ImageEnhance.Contrast),
        ("saturation", ImageEnhance.Color),
        ("sharpness", ImageEnhance.Sharpness),
    ):
        if factors[key] != 1.0:
            result = enhancer_class(result).enhance(factors[key])
    size = _save_image(result, target)
    return {
        "success": True,
        "input_path": normalized_input,
        "output_path": normalized_output,
        "adjustments": factors,
        "width": result.width,
        "height": result.height,
        "size": size,
        "output": f"已调整图片参数：{normalized_output}",
    }


def rotate_flip_image(
    thread_id: str,
    image_path: str,
    output_path: str,
    rotate_degrees: int = 0,
    flip_horizontal: bool = False,
    flip_vertical: bool = False,
) -> dict[str, Any]:
    if type(rotate_degrees) is not int or rotate_degrees not in {0, 90, 180, 270}:
        raise WorkspaceImageEditError("rotate_degrees 只能是 0、90、180 或 270。")
    if type(flip_horizontal) is not bool or type(flip_vertical) is not bool:
        raise WorkspaceImageEditError("翻转参数必须是布尔值。")
    workspace = get_session_workspace(thread_id)
    image, target, normalized_input, normalized_output = _load_for_edit(
        workspace, image_path, output_path
    )
    from PIL import Image

    result = image.rotate(-rotate_degrees, expand=True) if rotate_degrees else image
    if flip_horizontal:
        result = result.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if flip_vertical:
        result = result.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    size = _save_image(result, target)
    return {
        "success": True,
        "input_path": normalized_input,
        "output_path": normalized_output,
        "transforms": {
            "rotate": rotate_degrees,
            "flip_horizontal": flip_horizontal,
            "flip_vertical": flip_vertical,
        },
        "new_size": [result.width, result.height],
        "size": size,
        "output": f"已旋转或翻转图片：{normalized_output}",
    }


def add_watermark(
    thread_id: str,
    image_path: str,
    output_path: str,
    text: str,
    position: str = "bottom-right",
    opacity: float = 0.5,
    font_size: int = 36,
    color: str = "FFFFFF",
) -> dict[str, Any]:
    positions = {"center", "bottom-right", "bottom-left", "top-right", "top-left"}
    if position not in positions:
        raise WorkspaceImageEditError("水印位置参数无效。")
    if not str(text or "").strip():
        raise WorkspaceImageEditError("水印文本不能为空。")
    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)) or not 0.1 <= opacity <= 1.0:
        raise WorkspaceImageEditError("opacity 必须是 0.1 到 1.0 之间的数字。")
    if type(font_size) is not int or not 8 <= font_size <= 200:
        raise WorkspaceImageEditError("font_size 必须是 8 到 200 之间的整数。")
    color = _validate_hex_color(color)
    workspace = get_session_workspace(thread_id)
    image, target, normalized_input, normalized_output = _load_for_edit(
        workspace, image_path, output_path
    )
    from PIL import Image, ImageDraw, ImageFont

    base = image.convert("RGBA")
    layer = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    font = ImageFont.load_default(size=font_size)
    bounds = draw.textbbox((0, 0), text, font=font)
    text_width = bounds[2] - bounds[0]
    text_height = bounds[3] - bounds[1]
    padding = max(10, font_size // 2)
    locations = {
        "center": ((base.width - text_width) // 2, (base.height - text_height) // 2),
        "bottom-right": (base.width - text_width - padding, base.height - text_height - padding),
        "bottom-left": (padding, base.height - text_height - padding),
        "top-right": (base.width - text_width - padding, padding),
        "top-left": (padding, padding),
    }
    x, y = locations[position]
    rgb = tuple(int(color[index : index + 2], 16) for index in (0, 2, 4))
    alpha = round(opacity * 255)
    draw.text((x + 2, y + 2), text, font=font, fill=(0, 0, 0, alpha // 2))
    draw.text((x, y), text, font=font, fill=(*rgb, alpha))
    result = Image.alpha_composite(base, layer)
    size = _save_image(result, target)
    return {
        "success": True,
        "input_path": normalized_input,
        "output_path": normalized_output,
        "watermark_position": position,
        "width": result.width,
        "height": result.height,
        "size": size,
        "output": f"已添加图片水印：{normalized_output}",
    }


def overlay_images(
    thread_id: str,
    base_image_path: str,
    overlay_image_path: str,
    output_path: str,
    x: int = 0,
    y: int = 0,
    opacity: float = 1.0,
    resize_overlay: list[int] | None = None,
) -> dict[str, Any]:
    if type(x) is not int or type(y) is not int:
        raise WorkspaceImageEditError("x 和 y 必须是整数。")
    if isinstance(opacity, bool) or not isinstance(opacity, (int, float)) or not 0.0 <= opacity <= 1.0:
        raise WorkspaceImageEditError("opacity 必须是 0 到 1 之间的数字。")
    if resize_overlay is not None:
        if (
            not isinstance(resize_overlay, list)
            or len(resize_overlay) != 2
            or any(type(value) is not int or not 1 <= value <= 8_000 for value in resize_overlay)
        ):
            raise WorkspaceImageEditError("resize_overlay 必须是 [宽度, 高度]。")
    workspace = get_session_workspace(thread_id)
    base_path = _image_target(workspace, base_image_path)
    overlay_path = _image_target(workspace, overlay_image_path)
    output_name = _validate_image_path(output_path, output=True)
    target = workspace._target(output_name)
    Image = _pillow()
    with Image.open(base_path) as base_source, Image.open(overlay_path) as overlay_source:
        base = base_source.convert("RGBA")
        overlay = overlay_source.convert("RGBA")
        if resize_overlay:
            overlay = overlay.resize(tuple(resize_overlay), Image.Resampling.LANCZOS)
        if opacity != 1.0:
            alpha = overlay.getchannel("A").point(lambda value: round(value * opacity))
            overlay.putalpha(alpha)
        base.alpha_composite(overlay, dest=(x, y))
        size = _save_image(base, target)
        new_size = [base.width, base.height]
    return {
        "success": True,
        "base_image": str(base_image_path).replace("\\", "/"),
        "overlay_image": str(overlay_image_path).replace("\\", "/"),
        "output_path": output_name,
        "size": size,
        "new_size": new_size,
        "output": f"已叠加图片：{output_name}",
    }


def create_thumbnail(
    thread_id: str,
    image_path: str,
    output_path: str,
    max_size: int = 128,
    maintain_aspect: bool = True,
) -> dict[str, Any]:
    if type(max_size) is not int or not 1 <= max_size <= 8_000:
        raise WorkspaceImageEditError("max_size 必须是 1 到 8000 之间的整数。")
    if type(maintain_aspect) is not bool:
        raise WorkspaceImageEditError("maintain_aspect 必须是布尔值。")
    workspace = get_session_workspace(thread_id)
    image, target, normalized_input, normalized_output = _load_for_edit(
        workspace, image_path, output_path
    )
    original_size = image.size
    if maintain_aspect:
        image.thumbnail((max_size, max_size), _pillow().Resampling.LANCZOS)
    else:
        image = image.resize((max_size, max_size), _pillow().Resampling.LANCZOS)
    size = _save_image(image, target)
    return {
        "success": True,
        "input_path": normalized_input,
        "output_path": normalized_output,
        "original_size": list(original_size),
        "thumbnail_size": [image.width, image.height],
        "size": size,
        "output": f"已生成图片缩略图：{normalized_output}",
    }


def build_workspace_image_edit_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    def apply_filter_tool(image_path: str, output_path: str, filter_type: FilterType, intensity: float = 1.0) -> dict[str, Any]:
        return apply_image_filter(thread_id, image_path, output_path, filter_type, intensity)

    def apply_effect_tool(image_path: str, output_path: str, effect_type: EffectType) -> dict[str, Any]:
        return apply_image_effect(thread_id, image_path, output_path, effect_type)

    def adjust_tool(image_path: str, output_path: str, brightness: float = 1.0, contrast: float = 1.0, saturation: float = 1.0, sharpness: float = 1.0) -> dict[str, Any]:
        return adjust_image(thread_id, image_path, output_path, brightness, contrast, saturation, sharpness)

    def rotate_tool(image_path: str, output_path: str, rotate_degrees: int = 0, flip_horizontal: bool = False, flip_vertical: bool = False) -> dict[str, Any]:
        return rotate_flip_image(thread_id, image_path, output_path, rotate_degrees, flip_horizontal, flip_vertical)

    def watermark_tool(image_path: str, output_path: str, text: str, position: str = "bottom-right", opacity: float = 0.5, font_size: int = 36, color: str = "FFFFFF") -> dict[str, Any]:
        return add_watermark(thread_id, image_path, output_path, text, position, opacity, font_size, color)

    def overlay_tool(base_image_path: str, overlay_image_path: str, output_path: str, x: int = 0, y: int = 0, opacity: float = 1.0, resize_overlay: list[int] | None = None) -> dict[str, Any]:
        return overlay_images(thread_id, base_image_path, overlay_image_path, output_path, x, y, opacity, resize_overlay)

    def thumbnail_tool(image_path: str, output_path: str, max_size: int = 128, maintain_aspect: bool = True) -> dict[str, Any]:
        return create_thumbnail(thread_id, image_path, output_path, max_size, maintain_aspect)

    return [
        ToolDefinition("apply_image_filter", "对工作区图片应用模糊、锐化、浮雕等滤镜。", apply_filter_tool),
        ToolDefinition("apply_image_effect", "对工作区图片应用灰度、复古、反色等效果。", apply_effect_tool),
        ToolDefinition("adjust_image", "调整工作区图片的亮度、对比度、饱和度和锐度。", adjust_tool),
        ToolDefinition("rotate_flip_image", "旋转或水平、垂直翻转工作区图片。", rotate_tool),
        ToolDefinition("add_watermark", "向工作区图片添加文字水印。", watermark_tool),
        ToolDefinition("overlay_images", "将一张工作区图片以指定位置和透明度叠加到另一张图片。", overlay_tool),
        ToolDefinition("create_thumbnail", "为工作区图片生成缩略图。", thumbnail_tool),
    ]
