from __future__ import annotations

import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from tools.registry import ToolDefinition
from tools.workspace_files import (
    MAX_FILE_BYTES,
    SessionWorkspace,
    WorkspaceFileError,
    WorkspaceFileNotFoundError,
    get_session_workspace,
)


SUPPORTED_IMAGE_EXTENSIONS = {
    ".png",
    ".jpg",
    ".jpeg",
    ".webp",
    ".gif",
    ".bmp",
    ".tif",
    ".tiff",
}
MAX_IMAGE_DIMENSION = 8_000


class WorkspaceVisionError(WorkspaceFileError):
    """工作区图像操作错误。"""


def _pillow() -> Any:
    try:
        from PIL import Image
    except ImportError as exc:
        raise WorkspaceVisionError(
            "图像工具需要 Pillow；请先安装项目 requirements.txt 中的依赖。"
        ) from exc
    return Image


def _validate_image_path(path: str, *, output: bool = False) -> str:
    normalized = str(path or "").strip().replace("\\", "/")
    if not normalized:
        raise WorkspaceVisionError("图片路径不能为空。")
    if Path(normalized).suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
        kind = "输出图片" if output else "图片"
        raise WorkspaceVisionError(f"{kind}格式不受支持。")
    return normalized


def _image_target(workspace: SessionWorkspace, path: str, *, output: bool = False) -> Path:
    normalized = _validate_image_path(path, output=output)
    target = workspace._target(normalized)
    if not output and not target.is_file():
        raise WorkspaceFileNotFoundError("工作区图片不存在。")
    if not output and target.stat().st_size > MAX_FILE_BYTES:
        raise WorkspaceVisionError(f"图片不能超过 {MAX_FILE_BYTES} 字节。")
    return target


def _validate_dimension(value: int | None, name: str) -> int | None:
    if value is None:
        return None
    if type(value) is not int or not 1 <= value <= MAX_IMAGE_DIMENSION:
        raise WorkspaceVisionError(
            f"{name} 必须是 1 到 {MAX_IMAGE_DIMENSION} 之间的整数。"
        )
    return value


def _validate_quality(quality: int) -> int:
    if type(quality) is not int or not 1 <= quality <= 100:
        raise WorkspaceVisionError("quality 必须是 1 到 100 之间的整数。")
    return quality


def _save_image(image: Any, target: Path, *, quality: int = 85) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=".image-",
        suffix=target.suffix,
        dir=target.parent,
    )
    temporary = Path(temporary_name)
    try:
        import os

        os.close(descriptor)
        extension = target.suffix.lower()
        output_format = {
            ".jpg": "JPEG",
            ".jpeg": "JPEG",
            ".png": "PNG",
            ".webp": "WEBP",
            ".gif": "GIF",
            ".bmp": "BMP",
            ".tif": "TIFF",
            ".tiff": "TIFF",
        }[extension]
        save_image = image
        if output_format == "JPEG" and save_image.mode in {"RGBA", "LA", "P"}:
            save_image = save_image.convert("RGB")
        options: dict[str, Any] = {}
        if output_format == "JPEG":
            options["quality"] = quality
        elif output_format == "PNG":
            options["optimize"] = True
        save_image.save(temporary, output_format, **options)
        size = temporary.stat().st_size
        if size > MAX_FILE_BYTES:
            raise WorkspaceVisionError(f"输出图片不能超过 {MAX_FILE_BYTES} 字节。")
        temporary.replace(target)
        return size
    finally:
        temporary.unlink(missing_ok=True)


def get_image_info(thread_id: str, image_path: str) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    target = _image_target(workspace, image_path)
    Image = _pillow()
    try:
        with Image.open(target) as image:
            sample = image.convert("RGB").copy()
            sample.thumbnail((100, 100))
            pixels = (
                sample.get_flattened_data()
                if hasattr(sample, "get_flattened_data")
                else list(sample.getdata())
            )
            top_colors = [
                {"rgb": list(color), "count": count}
                for color, count in Counter(pixels).most_common(5)
            ]
            has_transparency = image.mode in {"RGBA", "LA"} or (
                image.mode == "P" and "transparency" in image.info
            )
            return {
                "success": True,
                "path": str(image_path).replace("\\", "/"),
                "width": image.width,
                "height": image.height,
                "format": image.format,
                "mode": image.mode,
                "file_size_bytes": target.stat().st_size,
                "has_transparency": has_transparency,
                "top_colors": top_colors,
                "dpi": image.info.get("dpi"),
                "output": f"已读取图片信息：{image_path}",
            }
    except OSError as exc:
        raise WorkspaceVisionError("图片无法读取或格式无效。") from exc


def resize_image(
    thread_id: str,
    image_path: str,
    output_path: str,
    width: int | None = None,
    height: int | None = None,
    quality: int = 85,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    source = _image_target(workspace, image_path)
    target = workspace._target(_validate_image_path(output_path, output=True))
    width = _validate_dimension(width, "width")
    height = _validate_dimension(height, "height")
    quality = _validate_quality(quality)
    if width is None and height is None:
        raise WorkspaceVisionError("width 和 height 至少需要提供一个。")
    Image = _pillow()
    with Image.open(source) as image:
        original = image.size
        if width is None:
            width = max(1, round(image.width * height / image.height))
        elif height is None:
            height = max(1, round(image.height * width / image.width))
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        size = _save_image(resized, target, quality=quality)
    return {
        "success": True,
        "input_path": str(image_path).replace("\\", "/"),
        "output_path": str(output_path).replace("\\", "/"),
        "original_size": list(original),
        "new_size": [width, height],
        "size": size,
        "output": f"已调整图片尺寸：{output_path}",
    }


def convert_image(
    thread_id: str,
    image_path: str,
    output_path: str,
    quality: int = 85,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    source = _image_target(workspace, image_path)
    target = workspace._target(_validate_image_path(output_path, output=True))
    quality = _validate_quality(quality)
    Image = _pillow()
    with Image.open(source) as image:
        input_format = image.format
        size = _save_image(image, target, quality=quality)
    return {
        "success": True,
        "input_path": str(image_path).replace("\\", "/"),
        "output_path": str(output_path).replace("\\", "/"),
        "input_format": input_format,
        "output_format": target.suffix.lstrip(".").upper(),
        "size": size,
        "output": f"已转换图片格式：{output_path}",
    }


def crop_image(
    thread_id: str,
    image_path: str,
    output_path: str,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    source = _image_target(workspace, image_path)
    target = workspace._target(_validate_image_path(output_path, output=True))
    values = (left, top, right, bottom)
    if any(type(value) is not int for value in values):
        raise WorkspaceVisionError("裁剪坐标必须是整数。")
    Image = _pillow()
    with Image.open(source) as image:
        if not (0 <= left < right <= image.width and 0 <= top < bottom <= image.height):
            raise WorkspaceVisionError(
                f"裁剪区域越界；图片尺寸为 {image.width}x{image.height}。"
            )
        cropped = image.crop((left, top, right, bottom))
        size = _save_image(cropped, target)
    return {
        "success": True,
        "input_path": str(image_path).replace("\\", "/"),
        "output_path": str(output_path).replace("\\", "/"),
        "original_size": [image.width, image.height],
        "crop_box": [left, top, right, bottom],
        "new_size": [right - left, bottom - top],
        "size": size,
        "output": f"已裁剪图片：{output_path}",
    }


def compare_images(
    thread_id: str,
    image1_path: str,
    image2_path: str,
) -> dict[str, Any]:
    workspace = get_session_workspace(thread_id)
    source1 = _image_target(workspace, image1_path)
    source2 = _image_target(workspace, image2_path)
    Image = _pillow()
    from PIL import ImageChops, ImageStat

    with Image.open(source1) as first, Image.open(source2) as second:
        original_size_same = first.size == second.size
        image1 = first.convert("RGB")
        image2 = second.convert("RGB")
        if image1.size != image2.size:
            size = (
                min(image1.width, image2.width),
                min(image1.height, image2.height),
            )
            image1 = image1.resize(size, Image.Resampling.LANCZOS)
            image2 = image2.resize(size, Image.Resampling.LANCZOS)
        difference = ImageChops.difference(image1, image2)
        mean_squared_error = sum(value * value for value in ImageStat.Stat(difference).mean) / 3
        similarity = max(0.0, 100.0 - (mean_squared_error / (255 * 255) * 100.0))
        return {
            "success": True,
            "image1_path": str(image1_path).replace("\\", "/"),
            "image2_path": str(image2_path).replace("\\", "/"),
            "same_size": original_size_same,
            "image1_size": list(image1.size),
            "image2_size": list(image2.size),
            "mse": round(mean_squared_error, 4),
            "similarity_percent": round(similarity, 2),
            "identical": mean_squared_error == 0,
            "output": "已完成图片比较。",
        }


def build_workspace_vision_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    def get_image_info_tool(image_path: str) -> dict[str, Any]:
        return get_image_info(thread_id, image_path)

    def resize_image_tool(
        image_path: str,
        output_path: str,
        width: int | None = None,
        height: int | None = None,
        quality: int = 85,
    ) -> dict[str, Any]:
        return resize_image(thread_id, image_path, output_path, width, height, quality)

    def convert_image_tool(
        image_path: str,
        output_path: str,
        quality: int = 85,
    ) -> dict[str, Any]:
        return convert_image(thread_id, image_path, output_path, quality)

    def crop_image_tool(
        image_path: str,
        output_path: str,
        left: int,
        top: int,
        right: int,
        bottom: int,
    ) -> dict[str, Any]:
        return crop_image(thread_id, image_path, output_path, left, top, right, bottom)

    def compare_images_tool(image1_path: str, image2_path: str) -> dict[str, Any]:
        return compare_images(thread_id, image1_path, image2_path)

    return [
        ToolDefinition(
            "get_image_info",
            "读取当前会话工作区图片的尺寸、格式、透明度和主色信息。",
            get_image_info_tool,
        ),
        ToolDefinition(
            "resize_image",
            "按指定宽度或高度调整工作区图片尺寸，并保持宽高比。",
            resize_image_tool,
        ),
        ToolDefinition(
            "convert_image",
            "将工作区图片转换为 PNG、JPEG、WEBP 等格式。",
            convert_image_tool,
        ),
        ToolDefinition(
            "crop_image",
            "按像素坐标裁剪工作区图片。",
            crop_image_tool,
        ),
        ToolDefinition(
            "compare_images",
            "比较两张工作区图片并返回尺寸、均方误差和相似度。",
            compare_images_tool,
        ),
    ]
