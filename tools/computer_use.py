from __future__ import annotations

import base64
import io
import time
from typing import Any, Callable

from agent.core.event_stream import get_emitter_sync
from tools.registry import ToolDefinition
from tools.screenshot_store import get_screenshot_store


class ComputerUseError(RuntimeError):
    """本地桌面自动化错误。"""


def _pyautogui() -> Any:
    try:
        import pyautogui
    except Exception as exc:
        raise ComputerUseError(
            "computer_use 需要 pyautogui；请执行 pip install pyautogui。"
        ) from exc
    pyautogui.FAILSAFE = True
    pyautogui.PAUSE = 0.1
    return pyautogui


def _emit(thread_id: str, event_type: str, data: dict[str, Any]) -> None:
    try:
        get_emitter_sync(thread_id).emit_sync(
            event_type,
            {"thread_id": thread_id, "tool": "computer_use", **data},
        )
    except Exception:
        pass


def _screen_size(gui: Any) -> tuple[int, int]:
    width, height = gui.size()
    return max(int(width), 1), max(int(height), 1)


def _point(gui: Any, x: int, y: int) -> tuple[int, int]:
    if type(x) is not int or type(y) is not int:
        raise ComputerUseError("屏幕坐标必须是整数。")
    width, height = _screen_size(gui)
    return max(0, min(x, width - 1)), max(0, min(y, height - 1))


def _capture(thread_id: str, gui: Any, action: str) -> dict[str, Any]:
    screenshot = gui.screenshot()
    buffer = io.BytesIO()
    screenshot.save(buffer, format="PNG")
    image_data = buffer.getvalue()
    encoded = base64.b64encode(image_data).decode("ascii")
    result: dict[str, Any] = {"image": encoded}
    try:
        saved = get_screenshot_store().save_screenshot(
            image_data,
            action=f"computer_{action}",
            thread_id=thread_id,
        )
        result.update(
            {
                "screenshot_url": saved.get("url"),
                "screenshot_filename": saved.get("filename"),
                "mime_type": saved.get("mime_type"),
            }
        )
        _emit(
            thread_id,
            "tool_screenshot",
            {
                "action": action,
                "url": saved.get("url"),
                "filename": saved.get("filename"),
                "mime_type": saved.get("mime_type"),
            },
        )
    except Exception as exc:
        result["screenshot_error"] = str(exc)
    return result


def _run_action(
    thread_id: str,
    action: str,
    arguments: dict[str, Any],
    operation: Callable[[Any], dict[str, Any]],
) -> dict[str, Any]:
    started_at = time.monotonic()
    _emit(thread_id, "tool_start", {"action": action, "args": arguments})
    try:
        result = operation(_pyautogui())
        result.setdefault("success", True)
        _emit(
            thread_id,
            "tool_result",
            {
                "action": action,
                "success": bool(result.get("success")),
                "duration_ms": round((time.monotonic() - started_at) * 1000, 2),
            },
        )
        return result
    except Exception as exc:
        message = str(exc)
        _emit(
            thread_id,
            "tool_error",
            {
                "action": action,
                "success": False,
                "error": message,
                "duration_ms": round((time.monotonic() - started_at) * 1000, 2),
            },
        )
        return {"success": False, "error": message}


def build_computer_use_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    normalized_thread_id = str(thread_id or "").strip() or "default"

    def computer_move_mouse(x: int, y: int) -> dict[str, Any]:
        def operation(gui: Any) -> dict[str, Any]:
            target_x, target_y = _point(gui, x, y)
            gui.moveTo(target_x, target_y, duration=0.2)
            return {
                "message": f"鼠标已移动到 ({target_x}, {target_y})。",
                "position": {"x": target_x, "y": target_y},
            }

        return _run_action(normalized_thread_id, "move_mouse", {"x": x, "y": y}, operation)

    def computer_click(
        x: int,
        y: int,
        button: str = "left",
        clicks: int = 1,
    ) -> dict[str, Any]:
        def operation(gui: Any) -> dict[str, Any]:
            target_x, target_y = _point(gui, x, y)
            selected_button = str(button or "left").lower()
            if selected_button not in {"left", "right", "middle"}:
                raise ComputerUseError("button 只能是 left、right 或 middle。")
            if type(clicks) is not int or not 1 <= clicks <= 3:
                raise ComputerUseError("clicks 必须是 1 到 3 之间的整数。")
            gui.click(target_x, target_y, clicks=clicks, button=selected_button)
            return {
                "message": f"已点击 ({target_x}, {target_y})。",
                "position": {"x": target_x, "y": target_y},
                **_capture(normalized_thread_id, gui, "click"),
            }

        return _run_action(
            normalized_thread_id,
            "click",
            {"x": x, "y": y, "button": button, "clicks": clicks},
            operation,
        )

    def computer_type(text: str, interval: float = 0.02) -> dict[str, Any]:
        def operation(gui: Any) -> dict[str, Any]:
            if not isinstance(text, str):
                raise ComputerUseError("text 必须是字符串。")
            if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not 0 <= interval <= 1:
                raise ComputerUseError("interval 必须是 0 到 1 之间的数字。")
            gui.write(text, interval=float(interval))
            return {
                "message": f"已输入 {len(text)} 个字符。",
                "text_length": len(text),
                **_capture(normalized_thread_id, gui, "type"),
            }

        return _run_action(
            normalized_thread_id,
            "type",
            {"text": text[:50] + "..." if len(text) > 50 else text},
            operation,
        )

    def computer_press(keys: str) -> dict[str, Any]:
        def operation(gui: Any) -> dict[str, Any]:
            normalized_keys = str(keys or "").lower().replace(" ", "")
            if not normalized_keys:
                raise ComputerUseError("keys 不能为空。")
            parts = normalized_keys.split("+")
            if any(not part for part in parts):
                raise ComputerUseError("keys 组合格式无效。")
            if len(parts) == 1:
                gui.press(parts[0])
            else:
                gui.hotkey(*parts)
            return {
                "message": f"已按下：{keys}",
                "keys": keys,
                **_capture(normalized_thread_id, gui, "press"),
            }

        return _run_action(normalized_thread_id, "press", {"keys": keys}, operation)

    def computer_scroll(direction: str, amount: int = 3) -> dict[str, Any]:
        def operation(gui: Any) -> dict[str, Any]:
            selected_direction = str(direction or "").lower()
            if selected_direction not in {"up", "down"}:
                raise ComputerUseError("direction 只能是 up 或 down。")
            if type(amount) is not int or not 1 <= amount <= 50:
                raise ComputerUseError("amount 必须是 1 到 50 之间的整数。")
            delta = amount if selected_direction == "up" else -amount
            gui.scroll(delta)
            return {
                "message": f"已向 {selected_direction} 滚动 {amount}。",
                "direction": selected_direction,
                "amount": amount,
                **_capture(normalized_thread_id, gui, "scroll"),
            }

        return _run_action(
            normalized_thread_id,
            "scroll",
            {"direction": direction, "amount": amount},
            operation,
        )

    def computer_screenshot() -> dict[str, Any]:
        def operation(gui: Any) -> dict[str, Any]:
            width, height = _screen_size(gui)
            return {
                "message": f"已捕获屏幕截图（{width}x{height}）。",
                "screen_size": {"width": width, "height": height},
                **_capture(normalized_thread_id, gui, "manual"),
            }

        return _run_action(normalized_thread_id, "screenshot", {}, operation)

    def computer_screen_info() -> dict[str, Any]:
        def operation(gui: Any) -> dict[str, Any]:
            width, height = _screen_size(gui)
            mouse_x, mouse_y = gui.position()
            return {
                "screen": {"width": width, "height": height},
                "mouse_position": {"x": int(mouse_x), "y": int(mouse_y)},
            }

        return _run_action(normalized_thread_id, "screen_info", {}, operation)

    def computer_drag(
        start_x: int,
        start_y: int,
        end_x: int,
        end_y: int,
        duration: float = 0.5,
    ) -> dict[str, Any]:
        def operation(gui: Any) -> dict[str, Any]:
            source_x, source_y = _point(gui, start_x, start_y)
            target_x, target_y = _point(gui, end_x, end_y)
            if isinstance(duration, bool) or not isinstance(duration, (int, float)) or not 0 <= duration <= 10:
                raise ComputerUseError("duration 必须是 0 到 10 之间的数字。")
            gui.moveTo(source_x, source_y)
            gui.dragTo(target_x, target_y, duration=float(duration), button="left")
            return {
                "message": f"已从 ({source_x}, {source_y}) 拖动到 ({target_x}, {target_y})。",
                "start": {"x": source_x, "y": source_y},
                "end": {"x": target_x, "y": target_y},
                **_capture(normalized_thread_id, gui, "drag"),
            }

        return _run_action(
            normalized_thread_id,
            "drag",
            {
                "start_x": start_x,
                "start_y": start_y,
                "end_x": end_x,
                "end_y": end_y,
            },
            operation,
        )

    description = "使用本地 pyautogui 执行桌面鼠标、键盘、滚动、拖拽和截图操作。"
    return [
        ToolDefinition("computer_move_mouse", description, computer_move_mouse),
        ToolDefinition("computer_click", description, computer_click),
        ToolDefinition("computer_type", description, computer_type),
        ToolDefinition("computer_press", description, computer_press),
        ToolDefinition("computer_scroll", description, computer_scroll),
        ToolDefinition("computer_screenshot", description, computer_screenshot),
        ToolDefinition("computer_screen_info", description, computer_screen_info),
        ToolDefinition("computer_drag", description, computer_drag),
    ]


__all__ = [
    "ComputerUseError",
    "build_computer_use_tool_definitions",
]
