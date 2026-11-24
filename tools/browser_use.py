from __future__ import annotations

import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import quote_plus, urlsplit

from agent.core.event_stream import get_emitter_sync
from tools.registry import ToolDefinition
from tools.screenshot_store import get_screenshot_store


class BrowserUseError(RuntimeError):
    """本地 Playwright 浏览器自动化错误。"""


def _playwright_sync() -> Any:
    try:
        from playwright.sync_api import sync_playwright
    except Exception as exc:
        raise BrowserUseError(
            "browser_use 需要 Playwright；请先安装依赖并执行 python -m playwright install chromium。"
        ) from exc
    return sync_playwright


@dataclass
class _BrowserUseSession:
    thread_id: str
    playwright: Any = None
    browser: Any = None
    context: Any = None
    page: Any = None
    _executor: ThreadPoolExecutor | None = field(
        default=None,
        init=False,
        repr=False,
    )
    _executor_lock: Any = field(
        default_factory=threading.RLock,
        init=False,
        repr=False,
    )
    _closed: bool = field(default=False, init=False, repr=False)

    def run(
        self,
        operation: Callable[[], Any],
        *,
        timeout_seconds: float | None = None,
    ) -> Any:
        with self._executor_lock:
            if self._closed:
                raise BrowserUseError("浏览器会话已关闭。")
            if self._executor is None:
                self._executor = ThreadPoolExecutor(
                    max_workers=1,
                    thread_name_prefix="browser-use",
                )
            future = self._executor.submit(operation)

        return future.result(timeout=timeout_seconds)

    def snapshot(self) -> dict[str, str] | None:
        def read_page() -> dict[str, str] | None:
            page = self.page
            if page is None:
                return None

            try:
                title = str(page.title() or "")
            except Exception:
                title = ""
            return {
                "url": str(getattr(page, "url", "") or ""),
                "title": title,
            }

        return self.run(read_page)

    def context_snapshot(
        self,
        *,
        max_chars: int = 3000,
        max_links: int = 5,
        timeout_seconds: float = 1.5,
    ) -> dict[str, Any] | None:
        def read_page() -> dict[str, Any] | None:
            page = self.page
            if page is None:
                return None

            try:
                title = str(page.title() or "")[:200]
            except Exception:
                title = ""
            try:
                text = str(
                    page.locator("body").inner_text(timeout=1000) or ""
                )[:max(0, int(max_chars))]
            except Exception:
                text = ""
            try:
                links = page.locator("a[href]").evaluate_all(
                    """(elements, limit) => elements.slice(0, limit).map(element => ({
                        text: (element.innerText || element.getAttribute('aria-label') || '').trim().slice(0, 160),
                        url: element.href
                    }))""",
                    max(0, int(max_links)),
                )
            except Exception:
                links = []

            return {
                "url": str(getattr(page, "url", "") or ""),
                "title": title,
                "text": text,
                "links": links if isinstance(links, list) else [],
            }

        try:
            return self.run(read_page, timeout_seconds=timeout_seconds)
        except Exception:
            return None

    def capture_screenshot(
        self,
        *,
        full_page: bool = True,
        quality: int = 85,
    ) -> dict[str, Any] | None:
        def capture() -> dict[str, Any] | None:
            page = self.page
            if page is None:
                return None

            metadata = self._page_metadata(page)
            normalized_quality = max(1, min(int(quality or 85), 100))
            image_data = page.screenshot(
                full_page=bool(full_page),
                type="jpeg",
                quality=normalized_quality,
            )
            return {
                "image_data": image_data,
                "metadata": metadata,
            }

        return self.run(capture)

    def perform_preview_action(
        self,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, str]:
        def perform() -> dict[str, str]:
            page = self.page
            if page is None:
                raise BrowserUseError("浏览器会话尚未启动。")

            action_name = str(action or "").strip().lower()
            if action_name == "mouse":
                mouse_type = str(payload.get("type") or "").strip().lower()
                if mouse_type not in {"click", "move", "down", "up"}:
                    raise ValueError(
                        f"Unsupported mouse action type: {mouse_type or '(missing)'}"
                    )

                button = str(payload.get("button") or "left").strip().lower()
                if button not in {"left", "middle", "right"}:
                    raise ValueError("Mouse button must be left, middle, or right.")

                if mouse_type in {"click", "move"}:
                    x = self._normalized_coordinate(payload.get("x"), "x")
                    y = self._normalized_coordinate(payload.get("y"), "y")
                    width, height = self._viewport_size(page)
                    x_px = min(int(x * width), width - 1)
                    y_px = min(int(y * height), height - 1)
                    if mouse_type == "click":
                        try:
                            clicks = int(payload.get("clicks", 1) or 1)
                        except (TypeError, ValueError):
                            clicks = 1
                        page.mouse.click(
                            x_px,
                            y_px,
                            button=button,
                            click_count=max(1, min(clicks, 10)),
                        )
                    else:
                        page.mouse.move(x_px, y_px)
                elif mouse_type == "down":
                    page.mouse.down(button=button)
                else:
                    page.mouse.up(button=button)

            elif action_name == "scroll":
                try:
                    dx = int(payload.get("dx", 0) or 0)
                    dy = int(payload.get("dy", 0) or 0)
                except (TypeError, ValueError) as exc:
                    raise ValueError("Scroll requires integer dx/dy.") from exc
                page.mouse.wheel(dx, dy)

            elif action_name == "keyboard":
                keyboard_type = str(payload.get("type") or "").strip().lower()
                if keyboard_type == "press":
                    key = str(payload.get("key") or "").strip()
                    if not key:
                        raise ValueError("Keyboard press requires a non-empty key.")
                    page.keyboard.press(key)
                elif keyboard_type == "type":
                    text = str(payload.get("text") or "")
                    if not text:
                        raise ValueError("Keyboard type requires a non-empty text.")
                    page.keyboard.type(text)
                else:
                    raise ValueError(
                        f"Unsupported keyboard action type: {keyboard_type or '(missing)'}"
                    )

            elif action_name == "navigate":
                target_url = str(payload.get("url") or "").strip()
                parsed = urlsplit(target_url)
                if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
                    raise ValueError("Only http(s) URLs with a host are allowed.")
                page.goto(
                    target_url,
                    wait_until="domcontentloaded",
                    timeout=30000,
                )

            else:
                raise ValueError(f"Unsupported action: {action_name or 'unknown'}")

            return self._page_metadata(page)

        return self.run(perform)

    @staticmethod
    def _page_metadata(page: Any) -> dict[str, str]:
        try:
            title = str(page.title() or "")
        except Exception:
            title = ""
        return {
            "url": str(getattr(page, "url", "") or ""),
            "title": title,
        }

    @staticmethod
    def _normalized_coordinate(value: Any, name: str) -> float:
        if isinstance(value, bool):
            raise ValueError("Mouse click/move requires numeric x/y in [0..1].")
        try:
            coordinate = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Mouse click/move requires numeric x/y in [0..1]."
            ) from exc
        if not 0.0 <= coordinate <= 1.0:
            raise ValueError(f"Mouse {name} must be between 0 and 1.")
        return coordinate

    @staticmethod
    def _viewport_size(page: Any) -> tuple[int, int]:
        viewport = getattr(page, "viewport_size", None)
        width = viewport.get("width") if isinstance(viewport, dict) else None
        height = viewport.get("height") if isinstance(viewport, dict) else None
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            measured = page.evaluate(
                "() => ({ w: window.innerWidth, h: window.innerHeight })"
            )
            if isinstance(measured, dict):
                width = measured.get("w") or measured.get("width")
                height = measured.get("h") or measured.get("height")
        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise RuntimeError("Could not determine browser viewport size")
        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid browser viewport size")
        return int(width), int(height)

    def ensure_page(self) -> Any:
        if self.context is not None:
            pages = list(getattr(self.context, "pages", []) or [])
            if self.page is None or self.page not in pages:
                self.page = pages[-1] if pages else self.context.new_page()
            return self.page

        sync_playwright = _playwright_sync()
        try:
            self.playwright = sync_playwright().start()
            self.browser = self.playwright.chromium.launch(headless=True)
            self.context = self.browser.new_context(
                viewport={"width": 1280, "height": 720}
            )
            self.page = self.context.new_page()
            return self.page
        except BrowserUseError:
            self._close_resources()
            raise
        except Exception as exc:
            self._close_resources()
            raise BrowserUseError(f"浏览器启动失败：{exc}") from exc

    def pages(self) -> list[Any]:
        if self.context is None:
            return []
        return list(getattr(self.context, "pages", []) or [])

    def close(self) -> None:
        with self._executor_lock:
            if self._closed:
                return
            self._closed = True
            executor = self._executor
            close_future = (
                executor.submit(self._close_resources)
                if executor is not None
                else None
            )

        if close_future is not None and executor is not None:
            try:
                close_future.result()
            finally:
                executor.shutdown(wait=True)
            return

        self._close_resources()

    def _close_resources(self) -> None:
        for item in (self.context, self.browser, self.playwright):
            if item is None:
                continue
            try:
                item.close() if item is not self.playwright else item.stop()
            except Exception:
                pass
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None


class BrowserUseSessionManager:
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._sessions: dict[str, _BrowserUseSession] = {}

    @staticmethod
    def _normalize(thread_id: str) -> str:
        return str(thread_id or "").strip() or "default"

    def get(self, thread_id: str) -> _BrowserUseSession:
        normalized = self._normalize(thread_id)
        with self._lock:
            session = self._sessions.get(normalized)
            if session is None:
                session = _BrowserUseSession(normalized)
                self._sessions[normalized] = session
            return session

    def peek(self, thread_id: str) -> _BrowserUseSession | None:
        normalized = self._normalize(thread_id)
        with self._lock:
            return self._sessions.get(normalized)

    def reset(self, thread_id: str) -> None:
        normalized = self._normalize(thread_id)
        with self._lock:
            session = self._sessions.pop(normalized, None)
        if session is not None:
            session.close()

    def reset_all(self) -> int:
        with self._lock:
            sessions = list(self._sessions.values())
            self._sessions.clear()

        for session in sessions:
            try:
                session.close()
            except Exception:
                pass

        return len(sessions)


browser_use_sessions = BrowserUseSessionManager()

_ACTION_ALIASES = {
    "navigate": "go_to_url",
    "go_to": "go_to_url",
    "back": "go_back",
    "click": "click_element",
    "input": "input_text",
    "type": "input_text",
    "scroll": "scroll_down",
    "search": "web_search",
    "extract": "extract_content",
}


def _emit(thread_id: str, event_type: str, data: dict[str, Any]) -> None:
    try:
        get_emitter_sync(thread_id).emit_sync(
            event_type,
            {"thread_id": thread_id, "tool": "browser_use", **data},
        )
    except Exception:
        pass


def _screenshot(
    thread_id: str,
    page: Any,
    *,
    action: str,
    full_page: bool = False,
) -> str | None:
    try:
        try:
            image = page.screenshot(
                full_page=bool(full_page),
                type="jpeg",
                quality=85,
            )
        except TypeError:
            image = page.screenshot(full_page=bool(full_page), type="jpeg")
        page_url = str(getattr(page, "url", "") or "")
        try:
            page_title = str(page.title() or "")
        except Exception:
            page_title = ""
        saved = get_screenshot_store().save_screenshot(
            image,
            action=f"browser_use_{action}",
            thread_id=thread_id,
            page_url=page_url,
            metadata={"url": page_url, "title": page_title},
        )
        _emit(
            thread_id,
            "tool_screenshot",
            {
                "action": action,
                "url": saved.get("url"),
                "filename": saved.get("filename"),
                "page_url": saved.get("page_url"),
                "title": page_title,
                "mime_type": saved.get("mime_type"),
            },
        )
        return str(saved.get("url") or "") or None
    except Exception:
        return None


def _page_info(page: Any) -> dict[str, Any]:
    try:
        title = str(page.title() or "")
    except Exception:
        title = ""
    result = {
        "url": str(getattr(page, "url", "") or ""),
        "title": title,
    }
    result["interactive_elements"] = _interactive_elements(page)
    return result


def _interactive_elements(page: Any, limit: int = 50) -> list[dict[str, Any]]:
    try:
        locator = page.locator(
            "a,button,input,textarea,select,[role='button'],[contenteditable='true']"
        )
        count = min(int(locator.count()), limit)
    except Exception:
        return []

    elements: list[dict[str, Any]] = []
    visible_index = 0
    for position in range(count):
        candidate = locator.nth(position)
        try:
            if not candidate.is_visible():
                continue
            details = candidate.evaluate(
                """
                element => ({
                    tag: element.tagName.toLowerCase(),
                    role: element.getAttribute('role') || '',
                    text: (element.innerText || element.value || '').trim().slice(0, 160),
                    aria_label: element.getAttribute('aria-label') || '',
                    placeholder: element.getAttribute('placeholder') || '',
                    type: element.getAttribute('type') || ''
                })
                """
            )
            if not isinstance(details, dict):
                details = {}
        except Exception:
            details = {}
        elements.append({"index": visible_index, **details})
        visible_index += 1
    return elements


def _element_by_index(page: Any, index: int) -> Any:
    if type(index) is not int or index < 0:
        raise BrowserUseError("index 必须是从 0 开始的非负整数。")
    locator = page.locator(
        "a,button,input,textarea,select,[role='button'],[contenteditable='true']"
    )
    visible_index = 0
    for position in range(locator.count()):
        candidate = locator.nth(position)
        if not candidate.is_visible():
            continue
        if visible_index == index:
            return candidate
        visible_index += 1
    raise BrowserUseError(f"未找到索引为 {index} 的可交互元素。")


def _browser_use_on_worker(
    thread_id: str,
    action: str,
    url: str | None = None,
    index: int | None = None,
    text: str | None = None,
    scroll_amount: int | None = None,
    tab_id: int | None = None,
    query: str | None = None,
    goal: str | None = None,
    keys: str | None = None,
    seconds: int | None = None,
) -> dict[str, Any]:
    normalized_thread_id = str(thread_id or "").strip() or "default"
    selected_action = str(action or "").strip().lower()
    selected_action = _ACTION_ALIASES.get(selected_action, selected_action)
    session = browser_use_sessions.get(normalized_thread_id)
    _emit(normalized_thread_id, "tool_start", {"action": selected_action})

    try:
        page = session.ensure_page()

        if selected_action == "go_to_url":
            target = str(url or "").strip()
            if not target:
                raise BrowserUseError("go_to_url 需要 url 参数。")
            page.goto(target, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_load_state()
            result = {
                "status": "ok",
                **_page_info(page),
                "screenshot": _screenshot(normalized_thread_id, page, action="navigate"),
            }
        elif selected_action == "go_back":
            page.go_back(wait_until="domcontentloaded", timeout=30000)
            result = {
                "status": "ok",
                "message": "back",
                **_page_info(page),
                "screenshot": _screenshot(normalized_thread_id, page, action="back"),
            }
        elif selected_action == "click_element":
            if index is None:
                raise BrowserUseError("click_element 需要 index 参数。")
            element = _element_by_index(page, index)
            element.click()
            result = {
                "status": "ok",
                "clicked": index,
                **_page_info(page),
                "screenshot": _screenshot(normalized_thread_id, page, action="click"),
            }
        elif selected_action == "input_text":
            if index is None or text is None:
                raise BrowserUseError("input_text 需要 index 和 text 参数。")
            _element_by_index(page, index).fill(str(text))
            result = {"status": "ok", "input": str(text), "index": index}
        elif selected_action in {"scroll_down", "scroll_up"}:
            amount = int(scroll_amount or 800)
            if amount < 1 or amount > 10_000:
                raise BrowserUseError("scroll_amount 必须是 1 到 10000 之间的整数。")
            delta = amount if selected_action == "scroll_down" else -amount
            page.evaluate("amount => window.scrollBy(0, amount)", delta)
            result = {
                "status": "ok",
                "scrolled": delta,
                "screenshot": _screenshot(normalized_thread_id, page, action=selected_action),
            }
        elif selected_action == "scroll_to_text":
            target_text = str(text or "").strip()
            if not target_text:
                raise BrowserUseError("scroll_to_text 需要 text 参数。")
            page.get_by_text(target_text, exact=False).first.scroll_into_view_if_needed()
            result = {
                "status": "ok",
                "text": target_text,
                "screenshot": _screenshot(normalized_thread_id, page, action="scroll_to_text"),
            }
        elif selected_action == "send_keys":
            if not str(keys or "").strip():
                raise BrowserUseError("send_keys 需要 keys 参数。")
            page.keyboard.press(str(keys))
            result = {"status": "ok", "keys": str(keys)}
        elif selected_action == "switch_tab":
            if type(tab_id) is not int or tab_id < 0:
                raise BrowserUseError("switch_tab 需要非负整数 tab_id。")
            pages = session.pages()
            if tab_id >= len(pages):
                raise BrowserUseError(f"标签页索引超出范围：{tab_id}")
            session.page = pages[tab_id]
            session.page.bring_to_front()
            result = {
                "status": "ok",
                "tab": tab_id,
                **_page_info(session.page),
                "screenshot": _screenshot(normalized_thread_id, session.page, action="switch_tab"),
            }
        elif selected_action == "open_tab":
            target = str(url or "").strip()
            if not target:
                raise BrowserUseError("open_tab 需要 url 参数。")
            session.page = session.context.new_page()
            session.page.goto(target, wait_until="domcontentloaded", timeout=30000)
            session.page.wait_for_load_state()
            result = {
                "status": "ok",
                "opened": target,
                "tab": max(len(session.pages()) - 1, 0),
                **_page_info(session.page),
                "screenshot": _screenshot(normalized_thread_id, session.page, action="open_tab"),
            }
        elif selected_action == "close_tab":
            pages = session.pages()
            if len(pages) <= 1:
                raise BrowserUseError("至少需要保留一个标签页。")
            session.page.close()
            session.page = pages[-2]
            result = {
                "status": "ok",
                "closed": True,
                **_page_info(session.page),
                "screenshot": _screenshot(normalized_thread_id, session.page, action="close_tab"),
            }
        elif selected_action == "wait":
            delay = int(seconds or 3)
            if delay < 0 or delay > 30:
                raise BrowserUseError("seconds 必须是 0 到 30 之间的整数。")
            page.wait_for_timeout(delay * 1000)
            result = {"status": "ok", "waited": delay}
        elif selected_action == "screenshot":
            result = {
                "status": "ok",
                "screenshot": _screenshot(
                    normalized_thread_id,
                    page,
                    action="screenshot",
                    full_page=True,
                ),
            }
        elif selected_action == "web_search":
            normalized_query = str(query or "").strip()
            if not normalized_query:
                raise BrowserUseError("web_search 需要 query 参数。")
            target = f"https://www.google.com/search?q={quote_plus(normalized_query)}"
            page.goto(target, wait_until="domcontentloaded", timeout=30000)
            result = {
                "status": "ok",
                **_page_info(page),
                "screenshot": _screenshot(normalized_thread_id, page, action="search"),
            }
        elif selected_action == "extract_content":
            content = str(page.content() or "")
            result = {
                "status": "ok",
                "goal": str(goal or ""),
                "content": content[:20_000],
                **_page_info(page),
            }
        else:
            raise BrowserUseError(f"不支持的浏览器操作：{selected_action}")

        _emit(normalized_thread_id, "tool_result", {"action": selected_action})
        return result
    except Exception as exc:
        message = str(exc)
        _emit(normalized_thread_id, "tool_error", {"action": selected_action, "error": message})
        return {"error": message}


def browser_use(
    thread_id: str,
    action: str,
    url: str | None = None,
    index: int | None = None,
    text: str | None = None,
    scroll_amount: int | None = None,
    tab_id: int | None = None,
    query: str | None = None,
    goal: str | None = None,
    keys: str | None = None,
    seconds: int | None = None,
) -> dict[str, Any]:
    normalized_thread_id = str(thread_id or "").strip() or "default"
    session = browser_use_sessions.get(normalized_thread_id)

    try:
        return session.run(
            lambda: _browser_use_on_worker(
                normalized_thread_id,
                action,
                url,
                index,
                text,
                scroll_amount,
                tab_id,
                query,
                goal,
                keys,
                seconds,
            )
        )
    except Exception as exc:
        message = str(exc)
        _emit(normalized_thread_id, "tool_error", {"action": action, "error": message})
        return {"error": message}


def build_browser_use_tool_definitions(thread_id: str) -> list[ToolDefinition]:
    normalized_thread_id = str(thread_id or "").strip() or "default"

    def browser_use_tool(
        action: str,
        url: str | None = None,
        index: int | None = None,
        text: str | None = None,
        scroll_amount: int | None = None,
        tab_id: int | None = None,
        query: str | None = None,
        goal: str | None = None,
        keys: str | None = None,
        seconds: int | None = None,
    ) -> dict[str, Any]:
        return browser_use(
            normalized_thread_id,
            action,
            url,
            index,
            text,
            scroll_amount,
            tab_id,
            query,
            goal,
            keys,
            seconds,
        )

    return [
        ToolDefinition(
            "browser_use",
            "使用本地 Playwright 浏览器执行导航、点击、输入、滚动、标签页和截图操作。"
            "动作支持 go_to_url/navigate、go_back/back、click_element/click、"
            "input_text/input、scroll_down/scroll_up、open_tab、switch_tab、"
            "close_tab、screenshot、web_search 和 extract_content。",
            browser_use_tool,
        )
    ]


__all__ = [
    "BrowserUseError",
    "BrowserUseSessionManager",
    "browser_use",
    "browser_use_sessions",
    "build_browser_use_tool_definitions",
]
