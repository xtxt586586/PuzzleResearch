from __future__ import annotations

from typing import Any

from tools.browser_session import browser_sessions
from tools.registry import ToolDefinition
from tools.screenshot_store import get_screenshot_store


def _page_result(
    page: Any,
    *,
    max_links: int = 10,
    max_chars: int = 1600,
) -> dict[str, Any]:
    return {
        "url": page.url,
        "title": page.title,
        "links": page.links[
            : max(int(max_links), 0)
        ],
        "text_excerpt": page.text[
            : max(int(max_chars), 1)
        ],
    }


def build_browser_tool_definitions(
    thread_id: str,
) -> list[ToolDefinition]:
    normalized_thread_id = (
        str(thread_id or "").strip()
        or "default"
    )

    def session():
        return browser_sessions.get(
            normalized_thread_id
        )

    def browser_search(
        query: str,
        engine: str = "duckduckgo",
        max_links: int = 10,
    ) -> dict[str, Any]:
        page = session().search(
            query,
            engine=engine,
        )

        return _page_result(
            page,
            max_links=max_links,
            max_chars=1200,
        )

    def browser_navigate(
        url: str,
        max_links: int = 10,
    ) -> dict[str, Any]:
        page = session().navigate(url)

        return _page_result(
            page,
            max_links=max_links,
        )

    def browser_click(
        index: int,
        max_links: int = 10,
    ) -> dict[str, Any]:
        current = session().current

        if current is None:
            raise ValueError(
                "当前没有页面，请先调用 browser_navigate。"
            )

        position = int(index) - 1

        if position < 0 or position >= len(
            current.links
        ):
            raise ValueError(
                f"链接索引超出范围：{index}"
            )

        target_url = current.links[position]["url"]
        page = session().navigate(target_url)

        result = _page_result(
            page,
            max_links=max_links,
        )
        result["clicked"] = current.links[position]

        return result

    def browser_back() -> dict[str, Any]:
        return _page_result(
            session().back()
        )

    def browser_extract_text(
        max_chars: int = 3000,
    ) -> dict[str, Any]:
        current = session().current

        if current is None:
            raise ValueError(
                "当前没有页面，请先调用 browser_navigate。"
            )

        return {
            "url": current.url,
            "title": current.title,
            "text": current.text[
                : max(int(max_chars), 1)
            ],
        }

    def browser_list_links(
        max_links: int = 10,
    ) -> dict[str, Any]:
        current = session().current

        if current is None:
            raise ValueError(
                "当前没有页面，请先调用 browser_navigate。"
            )

        return {
            "url": current.url,
            "title": current.title,
            "links": current.links[
                : max(int(max_links), 0)
            ],
        }

    def browser_reset() -> dict[str, Any]:
        browser_sessions.reset(
            normalized_thread_id
        )

        return {
            "status": "reset",
            "thread_id": normalized_thread_id,
        }

    def browser_screenshot(
        url: str | None = None,
        full_page: bool = True,
        wait_ms: int = 1000,
    ) -> dict[str, Any]:
        target = str(url or "").strip()

        if not target:
            current = session().current

            if current is None:
                raise ValueError(
                    "没有目标 URL，也没有当前页面。"
                )

            target = current.url

        try:
            from playwright.sync_api import (
                sync_playwright,
            )
        except Exception as exc:
            raise RuntimeError(
                "浏览器截图需要 Playwright。"
            ) from exc

        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(
                headless=True
            )

            try:
                page = browser.new_page(
                    viewport={
                        "width": 1280,
                        "height": 720,
                    }
                )
                page.goto(
                    target,
                    wait_until="networkidle",
                    timeout=30000,
                )

                if wait_ms > 0:
                    page.wait_for_timeout(
                        int(wait_ms)
                    )

                image_data = page.screenshot(
                    full_page=bool(full_page),
                    type="png",
                )
            finally:
                browser.close()

        saved = get_screenshot_store().save_screenshot(
            image_data,
            action="browser_screenshot",
            thread_id=normalized_thread_id,
            page_url=target,
        )

        return {
            "success": True,
            "screenshot_url": saved["url"],
            "filename": saved["filename"],
            "page_url": target,
        }

    return [
        ToolDefinition(
            name="browser_search",
            description="搜索网页并提取搜索结果链接。",
            handler=browser_search,
        ),
        ToolDefinition(
            name="browser_navigate",
            description="打开网页并提取标题、正文和链接。",
            handler=browser_navigate,
        ),
        ToolDefinition(
            name="browser_click",
            description="点击当前页面中指定索引的链接。",
            handler=browser_click,
        ),
        ToolDefinition(
            name="browser_back",
            description="返回浏览器会话中的上一页。",
            handler=browser_back,
        ),
        ToolDefinition(
            name="browser_extract_text",
            description="提取当前页面正文。",
            handler=browser_extract_text,
        ),
        ToolDefinition(
            name="browser_list_links",
            description="列出当前页面中的链接。",
            handler=browser_list_links,
        ),
        ToolDefinition(
            name="browser_reset",
            description="清空当前浏览器会话。",
            handler=browser_reset,
        ),
        ToolDefinition(
            name="browser_screenshot",
            description="使用 Playwright 对网页截图。",
            handler=browser_screenshot,
        ),
    ]