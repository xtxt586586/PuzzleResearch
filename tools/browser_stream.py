from __future__ import annotations

import base64
import asyncio
from typing import Any


class BrowserStreamError(RuntimeError):
    pass


class BrowserLiveView:
    def __init__(self) -> None:
        self._playwright: Any = None
        self._browser: Any = None
        self._page: Any = None
        self._lock = asyncio.Lock()

    async def start(self, url: str) -> dict[str, Any]:
        normalized_url = str(url or "").strip()

        if not normalized_url:
            raise BrowserStreamError(
                "浏览器页面 URL 不能为空。"
            )

        if self._page is not None:
            await self._page.goto(
                normalized_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )
            return await self.metadata()

        try:
            from playwright.async_api import (
                async_playwright,
            )
        except Exception as exc:
            raise BrowserStreamError(
                "浏览器实时预览需要 Playwright。"
            ) from exc

        try:
            self._playwright = (
                await async_playwright().start()
            )

            self._browser = (
                await self._playwright.chromium.launch(
                    headless=True
                )
            )

            self._page = (
                await self._browser.new_page(
                    viewport={
                        "width": 1280,
                        "height": 720,
                    }
                )
            )

            await self._page.goto(
                normalized_url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            return await self.metadata()

        except BrowserStreamError:
            await self.stop()
            raise

        except Exception as exc:
            await self.stop()
            raise BrowserStreamError(
                f"浏览器页面启动失败：{exc}"
            ) from exc

    async def metadata(self) -> dict[str, Any]:
        if self._page is None:
            return {
                "url": None,
                "title": "",
            }

        try:
            title = await self._page.title()
        except Exception:
            title = ""

        return {
            "url": str(
                getattr(
                    self._page,
                    "url",
                    "",
                )
                or ""
            ),
            "title": str(title or ""),
        }

    async def capture(
        self,
        *,
        full_page: bool = False,
        quality: int = 70,
    ) -> dict[str, Any]:
        async with self._lock:
            if self._page is None:
                raise BrowserStreamError(
                    "浏览器实时预览尚未启动。"
                )

            normalized_quality = max(
                1,
                min(int(quality or 70), 100),
            )

            try:
                image = await self._page.screenshot(
                    type="jpeg",
                    quality=normalized_quality,
                    full_page=bool(full_page),
                )

                return {
                    "data": base64.b64encode(
                        image
                    ).decode("ascii"),
                    "metadata": await self.metadata(),
                }

            except Exception as exc:
                raise BrowserStreamError(
                    f"浏览器截图失败：{exc}"
                ) from exc

    async def stop(self) -> None:
        page = self._page
        browser = self._browser
        playwright = self._playwright

        self._page = None
        self._browser = None
        self._playwright = None

        if page is not None:
            try:
                await page.close()
            except Exception:
                pass

        if browser is not None:
            try:
                await browser.close()
            except Exception:
                pass

        if playwright is not None:
            try:
                await playwright.stop()
            except Exception:
                pass


async def capture_url_screenshot(
    url: str,
    *,
    full_page: bool = True,
    quality: int = 85,
) -> dict[str, Any]:
    view = BrowserLiveView()

    try:
        await view.start(url)

        return await view.capture(
            full_page=full_page,
            quality=quality,
        )

    finally:
        await view.stop()