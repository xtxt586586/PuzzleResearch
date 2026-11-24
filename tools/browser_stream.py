from __future__ import annotations

import base64
import asyncio
from typing import Any
from urllib.parse import urlsplit


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

    async def perform_action(
        self,
        action: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        if self._page is None:
            raise BrowserStreamError(
                "浏览器实时预览尚未启动。"
            )

        page = self._page
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
                width, height = await self._viewport_size(page)
                x_px = min(int(x * width), width - 1)
                y_px = min(int(y * height), height - 1)

                if mouse_type == "click":
                    try:
                        clicks = int(payload.get("clicks", 1) or 1)
                    except (TypeError, ValueError):
                        clicks = 1
                    await page.mouse.click(
                        x_px,
                        y_px,
                        button=button,
                        click_count=max(1, min(clicks, 10)),
                    )
                else:
                    await page.mouse.move(x_px, y_px)
            elif mouse_type == "down":
                await page.mouse.down(button=button)
            else:
                await page.mouse.up(button=button)

        elif action_name == "scroll":
            try:
                dx = int(payload.get("dx", 0) or 0)
                dy = int(payload.get("dy", 0) or 0)
            except (TypeError, ValueError) as exc:
                raise ValueError("Scroll requires integer dx/dy.") from exc
            await page.mouse.wheel(dx, dy)

        elif action_name == "keyboard":
            keyboard_type = str(payload.get("type") or "").strip().lower()
            if keyboard_type == "press":
                key = str(payload.get("key") or "").strip()
                if not key:
                    raise ValueError("Keyboard press requires a non-empty key.")
                await page.keyboard.press(key)
            elif keyboard_type == "type":
                text = str(payload.get("text") or "")
                if not text:
                    raise ValueError("Keyboard type requires a non-empty text.")
                await page.keyboard.type(text)
            else:
                raise ValueError(
                    f"Unsupported keyboard action type: {keyboard_type or '(missing)'}"
                )

        elif action_name == "navigate":
            url = str(payload.get("url") or "").strip()
            parsed = urlsplit(url)
            if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
                raise ValueError("Only http(s) URLs with a host are allowed.")
            await page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

        else:
            raise ValueError(f"Unsupported action: {action_name or 'unknown'}")

        return await self.metadata()

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
    async def _viewport_size(page: Any) -> tuple[int, int]:
        viewport = getattr(page, "viewport_size", None)
        width = viewport.get("width") if isinstance(viewport, dict) else None
        height = viewport.get("height") if isinstance(viewport, dict) else None

        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            try:
                measured = await page.evaluate(
                    "() => ({ w: window.innerWidth, h: window.innerHeight })"
                )
            except Exception:
                measured = None
            if isinstance(measured, dict):
                width = measured.get("w") or measured.get("width")
                height = measured.get("h") or measured.get("height")

        if not isinstance(width, (int, float)) or not isinstance(height, (int, float)):
            raise RuntimeError("Could not determine browser viewport size")
        if width <= 0 or height <= 0:
            raise RuntimeError("Invalid browser viewport size")
        return int(width), int(height)

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
