from __future__ import annotations

from typing import Any

from agent.workflows.source_registry import SourceRegistry


def _text(value: Any) -> str:
    return str(value or "").strip() if isinstance(value, str) else ""


def extract_message_sources(
    scraped_content: list[dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    registry = SourceRegistry()
    sources: list[dict[str, Any]] = []
    positions: dict[str, int] = {}

    def register(url: str, title: str, provider: str, published: str) -> None:
        record = registry.register(url, title)
        if record is None:
            return
        if record.canonical_url not in positions:
            positions[record.canonical_url] = len(sources)
            item: dict[str, Any] = {
                "title": record.title or title,
                "url": record.canonical_url,
                "rawUrl": url,
                "domain": record.domain,
            }
            if provider:
                item["provider"] = provider
            if published:
                item["publishedDate"] = published
            sources.append(item)
            return

        current = sources[positions[record.canonical_url]]
        if not current.get("title") and title:
            current["title"] = title
        if provider and not current.get("provider"):
            current["provider"] = provider
        if published and not current.get("publishedDate"):
            current["publishedDate"] = published

    for run in scraped_content or []:
        if not isinstance(run, dict):
            continue
        run_url = _text(run.get("url"))
        run_title = _text(run.get("title"))
        run_provider = _text(run.get("provider"))
        run_published = _text(run.get("published_date") or run.get("publishedDate"))
        items = [run] if run_url else run.get("results", [])
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, dict):
                continue
            url = _text(item.get("url"))
            if not url:
                continue
            register(
                url,
                _text(item.get("title")) or run_title or "Untitled",
                _text(item.get("provider")) or run_provider,
                _text(item.get("published_date") or item.get("publishedDate")) or run_published,
            )
    return sources
