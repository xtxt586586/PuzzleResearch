from __future__ import annotations

import re
from typing import Any


_HEADING_RE = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*$")


def _paragraph_spans(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"\n\s*\n+", text):
        end = match.start()
        if text[start:end].strip():
            spans.append((start, end))
        start = match.end()
    if start < len(text) and text[start:].strip():
        spans.append((start, len(text)))
    return spans


def _heading(paragraph: str) -> tuple[int, str]:
    match = _HEADING_RE.match(paragraph.strip())
    if not match:
        return 0, ""
    title = re.sub(r"\s*#+\s*$", "", match.group(2)).strip()
    return len(match.group(1)), title


def split_into_passages(
    text: str,
    *,
    max_chars: int = 1200,
) -> list[dict[str, Any]]:
    """Split text into bounded passages while retaining source offsets."""
    if not text or int(max_chars) <= 0:
        return []

    budget = int(max_chars)
    spans = _paragraph_spans(text) or [(0, len(text))]
    passages: list[dict[str, Any]] = []
    chunk_start: int | None = None
    chunk_end: int | None = None
    chunk_heading: str | None = None
    chunk_path: list[str] | None = None
    heading_stack: list[tuple[int, str]] = []

    def flush() -> None:
        nonlocal chunk_start, chunk_end, chunk_heading, chunk_path
        if chunk_start is None or chunk_end is None:
            return
        value = text[chunk_start:chunk_end]
        if value.strip():
            item: dict[str, Any] = {
                "text": value,
                "start_char": chunk_start,
                "end_char": chunk_end,
            }
            if chunk_heading:
                item["heading"] = chunk_heading
            if chunk_path:
                item["heading_path"] = list(chunk_path)
            passages.append(item)
        chunk_start = None
        chunk_end = None
        chunk_heading = None
        chunk_path = None

    for start, end in spans:
        paragraph = text[start:end]
        level, title = _heading(paragraph)
        if title:
            flush()
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, title))
            chunk_start = start
            chunk_end = end
            chunk_heading = title
            chunk_path = [item[1] for item in heading_stack]
            continue

        if end - start > budget:
            flush()
            for offset in range(start, end, budget):
                sub_end = min(offset + budget, end)
                if text[offset:sub_end].strip():
                    item = {
                        "text": text[offset:sub_end],
                        "start_char": offset,
                        "end_char": sub_end,
                    }
                    if heading_stack:
                        item["heading"] = heading_stack[-1][1]
                        item["heading_path"] = [item[1] for item in heading_stack]
                    passages.append(item)
            continue

        if chunk_start is None:
            chunk_start = start
            chunk_end = end
            chunk_heading = heading_stack[-1][1] if heading_stack else None
            chunk_path = [item[1] for item in heading_stack] or None
            continue

        if end - chunk_start <= budget:
            chunk_end = end
            continue

        flush()
        chunk_start = start
        chunk_end = end
        chunk_heading = heading_stack[-1][1] if heading_stack else None
        chunk_path = [item[1] for item in heading_stack] or None

    flush()
    return passages


def passages_from_fetched_pages(
    pages: list[dict[str, Any]] | None,
    *,
    max_chars: int = 1200,
) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int]] = set()
    for page in pages or []:
        if not isinstance(page, dict):
            continue
        url = str(page.get("url", "")).strip()
        text = str(page.get("markdown") or page.get("text") or "")
        if not url or not text.strip():
            continue
        for passage in split_into_passages(text, max_chars=max_chars):
            key = (url, int(passage["start_char"]), int(passage["end_char"]))
            if key in seen:
                continue
            seen.add(key)
            output.append(
                {
                    **passage,
                    "url": url,
                    "page_title": page.get("title"),
                    "retrieved_at": page.get("retrieved_at"),
                    "method": page.get("method"),
                }
            )
    return output
