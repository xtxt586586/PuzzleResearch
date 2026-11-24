from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from xml.etree import ElementTree

import requests

from common.config import settings
from tools.search.base import SearchItem


class ArxivProvider:
    """arXiv Atom API 适配器。"""

    name = "arxiv"
    endpoint = "https://export.arxiv.org/api/query"
    _namespaces = {
        "atom": "http://www.w3.org/2005/Atom",
        "arxiv": "http://arxiv.org/schemas/atom",
    }

    def is_available(self) -> bool:
        return bool(getattr(settings, "arxiv_enabled", True))

    def search(
        self,
        query: str,
        max_results: int = 5,
        categories: list[str] | None = None,
        sort_by: str = "relevance",
        sort_order: str = "descending",
    ) -> list[SearchItem]:
        query = query.strip()
        if not query:
            raise ValueError("搜索关键词不能为空")
        if not self.is_available():
            raise RuntimeError("arXiv Provider 已禁用")

        search_query = query
        if categories:
            category_query = " OR ".join(f"cat:{item}" for item in categories)
            search_query = f"({query}) AND ({category_query})"

        response = requests.get(
            self.endpoint,
            params={
                "search_query": f"all:{search_query}",
                "start": 0,
                "max_results": max(1, min(int(max_results), 100)),
                "sortBy": sort_by,
                "sortOrder": sort_order,
            },
            headers={"User-Agent": "PuzzleResearch/1.0 (Research Tool)"},
            timeout=float(getattr(settings, "search_request_timeout_seconds", 30.0)),
        )
        if not response.ok:
            raise RuntimeError(f"arXiv 请求失败（HTTP {response.status_code}）")
        try:
            root = ElementTree.fromstring(response.text)
        except ElementTree.ParseError as exc:
            raise RuntimeError("arXiv 返回了无效 XML") from exc
        return self._parse_entries(root)

    def search_by_author(self, author: str, max_results: int = 20) -> list[SearchItem]:
        return self.search(f"au:{author}", max_results=max_results)

    def search_by_title(self, title: str, max_results: int = 10) -> list[SearchItem]:
        return self.search(f"ti:{title}", max_results=max_results)

    def get_paper_by_id(self, arxiv_id: str) -> SearchItem | None:
        cleaned_id = arxiv_id.replace("arXiv:", "").strip()
        if not cleaned_id:
            raise ValueError("arXiv ID 不能为空")
        results = self.search(cleaned_id, max_results=1)
        return results[0] if results else None

    def _parse_entries(self, root: ElementTree.Element) -> list[SearchItem]:
        results: list[SearchItem] = []
        for entry in root.findall("atom:entry", self._namespaces):
            title = self._text(entry, "atom:title").strip()
            summary = " ".join(self._text(entry, "atom:summary").split())
            entry_id = self._text(entry, "atom:id")
            arxiv_id = entry_id.split("/abs/")[-1] if entry_id else ""
            updated = self._text(entry, "atom:updated")
            pdf_url = ""
            for link in entry.findall("atom:link", self._namespaces):
                if link.get("title") == "pdf":
                    pdf_url = str(link.get("href") or "")
                    break

            authors = [
                self._text(author, "atom:name")
                for author in entry.findall("atom:author", self._namespaces)
                if self._text(author, "atom:name")
            ]
            categories = [
                str(category.get("term") or "")
                for category in entry.findall("atom:category", self._namespaces)
                if category.get("term")
            ]
            metadata = [
                f"作者：{', '.join(authors)}" if authors else "",
                f"分类：{', '.join(categories)}" if categories else "",
                f"PDF：{pdf_url}" if pdf_url else "",
                f"更新：{updated}" if updated else "",
            ]
            content = "\n".join([summary, *[item for item in metadata if item]]).strip()
            results.append(
                {
                    "title": title,
                    "url": f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else entry_id,
                    "snippet": summary[:500],
                    "content": content,
                    "provider": self.name,
                }
            )
        return results

    def _text(self, element: ElementTree.Element, path: str) -> str:
        child = element.find(path, self._namespaces)
        return str(child.text or "") if child is not None else ""
