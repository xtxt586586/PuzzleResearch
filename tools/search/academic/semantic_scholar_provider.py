from __future__ import annotations

from typing import Any

import requests

from common.config import settings
from tools.search.base import SearchItem


class SemanticScholarProvider:
    """Semantic Scholar Graph API 适配器。"""

    name = "semantic_scholar"
    endpoint = "https://api.semanticscholar.org/graph/v1"
    search_fields = (
        "paperId,title,abstract,year,citationCount,influentialCitationCount,"
        "authors,venue,publicationDate,openAccessPdf,fieldsOfStudy,tldr"
    )

    def is_available(self) -> bool:
        # API Key 只提高限额，未配置时仍可使用公共接口。
        return True

    def search(
        self,
        query: str,
        max_results: int = 5,
        year_range: tuple[int, int] | None = None,
        fields_of_study: list[str] | None = None,
        open_access_only: bool = False,
    ) -> list[SearchItem]:
        query = query.strip()
        if not query:
            raise ValueError("搜索关键词不能为空")

        params: dict[str, Any] = {
            "query": query,
            "limit": max(1, min(int(max_results), 100)),
            "fields": self.search_fields,
        }
        if year_range:
            params["year"] = f"{year_range[0]}-{year_range[1]}"
        if fields_of_study:
            params["fieldsOfStudy"] = ",".join(fields_of_study)
        if open_access_only:
            params["openAccessPdf"] = ""

        payload = self._get_json(f"{self.endpoint}/paper/search", params)
        raw_results = payload.get("data", []) if isinstance(payload, dict) else []
        return [self._normalize_result(item) for item in raw_results]

    def get_paper(self, paper_id: str) -> SearchItem | None:
        paper_id = paper_id.strip()
        if not paper_id:
            raise ValueError("论文 ID 不能为空")
        payload = self._get_json(
            f"{self.endpoint}/paper/{paper_id}",
            {"fields": self.search_fields},
        )
        return self._normalize_result(payload) if payload else None

    def search_author(self, author_name: str, max_results: int = 10) -> list[dict[str, Any]]:
        if not author_name.strip():
            raise ValueError("作者名称不能为空")
        payload = self._get_json(
            f"{self.endpoint}/author/search",
            {
                "query": author_name.strip(),
                "limit": max(1, min(int(max_results), 100)),
                "fields": "authorId,name,affiliations,paperCount,citationCount,hIndex",
            },
        )
        return payload.get("data", []) if isinstance(payload, dict) else []

    def _get_json(self, url: str, params: dict[str, Any]) -> Any:
        headers = {"User-Agent": "PuzzleResearch/1.0 (Research Tool)"}
        api_key = str(getattr(settings, "semantic_scholar_api_key", "") or "").strip()
        if api_key:
            headers["x-api-key"] = api_key
        response = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=float(getattr(settings, "search_request_timeout_seconds", 15.0)),
        )
        if not response.ok:
            raise RuntimeError(
                f"Semantic Scholar 请求失败（HTTP {response.status_code}）"
            )
        return response.json()

    def _normalize_result(self, paper: Any) -> SearchItem:
        data = paper if isinstance(paper, dict) else {}
        paper_id = str(data.get("paperId") or "")
        abstract = str(data.get("abstract") or "")
        tldr = data.get("tldr")
        if not abstract and isinstance(tldr, dict):
            abstract = str(tldr.get("text") or "")
        authors = [
            str(author.get("name") or "")
            for author in data.get("authors", [])
            if isinstance(author, dict) and author.get("name")
        ]
        metadata = [
            f"作者：{', '.join(authors)}" if authors else "",
            f"年份：{data.get('year')}" if data.get("year") else "",
            f"引用数：{data.get('citationCount')}" if data.get("citationCount") is not None else "",
            f"开放获取：{(data.get('openAccessPdf') or {}).get('url')}"
            if isinstance(data.get("openAccessPdf"), dict)
            and data.get("openAccessPdf", {}).get("url")
            else "",
        ]
        content = "\n".join([abstract, *[item for item in metadata if item]]).strip()
        return {
            "title": str(data.get("title") or ""),
            "url": f"https://www.semanticscholar.org/paper/{paper_id}",
            "snippet": abstract[:500],
            "content": content,
            "provider": self.name,
        }
