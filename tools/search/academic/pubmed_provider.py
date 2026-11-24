from __future__ import annotations

from typing import Any
from xml.etree import ElementTree

import requests

from common.config import settings
from tools.search.base import SearchItem


class PubMedProvider:
    """NCBI E-utilities PubMed 适配器。"""

    name = "pubmed"
    esearch_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    efetch_endpoint = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

    def is_available(self) -> bool:
        return bool(str(getattr(settings, "pubmed_email", "") or "").strip())

    def search(
        self,
        query: str,
        max_results: int = 5,
        date_range: tuple[str, str] | None = None,
        sort: str = "relevance",
    ) -> list[SearchItem]:
        query = query.strip()
        if not query:
            raise ValueError("搜索关键词不能为空")
        if not self.is_available():
            raise RuntimeError("未配置 PUBMED_EMAIL")

        common = self._common_params()
        search_params: dict[str, Any] = {
            **common,
            "db": "pubmed",
            "term": query,
            "retmax": max(1, min(int(max_results), 200)),
            "retmode": "json",
            "sort": sort,
        }
        if date_range:
            search_params.update(
                {"mindate": date_range[0], "maxdate": date_range[1], "datetype": "pdat"}
            )

        search_response = requests.get(
            self.esearch_endpoint,
            params=search_params,
            timeout=float(getattr(settings, "search_request_timeout_seconds", 15.0)),
        )
        self._ensure_ok(search_response, "PubMed 检索")
        search_payload = search_response.json()
        ids = (
            search_payload.get("esearchresult", {}).get("idlist", [])
            if isinstance(search_payload, dict)
            else []
        )
        if not ids:
            return []

        fetch_response = requests.get(
            self.efetch_endpoint,
            params={
                **common,
                "db": "pubmed",
                "id": ",".join(str(item) for item in ids),
                "retmode": "xml",
                "rettype": "abstract",
            },
            timeout=float(getattr(settings, "search_request_timeout_seconds", 30.0)),
        )
        self._ensure_ok(fetch_response, "PubMed 文献详情")
        try:
            root = ElementTree.fromstring(fetch_response.text)
        except ElementTree.ParseError as exc:
            raise RuntimeError("PubMed 返回了无效 XML") from exc
        return self._parse_articles(root)

    def search_by_author(self, author: str, max_results: int = 20) -> list[SearchItem]:
        return self.search(f"{author}[Author]", max_results=max_results)

    def search_by_mesh(self, mesh_term: str, max_results: int = 20) -> list[SearchItem]:
        return self.search(f"{mesh_term}[MeSH]", max_results=max_results)

    def get_article(self, pmid: str) -> SearchItem | None:
        results = self.search(f"{pmid}[PMID]", max_results=1)
        return results[0] if results else None

    def _common_params(self) -> dict[str, str]:
        params = {"email": str(settings.pubmed_email).strip()}
        api_key = str(getattr(settings, "pubmed_api_key", "") or "").strip()
        if api_key:
            params["api_key"] = api_key
        return params

    @staticmethod
    def _ensure_ok(response: Any, operation: str) -> None:
        if not response.ok:
            raise RuntimeError(f"{operation}请求失败（HTTP {response.status_code}）")

    def _parse_articles(self, root: ElementTree.Element) -> list[SearchItem]:
        results: list[SearchItem] = []
        for article in root.findall(".//PubmedArticle"):
            citation = article.find("MedlineCitation")
            if citation is None:
                continue
            article_node = citation.find("Article")
            if article_node is None:
                continue
            pmid = self._text(citation, "PMID")
            title = self._text(article_node, "ArticleTitle")
            abstract_parts = []
            abstract = article_node.find("Abstract")
            if abstract is not None:
                for item in abstract.findall("AbstractText"):
                    value = "".join(item.itertext()).strip()
                    label = str(item.get("Label") or "").strip()
                    if value:
                        abstract_parts.append(f"{label}: {value}" if label else value)
            summary = " ".join(abstract_parts)
            authors = []
            author_list = article_node.find("AuthorList")
            if author_list is not None:
                for author in author_list.findall("Author"):
                    last = self._text(author, "LastName")
                    first = self._text(author, "ForeName")
                    if last:
                        authors.append(f"{last} {first}".strip())
            journal = self._text(article_node, "Journal/Title")
            metadata = [
                f"作者：{', '.join(authors)}" if authors else "",
                f"期刊：{journal}" if journal else "",
            ]
            content = "\n".join([summary, *[item for item in metadata if item]]).strip()
            results.append(
                {
                    "title": title,
                    "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "snippet": summary[:500] or title,
                    "content": content,
                    "provider": self.name,
                }
            )
        return results

    @staticmethod
    def _text(element: ElementTree.Element, path: str) -> str:
        child = element.find(path)
        return "".join(child.itertext()).strip() if child is not None else ""
