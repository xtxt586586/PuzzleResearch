"""Deterministic quality assessment for research reports.

The assessor keeps the quality contract independent from any particular model.
Claims are checked with the local ClaimVerifier, while citation and source
signals are calculated from the report and collected evidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

from agent.workflows.claim_verifier import ClaimStatus, ClaimVerifier


@dataclass
class ClaimVerification:
    claim: str
    supported: bool
    supporting_sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    notes: str = ""


@dataclass
class QualityReport:
    claim_support_score: float = 0.0
    source_diversity_score: float = 0.0
    contradiction_free_score: float = 1.0
    citation_accuracy_score: float = 0.0
    citation_coverage_score: float = 1.0
    verified_claims: list[ClaimVerification] = field(default_factory=list)
    contradictions: list[str] = field(default_factory=list)
    unique_domains: list[str] = field(default_factory=list)
    missing_citations: list[str] = field(default_factory=list)
    overall_score: float = 0.0
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        claim_counts = {
            "total": len(self.verified_claims),
            "verified": sum(item.notes == ClaimStatus.VERIFIED.value for item in self.verified_claims),
            "unsupported": sum(item.notes == ClaimStatus.UNSUPPORTED.value for item in self.verified_claims),
            "contradicted": sum(item.notes == ClaimStatus.CONTRADICTED.value for item in self.verified_claims),
        }
        return {
            "claim_support_score": self.claim_support_score,
            "source_diversity_score": self.source_diversity_score,
            "contradiction_free_score": self.contradiction_free_score,
            "citation_accuracy_score": self.citation_accuracy_score,
            "citation_coverage_score": self.citation_coverage_score,
            "citation_coverage": self.citation_coverage_score,
            "overall_score": self.overall_score,
            "verified_claims_count": len(self.verified_claims),
            "supported_claims": claim_counts["verified"],
            "claim_verifier_total": claim_counts["total"],
            "claim_verifier_verified": claim_counts["verified"],
            "claim_verifier_unsupported": claim_counts["unsupported"],
            "claim_verifier_contradicted": claim_counts["contradicted"],
            "contradictions": list(self.contradictions),
            "unique_domains": list(self.unique_domains),
            "missing_citations": list(self.missing_citations[:5]),
            "recommendations": list(self.recommendations),
        }


class QualityAssessor:
    """Assess a report using evidence already collected by the workflow."""

    def __init__(self, llm: Any = None, config: dict[str, Any] | None = None) -> None:
        self.llm = llm
        self.config = config or {}

    def assess(
        self,
        report: str,
        scraped_content: list[dict[str, Any]],
        sources: list[Any] | None = None,
    ) -> QualityReport:
        report = str(report or "")
        scraped_content = list(scraped_content or [])
        urls = self._extract_urls(scraped_content, sources)

        quality = QualityReport()
        quality.unique_domains = self._get_unique_domains(urls)
        quality.source_diversity_score = self._calculate_diversity_score(
            quality.unique_domains
        )

        quality.verified_claims = self.check_claims(report, scraped_content)
        if quality.verified_claims:
            supported = sum(item.supported for item in quality.verified_claims)
            contradicted = sum(
                item.notes == ClaimStatus.CONTRADICTED.value
                for item in quality.verified_claims
            )
            quality.claim_support_score = max(
                0.0,
                (supported - 0.5 * contradicted) / len(quality.verified_claims),
            )
            quality.contradictions.extend(
                f"主张与证据矛盾：{item.claim[:120]}"
                for item in quality.verified_claims
                if item.notes == ClaimStatus.CONTRADICTED.value
            )

        quality.contradictions.extend(self.check_contradictions(report))
        quality.contradictions = list(dict.fromkeys(quality.contradictions))[:5]
        quality.contradiction_free_score = max(
            0.0,
            1.0 - 0.2 * len(quality.contradictions),
        )

        missing_accuracy, quality.citation_accuracy_score = self.check_citation_accuracy(
            report, urls
        )
        missing_coverage, quality.citation_coverage_score = self.check_citation_coverage(
            report,
            source_count=len(urls),
        )
        quality.missing_citations = list(
            dict.fromkeys([*missing_accuracy, *missing_coverage])
        )[:5]

        citation_score = (
            quality.citation_accuracy_score + quality.citation_coverage_score
        ) / 2.0
        quality.overall_score = self._clamp(
            quality.claim_support_score * 0.35
            + quality.source_diversity_score * 0.20
            + quality.contradiction_free_score * 0.25
            + citation_score * 0.20
        )
        quality.recommendations = self._generate_recommendations(quality)
        return quality

    def check_claims(
        self,
        report: str,
        scraped_content: list[dict[str, Any]],
        max_claims: int = 10,
    ) -> list[ClaimVerification]:
        checks = ClaimVerifier().verify_report(
            report=report,
            scraped_content=scraped_content,
            max_claims=max_claims,
        )
        return [
            ClaimVerification(
                claim=check.claim,
                supported=check.status == ClaimStatus.VERIFIED,
                supporting_sources=list(check.evidence_urls),
                confidence=min(1.0, check.score / 6.0) if check.score else 0.0,
                notes=check.status.value,
            )
            for check in checks
        ]

    def check_contradictions(self, report: str) -> list[str]:
        claims = ClaimVerifier().extract_claims(report, max_claims=20)
        contradictions: list[str] = []
        for index, left in enumerate(claims):
            for right in claims[index + 1 :]:
                left_tokens = ClaimVerifier._tokenize(left)
                right_tokens = ClaimVerifier._tokenize(right)
                if len(left_tokens & right_tokens) < 2:
                    continue
                if self._opposite_direction(left, right) or self._opposite_polarity(left, right):
                    contradictions.append(f"报告内部陈述可能矛盾：{left[:80]} / {right[:80]}")
        return list(dict.fromkeys(contradictions))[:5]

    def check_citation_accuracy(
        self,
        report: str,
        known_urls: list[str],
    ) -> tuple[list[str], float]:
        citations = [int(value) for value in re.findall(r"\[(?:来源\s*)?(\d+)\]", report)]
        url_citations = re.findall(r"(?:参考|来源|引用)[：:]?\s*(https?://\S+)", report)
        valid_count = sum(1 <= item <= len(known_urls) for item in citations)
        invalid_count = sum(item < 1 or item > len(known_urls) for item in citations)
        known_url_set = {str(url).rstrip(".,)") for url in known_urls}
        invalid_urls = [url for url in url_citations if url.rstrip(".,)") not in known_url_set]

        if not citations and not url_citations:
            return self._citation_candidates(report)[:5], 0.8

        total = len(citations) + len(url_citations)
        valid = valid_count + sum(
            url.rstrip(".,)") in known_url_set for url in url_citations
        )
        accuracy = valid / total if total else 0.0
        missing = self._citation_candidates(report)
        missing.extend(f"无效引用：{url}" for url in invalid_urls)
        if invalid_count:
            missing.append(f"存在 {invalid_count} 个超出来源范围的编号引用")
        return list(dict.fromkeys(missing))[:5], self._clamp(accuracy)

    def check_citation_coverage(
        self,
        report: str,
        source_count: int | None = None,
    ) -> tuple[list[str], float]:
        candidates = self._citation_candidates(report)
        if not candidates:
            return [], 1.0
        citation_pattern = re.compile(
            r"\[(?:来源\s*)?(\d+)\]|\[来源[：:].*?\]|https?://\S+",
            re.I,
        )
        cited = 0
        uncited: list[str] = []
        for item in candidates:
            match = citation_pattern.search(item)
            if not match:
                uncited.append(item)
                continue
            citation_index = match.group(1)
            if citation_index and source_count is not None:
                if not 1 <= int(citation_index) <= source_count:
                    uncited.append(item)
                    continue
            cited += 1
        return uncited[:5], self._clamp(cited / len(candidates))

    def check_source_diversity(self, sources: list[str]) -> tuple[list[str], float]:
        domains = self._get_unique_domains(sources)
        return domains, self._calculate_diversity_score(domains)

    def _citation_candidates(self, report: str) -> list[str]:
        candidates: list[str] = []
        markers = (
            r"\d{4}", r"\d+(?:\.\d+)?%", r"(?:研究|数据显示|统计|报告|发现|增长|下降)",
            r"(?:research|study|report|data|according to|shows|found)",
        )
        for sentence in re.split(r"(?<=[。！？.!?])\s+|\n+", report):
            text = sentence.strip()
            if len(text) >= 15 and any(re.search(marker, text, re.I) for marker in markers):
                candidates.append(text)
        return list(dict.fromkeys(candidates))

    def _extract_urls(
        self,
        scraped_content: list[dict[str, Any]],
        sources: list[Any] | None,
    ) -> list[str]:
        urls: list[str] = []
        for source in sources or []:
            url = source.get("url") if isinstance(source, dict) else source
            if isinstance(url, str) and url.strip() and url.strip() not in urls:
                urls.append(url.strip())
        for run in scraped_content:
            if not isinstance(run, dict):
                continue
            items = run.get("results") or ([run] if run.get("url") else [])
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "").strip()
                if url and url not in urls:
                    urls.append(url)
        return urls

    @staticmethod
    def _get_unique_domains(urls: Iterable[str]) -> list[str]:
        domains: set[str] = set()
        for url in urls:
            try:
                domain = urlparse(str(url)).netloc.lower()
            except ValueError:
                continue
            if domain.startswith("www."):
                domain = domain[4:]
            if domain:
                domains.add(domain)
        return sorted(domains)

    @staticmethod
    def _calculate_diversity_score(domains: list[str]) -> float:
        if not domains:
            return 0.0
        count = len(domains)
        base = 1.0 if count >= 10 else 0.8 if count >= 5 else 0.6 if count >= 3 else 0.4
        authoritative = ("wikipedia.org", ".gov", ".edu", "arxiv.org", "nature.com", "sciencedirect.com")
        bonus = min(0.2, 0.05 * sum(any(item in domain for item in authoritative) for domain in domains))
        return min(1.0, base + bonus)

    @staticmethod
    def _opposite_direction(left: str, right: str) -> bool:
        up = ("increase", "increased", "growth", "grow", "rise", "rose", "上升", "增长")
        down = ("decrease", "decreased", "decline", "drop", "下降", "减少")
        left_up = any(item in left.lower() for item in up)
        left_down = any(item in left.lower() for item in down)
        right_up = any(item in right.lower() for item in up)
        right_down = any(item in right.lower() for item in down)
        return (left_up and right_down) or (left_down and right_up)

    @staticmethod
    def _opposite_polarity(left: str, right: str) -> bool:
        markers = ("not", "no", "never", "didn't", "doesn't", "不是", "没有", "未", "无")
        return any(item in left.lower() for item in markers) != any(item in right.lower() for item in markers)

    @staticmethod
    def _clamp(value: float) -> float:
        return max(0.0, min(1.0, float(value)))

    def _generate_recommendations(self, quality: QualityReport) -> list[str]:
        recommendations: list[str] = []
        if quality.claim_support_score < 0.6:
            recommendations.append("为关键事实补充可验证来源，减少无证据主张")
        if quality.source_diversity_score < 0.5:
            recommendations.append("扩展来源域名，增加不同类型的权威来源")
        if quality.contradiction_free_score < 0.8:
            recommendations.append("检查并修正报告中的相互矛盾陈述")
        if quality.citation_accuracy_score < 0.6:
            recommendations.append("修正无效引用，并确保引用编号对应来源")
        if quality.citation_coverage_score < 0.6:
            recommendations.append("为关键数据、事实和结论补充引用")
        return recommendations or ["报告质量良好，可继续补充细节和案例"]
