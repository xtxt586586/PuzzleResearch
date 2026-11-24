from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agent.workflows.source_registry import SourceRegistry


class ClaimStatus(str, Enum):
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"


@dataclass
class ClaimCheck:
    claim: str
    status: ClaimStatus
    evidence_urls: list[str] = field(default_factory=list)
    evidence_passages: list[dict[str, Any]] = field(default_factory=list)
    score: float = 0.0
    notes: str = ""


_CLAIM_MARKERS = (
    "research", "study", "report", "data", "according to", "shows",
    "found", "研究", "报告", "数据显示", "统计", "增长", "下降",
)
_NEGATION_MARKERS = (
    "not", "no", "never", "without", "didn't", "doesn't", "isn't",
    "wasn't", "不是", "并非", "没有", "未", "无",
)
_UP_MARKERS = ("increase", "increased", "grow", "growth", "up", "rise", "rose", "增长", "上升")
_DOWN_MARKERS = ("decrease", "decreased", "decline", "down", "fell", "drop", "下降", "减少")
_STOPWORDS = {
    "the", "and", "that", "this", "with", "from", "into", "were", "was",
    "are", "for", "has", "have", "had", "will", "about", "在", "是", "了",
    "和", "与", "对", "将", "及",
}


class ClaimVerifier:
    def __init__(self, min_overlap_tokens: int = 2, *, max_evidence_per_claim: int = 5) -> None:
        self.min_overlap_tokens = max(1, int(min_overlap_tokens))
        self.max_evidence_per_claim = max(1, int(max_evidence_per_claim))

    def extract_claims(self, report: str, max_claims: int = 10) -> list[str]:
        if not report:
            return []
        candidates: list[str] = []
        in_references = False
        for line in report.splitlines():
            stripped = line.strip()
            heading = re.match(r"^#{1,6}\s*(.+?)\s*#*\s*$", stripped)
            if heading:
                title = heading.group(1).strip().lower().rstrip("：:")
                in_references = title in {"参考资料", "参考文献", "参考来源", "引用来源", "来源", "references", "sources", "bibliography"}
                continue
            if in_references:
                continue
            if re.match(r"^\s*(?:[-*+]\s*)?\[来源\s*\d+\]\s*.*https?://", stripped, re.I):
                continue
            candidates.extend(re.split(r"(?<=[。！？.!?])\s+", stripped))
        claims: list[str] = []
        seen: set[str] = set()
        for candidate in candidates:
            text = candidate.strip()
            if len(text) < 20:
                continue
            lowered = text.lower()
            has_signal = any(marker in lowered for marker in _CLAIM_MARKERS)
            has_number = bool(re.search(r"\d{2,4}|\d+%|\d+\.\d+", text))
            if not has_signal and not has_number:
                continue
            if lowered in seen:
                continue
            seen.add(lowered)
            claims.append(text)
            if len(claims) >= max(1, int(max_claims)):
                break
        return claims

    def verify_report(
        self,
        report: str,
        scraped_content: list[dict[str, Any]],
        max_claims: int = 10,
        passages: list[dict[str, Any]] | None = None,
    ) -> list[ClaimCheck]:
        evidence = self._extract_evidence(scraped_content, passages=passages)
        return [
            self.verify_claim(claim, evidence)
            for claim in self.extract_claims(report, max_claims=max_claims)
        ]

    def verify_claim(self, claim: str, evidence: list[dict[str, Any]]) -> ClaimCheck:
        claim_tokens = self._tokenize(claim)
        if not claim_tokens:
            return ClaimCheck(claim=claim, status=ClaimStatus.UNSUPPORTED, notes="no claim tokens")

        supported: list[tuple[int, str, dict[str, Any]]] = []
        contradicted: list[tuple[int, str, dict[str, Any]]] = []
        best_overlap = 0
        for item in evidence:
            url = str(item.get("url") or "unknown").strip() or "unknown"
            text = str(item.get("text") or "").strip()
            overlap = len(claim_tokens & self._tokenize(text))
            if overlap < self.min_overlap_tokens:
                continue
            best_overlap = max(best_overlap, overlap)
            passage = {"url": url}
            for key in ("snippet_hash", "quote"):
                value = str(item.get(key) or "").strip()
                if value:
                    passage[key] = value
            heading_path = item.get("heading_path")
            if isinstance(heading_path, list) and all(isinstance(value, str) for value in heading_path):
                passage["heading_path"] = heading_path
            (contradicted if self._is_contradiction(claim, text) else supported).append(
                (overlap, url, passage)
            )

        limit = self.max_evidence_per_claim
        contradicted.sort(key=lambda row: -row[0])
        supported.sort(key=lambda row: -row[0])
        if contradicted:
            rows = contradicted + supported
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.CONTRADICTED,
                evidence_urls=list(dict.fromkeys(row[1] for row in rows))[:limit],
                evidence_passages=[row[2] for row in rows[:limit]],
                score=float(best_overlap),
                notes="conflicting evidence found",
            )
        if supported:
            return ClaimCheck(
                claim=claim,
                status=ClaimStatus.VERIFIED,
                evidence_urls=list(dict.fromkeys(row[1] for row in supported))[:limit],
                evidence_passages=[row[2] for row in supported[:limit]],
                score=float(best_overlap),
                notes="supported by evidence",
            )
        return ClaimCheck(
            claim=claim,
            status=ClaimStatus.UNSUPPORTED,
            notes="no matching evidence",
        )

    def _extract_evidence(
        self,
        scraped_content: list[dict[str, Any]],
        *,
        passages: list[dict[str, Any]] | None = None,
    ) -> list[dict[str, Any]]:
        registry = SourceRegistry()
        evidence: list[dict[str, Any]] = []
        if passages:
            for passage in passages:
                if not isinstance(passage, dict):
                    continue
                url = str(passage.get("url") or "").strip()
                text = str(passage.get("text") or "").strip()
                if not url or not text:
                    continue
                evidence.append({
                    "url": registry.canonicalize_url(url) or url,
                    "text": text,
                    **{
                        key: passage[key]
                        for key in ("snippet_hash", "quote", "heading_path")
                        if passage.get(key)
                    },
                })
            return evidence

        for run in scraped_content or []:
            if not isinstance(run, dict):
                continue
            items = run.get("results") or ([run] if run.get("url") else [])
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url") or "unknown").strip() or "unknown"
                text = str(
                    item.get("raw_excerpt")
                    or item.get("content")
                    or item.get("summary")
                    or item.get("snippet")
                    or ""
                ).strip()
                if text:
                    evidence.append({
                        "url": registry.canonicalize_url(url) or url,
                        "text": text,
                    })
        return evidence

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        tokens = re.findall(r"[a-z0-9\u4e00-\u9fff]+", (text or "").lower())
        return {token for token in tokens if len(token) > 1 and token not in _STOPWORDS}

    @staticmethod
    def _has_negation(text: str) -> bool:
        lowered = (text or "").lower()
        return any(marker in lowered for marker in _NEGATION_MARKERS)

    @staticmethod
    def _trend_direction(text: str) -> int:
        lowered = (text or "").lower()
        up = any(marker in lowered for marker in _UP_MARKERS)
        down = any(marker in lowered for marker in _DOWN_MARKERS)
        return 1 if up and not down else -1 if down and not up else 0

    def _is_contradiction(self, claim: str, evidence: str) -> bool:
        if self._has_negation(claim) != self._has_negation(evidence):
            return True
        claim_direction = self._trend_direction(claim)
        evidence_direction = self._trend_direction(evidence)
        return (
            claim_direction != 0
            and evidence_direction != 0
            and claim_direction != evidence_direction
        )
