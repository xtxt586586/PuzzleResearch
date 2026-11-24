from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from pydantic import BaseModel, Field

from agent.prompts import get_prompt_manager


class ResearchDomain(str, Enum):
    SCIENTIFIC = "scientific"
    LEGAL = "legal"
    FINANCIAL = "financial"
    TECHNICAL = "technical"
    MEDICAL = "medical"
    BUSINESS = "business"
    HISTORICAL = "historical"
    GENERAL = "general"


@dataclass(frozen=True)
class DomainClassification:
    domain: ResearchDomain
    confidence: float
    reasoning: str
    search_hints: list[str] = field(default_factory=list)
    suggested_sources: list[str] = field(default_factory=list)
    language_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "search_hints": list(self.search_hints),
            "suggested_sources": list(self.suggested_sources),
            "language_hints": list(self.language_hints),
        }


class DomainClassificationResponse(BaseModel):
    domain: str = Field(default="general")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    reasoning: str = Field(default="")
    search_hints: list[str] = Field(default_factory=list)


DOMAIN_CONFIGS: dict[ResearchDomain, dict[str, list[str]]] = {
    ResearchDomain.SCIENTIFIC: {
        "suggested_sources": ["arxiv.org", "pubmed.ncbi.nlm.nih.gov", "nature.com"],
        "search_prefixes": ["research paper", "scientific study"],
        "language_hints": ["methodology", "hypothesis", "findings"],
    },
    ResearchDomain.LEGAL: {
        "suggested_sources": ["law.cornell.edu", "courtlistener.com", "gov"],
        "search_prefixes": ["legal analysis", "regulation"],
        "language_hints": ["statute", "precedent", "jurisdiction"],
    },
    ResearchDomain.FINANCIAL: {
        "suggested_sources": ["reuters.com", "sec.gov", "finance.yahoo.com"],
        "search_prefixes": ["financial analysis", "market analysis"],
        "language_hints": ["valuation", "market share", "fiscal"],
    },
    ResearchDomain.TECHNICAL: {
        "suggested_sources": ["github.com", "stackoverflow.com", "developer.mozilla.org"],
        "search_prefixes": ["technical documentation", "implementation"],
        "language_hints": ["algorithm", "architecture", "API"],
    },
    ResearchDomain.MEDICAL: {
        "suggested_sources": ["pubmed.ncbi.nlm.nih.gov", "who.int", "cdc.gov"],
        "search_prefixes": ["clinical study", "medical research"],
        "language_hints": ["clinical trial", "diagnosis", "treatment"],
    },
    ResearchDomain.BUSINESS: {
        "suggested_sources": ["crunchbase.com", "forbes.com", "businessinsider.com"],
        "search_prefixes": ["business analysis", "industry analysis"],
        "language_hints": ["revenue", "market share", "competitive analysis"],
    },
    ResearchDomain.HISTORICAL: {
        "suggested_sources": ["jstor.org", "archive.org", "britannica.com"],
        "search_prefixes": ["historical analysis", "history of"],
        "language_hints": ["era", "period", "historical context"],
    },
    ResearchDomain.GENERAL: {
        "suggested_sources": ["wikipedia.org", "britannica.com"],
        "search_prefixes": [],
        "language_hints": [],
    },
}

_KEYWORDS: tuple[tuple[ResearchDomain, tuple[str, ...]], ...] = (
    (ResearchDomain.MEDICAL, ("medical", "medicine", "health", "clinical", "disease", "treatment", "医疗", "医学", "健康", "临床", "疾病", "治疗")),
    (ResearchDomain.LEGAL, ("legal", "law", "regulation", "contract", "court", "法律", "法规", "合同", "法院")),
    (ResearchDomain.FINANCIAL, ("financial", "finance", "stock", "investment", "market", "valuation", "金融", "股票", "投资", "估值")),
    (ResearchDomain.TECHNICAL, ("software", "programming", "python", "api", "database", "algorithm", "engineering", "技术", "代码", "编程", "数据库", "算法", "工程")),
    (ResearchDomain.SCIENTIFIC, ("scientific", "research paper", "academic", "hypothesis", "methodology", "arxiv", "study", "科学", "研究", "论文", "学术", "假设", "实验")),
    (ResearchDomain.BUSINESS, ("business", "company", "industry", "revenue", "startup", "商业", "公司", "行业", "收入", "创业")),
    (ResearchDomain.HISTORICAL, ("history", "historical", "archive", "ancient", "历史", "古代", "档案")),
)

_SOURCE_PROVIDER_HINTS = {
    "arxiv": "arxiv",
    "pubmed": "pubmed",
    "nature.com": "semantic_scholar",
    "sciencedirect.com": "semantic_scholar",
    "jstor.org": "semantic_scholar",
    "reuters.com": "serper",
    "bloomberg.com": "serper",
    "wsj.com": "serper",
    "sec.gov": "tavily",
    "github.com": "duckduckgo",
    "stackoverflow.com": "duckduckgo",
    "developer.mozilla.org": "duckduckgo",
    "who.int": "tavily",
    "cdc.gov": "tavily",
    "wikipedia.org": "tavily",
}

_DOMAIN_PROVIDER_DEFAULTS = {
    ResearchDomain.SCIENTIFIC: ["arxiv", "pubmed", "semantic_scholar", "tavily"],
    ResearchDomain.MEDICAL: ["pubmed", "semantic_scholar", "tavily", "serper"],
    ResearchDomain.TECHNICAL: ["duckduckgo", "tavily", "serper"],
    ResearchDomain.FINANCIAL: ["serper", "tavily", "brave"],
    ResearchDomain.LEGAL: ["tavily", "serper", "duckduckgo"],
    ResearchDomain.BUSINESS: ["serper", "tavily"],
    ResearchDomain.HISTORICAL: ["tavily", "duckduckgo", "serper"],
    ResearchDomain.GENERAL: ["tavily", "duckduckgo", "serper"],
}


class DomainClassifier:
    """可选 LLM 分类器，始终保留本地关键词降级路径。"""

    def __init__(self, llm: Any | None = None, config: Any | None = None) -> None:
        self.llm = llm
        self.config = config

    def classify(self, query: str, images: list[dict[str, Any]] | None = None) -> DomainClassification:
        if self.llm is not None:
            try:
                response = self._classify_with_llm(query)
                domain = ResearchDomain(str(response.domain).strip().lower())
                config = DOMAIN_CONFIGS[domain]
                return DomainClassification(
                    domain=domain,
                    confidence=float(response.confidence),
                    reasoning=str(response.reasoning or "").strip(),
                    search_hints=[str(item) for item in response.search_hints or []],
                    suggested_sources=list(config["suggested_sources"]),
                    language_hints=list(config["language_hints"]),
                )
            except Exception:
                # 远程模型不可用时继续使用本地规则，保证搜索仍可执行。
                pass

        return self._classify_locally(query)

    def _classify_with_llm(self, query: str) -> DomainClassificationResponse:
        structured_llm = self.llm.with_structured_output(DomainClassificationResponse)
        prompt = (
            f"{get_prompt_manager(self.config).get_prompt('domain_classifier')}\n"
            f"研究问题：{query}"
        )
        if self.config is None:
            response = structured_llm.invoke(prompt)
        else:
            response = structured_llm.invoke(prompt, config=self.config)
        if isinstance(response, DomainClassificationResponse):
            return response
        if isinstance(response, dict):
            return DomainClassificationResponse.model_validate(response)
        return DomainClassificationResponse.model_validate(response)

    def _classify_locally(self, query: str) -> DomainClassification:
        normalized = " ".join(str(query or "").lower().split())
        scores = {
            domain: sum(1 for keyword in keywords if keyword in normalized)
            for domain, keywords in _KEYWORDS
        }
        domain, score = max(scores.items(), key=lambda item: item[1])
        if score == 0:
            domain = ResearchDomain.GENERAL
            confidence = 0.5
            reasoning = "未识别到明确领域关键词，使用通用研究来源。"
        else:
            confidence = min(0.95, 0.55 + score * 0.1)
            reasoning = f"根据查询中的领域关键词，将问题归类为 {domain.value}。"
        config = DOMAIN_CONFIGS[domain]
        return DomainClassification(
            domain=domain,
            confidence=confidence,
            reasoning=reasoning,
            search_hints=list(config["search_prefixes"]),
            suggested_sources=list(config["suggested_sources"]),
            language_hints=list(config["language_hints"]),
        )

    def enhance_query(self, query: str, classification: DomainClassification) -> str:
        normalized = str(query or "").lower()
        for prefix in DOMAIN_CONFIGS[classification.domain]["search_prefixes"]:
            if prefix.lower() not in normalized:
                return f"{prefix} {query}".strip()
        return query


def build_provider_profile(
    suggested_sources: Iterable[str] | None,
    domain: ResearchDomain = ResearchDomain.GENERAL,
) -> list[str]:
    profile: list[str] = []

    def add_provider(value: str | None) -> None:
        normalized = str(value or "").strip().lower()
        if normalized and normalized not in profile:
            profile.append(normalized)

    for source in suggested_sources or []:
        source_text = str(source or "").strip().lower()
        for hint, provider in _SOURCE_PROVIDER_HINTS.items():
            if hint in source_text:
                add_provider(provider)

    for provider in _DOMAIN_PROVIDER_DEFAULTS.get(domain, _DOMAIN_PROVIDER_DEFAULTS[ResearchDomain.GENERAL]):
        add_provider(provider)
    return profile


def classify_domain(query: str) -> DomainClassification:
    return DomainClassifier().classify(query)
