from __future__ import annotations

from typing import Any

from src.documents.models import SourceDocument
from src.extractions.models import LLMExtraction
from src.extractions.validation import optional_int


RISK_KEYWORDS = {
    "offering": "Text mentions an offering, which may imply financing or dilution risk.",
    "dilution": "Text mentions dilution.",
    "bankruptcy": "Text mentions bankruptcy.",
    "going concern": "Text mentions going concern risk.",
    "default": "Text mentions default risk.",
    "investigation": "Text mentions an investigation.",
    "subpoena": "Text mentions a subpoena.",
    "restatement": "Text mentions a restatement.",
}

POSITIVE_KEYWORDS = {
    "raises guidance": "Text mentions raised guidance.",
    "raise guidance": "Text mentions raised guidance.",
    "record revenue": "Text mentions record revenue.",
    "beat expectations": "Text mentions beating expectations.",
    "beats expectations": "Text mentions beating expectations.",
    "revenue growth": "Text mentions revenue growth.",
    "profitability": "Text mentions profitability.",
    "contract win": "Text mentions a contract win.",
}


def _doc_value(document: SourceDocument | dict[str, Any], field: str, default: Any = None) -> Any:
    if isinstance(document, SourceDocument):
        return getattr(document, field, default)
    return document.get(field, default)


def _find_matches(text: str, keyword_map: dict[str, str]) -> list[tuple[str, str]]:
    lowered = text.lower()
    return [(keyword, reason) for keyword, reason in keyword_map.items() if keyword in lowered]


def _snippet_for_keyword(text: str, keyword: str, radius: int = 90) -> str:
    lowered = text.lower()
    index = lowered.find(keyword.lower())
    if index < 0:
        return ""
    start = max(0, index - radius)
    end = min(len(text), index + len(keyword) + radius)
    snippet = text[start:end].replace("\n", " ").strip()
    return snippet


def _event_type_for_document(document_type: str, risk_matches: list[tuple[str, str]]) -> str:
    risk_keywords = {keyword for keyword, _ in risk_matches}
    if {"offering", "dilution"} & risk_keywords:
        return "dilution"
    if {"investigation", "subpoena"} & risk_keywords:
        return "legal_regulatory"
    if document_type == "sec_filing":
        return "sec_filing"
    if document_type == "news_article":
        return "news"
    if document_type == "earnings_transcript":
        return "earnings"
    return "unknown"


def _extraction_type_for_document(document_type: str) -> str:
    if document_type == "sec_filing":
        return "sec_filing_review"
    if document_type == "news_article":
        return "news_review"
    if document_type == "earnings_transcript":
        return "earnings_tone"
    return "general_document_review"


def run_fallback_extraction(document: SourceDocument | dict[str, Any]) -> LLMExtraction:
    """Return a deterministic extraction-shaped object for pipeline testing only."""
    cleaned_text = str(_doc_value(document, "cleaned_text", "") or "")
    raw_text = str(_doc_value(document, "raw_text", "") or "")
    text = cleaned_text or raw_text
    document_type = str(_doc_value(document, "document_type", "other") or "other")
    ticker = str(_doc_value(document, "ticker", "UNKNOWN") or "UNKNOWN")
    document_id = int(_doc_value(document, "document_id", 0) or 0)
    catalyst_id = _doc_value(document, "catalyst_id", None)

    risk_matches = _find_matches(text, RISK_KEYWORDS)
    positive_matches = _find_matches(text, POSITIVE_KEYWORDS)

    risk_severity = min(10, 3 + len(risk_matches) * 2) if risk_matches else 0
    catalyst_strength = min(10, 2 + len(positive_matches) * 2) if positive_matches else 0

    if risk_matches and positive_matches:
        sentiment = "mixed"
    elif risk_matches:
        sentiment = "negative"
    elif positive_matches:
        sentiment = "positive"
    else:
        sentiment = "unknown"

    confidence = 0.35 if (risk_matches or positive_matches) else 0.15
    evidence_sufficiency = "limited" if (risk_matches or positive_matches) else "insufficient"
    document_relevance = "uncertain" if text.strip() else "unknown"
    proposed_effect = 0
    if positive_matches:
        proposed_effect += min(6, len(positive_matches) * 2)
    if risk_matches:
        proposed_effect -= min(10, len(risk_matches) * 3)
    proposed_effect = max(-15, min(10, proposed_effect))

    positive_points = [reason for _, reason in positive_matches]
    risks = [reason for _, reason in risk_matches]
    evidence = [
        snippet
        for keyword, _ in [*positive_matches, *risk_matches]
        if (snippet := _snippet_for_keyword(text, keyword))
    ][:6]

    if not text.strip():
        risks.append("No source text was available for fallback extraction.")

    warning = (
        "Fallback keyword extraction only; not real LLM analysis. "
        "For pipeline testing and manual review preparation only. No buy/sell recommendation is generated."
    )

    if sentiment == "unknown":
        short_summary = "Fallback extractor found no strong positive or risk keywords."
    elif sentiment == "positive":
        short_summary = "Fallback extractor found simple positive catalyst keywords."
    elif sentiment == "negative":
        short_summary = "Fallback extractor found simple risk keywords."
    else:
        short_summary = "Fallback extractor found both positive catalyst and risk keywords."

    return LLMExtraction(
        document_id=document_id,
        catalyst_id=optional_int(catalyst_id),
        ticker=ticker,
        provider="fallback",
        model_name=None,
        extraction_type=_extraction_type_for_document(document_type),
        event_type_detected=_event_type_for_document(document_type, risk_matches),
        sentiment_label=sentiment,
        catalyst_strength=catalyst_strength,
        risk_severity=risk_severity,
        confidence=confidence,
        document_relevance=document_relevance,
        evidence_sufficiency=evidence_sufficiency,
        time_horizon="unknown",
        key_positive_points=positive_points,
        key_risks=risks,
        evidence_snippets=evidence,
        short_summary=short_summary,
        detailed_summary=(
            "This deterministic fallback used keyword rules to produce an extraction-shaped record. "
            "It is designed to test storage and review workflows without calling a real LLM provider."
        ),
        proposed_score_effect=proposed_effect,
        review_status="pending_review",
        prompt_version="fallback_v1",
        extraction_warnings=warning,
    )
