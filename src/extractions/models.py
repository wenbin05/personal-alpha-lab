from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


ExtractionProvider = Literal["fallback", "openai_compatible", "other"]

ExtractionType = Literal[
    "catalyst_analysis",
    "risk_analysis",
    "earnings_tone",
    "sec_filing_review",
    "news_review",
    "forum_sentiment",
    "general_document_review",
]

DetectedEventType = Literal[
    "earnings",
    "sec_filing",
    "news",
    "analyst",
    "corporate_action",
    "dilution",
    "insider_activity",
    "product_launch",
    "guidance_update",
    "legal_regulatory",
    "macro_sensitive",
    "other",
    "unknown",
]

ExtractionSentiment = Literal["positive", "neutral", "negative", "mixed", "unknown"]

TimeHorizon = Literal["intraday", "short_term", "medium_term", "long_term", "unknown"]

ReviewStatus = Literal["pending_review", "approved", "rejected", "superseded"]

DocumentRelevance = Literal["relevant", "uncertain", "irrelevant", "unknown"]

EvidenceSufficiency = Literal["sufficient", "limited", "insufficient", "unknown"]

EXTRACTION_PROVIDERS = ["fallback", "openai_compatible", "other"]

EXTRACTION_TYPES = [
    "catalyst_analysis",
    "risk_analysis",
    "earnings_tone",
    "sec_filing_review",
    "news_review",
    "forum_sentiment",
    "general_document_review",
]

DETECTED_EVENT_TYPES = [
    "earnings",
    "sec_filing",
    "news",
    "analyst",
    "corporate_action",
    "dilution",
    "insider_activity",
    "product_launch",
    "guidance_update",
    "legal_regulatory",
    "macro_sensitive",
    "other",
    "unknown",
]

EXTRACTION_SENTIMENTS = ["positive", "neutral", "negative", "mixed", "unknown"]

TIME_HORIZONS = ["intraday", "short_term", "medium_term", "long_term", "unknown"]

REVIEW_STATUSES = ["pending_review", "approved", "rejected", "superseded"]

DOCUMENT_RELEVANCE_LABELS = ["relevant", "uncertain", "irrelevant", "unknown"]

EVIDENCE_SUFFICIENCY_LABELS = ["sufficient", "limited", "insufficient", "unknown"]


class LLMExtraction(BaseModel):
    extraction_id: int | None = None
    document_id: int
    catalyst_id: int | None = None
    ticker: str
    provider: ExtractionProvider = "fallback"
    model_name: str | None = None
    extraction_type: ExtractionType = "general_document_review"
    event_type_detected: DetectedEventType = "unknown"
    sentiment_label: ExtractionSentiment = "unknown"
    catalyst_strength: int = Field(default=0, ge=0, le=10)
    risk_severity: int = Field(default=0, ge=0, le=10)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    document_relevance: DocumentRelevance = "unknown"
    evidence_sufficiency: EvidenceSufficiency = "unknown"
    time_horizon: TimeHorizon = "unknown"
    key_positive_points: list[str] = Field(default_factory=list)
    key_risks: list[str] = Field(default_factory=list)
    evidence_snippets: list[str] = Field(default_factory=list)
    short_summary: str = ""
    detailed_summary: str = ""
    proposed_score_effect: int = Field(default=0, ge=-15, le=10)
    review_status: ReviewStatus = "pending_review"
    reviewer_note: str = ""
    reviewed_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_llm_response_json: str | None = None
    prompt_version: str = "fallback_v1"
    extraction_warnings: str = ""

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        cleaned = (value or "").strip().upper()
        if not cleaned:
            raise ValueError("Ticker is required.")
        return cleaned

    @field_validator("model_name", "raw_llm_response_json")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        cleaned = (value or "").strip()
        return cleaned or None

    @field_validator("reviewer_note", "short_summary", "detailed_summary", "prompt_version", "extraction_warnings")
    @classmethod
    def clean_text(cls, value: str | None) -> str:
        return (value or "").strip()

    @field_validator("key_positive_points", "key_risks", "evidence_snippets", mode="before")
    @classmethod
    def coerce_list(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, list):
            return [str(item).strip() for item in value if str(item).strip()]
        if isinstance(value, (tuple, set)):
            return [str(item).strip() for item in value if str(item).strip()]
        text = str(value).strip()
        return [text] if text else []
