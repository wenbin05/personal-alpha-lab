from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any, Protocol

from src.annotations.models import ANNOTATION_EVENT_TYPES, ANNOTATION_SENTIMENT_LABELS, normalize_tags, utc_now


CANDIDATE_STATUSES = ("staged", "accepted", "rejected", "duplicate", "imported")
NEWS_EVENT_PROVIDER_VERSION = "news_event_provider_v1"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def normalize_title(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"https?://\\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\\s+", " ", text).strip()


def normalize_source_url(value: str | None) -> str | None:
    text = (value or "").strip()
    if not text:
        return None
    return text.rstrip("/").lower()


def stable_text_hash(value: str | None) -> str | None:
    text = re.sub(r"\\s+", " ", (value or "").strip())
    if not text:
        return None
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchEventAnnotationCandidate:
    ticker: str
    event_date: date
    available_at: datetime
    event_type: str
    title: str
    summary: str = ""
    source: str = "csv_manual"
    source_url: str | None = None
    evidence_text: str = ""
    sentiment_label: str = "unknown"
    strength: int = 0
    confidence: float = 0.0
    tags: list[str] = field(default_factory=list)
    provider: str = "csv_manual"
    provider_metadata: dict[str, Any] = field(default_factory=dict)
    document_type: str | None = None
    published_at: datetime | None = None
    raw_text: str = ""
    cleaned_text: str = ""
    status: str = "staged"
    candidate_id: int | None = None
    duplicate_of_annotation_id: int | None = None
    duplicate_of_candidate_id: int | None = None
    duplicate_reason: str | None = None
    rejection_reason: str | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    reviewed_at: datetime | None = None
    imported_annotation_id: int | None = None
    source_document_id: int | None = None

    def normalized(self) -> "ResearchEventAnnotationCandidate":
        event_type = (self.event_type or "other").strip().lower()
        if event_type not in ANNOTATION_EVENT_TYPES:
            event_type = "other"
        sentiment = (self.sentiment_label or "unknown").strip().lower()
        if sentiment not in ANNOTATION_SENTIMENT_LABELS:
            sentiment = "unknown"
        status = (self.status or "staged").strip().lower()
        if status not in CANDIDATE_STATUSES:
            status = "staged"
        available_at = self.available_at
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=UTC)
        return ResearchEventAnnotationCandidate(
            candidate_id=self.candidate_id,
            ticker=(self.ticker or "").strip().upper(),
            event_date=self.event_date,
            available_at=available_at.astimezone(UTC),
            event_type=event_type,
            title=(self.title or "").strip(),
            summary=(self.summary or "").strip(),
            source=(self.source or "csv_manual").strip() or "csv_manual",
            source_url=(self.source_url or "").strip() or None,
            evidence_text=(self.evidence_text or "").strip(),
            sentiment_label=sentiment,
            strength=max(0, min(10, int(self.strength))),
            confidence=max(0.0, min(1.0, float(self.confidence))),
            tags=normalize_tags(self.tags),
            provider=(self.provider or "csv_manual").strip() or "csv_manual",
            provider_metadata=dict(self.provider_metadata or {}),
            document_type=(self.document_type or "").strip().lower() or None,
            published_at=(
                self.published_at.replace(tzinfo=UTC).astimezone(UTC)
                if self.published_at is not None and self.published_at.tzinfo is None
                else self.published_at.astimezone(UTC)
                if self.published_at is not None
                else None
            ),
            raw_text=(self.raw_text or "").strip(),
            cleaned_text=(self.cleaned_text or "").strip(),
            status=status,
            duplicate_of_annotation_id=self.duplicate_of_annotation_id,
            duplicate_of_candidate_id=self.duplicate_of_candidate_id,
            duplicate_reason=self.duplicate_reason,
            rejection_reason=self.rejection_reason,
            created_at=self.created_at,
            updated_at=self.updated_at,
            reviewed_at=self.reviewed_at,
            imported_annotation_id=self.imported_annotation_id,
            source_document_id=self.source_document_id,
        )


def candidate_dedupe_key(candidate: ResearchEventAnnotationCandidate) -> str:
    item = candidate.normalized()
    parts = [
        item.ticker,
        item.event_date.isoformat(),
        item.event_type,
        normalize_title(item.title),
        normalize_source_url(item.source_url) or "",
        stable_text_hash(item.evidence_text) or "",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class NewsEventProvider(Protocol):
    provider_name: str

    def get_events(self, ticker: str, start_date: date, end_date: date) -> list[ResearchEventAnnotationCandidate]:
        """Return provider-normalized research annotation candidates without importing them."""
        ...


class EmptyNewsEventProvider:
    """Safe default provider: no crawling, no API calls, no side effects."""

    provider_name = "empty_placeholder"

    def get_events(self, ticker: str, start_date: date, end_date: date) -> list[ResearchEventAnnotationCandidate]:
        return []
