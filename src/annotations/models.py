from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any


ANNOTATION_EVENT_TYPES = (
    "earnings",
    "sec_filing",
    "news",
    "analyst",
    "manual_note",
    "corporate_action",
    "product_launch",
    "guidance_update",
    "legal_regulatory",
    "macro_sensitive",
    "management_change",
    "partnership",
    "financing",
    "other",
)

ANNOTATION_SENTIMENT_LABELS = ("positive", "neutral", "negative", "mixed", "unknown")


def utc_now() -> datetime:
    return datetime.now(UTC)


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def normalize_tags(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        raw = value
    elif isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("["):
            try:
                parsed = json.loads(text)
                raw = parsed if isinstance(parsed, list) else [text]
            except Exception:
                raw = [part.strip() for part in text.replace(";", ",").split(",")]
        else:
            raw = [part.strip() for part in text.replace(";", ",").split(",")]
    else:
        raw = [str(value)]
    return sorted({str(item).strip().lower() for item in raw if str(item).strip()})


@dataclass(frozen=True)
class ResearchEventAnnotation:
    ticker: str
    event_date: date
    available_at: datetime
    event_type: str = "other"
    sentiment_label: str = "unknown"
    strength: int = 0
    confidence: float = 0.0
    source: str = "manual"
    source_url: str | None = None
    title: str = ""
    summary: str = ""
    evidence_text: str = ""
    tags: list[str] = field(default_factory=list)
    source_document_id: int | None = None
    annotation_id: int | None = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    research_only: bool = True
    scanner_scoring_effect: int = 0

    def normalized(self) -> "ResearchEventAnnotation":
        ticker = self.ticker.strip().upper()
        event_type = self.event_type.strip().lower() if self.event_type else "other"
        if event_type not in ANNOTATION_EVENT_TYPES:
            event_type = "other"
        sentiment = self.sentiment_label.strip().lower() if self.sentiment_label else "unknown"
        if sentiment not in ANNOTATION_SENTIMENT_LABELS:
            sentiment = "unknown"
        available_at = self.available_at
        if available_at.tzinfo is None:
            available_at = available_at.replace(tzinfo=UTC)
        return ResearchEventAnnotation(
            annotation_id=self.annotation_id,
            ticker=ticker,
            event_date=self.event_date,
            available_at=available_at.astimezone(UTC),
            event_type=event_type,
            sentiment_label=sentiment,
            strength=max(0, min(10, int(self.strength))),
            confidence=max(0.0, min(1.0, float(self.confidence))),
            source=(self.source or "manual").strip() or "manual",
            source_url=(self.source_url or "").strip() or None,
            title=(self.title or "").strip(),
            summary=(self.summary or "").strip(),
            evidence_text=(self.evidence_text or "").strip(),
            tags=normalize_tags(self.tags),
            source_document_id=self.source_document_id,
            created_at=self.created_at,
            updated_at=self.updated_at,
            research_only=True,
            scanner_scoring_effect=0,
        )


def annotation_dedupe_key(annotation: ResearchEventAnnotation) -> str:
    item = annotation.normalized()
    parts = [
        item.ticker,
        item.event_date.isoformat(),
        item.available_at.isoformat(timespec="seconds"),
        item.event_type,
        item.sentiment_label,
        item.source.lower(),
        (item.source_url or "").lower(),
        item.title.lower(),
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
