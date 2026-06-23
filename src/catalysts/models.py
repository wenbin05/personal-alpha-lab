from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


EventType = Literal[
    "earnings",
    "sec_filing",
    "news",
    "analyst",
    "manual_note",
    "corporate_action",
    "other",
]

SentimentLabel = Literal["positive", "neutral", "negative", "unknown"]

EVENT_TYPES = [
    "earnings",
    "sec_filing",
    "news",
    "analyst",
    "manual_note",
    "corporate_action",
    "other",
]

SENTIMENT_LABELS = ["positive", "neutral", "negative", "unknown"]


class CatalystEvent(BaseModel):
    id: int | None = None
    ticker: str
    event_date: date
    event_time: str | None = None
    event_type: EventType = "other"
    title: str
    summary: str = ""
    source: str = "manual"
    source_url: str | None = None
    sentiment_label: SentimentLabel = "unknown"
    catalyst_strength: int = Field(default=0, ge=0, le=10)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_manual: bool = False
    available_at: datetime | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_payload_json: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Catalyst title is required.")
        return cleaned

    @field_validator("summary", "source")
    @classmethod
    def clean_optional_text(cls, value: str) -> str:
        return (value or "").strip()
