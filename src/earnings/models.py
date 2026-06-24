from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


EarningsTiming = Literal["before_market", "after_market", "during_market", "unknown"]
EarningsDataQuality = Literal["ok", "partial", "unavailable", "error"]

EARNINGS_TIMINGS = ["before_market", "after_market", "during_market", "unknown"]
EARNINGS_DATA_QUALITY_STATUSES = ["ok", "partial", "unavailable", "error"]


class EarningsEvent(BaseModel):
    earnings_event_id: int | None = None
    ticker: str
    fiscal_period_end: date | None = None
    announced_at: datetime | None = None
    available_at: datetime
    timing: EarningsTiming = "unknown"
    eps_estimate: float | None = None
    eps_actual: float | None = None
    eps_surprise: float | None = None
    eps_surprise_percent: float | None = None
    revenue_estimate: float | None = None
    revenue_actual: float | None = None
    revenue_surprise_percent: float | None = None
    currency: str | None = None
    provider: str
    provider_event_id: str | None = None
    fetched_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_payload_json: str | None = None
    raw_payload_hash: str | None = None
    data_quality_status: EarningsDataQuality = "partial"
    warnings: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @field_validator("provider")
    @classmethod
    def clean_provider(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Earnings provider is required.")
        return cleaned

    @field_validator("currency")
    @classmethod
    def clean_currency(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().upper()
        return cleaned or None

    def model_payload(self) -> dict[str, Any]:
        return self.model_dump(mode="json")

