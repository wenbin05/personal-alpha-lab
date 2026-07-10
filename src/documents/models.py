from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator


DocumentType = Literal[
    "sec_filing",
    "news_article",
    "earnings_transcript",
    "company_ir_press_release",
    "manual_text",
    "imported_csv",
    "other",
]

DocumentSource = Literal[
    "SEC",
    "manual",
    "csv_import",
    "company_ir_press_release",
    "provider_placeholder",
    "other",
]

ParsingStatus = Literal["success", "failed", "partial", "not_attempted"]

DOCUMENT_TYPES = [
    "sec_filing",
    "news_article",
    "earnings_transcript",
    "company_ir_press_release",
    "manual_text",
    "imported_csv",
    "other",
]

DOCUMENT_SOURCES = ["SEC", "manual", "csv_import", "company_ir_press_release", "provider_placeholder", "other"]

PARSING_STATUSES = ["success", "failed", "partial", "not_attempted"]


class SourceDocument(BaseModel):
    document_id: int | None = None
    ticker: str
    catalyst_id: int | None = None
    document_type: DocumentType = "manual_text"
    source: DocumentSource = "manual"
    source_url: str | None = None
    accession_number: str | None = None
    filing_type: str | None = None
    title: str
    published_at: date | datetime | None = None
    raw_text: str = ""
    cleaned_text: str = ""
    text_hash: str = ""
    parsing_status: ParsingStatus = "not_attempted"
    warnings: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    raw_payload_json: str | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        cleaned = value.strip().upper()
        if not cleaned:
            raise ValueError("Ticker is required.")
        return cleaned

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Document title is required.")
        return cleaned

    @field_validator("source_url", "accession_number", "filing_type", "raw_payload_json")
    @classmethod
    def clean_optional_text(cls, value: str | None) -> str | None:
        cleaned = (value or "").strip()
        return cleaned or None

    @field_validator("raw_text", "cleaned_text", "warnings")
    @classmethod
    def normalize_text_fields(cls, value: str | None) -> str:
        return (value or "").strip()
