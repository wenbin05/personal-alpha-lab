from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from src.catalysts.models import CatalystEvent, SENTIMENT_LABELS
from src.documents.models import DOCUMENT_TYPES, SourceDocument
from src.documents.repository import build_source_document
from src.documents.text_cleaning import preview_text


@dataclass
class DocumentImportRow:
    document: SourceDocument
    catalyst: CatalystEvent | None = None


@dataclass
class DocumentImportResult:
    rows: list[DocumentImportRow] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _as_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _parse_date(value: Any, row_number: int, errors: list[str]) -> date | None:
    text = _as_text(value)
    if not text:
        return None
    try:
        return pd.to_datetime(text).date()
    except Exception:
        errors.append(f"Row {row_number}: invalid published_at/date value '{text}'.")
        return None


def _parse_float(value: Any, default: float) -> float:
    text = _as_text(value)
    if not text:
        return default
    try:
        return float(text)
    except Exception:
        return default


def _parse_int(value: Any, default: int) -> int:
    text = _as_text(value)
    if not text:
        return default
    try:
        return int(float(text))
    except Exception:
        return default


def _document_type(value: Any, row_number: int, errors: list[str]) -> str | None:
    doc_type = _as_text(value) or "imported_csv"
    if doc_type not in DOCUMENT_TYPES:
        errors.append(f"Row {row_number}: unsupported document_type '{doc_type}'.")
        return None
    return doc_type


def _event_type_for_document(document_type: str) -> str:
    if document_type == "sec_filing":
        return "sec_filing"
    if document_type == "news_article":
        return "news"
    if document_type == "earnings_transcript":
        return "earnings"
    if document_type in {"manual_text", "imported_csv"}:
        return "manual_note"
    return "other"


def parse_document_import_frame(df: pd.DataFrame) -> DocumentImportResult:
    result = DocumentImportResult()
    if df is None or df.empty:
        result.errors.append("CSV import is empty.")
        return result

    normalized = df.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    text_column = "text" if "text" in normalized.columns else "raw_text" if "raw_text" in normalized.columns else None
    if text_column is None:
        result.errors.append("CSV must include a text or raw_text column.")
        return result

    seen_source_urls: set[str] = set()
    seen_text_hashes: set[str] = set()

    for idx, row in normalized.iterrows():
        row_number = int(idx) + 2
        row_errors: list[str] = []
        ticker = _as_text(row.get("ticker")).upper()
        raw_text = _as_text(row.get(text_column))
        if not ticker:
            row_errors.append(f"Row {row_number}: ticker is required.")
        if not raw_text:
            row_errors.append(f"Row {row_number}: text is required.")
        doc_type = _document_type(row.get("document_type"), row_number, row_errors)
        published_value = row.get("published_at") if "published_at" in normalized.columns else row.get("date")
        if not _as_text(published_value) and "date" in normalized.columns:
            published_value = row.get("date")
        published_at = _parse_date(published_value, row_number, row_errors)
        if row_errors:
            result.errors.extend(row_errors)
            continue

        title = _as_text(row.get("title")) or f"Imported {doc_type} document"
        source_url = _as_text(row.get("source_url")) or None
        original_source = _as_text(row.get("source")) or "csv_import"
        raw_payload = {
            "csv_row": row_number,
            "original_source": original_source,
            "columns": {key: _as_text(value) for key, value in row.items()},
        }
        try:
            document = build_source_document(
                ticker=ticker,
                document_type=doc_type or "imported_csv",
                title=title,
                raw_text=raw_text,
                source="csv_import",
                source_url=source_url,
                published_at=published_at,
                warnings=f"Imported from CSV source '{original_source}'.",
                raw_payload_json=json.dumps(raw_payload, default=str),
            )
        except Exception as exc:
            result.errors.append(f"Row {row_number}: could not build document: {exc}")
            continue

        if document.source_url:
            source_url_key = document.source_url.strip().lower()
            if source_url_key in seen_source_urls:
                result.warnings.append(f"Row {row_number}: duplicate source_url in CSV import.")
            seen_source_urls.add(source_url_key)
        if document.text_hash in seen_text_hashes:
            result.warnings.append(f"Row {row_number}: duplicate text content in CSV import.")
        seen_text_hashes.add(document.text_hash)

        catalyst = None
        has_catalyst_fields = any(_as_text(row.get(column)) for column in ["sentiment_label", "catalyst_strength", "confidence"])
        if has_catalyst_fields:
            sentiment = _as_text(row.get("sentiment_label")) or "unknown"
            if sentiment not in SENTIMENT_LABELS:
                result.warnings.append(f"Row {row_number}: unsupported sentiment '{sentiment}', using unknown.")
                sentiment = "unknown"
            strength = max(0, min(10, _parse_int(row.get("catalyst_strength"), 0)))
            confidence = max(0.0, min(1.0, _parse_float(row.get("confidence"), 0.5)))
            catalyst = CatalystEvent(
                ticker=ticker,
                event_date=published_at or datetime.now(UTC).date(),
                event_type=_event_type_for_document(doc_type or "imported_csv"),
                title=title,
                summary=preview_text(raw_text, limit=500),
                source="csv_import",
                source_url=source_url,
                sentiment_label=sentiment,
                catalyst_strength=strength,
                confidence=confidence,
                is_manual=True,
                created_at=datetime.now(UTC),
                updated_at=datetime.now(UTC),
                raw_payload_json=json.dumps(raw_payload, default=str),
            )

        result.rows.append(DocumentImportRow(document=document, catalyst=catalyst))

    return result
