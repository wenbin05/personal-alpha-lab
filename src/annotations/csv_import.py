from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Any

import pandas as pd

from src.annotations.models import (
    ANNOTATION_EVENT_TYPES,
    ANNOTATION_SENTIMENT_LABELS,
    ResearchEventAnnotation,
    annotation_dedupe_key,
    normalize_tags,
)


REQUIRED_COLUMNS = {"ticker", "event_date"}


@dataclass(frozen=True)
class AnnotationImportError:
    row_number: int
    message: str


@dataclass(frozen=True)
class AnnotationImportWarning:
    row_number: int
    message: str


@dataclass(frozen=True)
class AnnotationImportResult:
    annotations: list[ResearchEventAnnotation] = field(default_factory=list)
    errors: list[AnnotationImportError] = field(default_factory=list)
    warnings: list[AnnotationImportWarning] = field(default_factory=list)


def _cell(row: pd.Series, column: str, default: Any = "") -> Any:
    if column not in row:
        return default
    value = row[column]
    if pd.isna(value):
        return default
    return value


def _parse_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _parse_available_at(value: Any, event_date: date) -> datetime | None:
    if value is None or str(value).strip() == "":
        return datetime.combine(event_date, time(23, 59, 59), tzinfo=UTC)
    text = str(value).strip()
    parsed = pd.to_datetime(text, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    if len(text) <= 10:
        return datetime.combine(parsed.date(), time(23, 59, 59), tzinfo=UTC)
    return parsed.to_pydatetime().astimezone(UTC)


def _parse_int(value: Any, default: int = 0) -> int | None:
    try:
        return int(float(value))
    except Exception:
        return default if value in (None, "") else None


def _parse_float(value: Any, default: float = 0.0) -> float | None:
    try:
        return float(value)
    except Exception:
        return default if value in (None, "") else None


def parse_annotation_import_frame(frame: pd.DataFrame) -> AnnotationImportResult:
    """Validate and parse research-only historical annotation CSV rows."""
    if frame is None or frame.empty:
        return AnnotationImportResult(errors=[AnnotationImportError(row_number=0, message="CSV is empty.")])
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    missing = sorted(REQUIRED_COLUMNS - set(normalized.columns))
    if missing:
        return AnnotationImportResult(
            errors=[AnnotationImportError(row_number=0, message=f"Missing required columns: {', '.join(missing)}")]
        )

    annotations: list[ResearchEventAnnotation] = []
    errors: list[AnnotationImportError] = []
    warnings: list[AnnotationImportWarning] = []
    seen_keys: set[str] = set()

    for idx, row in normalized.iterrows():
        row_number = int(idx) + 2
        ticker = str(_cell(row, "ticker", "")).strip().upper()
        if not ticker:
            errors.append(AnnotationImportError(row_number, "Missing ticker."))
            continue
        event_date = _parse_date(_cell(row, "event_date", ""))
        if event_date is None:
            errors.append(AnnotationImportError(row_number, "Invalid event_date."))
            continue
        available_at = _parse_available_at(_cell(row, "available_at", ""), event_date)
        if available_at is None:
            errors.append(AnnotationImportError(row_number, "Invalid available_at."))
            continue

        event_type = str(_cell(row, "event_type", "other")).strip().lower() or "other"
        if event_type not in ANNOTATION_EVENT_TYPES:
            errors.append(AnnotationImportError(row_number, f"Unsupported event_type: {event_type}."))
            continue
        sentiment = str(_cell(row, "sentiment_label", "unknown")).strip().lower() or "unknown"
        if sentiment not in ANNOTATION_SENTIMENT_LABELS:
            errors.append(AnnotationImportError(row_number, f"Invalid sentiment_label: {sentiment}."))
            continue
        strength = _parse_int(_cell(row, "strength", 0), default=0)
        if strength is None or strength < 0 or strength > 10:
            errors.append(AnnotationImportError(row_number, "Strength must be an integer from 0 to 10."))
            continue
        confidence = _parse_float(_cell(row, "confidence", 0.0), default=0.0)
        if confidence is None or confidence < 0 or confidence > 1:
            errors.append(AnnotationImportError(row_number, "Confidence must be a float from 0.0 to 1.0."))
            continue
        if available_at.date() < event_date:
            warnings.append(AnnotationImportWarning(row_number, "available_at is before event_date; feature activation still uses available_at and event_date."))

        annotation = ResearchEventAnnotation(
            ticker=ticker,
            event_date=event_date,
            available_at=available_at,
            event_type=event_type,
            sentiment_label=sentiment,
            strength=strength,
            confidence=confidence,
            source=str(_cell(row, "source", "csv_import")).strip() or "csv_import",
            source_url=str(_cell(row, "source_url", "")).strip() or None,
            title=str(_cell(row, "title", "")).strip(),
            summary=str(_cell(row, "summary", "")).strip(),
            evidence_text=str(_cell(row, "evidence_text", "")).strip(),
            tags=normalize_tags(_cell(row, "tags", "")),
        )
        key = annotation_dedupe_key(annotation)
        if key in seen_keys:
            errors.append(AnnotationImportError(row_number, "Duplicate annotation event key in CSV."))
            continue
        seen_keys.add(key)
        annotations.append(annotation)

    return AnnotationImportResult(annotations=annotations, errors=errors, warnings=warnings)

