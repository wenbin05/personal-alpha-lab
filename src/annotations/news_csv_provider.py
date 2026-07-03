from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time
from typing import Any

import pandas as pd

from src.annotations.models import ANNOTATION_EVENT_TYPES, ANNOTATION_SENTIMENT_LABELS, normalize_tags
from src.annotations.news_events import NewsEventProvider, ResearchEventAnnotationCandidate


REQUIRED_CANDIDATE_COLUMNS = {"ticker", "event_date", "title"}


@dataclass(frozen=True)
class CandidateImportError:
    row_number: int
    message: str


@dataclass(frozen=True)
class CandidateImportWarning:
    row_number: int
    message: str


@dataclass(frozen=True)
class CandidateImportResult:
    candidates: list[ResearchEventAnnotationCandidate] = field(default_factory=list)
    errors: list[CandidateImportError] = field(default_factory=list)
    warnings: list[CandidateImportWarning] = field(default_factory=list)


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


def _parse_metadata(value: Any) -> dict[str, Any]:
    if value is None or str(value).strip() == "":
        return {}
    text = str(value).strip()
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {"value": parsed}
    except Exception:
        return {"raw": text}


def parse_candidate_import_frame(frame: pd.DataFrame, provider: str = "csv_manual") -> CandidateImportResult:
    """Validate a provider-style news/event candidate CSV without importing annotations."""
    if frame is None or frame.empty:
        return CandidateImportResult(errors=[CandidateImportError(row_number=0, message="CSV is empty.")])
    normalized = frame.copy()
    normalized.columns = [str(column).strip().lower() for column in normalized.columns]
    missing = sorted(REQUIRED_CANDIDATE_COLUMNS - set(normalized.columns))
    if missing:
        return CandidateImportResult(
            errors=[CandidateImportError(row_number=0, message=f"Missing required columns: {', '.join(missing)}")]
        )

    candidates: list[ResearchEventAnnotationCandidate] = []
    errors: list[CandidateImportError] = []
    warnings: list[CandidateImportWarning] = []
    seen_keys: set[tuple[str, str, str]] = set()

    for idx, row in normalized.iterrows():
        row_number = int(idx) + 2
        ticker = str(_cell(row, "ticker", "")).strip().upper()
        if not ticker:
            errors.append(CandidateImportError(row_number, "Missing ticker."))
            continue
        event_date = _parse_date(_cell(row, "event_date", ""))
        if event_date is None:
            errors.append(CandidateImportError(row_number, "Invalid event_date."))
            continue
        available_at = _parse_available_at(_cell(row, "available_at", ""), event_date)
        if available_at is None:
            errors.append(CandidateImportError(row_number, "Invalid available_at."))
            continue
        title = str(_cell(row, "title", "")).strip()
        if not title:
            errors.append(CandidateImportError(row_number, "Missing title."))
            continue
        event_type = str(_cell(row, "event_type", "news")).strip().lower() or "news"
        if event_type not in ANNOTATION_EVENT_TYPES:
            errors.append(CandidateImportError(row_number, f"Unsupported event_type: {event_type}."))
            continue
        sentiment = str(_cell(row, "sentiment_label", "unknown")).strip().lower() or "unknown"
        if sentiment not in ANNOTATION_SENTIMENT_LABELS:
            errors.append(CandidateImportError(row_number, f"Invalid sentiment_label: {sentiment}."))
            continue
        strength = _parse_int(_cell(row, "strength", 0), default=0)
        if strength is None or strength < 0 or strength > 10:
            errors.append(CandidateImportError(row_number, "Strength must be an integer from 0 to 10."))
            continue
        confidence = _parse_float(_cell(row, "confidence", 0.0), default=0.0)
        if confidence is None or confidence < 0 or confidence > 1:
            errors.append(CandidateImportError(row_number, "Confidence must be a float from 0.0 to 1.0."))
            continue
        if available_at.date() < event_date:
            warnings.append(CandidateImportWarning(row_number, "available_at is before event_date; feature activation uses both fields."))

        source = str(_cell(row, "source", provider)).strip() or provider
        source_url = str(_cell(row, "source_url", "")).strip() or None
        evidence_text = str(_cell(row, "evidence_text", "")).strip()
        metadata = _parse_metadata(_cell(row, "provider_metadata_json", _cell(row, "metadata", "")))
        metadata.setdefault("csv_row_number", row_number)
        candidate = ResearchEventAnnotationCandidate(
            ticker=ticker,
            event_date=event_date,
            available_at=available_at,
            event_type=event_type,
            title=title,
            summary=str(_cell(row, "summary", "")).strip(),
            source=source,
            source_url=source_url,
            evidence_text=evidence_text,
            sentiment_label=sentiment,
            strength=strength,
            confidence=confidence,
            tags=normalize_tags(_cell(row, "tags", "")),
            provider=provider,
            provider_metadata=metadata,
        )
        row_key = (candidate.ticker, candidate.event_date.isoformat(), candidate.title.strip().lower())
        if row_key in seen_keys:
            errors.append(CandidateImportError(row_number, "Duplicate candidate key in CSV."))
            continue
        seen_keys.add(row_key)
        candidates.append(candidate)

    return CandidateImportResult(candidates=candidates, errors=errors, warnings=warnings)


class CsvManualNewsEventProvider:
    provider_name = "csv_manual"

    def __init__(self, frame: pd.DataFrame):
        self.frame = frame

    def get_events(self, ticker: str, start_date: date, end_date: date) -> list[ResearchEventAnnotationCandidate]:
        result = parse_candidate_import_frame(self.frame, provider=self.provider_name)
        ticker_upper = ticker.strip().upper()
        return [
            candidate
            for candidate in result.candidates
            if candidate.ticker == ticker_upper and start_date <= candidate.event_date <= end_date
        ]

