from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as dt_time, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.earnings.models import EarningsEvent


MARKET_TZ = ZoneInfo("America/New_York")


@dataclass
class EarningsCSVImportResult:
    events: list[EarningsEvent] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value) or str(value).strip() == "":
            return None
        return float(value)
    except Exception:
        return None


def _parse_datetime(value: Any) -> datetime | None:
    parsed = pd.to_datetime(value, utc=False, errors="coerce")
    if pd.isna(parsed):
        return None
    if getattr(parsed, "tzinfo", None) is None:
        return parsed.to_pydatetime().replace(tzinfo=MARKET_TZ).astimezone(UTC)
    return parsed.to_pydatetime().astimezone(UTC)


def _parse_date(value: Any) -> date | None:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date()


def _available_from_row(row: dict[str, Any], announced_at: datetime | None, fiscal_period_end: date | None) -> tuple[datetime | None, list[str]]:
    warnings: list[str] = []
    available_at = _parse_datetime(row.get("available_at"))
    if available_at is not None:
        return available_at, warnings
    if announced_at is not None:
        return announced_at, warnings
    if fiscal_period_end is not None:
        warnings.append("available_at and announced_at missing; using conservative fiscal period end + 45 days.")
        return datetime.combine(fiscal_period_end + timedelta(days=45), dt_time(16, 1), tzinfo=MARKET_TZ).astimezone(UTC), warnings
    return None, warnings


def parse_earnings_import_frame(frame: pd.DataFrame, provider: str = "csv_import") -> EarningsCSVImportResult:
    if frame is None or frame.empty:
        return EarningsCSVImportResult(errors=["CSV is empty."])
    normalized = frame.rename(columns={column: str(column).strip().lower() for column in frame.columns})
    result = EarningsCSVImportResult()
    for row_number, row in enumerate(normalized.to_dict(orient="records"), start=2):
        ticker = str(row.get("ticker") or "").strip().upper()
        if not ticker:
            result.errors.append(f"Row {row_number}: missing ticker.")
            continue
        fiscal_period_end = _parse_date(row.get("fiscal_period_end"))
        announced_at = _parse_datetime(row.get("announced_at"))
        available_at, warnings = _available_from_row(row, announced_at, fiscal_period_end)
        if available_at is None:
            result.errors.append(f"Row {row_number}: missing valid available_at, announced_at, or fiscal_period_end.")
            continue
        raw_payload = json.dumps(row, ensure_ascii=False, sort_keys=True, default=str)
        try:
            result.events.append(
                EarningsEvent(
                    ticker=ticker,
                    fiscal_period_end=fiscal_period_end,
                    announced_at=announced_at,
                    available_at=available_at,
                    timing=str(row.get("timing") or "unknown"),
                    eps_estimate=_safe_float(row.get("eps_estimate")),
                    eps_actual=_safe_float(row.get("eps_actual")),
                    eps_surprise=_safe_float(row.get("eps_surprise")),
                    eps_surprise_percent=_safe_float(row.get("eps_surprise_percent")),
                    revenue_estimate=_safe_float(row.get("revenue_estimate")),
                    revenue_actual=_safe_float(row.get("revenue_actual")),
                    revenue_surprise_percent=_safe_float(row.get("revenue_surprise_percent")),
                    currency=str(row.get("currency") or "USD"),
                    provider=provider,
                    provider_event_id=str(row.get("provider_event_id") or f"csv:{ticker}:{row_number}"),
                    fetched_at=datetime.now(UTC),
                    raw_payload_json=raw_payload,
                    data_quality_status="partial" if warnings else "ok",
                    warnings=warnings,
                )
            )
        except Exception as exc:
            result.errors.append(f"Row {row_number}: {exc}")
    return result

