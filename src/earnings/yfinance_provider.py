from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time as dt_time, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from src.earnings.models import EarningsEvent
from src.earnings.repository import cache_provider_response, get_cached_provider_response


YFINANCE_EARNINGS_PROVIDER = "yfinance_earnings_history"
YFINANCE_EARNINGS_CACHE_VERSION = "yf_earnings_v1"
MARKET_TZ = ZoneInfo("America/New_York")


@dataclass
class EarningsProviderResult:
    events: list[EarningsEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


class YFinanceHistoricalEarningsProvider:
    """Best-effort historical earnings provider.

    yfinance earnings coverage varies by version and ticker. The provider keeps
    missing EPS/revenue fields missing, records warnings, and never fabricates
    revenue values or future scheduled-event knowledge.
    """

    name = YFINANCE_EARNINGS_PROVIDER

    def __init__(self, db_path: str | Path | None = None, min_interval_seconds: float = 0.25) -> None:
        self.db_path = Path(db_path) if db_path is not None else None
        self.min_interval_seconds = float(min_interval_seconds)
        self._last_request_at = 0.0

    def _get_ticker(self, ticker: str) -> Any:
        import yfinance as yf

        return yf.Ticker(ticker)

    def _cache_key(self, ticker: str, start_date: date, end_date: date) -> str:
        return f"{YFINANCE_EARNINGS_CACHE_VERSION}:{ticker.upper()}:{start_date.isoformat()}:{end_date.isoformat()}"

    def _wait(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def fetch_earnings_events(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        use_cache: bool = True,
    ) -> EarningsProviderResult:
        ticker = ticker.upper().strip()
        cache_key = self._cache_key(ticker, start_date, end_date)
        if use_cache and self.db_path is not None:
            cached = get_cached_provider_response(self.db_path, cache_key)
            if cached and cached.get("status") in {"ok", "empty"} and cached.get("response_json"):
                payload = json.loads(str(cached["response_json"]))
                result = self._events_from_payload(ticker, start_date, end_date, payload)
                result.metadata.update({"source": "cache", "cache_key": cache_key, "fetched_at": cached.get("fetched_at")})
                return result

        warnings: list[str] = []
        payload: dict[str, Any] = {"ticker": ticker, "earnings_dates": [], "earnings_history": [], "warnings": []}
        try:
            yf_ticker = self._get_ticker(ticker)
        except Exception as exc:
            message = f"yfinance ticker initialization failed for {ticker}: {exc}"
            if self.db_path is not None:
                cache_provider_response(self.db_path, cache_key, self.name, ticker, None, "error", message)
            return EarningsProviderResult(warnings=[message], metadata={"source": "download", "cache_key": cache_key})

        self._wait()
        try:
            limit = max(12, int((end_date - start_date).days / 90) + 8)
            earnings_dates = yf_ticker.get_earnings_dates(limit=limit)
            if isinstance(earnings_dates, pd.DataFrame) and not earnings_dates.empty:
                payload["earnings_dates"] = _frame_records(earnings_dates)
        except Exception as exc:
            warnings.append(f"yfinance get_earnings_dates unavailable for {ticker}: {exc}")

        self._wait()
        try:
            earnings_history = yf_ticker.get_earnings_history()
            if isinstance(earnings_history, pd.DataFrame) and not earnings_history.empty:
                payload["earnings_history"] = _frame_records(earnings_history)
        except Exception as exc:
            warnings.append(f"yfinance get_earnings_history unavailable for {ticker}: {exc}")

        payload["warnings"] = warnings
        payload_json = json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
        if self.db_path is not None:
            status = "ok" if payload["earnings_dates"] or payload["earnings_history"] else "empty"
            cache_provider_response(self.db_path, cache_key, self.name, ticker, payload_json, status, None)
        result = self._events_from_payload(ticker, start_date, end_date, payload)
        result.metadata.update({"source": "download", "cache_key": cache_key})
        return result

    def _events_from_payload(
        self,
        ticker: str,
        start_date: date,
        end_date: date,
        payload: dict[str, Any],
    ) -> EarningsProviderResult:
        warnings = list(payload.get("warnings") or [])
        source_records = payload.get("earnings_dates") or []
        source_name = "earnings_dates"
        if not source_records:
            source_records = payload.get("earnings_history") or []
            source_name = "earnings_history"
        events: list[EarningsEvent] = []
        fetched_at = datetime.now(UTC)
        base_ids = [_base_provider_event_id(source_name, idx, record) for idx, record in enumerate(source_records)]
        base_counts = Counter(base_ids)
        seen: dict[str, int] = defaultdict(int)
        for idx, record in enumerate(source_records):
            base_id = base_ids[idx]
            seen[base_id] += 1
            provider_event_id = base_id
            if base_counts[base_id] > 1 and seen[base_id] < base_counts[base_id]:
                provider_event_id = f"{base_id}:duplicate_{seen[base_id]}"
            event = _record_to_event(ticker, source_name, idx, record, fetched_at, provider_event_id=provider_event_id)
            if event is None:
                continue
            available_date = event.available_at.date()
            if start_date <= available_date <= end_date:
                events.append(event)
        if not events:
            warnings.append(f"No historical earnings events returned for {ticker} in requested range.")
        return EarningsProviderResult(
            events=events,
            warnings=warnings,
            metadata={"provider": self.name, "record_source": source_name, "raw_records": len(source_records)},
        )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    normalized = frame.copy()
    index_name = normalized.index.name or "index"
    normalized = normalized.reset_index().rename(columns={index_name: "event_index"})
    return normalized.to_dict(orient="records")


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _first_present(record: dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in record and record[name] is not None:
            return record[name]
    return None


def _base_provider_event_id(source_name: str, idx: int, record: dict[str, Any]) -> str:
    event_index = _first_present(record, ["event_index", "Earnings Date", "Earnings Date"])
    return f"{source_name}:{event_index or idx}"


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


def _timing_and_available(announced_at: datetime | None, fiscal_period_end: date | None) -> tuple[str, datetime, list[str]]:
    warnings: list[str] = []
    if announced_at is not None:
        local = announced_at.astimezone(MARKET_TZ)
        local_clock = local.time()
        if local_clock < dt_time(9, 30):
            return "before_market", announced_at, warnings
        if local_clock >= dt_time(16, 0):
            return "after_market", announced_at, warnings
        return "during_market", announced_at, warnings

    base_date = fiscal_period_end or datetime.now(UTC).date()
    available_local = datetime.combine(base_date + timedelta(days=45), dt_time(16, 1), tzinfo=MARKET_TZ)
    warnings.append(
        "Announcement timestamp unavailable from provider; availability conservatively approximated from fiscal period end."
    )
    return "unknown", available_local.astimezone(UTC), warnings


def _record_to_event(
    ticker: str,
    source_name: str,
    idx: int,
    record: dict[str, Any],
    fetched_at: datetime,
    provider_event_id: str | None = None,
) -> EarningsEvent | None:
    event_index = _first_present(record, ["event_index", "Earnings Date", "Earnings Date"])
    fiscal_period_end = _parse_date(event_index)
    announced_at: datetime | None = None
    if source_name == "earnings_dates":
        announced_at = _parse_datetime(event_index)
        fiscal_period_end = None
    eps_estimate = _safe_float(_first_present(record, ["EPS Estimate", "epsEstimate", "eps_estimate"]))
    eps_actual = _safe_float(_first_present(record, ["Reported EPS", "epsActual", "eps_actual"]))
    eps_surprise = _safe_float(_first_present(record, ["EPS Difference", "epsDifference", "eps_surprise"]))
    eps_surprise_percent = _safe_float(_first_present(record, ["Surprise(%)", "surprisePercent", "eps_surprise_percent"]))
    revenue_estimate = _safe_float(_first_present(record, ["Revenue Estimate", "revenueEstimate", "revenue_estimate"]))
    revenue_actual = _safe_float(_first_present(record, ["Reported Revenue", "revenueActual", "revenue_actual"]))
    revenue_surprise_percent = _safe_float(
        _first_present(record, ["Revenue Surprise(%)", "revenueSurprisePercent", "revenue_surprise_percent"])
    )
    timing, available_at, timing_warnings = _timing_and_available(announced_at, fiscal_period_end)
    if eps_surprise is None and eps_actual is not None and eps_estimate is not None:
        eps_surprise = eps_actual - eps_estimate
    warnings = list(timing_warnings)
    if eps_actual is None:
        warnings.append("EPS actual unavailable from provider.")
    if eps_estimate is None:
        warnings.append("EPS estimate unavailable from provider.")
    if revenue_actual is None or revenue_estimate is None:
        warnings.append("Revenue fields unavailable from provider.")
    quality = "ok" if not warnings else "partial"
    raw_payload = json.dumps(record, ensure_ascii=False, sort_keys=True, default=str)
    provider_event_id = provider_event_id or f"{source_name}:{event_index or idx}"
    try:
        return EarningsEvent(
            ticker=ticker,
            fiscal_period_end=fiscal_period_end,
            announced_at=announced_at,
            available_at=available_at,
            timing=timing,
            eps_estimate=eps_estimate,
            eps_actual=eps_actual,
            eps_surprise=eps_surprise,
            eps_surprise_percent=eps_surprise_percent,
            revenue_estimate=revenue_estimate,
            revenue_actual=revenue_actual,
            revenue_surprise_percent=revenue_surprise_percent,
            currency=str(record.get("currency") or "USD"),
            provider=YFINANCE_EARNINGS_PROVIDER,
            provider_event_id=provider_event_id,
            fetched_at=fetched_at,
            raw_payload_json=raw_payload,
            data_quality_status=quality,
            warnings=warnings,
        )
    except Exception:
        return None
