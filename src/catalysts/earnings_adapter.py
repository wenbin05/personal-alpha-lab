from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from src.catalysts.models import CatalystEvent


@dataclass
class EarningsResult:
    events: list[CatalystEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class YFinanceEarningsProvider:
    name = "yfinance_earnings"

    def _get_ticker(self, ticker: str) -> Any:
        import yfinance as yf

        return yf.Ticker(ticker)

    def fetch_earnings_events(self, ticker: str, limit: int = 6) -> EarningsResult:
        ticker = ticker.upper().strip()
        try:
            yf_ticker = self._get_ticker(ticker)
        except Exception as exc:
            return EarningsResult(warnings=[f"Earnings provider unavailable for {ticker}: {exc}"])

        events: list[CatalystEvent] = []
        warnings: list[str] = []

        try:
            earnings_dates = yf_ticker.get_earnings_dates(limit=limit)
        except Exception as exc:
            earnings_dates = None
            warnings.append(f"yfinance earnings dates unavailable for {ticker}: {exc}")

        if isinstance(earnings_dates, pd.DataFrame) and not earnings_dates.empty:
            for idx, row in earnings_dates.reset_index().iterrows():
                event_dt = pd.to_datetime(row.iloc[0], errors="coerce")
                if pd.isna(event_dt):
                    continue
                event_date = event_dt.date()
                eps_estimate = _safe_value(row.get("EPS Estimate"))
                reported_eps = _safe_value(row.get("Reported EPS"))
                surprise = _safe_value(row.get("Surprise(%)"))
                summary_parts = []
                if eps_estimate is not None:
                    summary_parts.append(f"EPS estimate: {eps_estimate}")
                if reported_eps is not None:
                    summary_parts.append(f"reported EPS: {reported_eps}")
                if surprise is not None:
                    summary_parts.append(f"surprise: {surprise}")
                summary = "; ".join(summary_parts) or "Earnings event metadata from yfinance."
                title = "Upcoming earnings" if event_date >= date.today() else "Recent earnings"
                events.append(
                    CatalystEvent(
                        ticker=ticker,
                        event_date=event_date,
                        event_time=event_dt.strftime("%H:%M:%S") if event_dt.time() else None,
                        event_type="earnings",
                        title=title,
                        summary=summary,
                        source="yfinance",
                        sentiment_label="unknown",
                        catalyst_strength=3,
                        confidence=0.45,
                        is_manual=False,
                        created_at=datetime.now(UTC),
                        updated_at=datetime.now(UTC),
                        raw_payload_json=json.dumps(row.to_dict(), default=str),
                    )
                )

        if not events:
            calendar_event = self._calendar_event(ticker, yf_ticker)
            if calendar_event is not None:
                events.append(calendar_event)

        if not events:
            warnings.append(f"Earnings data unavailable or empty for {ticker}. Manual entry is supported.")

        return EarningsResult(events=events[:limit], warnings=warnings)

    def _calendar_event(self, ticker: str, yf_ticker: Any) -> CatalystEvent | None:
        try:
            calendar_raw = yf_ticker.calendar
        except Exception:
            return None

        if calendar_raw is None:
            return None
        if isinstance(calendar_raw, pd.DataFrame):
            calendar_data = calendar_raw.to_dict()
        elif isinstance(calendar_raw, dict):
            calendar_data = calendar_raw
        else:
            return None

        raw_date = _first_calendar_value(calendar_data, "Earnings Date")
        if raw_date is None:
            return None
        event_dt = pd.to_datetime(raw_date, errors="coerce")
        if pd.isna(event_dt):
            return None
        return CatalystEvent(
            ticker=ticker,
            event_date=event_dt.date(),
            event_time=event_dt.strftime("%H:%M:%S") if event_dt.time() else None,
            event_type="earnings",
            title="Earnings date from yfinance calendar",
            summary="Provider supplied an earnings date, but estimates/actuals may be unavailable.",
            source="yfinance",
            sentiment_label="unknown",
            catalyst_strength=3,
            confidence=0.35,
            is_manual=False,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            raw_payload_json=json.dumps(calendar_data, default=str),
        )


def _safe_value(value: Any) -> Any:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _first_calendar_value(calendar_data: dict[str, Any], key: str) -> Any:
    value = calendar_data.get(key)
    if isinstance(value, dict):
        return next(iter(value.values()), None)
    if isinstance(value, (list, tuple)) and value:
        return value[0]
    return value
