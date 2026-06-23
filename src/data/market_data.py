from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage
from src.utils.trading_calendar import calendar_source, latest_expected_trading_day

logger = logging.getLogger(__name__)


class MarketDataError(RuntimeError):
    pass


@dataclass(frozen=True)
class HistoryFetchResult:
    data: pd.DataFrame
    metadata: dict[str, Any]


class YFinanceProvider:
    name = "yfinance"

    def download_history(
        self,
        ticker: str,
        period: str = "2y",
        start: date | str | None = None,
        end: date | str | None = None,
    ) -> pd.DataFrame:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise MarketDataError("yfinance is not installed. Run pip install -r requirements.txt.") from exc

        try:
            kwargs: dict[str, Any] = {"interval": "1d", "auto_adjust": False}
            if start is not None or end is not None:
                if start is not None:
                    kwargs["start"] = pd.to_datetime(start).date().isoformat()
                if end is not None:
                    # yfinance treats end as exclusive, so request through the
                    # next calendar day to include the intended final session.
                    kwargs["end"] = (pd.to_datetime(end).date() + timedelta(days=1)).isoformat()
            else:
                kwargs["period"] = period
            df = yf.Ticker(ticker).history(**kwargs)
        except Exception as exc:
            raise MarketDataError(f"Could not download {ticker}: {exc}") from exc

        return storage.normalize_ohlcv(df)

    def company_info(self, ticker: str) -> dict:
        try:
            import yfinance as yf

            raw = yf.Ticker(ticker).info or {}
        except Exception:
            return {}
        keys = [
            "longName",
            "shortName",
            "sector",
            "industry",
            "marketCap",
            "beta",
            "website",
            "quoteType",
        ]
        return {key: raw.get(key) for key in keys if raw.get(key) is not None}


def get_provider(name: str = "yfinance") -> YFinanceProvider:
    if name.lower() != "yfinance":
        logger.warning("Unsupported provider '%s'; falling back to yfinance.", name)
    return YFinanceProvider()


def _last_expected_daily_bar_date(today: date | None = None) -> date:
    """Compatibility wrapper around the U.S. equities trading-calendar helper."""
    return latest_expected_trading_day(today or datetime.now(UTC))


def cache_is_fresh(
    df: pd.DataFrame,
    max_age_hours: int = 18,
    reference_datetime: datetime | date | None = None,
) -> bool:
    if df.empty:
        return False
    latest_date = pd.to_datetime(df.index.max()).date()
    return latest_date >= latest_expected_trading_day(reference_datetime)


def minimum_bars_for_period(period: str) -> int:
    """Approximate the minimum daily bars needed before cached data satisfies a request."""
    normalized = (period or "").strip().lower()
    if normalized in {"max", "ytd"}:
        return 0

    match = re.fullmatch(r"(\d+)(d|mo|y)", normalized)
    if not match:
        return 0

    amount = int(match.group(1))
    unit = match.group(2)
    if unit == "d":
        expected = amount
    elif unit == "mo":
        expected = amount * 21
    else:
        expected = amount * 252
    return int(expected * 0.8)


def cache_satisfies_period(df: pd.DataFrame, period: str) -> bool:
    if df.empty or "close" not in df.columns:
        return False
    minimum_bars = minimum_bars_for_period(period)
    return len(df.dropna(subset=["close"])) >= minimum_bars


def get_history_with_metadata(
    ticker: str,
    db_path: str | Path,
    provider_name: str = "yfinance",
    period: str = "2y",
    refresh: bool = False,
) -> HistoryFetchResult:
    ticker = ticker.upper().strip()
    cached = storage.load_ohlcv(db_path, ticker)
    cached_rows = int(len(cached))
    cached_latest = None if cached.empty else str(pd.to_datetime(cached.index.max()).date())
    cached_fresh = cache_is_fresh(cached)
    cached_period_ok = cache_satisfies_period(cached, period)
    expected_latest = latest_expected_trading_day()
    base_metadata: dict[str, Any] = {
        "ticker": ticker,
        "requested_period": period,
        "provider": provider_name,
        "calendar_source": calendar_source(),
        "latest_expected_trading_day": expected_latest.isoformat(),
        "refresh_requested": refresh,
        "cached_rows_before": cached_rows,
        "cached_latest_date_before": cached_latest,
        "cache_fresh_before": cached_fresh,
        "cache_satisfies_requested_period_before": cached_period_ok,
        "minimum_bars_for_requested_period": minimum_bars_for_period(period),
        "source": "cache",
        "download_error": None,
    }

    if not refresh and cached_fresh and cached_period_ok:
        return HistoryFetchResult(cached, base_metadata)

    provider = get_provider(provider_name)
    try:
        downloaded = provider.download_history(ticker, period=period)
    except MarketDataError as exc:
        if not cached.empty:
            metadata = {
                **base_metadata,
                "source": "cache_fallback_after_download_error",
                "download_error": f"Market data download failed; using cached data. {exc}",
            }
            return HistoryFetchResult(cached, metadata)
        metadata = {
            **base_metadata,
            "source": "download_error_no_cache",
            "download_error": f"Market data download failed and no cached data is available. {exc}",
        }
        return HistoryFetchResult(pd.DataFrame(), metadata)

    if not downloaded.empty:
        storage.upsert_ohlcv(db_path, ticker, downloaded)
        fresh = storage.load_ohlcv(db_path, ticker)
        metadata = {
            **base_metadata,
            "source": "fresh_download",
            "downloaded_rows": int(len(downloaded)),
        }
        return HistoryFetchResult(fresh, metadata)

    metadata = {
        **base_metadata,
        "source": "empty_download_cache_fallback" if not cached.empty else "empty_download_no_data",
        "downloaded_rows": 0,
    }
    return HistoryFetchResult(cached, metadata)


def get_history(
    ticker: str,
    db_path: str | Path,
    provider_name: str = "yfinance",
    period: str = "2y",
    refresh: bool = False,
) -> pd.DataFrame:
    return get_history_with_metadata(ticker, db_path, provider_name, period, refresh).data


def get_histories(
    tickers: list[str],
    db_path: str | Path,
    provider_name: str = "yfinance",
    period: str = "2y",
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    histories: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        try:
            histories[ticker.upper()] = get_history(ticker, db_path, provider_name, period, refresh)
        except Exception as exc:
            logger.warning("Skipping %s: %s", ticker, exc)
            histories[ticker.upper()] = pd.DataFrame()
    return histories


def get_company_info(ticker: str, provider_name: str = "yfinance") -> dict:
    return get_provider(provider_name).company_info(ticker.upper().strip())
