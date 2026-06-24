from __future__ import annotations

from bisect import bisect_right
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage
from src.utils.trading_calendar import previous_trading_day, trading_days_between


EARNINGS_FEATURE_WINDOWS = (1, 5, 20)


def _as_of_after_close(trading_date: date) -> datetime:
    return datetime.combine(trading_date, time(23, 59, 59), tzinfo=UTC)


def _sessions_ago(trading_date: date, sessions: int) -> date:
    current = trading_date
    for _ in range(max(0, int(sessions) - 1)):
        current = previous_trading_day(current)
    return current


def earnings_feature_base() -> dict[str, Any]:
    return {
        "earnings_event_present_1s": False,
        "earnings_event_present_5s": False,
        "earnings_event_present_20s": False,
        "sessions_since_latest_earnings": None,
        "latest_eps_surprise_percent": None,
        "latest_eps_surprise_direction": 0,
        "latest_revenue_surprise_percent": None,
        "earnings_timing_known": False,
        "earnings_data_available": False,
        "earnings_event_count_20s": 0,
        "earnings_missing_eps_actual_count_20s": 0,
        "earnings_missing_eps_estimate_count_20s": 0,
        "earnings_missing_revenue_count_20s": 0,
        "earnings_latest_provider": None,
        "earnings_latest_data_quality_status": None,
        "earnings_latest_warnings": [],
        "earnings_available_event_ids": [],
    }


def _compact_earnings_frame(db_path: str | Path, ticker: str) -> pd.DataFrame:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT earnings_event_id, ticker, fiscal_period_end, announced_at, available_at, timing,
                   eps_estimate, eps_actual, eps_surprise_percent, revenue_estimate,
                   revenue_actual, revenue_surprise_percent, provider, data_quality_status, warnings
            FROM earnings_events
            WHERE ticker = ?
            ORDER BY datetime(available_at), earnings_event_id
            """,
            conn,
            params=(ticker.upper(),),
        )
    if frame.empty:
        return frame
    frame["available_at_parsed"] = pd.to_datetime(frame["available_at"], utc=True, errors="coerce")
    frame["available_date"] = frame["available_at_parsed"].dt.date
    return frame.dropna(subset=["available_at_parsed", "available_date"]).copy()


def _direction(value: Any) -> int:
    try:
        if value is None or pd.isna(value):
            return 0
        number = float(value)
    except Exception:
        return 0
    if number > 0:
        return 1
    if number < 0:
        return -1
    return 0


def _trading_sessions_since(start_date: date, end_date: date) -> int:
    if start_date > end_date:
        return 0
    return max(0, len(trading_days_between(start_date, end_date)) - 1)


def _is_missing(value: Any) -> bool:
    try:
        return bool(value is None or pd.isna(value))
    except Exception:
        return value is None


def precompute_earnings_features_for_dates(
    db_path: str | Path,
    ticker: str,
    snapshot_dates: list[date],
) -> dict[date, dict[str, Any]]:
    dates = sorted(set(snapshot_dates))
    if not dates:
        return {}
    events = _compact_earnings_frame(db_path, ticker)
    if events.empty:
        return {trading_date: earnings_feature_base() for trading_date in dates}

    records: list[dict[str, Any]] = []
    for row in events.itertuples(index=False):
        records.append(
            {
                "earnings_event_id": int(row.earnings_event_id),
                "available_at": row.available_at_parsed,
                "available_date": row.available_date,
                "timing": row.timing,
                "eps_actual": row.eps_actual,
                "eps_estimate": row.eps_estimate,
                "eps_surprise_percent": row.eps_surprise_percent,
                "revenue_estimate": row.revenue_estimate,
                "revenue_actual": row.revenue_actual,
                "revenue_surprise_percent": row.revenue_surprise_percent,
                "provider": row.provider,
                "data_quality_status": row.data_quality_status,
                "warnings": row.warnings,
            }
        )
    records.sort(key=lambda item: (item["available_at"], item["earnings_event_id"]))
    event_dates = sorted({item["available_date"] for item in records})
    outputs: dict[date, dict[str, Any]] = {}
    available_idx = 0
    available_records: list[dict[str, Any]] = []
    for trading_date in dates:
        output = earnings_feature_base()
        as_of = pd.Timestamp(_as_of_after_close(trading_date))
        while available_idx < len(records) and records[available_idx]["available_at"] <= as_of:
            available_records.append(records[available_idx])
            available_idx += 1
        if not available_records:
            outputs[trading_date] = output
            continue

        output["earnings_data_available"] = True
        latest_idx = bisect_right(event_dates, trading_date)
        latest_date = event_dates[latest_idx - 1] if latest_idx > 0 else None
        latest_row = available_records[-1]
        for window in EARNINGS_FEATURE_WINDOWS:
            start_date = _sessions_ago(trading_date, window)
            output[f"earnings_event_present_{window}s"] = any(
                start_date <= item["available_date"] <= trading_date for item in available_records
            )
        window_20_start = _sessions_ago(trading_date, 20)
        window_20 = [item for item in available_records if window_20_start <= item["available_date"] <= trading_date]
        if latest_date is not None:
            output["sessions_since_latest_earnings"] = _trading_sessions_since(latest_date, trading_date)
        output["latest_eps_surprise_percent"] = (
            None if _is_missing(latest_row.get("eps_surprise_percent")) else float(latest_row.get("eps_surprise_percent"))
        )
        output["latest_eps_surprise_direction"] = _direction(latest_row.get("eps_surprise_percent"))
        output["latest_revenue_surprise_percent"] = (
            None
            if _is_missing(latest_row.get("revenue_surprise_percent"))
            else float(latest_row.get("revenue_surprise_percent"))
        )
        output["earnings_timing_known"] = str(latest_row.get("timing") or "unknown") != "unknown"
        output["earnings_event_count_20s"] = int(len(window_20))
        output["earnings_missing_eps_actual_count_20s"] = sum(1 for item in window_20 if _is_missing(item.get("eps_actual")))
        output["earnings_missing_eps_estimate_count_20s"] = sum(1 for item in window_20 if _is_missing(item.get("eps_estimate")))
        output["earnings_missing_revenue_count_20s"] = sum(
            1 for item in window_20 if _is_missing(item.get("revenue_actual")) or _is_missing(item.get("revenue_estimate"))
        )
        output["earnings_latest_provider"] = latest_row.get("provider")
        output["earnings_latest_data_quality_status"] = latest_row.get("data_quality_status")
        output["earnings_latest_warnings"] = latest_row.get("warnings") or "[]"
        output["earnings_available_event_ids"] = [int(item["earnings_event_id"]) for item in available_records]
        outputs[trading_date] = output
    return outputs
