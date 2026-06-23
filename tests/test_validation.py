from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from src.validation.debug import (
    build_validation_report_from_data,
    latest_ohlcv_rows,
    missing_value_counts,
    ohlcv_metadata,
    validation_warnings,
)


def sample_ohlcv(rows: int = 260, start: float = 100.0, step: float = 0.2) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    close = start + np.arange(rows) * step
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "adj_close": close,
            "volume": np.full(rows, 1_000_000.0),
        },
        index=dates,
    )


def test_ohlcv_metadata_and_latest_rows() -> None:
    df = sample_ohlcv(10)
    metadata = ohlcv_metadata(
        df,
        "2y",
        {
            "source": "cache",
            "cache_satisfies_requested_period_before": False,
            "minimum_bars_for_requested_period": 403,
        },
        today=date(2024, 1, 20),
    )
    latest = latest_ohlcv_rows(df)

    assert metadata["requested_period"] == "2y"
    assert metadata["rows"] == 10
    assert metadata["data_source"] == "cache"
    assert metadata["latest_expected_trading_day"] == "2024-01-19"
    assert metadata["calendar_source"]
    assert len(latest) == 5
    assert "Close" in latest.columns


def test_missing_value_counts_include_missing_columns() -> None:
    df = sample_ohlcv(5).drop(columns=["adj_close"])
    df.loc[df.index[-1], "volume"] = np.nan

    counts = missing_value_counts(df)

    assert counts["Adj Close"] == 5
    assert counts["Volume"] == 1


def test_validation_warnings_detect_quality_issues() -> None:
    df = sample_ohlcv(40)
    df.loc[df.index[-1], "volume"] = 0
    df.loc[df.index[-2], "close"] = -1
    features = {
        "has_data": True,
        "ma_200": None,
        "relative_strength_20d": None,
        "relative_strength_60d": None,
    }
    metadata = ohlcv_metadata(
        df,
        "2y",
        {"cached_rows_before": 40, "minimum_bars_for_requested_period": 403},
        today=date(2025, 1, 1),
    )

    warnings = validation_warnings(df, pd.DataFrame(), features, metadata, "2y", today=date(2025, 1, 1))
    warning_names = {warning["name"] for warning in warnings}

    assert "insufficient_history_for_200d_ma" in warning_names
    assert "stale_latest_date" in warning_names
    assert "zero_volume" in warning_names
    assert "suspicious_price_values" in warning_names
    assert "failed_spy_comparison" in warning_names
    assert "cache_period_shorter_than_requested_period" in warning_names
    stale_warning = next(warning for warning in warnings if warning["name"] == "stale_latest_date")
    assert "expected U.S. trading-day bar" in stale_warning["message"]


def test_validation_warnings_detect_missing_required_columns() -> None:
    df = sample_ohlcv(260).drop(columns=["open", "volume"])
    features = {
        "has_data": True,
        "ma_200": 100.0,
        "relative_strength_20d": 0.01,
        "relative_strength_60d": 0.02,
    }
    metadata = ohlcv_metadata(df, "2y", {"cached_rows_before": len(df)}, today=date(2024, 12, 31))

    warnings = validation_warnings(df, sample_ohlcv(260), features, metadata, "2y", today=date(2024, 12, 31))
    warning_names = {warning["name"] for warning in warnings}

    assert "missing_required_ohlcv_columns" in warning_names
    assert "missing_volume" in warning_names


def test_build_validation_report_from_data_contains_score_and_feature_tables() -> None:
    ticker_df = sample_ohlcv()
    spy_df = sample_ohlcv(start=90.0, step=0.1)
    report = build_validation_report_from_data(
        "AAA",
        ticker_df,
        spy_df,
        {"regime": "Risk-On"},
        {"source": "cache", "cached_rows_before": len(ticker_df), "minimum_bars_for_requested_period": 403},
        requested_period="2y",
        today=date(2024, 12, 31),
    )

    assert report["ticker"] == "AAA"
    assert report["metadata"]["rows"] == len(ticker_df)
    assert "score" in report["score_result"]
    assert not report["feature_table"].empty
    assert not report["score_breakdown_table"].empty
