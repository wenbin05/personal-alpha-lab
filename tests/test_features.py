from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.momentum import compute_price_features
from src.features.momentum import compute_relative_strength
from src.features.momentum import compute_relative_strength_from_prices
from src.features.volume import compute_volume_features


def sample_ohlcv(rows: int = 260, start: float = 50.0, step: float = 0.2) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    close = start + np.arange(rows) * step
    return pd.DataFrame(
        {
            "open": close * 0.99,
            "high": close * 1.01,
            "low": close * 0.98,
            "close": close,
            "adj_close": close,
            "volume": np.full(rows, 1_000_000),
        },
        index=dates,
    )


def test_moving_average_features_work() -> None:
    df = sample_ohlcv()
    features = compute_price_features(df)

    assert features["has_data"] is True
    assert features["ma_50"] is not None
    assert features["ma_200"] is not None
    assert features["above_50d_ma"] is True
    assert features["above_200d_ma"] is True
    assert features["ret_20d"] > 0


def test_volume_features_work() -> None:
    df = sample_ohlcv()
    df.loc[df.index[-1], "volume"] = 2_000_000
    features = compute_volume_features(df)

    assert features["avg_volume_20d"] > 0
    assert features["current_volume"] == 2_000_000
    assert features["volume_ratio_20d"] > 1
    assert features["avg_dollar_volume_20d"] > 10_000_000


def test_feature_calculations_are_exact() -> None:
    dates = pd.bdate_range("2024-01-01", periods=260)
    close = np.arange(1, 261, dtype=float)
    volume = np.full(260, 100.0)
    volume[-1] = 300.0
    df = pd.DataFrame(
        {
            "open": close,
            "high": close,
            "low": close,
            "close": close,
            "adj_close": close,
            "volume": volume,
        },
        index=dates,
    )

    price_features = compute_price_features(df)
    volume_features = compute_volume_features(df)

    assert price_features["ret_20d"] == np.float64(close[-1] / close[-21] - 1)
    assert price_features["ret_60d"] == np.float64(close[-1] / close[-61] - 1)
    assert price_features["ma_50"] == np.mean(close[-50:])
    assert price_features["ma_200"] == np.mean(close[-200:])
    assert volume_features["avg_volume_20d"] == 100.0
    assert volume_features["volume_ratio_20d"] == 3.0
    assert volume_features["avg_dollar_volume_20d"] == 100.0 * close[-1]


def test_relative_strength_vs_spy() -> None:
    rs = compute_relative_strength(
        {"ret_20d": 0.10, "ret_60d": 0.25},
        {"ret_20d": 0.04, "ret_60d": 0.10},
    )

    assert rs["relative_strength_20d"] == pytest.approx(0.06)
    assert rs["relative_strength_60d"] == pytest.approx(0.15)
    assert rs["relative_strength_score_raw"] > 0


def test_relative_strength_uses_aligned_dates() -> None:
    dates = pd.bdate_range("2024-01-01", periods=70)
    ticker_close = np.linspace(100, 170, len(dates))
    spy_close = np.linspace(100, 135, len(dates))
    ticker_df = pd.DataFrame({"close": ticker_close}, index=dates)
    spy_dates = dates.delete([3, 7, 11])
    spy_df = pd.DataFrame({"close": spy_close[: len(spy_dates)]}, index=spy_dates)

    aligned = pd.concat(
        [ticker_df["close"].rename("ticker"), spy_df["close"].rename("spy")],
        axis=1,
        join="inner",
    ).dropna()
    rs = compute_relative_strength_from_prices(ticker_df, spy_df)
    expected_20 = (aligned["ticker"].iloc[-1] / aligned["ticker"].iloc[-21] - 1) - (
        aligned["spy"].iloc[-1] / aligned["spy"].iloc[-21] - 1
    )

    assert rs["relative_strength_20d"] == pytest.approx(expected_20)
