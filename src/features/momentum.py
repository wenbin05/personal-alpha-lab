from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _pct_change(series: pd.Series, periods: int) -> float | None:
    if len(series.dropna()) <= periods:
        return None
    value = series.pct_change(periods).iloc[-1]
    return _safe_float(value)


def add_technical_columns(df: pd.DataFrame) -> pd.DataFrame:
    enriched = df.copy()
    if enriched.empty:
        return enriched
    enriched = enriched.sort_index()
    enriched["daily_return"] = enriched["close"].pct_change()
    enriched["ret_5d"] = enriched["close"].pct_change(5)
    enriched["ret_20d"] = enriched["close"].pct_change(20)
    enriched["ret_60d"] = enriched["close"].pct_change(60)
    enriched["ret_120d"] = enriched["close"].pct_change(120)
    enriched["volatility_20d"] = enriched["daily_return"].rolling(20).std() * math.sqrt(252)
    enriched["ma_20"] = enriched["close"].rolling(20).mean()
    enriched["ma_50"] = enriched["close"].rolling(50).mean()
    enriched["ma_200"] = enriched["close"].rolling(200).mean()
    enriched["distance_20d_ma"] = enriched["close"] / enriched["ma_20"] - 1
    enriched["distance_50d_ma"] = enriched["close"] / enriched["ma_50"] - 1
    enriched["above_50d_ma"] = enriched["close"] > enriched["ma_50"]
    enriched["above_200d_ma"] = enriched["close"] > enriched["ma_200"]
    return enriched


def compute_price_features(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty or "close" not in df.columns:
        return {
            "has_data": False,
            "data_quality": "missing",
            "last_price": None,
        }

    enriched = add_technical_columns(df)
    close = enriched["close"].dropna()
    if close.empty:
        return {"has_data": False, "data_quality": "missing_close", "last_price": None}

    last = enriched.iloc[-1]
    features: dict[str, Any] = {
        "has_data": True,
        "data_quality": "ok" if len(enriched) >= 200 else "limited_history",
        "bars": int(len(enriched)),
        "last_price": _safe_float(last.get("close")),
        "daily_return": _safe_float(last.get("daily_return")),
        "ret_5d": _pct_change(close, 5),
        "ret_20d": _pct_change(close, 20),
        "ret_60d": _pct_change(close, 60),
        "ret_120d": _pct_change(close, 120),
        "volatility_20d": _safe_float(last.get("volatility_20d")),
        "ma_20": _safe_float(last.get("ma_20")),
        "ma_50": _safe_float(last.get("ma_50")),
        "ma_200": _safe_float(last.get("ma_200")),
        "distance_20d_ma": _safe_float(last.get("distance_20d_ma")),
        "distance_50d_ma": _safe_float(last.get("distance_50d_ma")),
        "above_50d_ma": bool(last.get("above_50d_ma")) if not pd.isna(last.get("above_50d_ma")) else False,
        "above_200d_ma": bool(last.get("above_200d_ma")) if not pd.isna(last.get("above_200d_ma")) else False,
    }
    if len(enriched) < 50:
        features["data_quality"] = "not_enough_history"
    return features


def compute_relative_strength(ticker_features: dict, spy_features: dict) -> dict[str, float | None]:
    ret_20 = ticker_features.get("ret_20d")
    ret_60 = ticker_features.get("ret_60d")
    spy_20 = spy_features.get("ret_20d")
    spy_60 = spy_features.get("ret_60d")
    rs_20 = None if ret_20 is None or spy_20 is None else float(ret_20) - float(spy_20)
    rs_60 = None if ret_60 is None or spy_60 is None else float(ret_60) - float(spy_60)
    score = None
    if rs_20 is not None and rs_60 is not None:
        score = float(np.clip((rs_20 * 1.5 + rs_60) / 0.20, -1, 1))
    return {
        "relative_strength_20d": rs_20,
        "relative_strength_60d": rs_60,
        "relative_strength_score_raw": score,
    }


def compute_relative_strength_from_prices(
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
) -> dict[str, float | None]:
    """Compare ticker returns against SPY over the same aligned trading-date windows."""
    if (
        ticker_df is None
        or spy_df is None
        or ticker_df.empty
        or spy_df.empty
        or "close" not in ticker_df.columns
        or "close" not in spy_df.columns
    ):
        return {
            "relative_strength_20d": None,
            "relative_strength_60d": None,
            "relative_strength_score_raw": None,
        }

    aligned = pd.concat(
        [
            ticker_df.sort_index()["close"].rename("ticker"),
            spy_df.sort_index()["close"].rename("spy"),
        ],
        axis=1,
        join="inner",
    ).dropna()

    def aligned_rs(periods: int) -> float | None:
        if len(aligned) <= periods:
            return None
        ticker_ret = aligned["ticker"].iloc[-1] / aligned["ticker"].iloc[-periods - 1] - 1
        spy_ret = aligned["spy"].iloc[-1] / aligned["spy"].iloc[-periods - 1] - 1
        return float(ticker_ret - spy_ret)

    rs_20 = aligned_rs(20)
    rs_60 = aligned_rs(60)
    score = None
    if rs_20 is not None and rs_60 is not None:
        score = float(np.clip((rs_20 * 1.5 + rs_60) / 0.20, -1, 1))
    return {
        "relative_strength_20d": rs_20,
        "relative_strength_60d": rs_60,
        "relative_strength_score_raw": score,
    }
