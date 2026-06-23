from __future__ import annotations

from typing import Any

import pandas as pd


def compute_volume_features(df: pd.DataFrame) -> dict[str, Any]:
    if df is None or df.empty or "volume" not in df.columns or "close" not in df.columns:
        return {
            "avg_volume_20d": None,
            "volume_ratio_20d": None,
            "avg_dollar_volume_20d": None,
            "volume_anomaly": False,
        }

    ordered = df.sort_index()
    avg_volume = ordered["volume"].shift(1).rolling(20).mean().iloc[-1]
    last_volume = ordered["volume"].iloc[-1]
    last_price = ordered["close"].iloc[-1]
    volume_ratio = None
    if pd.notna(avg_volume) and avg_volume > 0:
        volume_ratio = float(last_volume / avg_volume)
    avg_dollar_volume = None
    if pd.notna(avg_volume) and pd.notna(last_price):
        avg_dollar_volume = float(avg_volume * last_price)

    return {
        "current_volume": None if pd.isna(last_volume) else float(last_volume),
        "avg_volume_20d": None if pd.isna(avg_volume) else float(avg_volume),
        "volume_ratio_20d": volume_ratio,
        "avg_dollar_volume_20d": avg_dollar_volume,
        "volume_anomaly": bool(volume_ratio is not None and volume_ratio > 1.5),
    }
