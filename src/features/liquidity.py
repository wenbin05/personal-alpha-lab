from __future__ import annotations

from typing import Any


def compute_liquidity_features(
    price_features: dict[str, Any],
    volume_features: dict[str, Any],
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
) -> dict[str, Any]:
    price = price_features.get("last_price")
    adv = volume_features.get("avg_dollar_volume_20d")
    price_ok = price is not None and price >= min_price
    adv_ok = adv is not None and adv >= min_avg_dollar_volume

    if price_ok and adv_ok:
        score = 1.0
        label = "Acceptable"
    elif price_ok and adv is not None and adv >= min_avg_dollar_volume * 0.35:
        score = 0.55
        label = "Thin"
    else:
        score = 0.15
        label = "Liquidity Too Low"

    return {
        "price_ok": bool(price_ok),
        "avg_dollar_volume_ok": bool(adv_ok),
        "liquidity_score_raw": score,
        "liquidity_label": label,
    }

