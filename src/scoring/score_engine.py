from __future__ import annotations

from typing import Any

import numpy as np

from src.features.catalyst import get_catalyst_features
from src.features.liquidity import compute_liquidity_features
from src.features.momentum import compute_price_features, compute_relative_strength_from_prices
from src.features.options_placeholder import get_options_features
from src.features.volume import compute_volume_features
from src.scoring.risk_rules import clamp, risk_reward_score


WEIGHTS = {
    "market_regime": 15,
    "momentum_trend": 20,
    "relative_strength": 15,
    "volume_anomaly": 10,
    "liquidity_quality": 15,
    "catalyst": 10,
    "options": 10,
    "risk_reward": 5,
}


def score_label(score: float) -> str:
    if score >= 80:
        return "Strong Watch"
    if score >= 65:
        return "Watch"
    if score >= 50:
        return "Neutral"
    if score >= 35:
        return "Weak"
    return "Avoid"


def _num(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or np.isnan(value):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def market_regime_component(regime: str) -> tuple[float, str]:
    if regime == "Risk-On":
        return 15.0, "Risk-on tape supports long momentum ideas."
    if regime == "Small-Cap Rotation":
        return 13.0, "Small-cap rotation regime supports selective risk appetite."
    if regime == "Neutral":
        return 9.0, "Neutral tape allows selective setups but deserves sizing discipline."
    if regime == "Choppy":
        return 6.0, "Choppy tape reduces conviction for breakouts."
    return 3.0, "Risk-off tape is a headwind for long alpha ideas."


def momentum_component(features: dict[str, Any]) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    ret_20 = features.get("ret_20d")
    ret_60 = features.get("ret_60d")

    if features.get("above_50d_ma"):
        score += 5
        reasons.append("Price is above the 50-day moving average.")
    if features.get("above_200d_ma"):
        score += 5
        reasons.append("Price is above the 200-day moving average.")
    if ret_20 is not None:
        if ret_20 > 0.10:
            score += 5
            reasons.append("20-day return is strong.")
        elif ret_20 > 0:
            score += 3
            reasons.append("20-day return is positive.")
        elif ret_20 < -0.10:
            reasons.append("20-day return is sharply negative.")
    if ret_60 is not None:
        if ret_60 > 0.20:
            score += 5
            reasons.append("60-day return shows strong intermediate momentum.")
        elif ret_60 > 0:
            score += 3
            reasons.append("60-day return is positive.")
        elif ret_60 < -0.15:
            reasons.append("60-day return is weak.")
    return clamp(score, 0, WEIGHTS["momentum_trend"]), reasons


def relative_strength_component(rs_features: dict[str, Any]) -> tuple[float, list[str]]:
    rs_20 = rs_features.get("relative_strength_20d")
    rs_60 = rs_features.get("relative_strength_60d")
    reasons: list[str] = []
    score = 7.5
    if rs_20 is not None:
        score += clamp(rs_20 / 0.10 * 4.5, -4.5, 4.5)
        if rs_20 > 0:
            reasons.append("20-day relative strength is beating SPY.")
        else:
            reasons.append("20-day relative strength is lagging SPY.")
    if rs_60 is not None:
        score += clamp(rs_60 / 0.15 * 3.0, -3.0, 3.0)
        if rs_60 > 0:
            reasons.append("60-day relative strength is beating SPY.")
    return clamp(score, 0, WEIGHTS["relative_strength"]), reasons


def volume_component(volume_features: dict[str, Any]) -> tuple[float, list[str]]:
    ratio = volume_features.get("volume_ratio_20d")
    if ratio is None:
        return 3.0, ["Volume ratio is unavailable."]
    score = clamp((ratio - 0.75) / 1.25 * 10, 0, 10)
    if ratio >= 1.5:
        reason = f"Volume is elevated at {ratio:.1f}x the 20-day average."
    elif ratio >= 1.0:
        reason = f"Volume is normal to mildly supportive at {ratio:.1f}x average."
    else:
        reason = f"Volume is below average at {ratio:.1f}x."
    return score, [reason]


def liquidity_component(liquidity_features: dict[str, Any]) -> tuple[float, list[str]]:
    raw = float(liquidity_features.get("liquidity_score_raw", 0.15))
    score = raw * WEIGHTS["liquidity_quality"]
    label = liquidity_features.get("liquidity_label", "Unknown")
    return score, [f"Liquidity label: {label}."]


def apply_penalties(features: dict[str, Any], liquidity_features: dict[str, Any]) -> list[dict[str, Any]]:
    penalties: list[dict[str, Any]] = []
    price = features.get("last_price")
    adv = features.get("avg_dollar_volume_20d")

    if not features.get("has_data"):
        penalties.append({"name": "missing_data", "amount": -20, "reason": "Missing or unusable market data."})
    elif features.get("data_quality") in {"limited_history", "not_enough_history"}:
        penalties.append({"name": "limited_history", "amount": -5, "reason": "Historical data is limited."})

    if price is not None and price < 5:
        penalties.append({"name": "low_price", "amount": -10, "reason": "Price is below $5."})

    if adv is not None and adv < 10_000_000:
        penalties.append({"name": "low_liquidity", "amount": -20, "reason": "Average daily dollar volume is below $10M."})
    elif not liquidity_features.get("avg_dollar_volume_ok", False):
        penalties.append({"name": "liquidity_unknown", "amount": -8, "reason": "Average dollar volume is missing or insufficient."})

    if features.get("above_200d_ma") is False:
        penalties.append({"name": "below_200d_ma", "amount": -10, "reason": "Price is below the 200-day moving average."})

    distance_20 = features.get("distance_20d_ma")
    if distance_20 is not None:
        if distance_20 > 0.25:
            penalties.append({"name": "extreme_overextension", "amount": -15, "reason": "Price is more than 25% above the 20-day moving average."})
        elif distance_20 > 0.15:
            penalties.append({"name": "overextension", "amount": -8, "reason": "Price is extended above the 20-day moving average."})
        elif distance_20 > 0.10:
            penalties.append({"name": "mild_overextension", "amount": -5, "reason": "Price is mildly extended above the 20-day moving average."})

    return penalties


def score_ticker_from_features(
    ticker: str,
    features: dict[str, Any],
    regime: dict[str, Any],
    catalyst_features: dict[str, Any] | None = None,
    options_features: dict[str, Any] | None = None,
) -> dict[str, Any]:
    catalyst_features = catalyst_features or {
        "catalyst_score": 0,
        "catalyst_penalty": 0,
        "catalyst_note": "",
        "catalyst_warnings": [],
    }
    options_features = options_features or get_options_features(ticker)

    breakdown: dict[str, float] = {}
    reasons: list[str] = []

    regime_score, regime_reason = market_regime_component(regime.get("regime", "Neutral"))
    breakdown["market_regime"] = regime_score
    reasons.append(regime_reason)

    momentum_score, momentum_reasons = momentum_component(features)
    breakdown["momentum_trend"] = momentum_score
    reasons.extend(momentum_reasons)

    rs_score, rs_reasons = relative_strength_component(features)
    breakdown["relative_strength"] = rs_score
    reasons.extend(rs_reasons)

    volume_score, volume_reasons = volume_component(features)
    breakdown["volume_anomaly"] = volume_score
    reasons.extend(volume_reasons)

    liquidity_score, liquidity_reasons = liquidity_component(features)
    breakdown["liquidity_quality"] = liquidity_score
    reasons.extend(liquidity_reasons)

    catalyst_score = clamp(_num(catalyst_features.get("catalyst_score")), 0, 10)
    breakdown["catalyst"] = catalyst_score
    if catalyst_features.get("catalyst_reasons"):
        reasons.extend(catalyst_features["catalyst_reasons"])
    elif catalyst_features.get("has_manual_catalyst"):
        reasons.append(f"Manual catalyst note: {catalyst_features.get('catalyst_note')}")
    elif catalyst_features.get("has_catalyst"):
        reasons.append("Catalyst events are present but neutral/unknown for scoring.")
    else:
        reasons.append("No catalyst events found; catalyst contribution is neutral.")

    options_score = clamp(_num(options_features.get("options_score"), 5.0), 0, 10)
    breakdown["options"] = options_score
    reasons.append(options_features.get("options_signal", "Options data is neutral placeholder."))

    risk_score, risk_reasons, risk_label = risk_reward_score(features)
    breakdown["risk_reward"] = risk_score
    reasons.extend(risk_reasons)

    liquidity_features = {
        "avg_dollar_volume_ok": features.get("avg_dollar_volume_ok", False),
        "liquidity_label": features.get("liquidity_label", "Unknown"),
    }
    penalties = apply_penalties(features, liquidity_features)
    catalyst_penalty = _num(catalyst_features.get("catalyst_penalty"), 0.0)
    if catalyst_penalty < 0:
        penalties.append(
            {
                "name": "negative_catalyst",
                "amount": round(catalyst_penalty, 2),
                "reason": "Recent negative catalyst data reduced the score.",
            }
        )
    raw_score = sum(breakdown.values())
    penalty_total = sum(penalty["amount"] for penalty in penalties)
    final_score = round(clamp(raw_score + penalty_total, 0, 100), 1)

    return {
        "ticker": ticker.upper(),
        "score": final_score,
        "label": score_label(final_score),
        "risk_label": risk_label if features.get("liquidity_label") != "Liquidity Too Low" else "Liquidity Too Low",
        "breakdown": {key: round(value, 2) for key, value in breakdown.items()},
        "penalties": penalties,
        "reasons": reasons,
        "features": features,
        "catalyst_features": catalyst_features,
    }


def build_feature_set(
    ticker: str,
    ticker_df,
    spy_df,
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
) -> dict[str, Any]:
    price_features = compute_price_features(ticker_df)
    volume_features = compute_volume_features(ticker_df)
    liquidity_features = compute_liquidity_features(
        price_features,
        volume_features,
        min_price=min_price,
        min_avg_dollar_volume=min_avg_dollar_volume,
    )
    rs_features = compute_relative_strength_from_prices(ticker_df, spy_df)
    return {
        "ticker": ticker.upper(),
        **price_features,
        **volume_features,
        **liquidity_features,
        **rs_features,
    }


def score_ticker(
    ticker: str,
    ticker_df,
    spy_df,
    regime: dict[str, Any],
    manual_catalysts=None,
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
) -> dict[str, Any]:
    features = build_feature_set(ticker, ticker_df, spy_df, min_price, min_avg_dollar_volume)
    catalyst_features = get_catalyst_features(ticker, manual_catalysts)
    options_features = get_options_features(ticker)
    return score_ticker_from_features(ticker, features, regime, catalyst_features, options_features)


def flatten_score_result(result: dict[str, Any], company_name: str = "") -> dict[str, Any]:
    features = result.get("features", {})
    reasons = result.get("reasons", [])
    return {
        "ticker": result.get("ticker"),
        "company_name": company_name,
        "last_price": features.get("last_price"),
        "20d_return": features.get("ret_20d"),
        "60d_return": features.get("ret_60d"),
        "above_50d_ma": features.get("above_50d_ma"),
        "above_200d_ma": features.get("above_200d_ma"),
        "relative_strength_vs_spy": features.get("relative_strength_20d"),
        "volume_ratio_vs_20d": features.get("volume_ratio_20d"),
        "avg_daily_dollar_volume": features.get("avg_dollar_volume_20d"),
        "catalyst_score": result.get("breakdown", {}).get("catalyst"),
        "alpha_score": result.get("score"),
        "score": result.get("score"),
        "label": result.get("label"),
        "risk_label": result.get("risk_label"),
        "reasons": "; ".join(reasons[:4]),
        "full_result": result,
    }
