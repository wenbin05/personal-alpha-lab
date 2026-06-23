from __future__ import annotations

from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def calculate_position_size(
    portfolio_size: float,
    risk_per_trade_pct: float,
    entry_price: float,
    stop_loss_price: float,
    max_position_pct: float = 10.0,
) -> dict[str, Any]:
    if portfolio_size <= 0:
        raise ValueError("portfolio_size must be positive.")
    if entry_price <= 0:
        raise ValueError("entry_price must be positive.")
    if stop_loss_price <= 0:
        raise ValueError("stop_loss_price must be positive.")

    risk_per_share = abs(entry_price - stop_loss_price)
    max_dollar_risk = portfolio_size * (risk_per_trade_pct / 100)
    max_position_value = portfolio_size * (max_position_pct / 100)

    if risk_per_share == 0:
        shares_by_risk = 0
    else:
        shares_by_risk = int(max_dollar_risk // risk_per_share)
    shares_by_position_cap = int(max_position_value // entry_price)
    shares_allowed = max(0, min(shares_by_risk, shares_by_position_cap))
    position_value = shares_allowed * entry_price
    actual_dollar_risk = shares_allowed * risk_per_share
    portfolio_pct = 0 if portfolio_size == 0 else position_value / portfolio_size * 100

    warnings: list[str] = []
    if shares_allowed == 0:
        warnings.append("Position size rounds to zero with the current stop and risk budget.")
    if shares_by_risk > shares_by_position_cap:
        warnings.append("Position capped by max position size rather than risk per trade.")
    if portfolio_pct > max_position_pct:
        warnings.append("Position exceeds configured max position percentage.")

    return {
        "max_dollar_risk": round(max_dollar_risk, 2),
        "risk_per_share": round(risk_per_share, 4),
        "shares_allowed": shares_allowed,
        "position_value": round(position_value, 2),
        "actual_dollar_risk": round(actual_dollar_risk, 2),
        "portfolio_pct": round(portfolio_pct, 2),
        "warnings": warnings,
    }


def stop_candidates(features: dict[str, Any]) -> dict[str, float | None]:
    price = features.get("last_price")
    ma20 = features.get("ma_20")
    if price is None:
        return {"below_20d_ma": None, "fixed_7_pct": None}
    fixed = float(price) * 0.93
    below_ma = None if ma20 is None else float(ma20) * 0.99
    return {
        "below_20d_ma": below_ma,
        "fixed_7_pct": fixed,
    }


def risk_reward_score(features: dict[str, Any]) -> tuple[float, list[str], str]:
    """Return a 0-5 score, risk notes, and human label."""
    reasons: list[str] = []
    score = 3.0
    label = "Normal"
    distance_20 = features.get("distance_20d_ma")
    above_200 = features.get("above_200d_ma")
    volatility = features.get("volatility_20d")

    if above_200:
        score += 1.0
        reasons.append("Price is above the 200-day moving average.")
    else:
        score -= 1.0
        reasons.append("Price is below or missing the 200-day moving average.")

    if distance_20 is not None:
        if distance_20 > 0.18:
            score -= 2.0
            label = "Extended"
            reasons.append("Price is extremely extended above the 20-day moving average.")
        elif distance_20 > 0.10:
            score -= 1.0
            label = "Slightly Extended"
            reasons.append("Price is extended above the 20-day moving average.")
        elif -0.05 <= distance_20 <= 0.08:
            score += 1.0
            reasons.append("Price is near a manageable distance from the 20-day moving average.")

    if volatility is not None and volatility > 0.75:
        score -= 1.0
        label = "High Volatility"
        reasons.append("20-day annualized volatility is high.")

    return clamp(score, 0.0, 5.0), reasons, label

