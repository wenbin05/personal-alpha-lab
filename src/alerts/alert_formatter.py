from __future__ import annotations

from typing import Any


def suggested_watch_action(result: dict[str, Any]) -> str:
    score = float(result.get("score", 0) or 0)
    risk_label = result.get("risk_label", "")
    features = result.get("features", {})
    distance_20 = features.get("distance_20d_ma")

    if risk_label == "Liquidity Too Low":
        return "Liquidity Too Low"
    if score >= 80 and distance_20 is not None and distance_20 > 0.10:
        return "Wait for Pullback"
    if score >= 80:
        return "Strong Watch"
    if score >= 65:
        return "Watch"
    if score < 50:
        return "Avoid"
    return "Watch"


def format_alert(result: dict[str, Any]) -> str:
    ticker = result.get("ticker", "")
    score = result.get("score", 0)
    label = result.get("label", "Neutral")
    features = result.get("features", {})
    reasons = result.get("reasons", [])
    risk = result.get("risk_label", "Normal")
    action = suggested_watch_action(result)

    compact_reasons = []
    if features.get("above_50d_ma") and features.get("above_200d_ma"):
        compact_reasons.append("price above 50D/200D")
    ret_20 = features.get("ret_20d")
    if ret_20 is not None and ret_20 > 0:
        compact_reasons.append("20D return positive")
    volume_ratio = features.get("volume_ratio_20d")
    if volume_ratio is not None:
        compact_reasons.append(f"volume {volume_ratio:.1f}x average")
    rs = features.get("relative_strength_20d")
    if rs is not None and rs > 0:
        compact_reasons.append("relative strength beating SPY")
    if not compact_reasons and reasons:
        compact_reasons = reasons[:3]

    return (
        f"{label.upper()}: {ticker}\n"
        f"Score: {score}/100\n"
        f"Reason: {', '.join(compact_reasons) or 'No strong rule-based drivers detected.'}.\n"
        f"Risk: {risk}.\n"
        f"Suggested action: {action}. Not financial advice."
    )

