from __future__ import annotations

from typing import Any

import pandas as pd

from src.features.momentum import compute_price_features


def _ratio_return(numerator: pd.DataFrame, denominator: pd.DataFrame, periods: int) -> float | None:
    if numerator is None or denominator is None or numerator.empty or denominator.empty:
        return None
    aligned = pd.concat(
        [numerator["close"].rename("num"), denominator["close"].rename("den")],
        axis=1,
        join="inner",
    ).dropna()
    if len(aligned) <= periods:
        return None
    ratio = aligned["num"] / aligned["den"]
    return float(ratio.pct_change(periods).iloc[-1])


def classify_market_regime(histories: dict[str, pd.DataFrame]) -> dict[str, Any]:
    spy = histories.get("SPY", pd.DataFrame())
    qqq = histories.get("QQQ", pd.DataFrame())
    iwm = histories.get("IWM", pd.DataFrame())
    vix = histories.get("^VIX", pd.DataFrame())

    spy_features = compute_price_features(spy)
    qqq_features = compute_price_features(qqq)
    iwm_features = compute_price_features(iwm)

    qqq_spy_rs_20 = _ratio_return(qqq, spy, 20)
    qqq_spy_rs_60 = _ratio_return(qqq, spy, 60)
    iwm_spy_rs_20 = _ratio_return(iwm, spy, 20)
    iwm_spy_rs_60 = _ratio_return(iwm, spy, 60)

    vix_last = None
    if vix is not None and not vix.empty and "close" in vix.columns:
        try:
            vix_last = float(vix["close"].dropna().iloc[-1])
        except Exception:
            vix_last = None

    spy_above_50 = bool(spy_features.get("above_50d_ma"))
    spy_above_200 = bool(spy_features.get("above_200d_ma"))
    spy_50_above_200 = (
        spy_features.get("ma_50") is not None
        and spy_features.get("ma_200") is not None
        and spy_features["ma_50"] > spy_features["ma_200"]
    )
    qqq_rs_positive = qqq_spy_rs_20 is not None and qqq_spy_rs_20 > 0
    vix_elevated = vix_last is not None and vix_last >= 25
    iwm_rotation = (
        iwm_spy_rs_20 is not None
        and iwm_spy_rs_20 > 0
        and (iwm_spy_rs_60 is None or iwm_spy_rs_20 > iwm_spy_rs_60)
        and bool(iwm_features.get("above_50d_ma"))
    )

    reasons: list[str] = []
    warnings: list[str] = []
    confidence = "normal"

    if not spy_features.get("has_data"):
        warnings.append("SPY data is missing, so market regime cannot be calculated reliably.")
        confidence = "low"
    elif spy_features.get("bars", 0) < 200 or spy_features.get("ma_200") is None:
        warnings.append("SPY has insufficient history for a reliable 200-day trend read.")
        confidence = "low"

    if qqq_spy_rs_20 is None:
        warnings.append("QQQ/SPY relative strength is unavailable.")
        if confidence != "low":
            confidence = "medium"
    if iwm_spy_rs_20 is None:
        warnings.append("IWM/SPY relative strength is unavailable.")
        if confidence != "low":
            confidence = "medium"
    if vix_last is None:
        warnings.append("VIX data is unavailable; volatility is treated as neutral.")
        if confidence == "normal":
            confidence = "medium"
    if spy_above_50:
        reasons.append("SPY is above its 50-day moving average.")
    else:
        reasons.append("SPY is not above its 50-day moving average.")
    if spy_above_200:
        reasons.append("SPY is above its 200-day moving average.")
    else:
        reasons.append("SPY is not above its 200-day moving average.")
    if spy_50_above_200:
        reasons.append("SPY 50-day moving average is above the 200-day moving average.")
    if qqq_rs_positive:
        reasons.append("QQQ/SPY relative strength is positive over 20 trading days.")
    if iwm_rotation:
        reasons.append("IWM is above its 50-day moving average and improving versus SPY.")
    if vix_last is None:
        reasons.append("VIX data is unavailable, so volatility is treated as neutral.")
    elif vix_elevated:
        reasons.append(f"VIX is elevated at {vix_last:.1f}.")
    else:
        reasons.append(f"VIX is not elevated at {vix_last:.1f}.")

    if confidence == "low":
        regime = "Neutral"
        reasons.append("Core regime inputs are unreliable; regime defaults conservatively to Neutral.")
    elif not spy_above_200 or vix_elevated:
        regime = "Risk-Off"
    elif iwm_rotation:
        regime = "Small-Cap Rotation"
    elif spy_above_50 and spy_50_above_200 and qqq_rs_positive and not vix_elevated:
        regime = "Risk-On"
    elif spy_above_50 and spy_above_200:
        regime = "Neutral"
    else:
        regime = "Choppy"

    return {
        "regime": regime,
        "spy_features": spy_features,
        "qqq_features": qqq_features,
        "iwm_features": iwm_features,
        "qqq_spy_rs_20": qqq_spy_rs_20,
        "qqq_spy_rs_60": qqq_spy_rs_60,
        "iwm_spy_rs_20": iwm_spy_rs_20,
        "iwm_spy_rs_60": iwm_spy_rs_60,
        "vix": vix_last,
        "vix_elevated": vix_elevated,
        "confidence": confidence,
        "warnings": warnings,
        "reasons": reasons,
    }
