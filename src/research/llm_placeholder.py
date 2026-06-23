from __future__ import annotations

import os
from typing import Any


def summarize_ticker_features(ticker: str, features: dict[str, Any], score_result: dict[str, Any]) -> str:
    """Deterministic v1 summary with a future LLM extension point."""
    has_key = bool(os.getenv("OPENAI_API_KEY"))
    label = score_result.get("label", "Unknown")
    score = score_result.get("score", 0)
    reasons = score_result.get("reasons", [])[:5]
    price = features.get("last_price")
    ret_20 = features.get("ret_20d")
    ret_60 = features.get("ret_60d")
    volume_ratio = features.get("volume_ratio_20d")
    liquidity = features.get("liquidity_label", "Unknown")

    summary_parts = [
        f"{ticker.upper()} is currently rated {label} with an alpha score of {score}/100.",
    ]
    if price is not None:
        summary_parts.append(f"Last price is approximately ${price:.2f}.")
    if ret_20 is not None and ret_60 is not None:
        summary_parts.append(f"Momentum: 20D return {ret_20:.1%}, 60D return {ret_60:.1%}.")
    if volume_ratio is not None:
        summary_parts.append(f"Volume is {volume_ratio:.1f}x its 20D average.")
    summary_parts.append(f"Liquidity is classified as {liquidity}.")
    if reasons:
        summary_parts.append("Key drivers: " + " ".join(reasons))
    if has_key:
        summary_parts.append("OPENAI_API_KEY is present, but v1 intentionally uses this rule-based summary hook.")
    else:
        summary_parts.append("No LLM key is configured, so this is a deterministic rule-based summary.")
    summary_parts.append("Research only; not financial advice.")
    return " ".join(summary_parts)

