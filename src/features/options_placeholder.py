from __future__ import annotations

from typing import Any


def get_options_features(ticker: str) -> dict[str, Any]:
    """Neutral options hook for future Tradier, IBKR, Polygon, or similar adapters."""
    return {
        "ticker": ticker.upper(),
        "options_score": 5.0,
        "options_signal": "Neutral placeholder",
        "call_put_volume_ratio": None,
        "iv_rank": None,
        "open_interest_change": None,
        "provider": "placeholder",
    }

