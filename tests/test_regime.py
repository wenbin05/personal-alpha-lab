from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.regime import classify_market_regime


def sample_ohlcv(rows: int = 260, start: float = 100.0, step: float = 0.2) -> pd.DataFrame:
    dates = pd.bdate_range("2024-01-01", periods=rows)
    close = start + np.arange(rows) * step
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(rows, 1_000_000),
        },
        index=dates,
    )


def test_regime_defaults_neutral_with_warning_when_spy_missing() -> None:
    regime = classify_market_regime({})

    assert regime["regime"] == "Neutral"
    assert regime["confidence"] == "low"
    assert regime["warnings"]


def test_regime_treats_missing_vix_as_medium_confidence_not_crash() -> None:
    histories = {
        "SPY": sample_ohlcv(),
        "QQQ": sample_ohlcv(start=100, step=0.4),
        "IWM": sample_ohlcv(start=90, step=0.1),
    }

    regime = classify_market_regime(histories)

    assert regime["regime"] in {"Risk-On", "Neutral", "Choppy", "Risk-Off", "Small-Cap Rotation"}
    assert regime["confidence"] in {"normal", "medium"}
    assert any("VIX" in warning for warning in regime["warnings"])
