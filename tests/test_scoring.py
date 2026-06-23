from __future__ import annotations

from src.scoring.risk_rules import calculate_position_size
from src.scoring.score_engine import score_ticker_from_features


def base_features() -> dict:
    return {
        "has_data": True,
        "data_quality": "ok",
        "last_price": 50.0,
        "ret_20d": 0.12,
        "ret_60d": 0.25,
        "above_50d_ma": True,
        "above_200d_ma": True,
        "relative_strength_20d": 0.05,
        "relative_strength_60d": 0.08,
        "volume_ratio_20d": 1.8,
        "avg_dollar_volume_20d": 50_000_000,
        "avg_dollar_volume_ok": True,
        "liquidity_score_raw": 1.0,
        "liquidity_label": "Acceptable",
        "distance_20d_ma": 0.04,
        "volatility_20d": 0.35,
    }


def test_score_calculation_returns_0_to_100() -> None:
    result = score_ticker_from_features("XYZ", base_features(), {"regime": "Risk-On"})
    assert 0 <= result["score"] <= 100
    assert result["label"] in {"Strong Watch", "Watch", "Neutral", "Weak", "Avoid"}
    assert result["breakdown"]
    assert result["reasons"]


def test_low_liquidity_penalty_works() -> None:
    features = base_features()
    features["avg_dollar_volume_20d"] = 500_000
    features["avg_dollar_volume_ok"] = False
    features["liquidity_score_raw"] = 0.15
    features["liquidity_label"] = "Liquidity Too Low"
    result = score_ticker_from_features("THIN", features, {"regime": "Risk-On"})

    assert any(p["name"] == "low_liquidity" for p in result["penalties"])
    assert result["score"] < 80


def test_low_price_below_200_and_overextension_penalties_work() -> None:
    features = base_features()
    features["last_price"] = 4.0
    features["above_200d_ma"] = False
    features["distance_20d_ma"] = 0.30

    result = score_ticker_from_features("RISKY", features, {"regime": "Risk-On"})
    penalty_names = {penalty["name"] for penalty in result["penalties"]}

    assert "low_price" in penalty_names
    assert "below_200d_ma" in penalty_names
    assert "extreme_overextension" in penalty_names


def test_missing_data_does_not_crash() -> None:
    result = score_ticker_from_features("BAD", {"has_data": False, "last_price": None}, {"regime": "Neutral"})

    assert 0 <= result["score"] <= 100
    assert any(p["name"] == "missing_data" for p in result["penalties"])


def test_risk_position_sizing_works() -> None:
    sizing = calculate_position_size(100_000, 1.0, 50.0, 45.0, 10.0)

    assert sizing["max_dollar_risk"] == 1000
    assert sizing["shares_allowed"] == 200
    assert sizing["position_value"] == 10_000
