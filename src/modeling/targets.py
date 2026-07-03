from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


RAW_TARGET_1_SESSION = "label_1_session_excess_return"
RAW_TARGET_5_SESSION = "label_5_session_excess_return"
RAW_TARGET_20_SESSION = "label_20_session_excess_return"


@dataclass(frozen=True)
class TargetDefinition:
    name: str
    display_name: str
    base_label_column: str
    horizon_sessions: int
    target_type: str
    task: str
    allowed_models: tuple[str, ...]
    metadata: dict[str, Any]


TARGET_DEFINITIONS: dict[str, TargetDefinition] = {
    RAW_TARGET_1_SESSION: TargetDefinition(
        name=RAW_TARGET_1_SESSION,
        display_name="Raw 1-session SPY excess return",
        base_label_column=RAW_TARGET_1_SESSION,
        horizon_sessions=1,
        target_type="raw_excess_return",
        task="regression",
        allowed_models=("zero_prediction", "train_mean", "ridge_regression"),
        metadata={"transformation": "none"},
    ),
    RAW_TARGET_5_SESSION: TargetDefinition(
        name=RAW_TARGET_5_SESSION,
        display_name="Raw 5-session SPY excess return",
        base_label_column=RAW_TARGET_5_SESSION,
        horizon_sessions=5,
        target_type="raw_excess_return",
        task="regression",
        allowed_models=("zero_prediction", "train_mean", "ridge_regression"),
        metadata={"transformation": "none"},
    ),
    RAW_TARGET_20_SESSION: TargetDefinition(
        name=RAW_TARGET_20_SESSION,
        display_name="Raw 20-session SPY excess return",
        base_label_column=RAW_TARGET_20_SESSION,
        horizon_sessions=20,
        target_type="raw_excess_return",
        task="regression",
        allowed_models=("zero_prediction", "train_mean", "ridge_regression"),
        metadata={"transformation": "none"},
    ),
    "label_5_session_excess_return_winsorized_q01_q99": TargetDefinition(
        name="label_5_session_excess_return_winsorized_q01_q99",
        display_name="Winsorized 5-session SPY excess return",
        base_label_column=RAW_TARGET_5_SESSION,
        horizon_sessions=5,
        target_type="winsorized_excess_return",
        task="regression",
        allowed_models=("zero_prediction", "train_mean", "ridge_regression"),
        metadata={
            "transformation": "clip raw excess return using train-fold quantiles only",
            "lower_quantile": 0.01,
            "upper_quantile": 0.99,
            "fold_local": True,
        },
    ),
    "label_5_session_excess_return_vol_norm_20d": TargetDefinition(
        name="label_5_session_excess_return_vol_norm_20d",
        display_name="Volatility-normalized 5-session SPY excess return",
        base_label_column=RAW_TARGET_5_SESSION,
        horizon_sessions=5,
        target_type="volatility_normalized_excess_return",
        task="regression",
        allowed_models=("zero_prediction", "train_mean", "ridge_regression"),
        metadata={
            "transformation": "raw excess return divided by trailing 20D annualized volatility scaled to horizon",
            "volatility_feature": "volatility_20d",
            "volatility_is_point_in_time": True,
        },
    ),
    "label_1_session_excess_return_vol_norm_20d": TargetDefinition(
        name="label_1_session_excess_return_vol_norm_20d",
        display_name="Volatility-normalized 1-session SPY excess return",
        base_label_column=RAW_TARGET_1_SESSION,
        horizon_sessions=1,
        target_type="volatility_normalized_excess_return",
        task="regression",
        allowed_models=("zero_prediction", "train_mean", "ridge_regression"),
        metadata={
            "transformation": "raw excess return divided by trailing 20D annualized volatility scaled to horizon",
            "volatility_feature": "volatility_20d",
            "volatility_is_point_in_time": True,
        },
    ),
    "label_20_session_excess_return_vol_norm_20d": TargetDefinition(
        name="label_20_session_excess_return_vol_norm_20d",
        display_name="Volatility-normalized 20-session SPY excess return",
        base_label_column=RAW_TARGET_20_SESSION,
        horizon_sessions=20,
        target_type="volatility_normalized_excess_return",
        task="regression",
        allowed_models=("zero_prediction", "train_mean", "ridge_regression"),
        metadata={
            "transformation": "raw excess return divided by trailing 20D annualized volatility scaled to horizon",
            "volatility_feature": "volatility_20d",
            "volatility_is_point_in_time": True,
        },
    ),
    "label_5_session_excess_return_cs_rank_pct": TargetDefinition(
        name="label_5_session_excess_return_cs_rank_pct",
        display_name="Cross-sectional rank percentile of 5-session SPY excess return",
        base_label_column=RAW_TARGET_5_SESSION,
        horizon_sessions=5,
        target_type="cross_sectional_rank_percentile",
        task="regression",
        allowed_models=("zero_prediction", "train_mean", "ridge_regression"),
        metadata={
            "transformation": "rank raw future returns within the same snapshot date only",
            "rank_method": "average",
            "range": "0_to_1",
        },
    ),
    "label_5_session_excess_return_top_bottom_q20": TargetDefinition(
        name="label_5_session_excess_return_top_bottom_q20",
        display_name="Top/bottom quintile 5-session SPY excess return",
        base_label_column=RAW_TARGET_5_SESSION,
        horizon_sessions=5,
        target_type="binary_top_bottom_quantile",
        task="classification",
        allowed_models=("zero_prediction", "train_mean", "logistic_regression"),
        metadata={
            "transformation": "classify same-date top quintile as 1, bottom quintile as 0, middle omitted",
            "lower_quantile": 0.20,
            "upper_quantile": 0.80,
            "middle_bucket": "ignored",
        },
    ),
}


TARGET_ORDER = [
    RAW_TARGET_5_SESSION,
    "label_5_session_excess_return_winsorized_q01_q99",
    "label_5_session_excess_return_vol_norm_20d",
    "label_5_session_excess_return_cs_rank_pct",
    "label_5_session_excess_return_top_bottom_q20",
    "label_1_session_excess_return_vol_norm_20d",
    "label_20_session_excess_return_vol_norm_20d",
    RAW_TARGET_1_SESSION,
    RAW_TARGET_20_SESSION,
]

TARGET_ENGINEERING_TARGETS = [
    "label_5_session_excess_return_winsorized_q01_q99",
    "label_5_session_excess_return_vol_norm_20d",
    "label_5_session_excess_return_cs_rank_pct",
    "label_5_session_excess_return_top_bottom_q20",
]


def get_target_definition(target_name: str) -> TargetDefinition:
    if target_name not in TARGET_DEFINITIONS:
        raise ValueError(f"Unknown target definition: {target_name}")
    return TARGET_DEFINITIONS[target_name]


def target_options() -> list[str]:
    return list(TARGET_ORDER)


def target_metadata(definition: TargetDefinition) -> dict[str, Any]:
    return {
        "target_name": definition.name,
        "display_name": definition.display_name,
        "base_label_column": definition.base_label_column,
        "horizon_sessions": definition.horizon_sessions,
        "target_type": definition.target_type,
        "task": definition.task,
        "allowed_models": list(definition.allowed_models),
        **definition.metadata,
    }


def _trading_dates(metadata: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(metadata["trading_date"]).dt.date


def cross_sectional_rank_target(raw_y: pd.Series, metadata: pd.DataFrame) -> pd.Series:
    frame = pd.DataFrame({"trading_date": _trading_dates(metadata), "target": pd.to_numeric(raw_y, errors="coerce")}, index=raw_y.index)
    ranked = frame.groupby("trading_date")["target"].rank(method="average", pct=True)
    return ranked.astype(float)


def binary_top_bottom_target(
    raw_y: pd.Series,
    metadata: pd.DataFrame,
    lower_quantile: float = 0.20,
    upper_quantile: float = 0.80,
) -> pd.Series:
    frame = pd.DataFrame({"trading_date": _trading_dates(metadata), "target": pd.to_numeric(raw_y, errors="coerce")}, index=raw_y.index)
    output = pd.Series(np.nan, index=raw_y.index, dtype=float)
    for _, group in frame.dropna(subset=["target"]).groupby("trading_date"):
        if len(group) < 3:
            continue
        lower = group["target"].quantile(lower_quantile)
        upper = group["target"].quantile(upper_quantile)
        output.loc[group.index[group["target"].le(lower)]] = 0.0
        output.loc[group.index[group["target"].ge(upper)]] = 1.0
    return output


def volatility_normalized_target(
    raw_y: pd.Series,
    X: pd.DataFrame,
    horizon_sessions: int,
    volatility_column: str = "volatility_20d",
) -> pd.Series:
    if volatility_column not in X.columns:
        return pd.Series(np.nan, index=raw_y.index, dtype=float)
    raw = pd.to_numeric(raw_y, errors="coerce")
    annualized_vol = pd.to_numeric(X[volatility_column], errors="coerce")
    horizon_vol = annualized_vol * math.sqrt(float(horizon_sessions) / 252.0)
    horizon_vol = horizon_vol.where(horizon_vol.gt(0))
    return (raw / horizon_vol).replace([np.inf, -np.inf], np.nan).astype(float)


def precompute_target_series(
    definition: TargetDefinition,
    raw_y: pd.Series,
    X: pd.DataFrame,
    metadata: pd.DataFrame,
) -> pd.Series:
    if definition.target_type in {"raw_excess_return", "winsorized_excess_return"}:
        return pd.to_numeric(raw_y, errors="coerce").astype(float)
    if definition.target_type == "volatility_normalized_excess_return":
        return volatility_normalized_target(raw_y, X, definition.horizon_sessions)
    if definition.target_type == "cross_sectional_rank_percentile":
        return cross_sectional_rank_target(raw_y, metadata)
    if definition.target_type == "binary_top_bottom_quantile":
        return binary_top_bottom_target(
            raw_y,
            metadata,
            lower_quantile=float(definition.metadata.get("lower_quantile", 0.20)),
            upper_quantile=float(definition.metadata.get("upper_quantile", 0.80)),
        )
    raise ValueError(f"Unsupported target type: {definition.target_type}")


def transform_target_for_split(
    definition: TargetDefinition,
    target_series: pd.Series,
    train_mask: pd.Series,
    eval_mask: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, Any]]:
    y_train = pd.to_numeric(target_series.loc[train_mask], errors="coerce").astype(float)
    y_eval = pd.to_numeric(target_series.loc[eval_mask], errors="coerce").astype(float)
    metadata: dict[str, Any] = {}
    if definition.target_type == "winsorized_excess_return":
        clean_train = y_train.dropna()
        if clean_train.empty:
            return y_train, y_eval, {"warning": "No train labels available for winsorization."}
        lower_q = float(definition.metadata.get("lower_quantile", 0.01))
        upper_q = float(definition.metadata.get("upper_quantile", 0.99))
        lower = float(clean_train.quantile(lower_q))
        upper = float(clean_train.quantile(upper_q))
        y_train = y_train.clip(lower=lower, upper=upper)
        y_eval = y_eval.clip(lower=lower, upper=upper)
        metadata = {
            "winsor_lower_quantile": lower_q,
            "winsor_upper_quantile": upper_q,
            "winsor_lower_value": lower,
            "winsor_upper_value": upper,
            "winsor_threshold_source": "train_fold_only",
        }
    return y_train, y_eval, metadata


def target_distribution_row(name: str, values: pd.Series) -> dict[str, Any]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"target": name, "n": 0}
    return {
        "target": name,
        "n": int(len(clean)),
        "mean": float(clean.mean()),
        "median": float(clean.median()),
        "std": float(clean.std(ddof=0)),
        "skew": None if clean.std(ddof=0) == 0 else float(clean.skew()),
        "kurtosis": None if clean.std(ddof=0) == 0 else float(clean.kurtosis()),
        "positive_rate": float(clean.gt(0).mean()),
        "p01": float(clean.quantile(0.01)),
        "p05": float(clean.quantile(0.05)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
    }
