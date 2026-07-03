from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, roc_auc_score


def _finite_arrays(y_true: Any, y_pred: Any) -> tuple[np.ndarray, np.ndarray]:
    frame = pd.DataFrame({"y_true": y_true, "y_pred": y_pred}).replace([np.inf, -np.inf], np.nan).dropna()
    return frame["y_true"].to_numpy(dtype=float), frame["y_pred"].to_numpy(dtype=float)


def _corr(y_true: np.ndarray, y_pred: np.ndarray, method: str) -> float | None:
    if len(y_true) < 2 or np.nanstd(y_true) == 0 or np.nanstd(y_pred) == 0:
        return None
    return float(pd.Series(y_true).corr(pd.Series(y_pred), method=method))


def mean_daily_cross_sectional_ic(frame: pd.DataFrame, method: str = "spearman") -> float | None:
    if frame.empty or not {"trading_date", "y_true", "y_pred"}.issubset(frame.columns):
        return None
    values: list[float] = []
    for _, group in frame.dropna(subset=["y_true", "y_pred"]).groupby("trading_date"):
        if len(group) < 2:
            continue
        if group["y_true"].nunique() < 2 or group["y_pred"].nunique() < 2:
            continue
        value = group["y_true"].corr(group["y_pred"], method=method)
        if pd.notna(value):
            values.append(float(value))
    if not values:
        return None
    return float(np.mean(values))


def regression_metrics(
    y_true: Any,
    y_pred: Any,
    train_mean: float,
    prediction_frame: pd.DataFrame | None = None,
) -> dict[str, float | int | None]:
    y_true_arr, y_pred_arr = _finite_arrays(y_true, y_pred)
    if len(y_true_arr) == 0:
        return {"n": 0}
    errors = y_pred_arr - y_true_arr
    baseline_errors = np.full_like(y_true_arr, float(train_mean)) - y_true_arr
    sse_model = float(np.sum(errors**2))
    sse_baseline = float(np.sum(baseline_errors**2))
    directional = np.sign(y_true_arr) == np.sign(y_pred_arr)
    metrics: dict[str, float | int | None] = {
        "n": int(len(y_true_arr)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "oos_r2_vs_train_mean": None if sse_baseline == 0 else float(1 - sse_model / sse_baseline),
        "pearson_correlation": _corr(y_true_arr, y_pred_arr, "pearson"),
        "spearman_ic": _corr(y_true_arr, y_pred_arr, "spearman"),
        "directional_accuracy": float(np.mean(directional)),
    }
    if prediction_frame is not None and not prediction_frame.empty:
        metrics["mean_daily_cross_sectional_ic"] = mean_daily_cross_sectional_ic(prediction_frame, "spearman")
    else:
        metrics["mean_daily_cross_sectional_ic"] = None
    return metrics


def classification_metrics(y_true_binary: Any, y_score: Any, y_pred_label: Any) -> dict[str, float | int | None]:
    frame = pd.DataFrame({"y_true": y_true_binary, "y_score": y_score, "y_pred": y_pred_label}).dropna()
    if frame.empty:
        return {"n": 0}
    y_true = frame["y_true"].astype(int).to_numpy()
    y_pred = frame["y_pred"].astype(int).to_numpy()
    y_score_arr = frame["y_score"].astype(float).to_numpy()
    class_counts = pd.Series(y_true).value_counts().to_dict()
    metrics: dict[str, float | int | None] = {
        "n": int(len(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "directional_accuracy": float(np.mean(y_true == y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)) if len(set(y_true)) > 1 else None,
        "roc_auc": float(roc_auc_score(y_true, y_score_arr)) if len(set(y_true)) > 1 else None,
        "positive_count": int(class_counts.get(1, 0)),
        "negative_count": int(class_counts.get(0, 0)),
    }
    return metrics

