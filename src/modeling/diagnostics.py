from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import storage
from src.datasets.repository import flatten_saved_dataset
from src.datasets.training_loader import load_training_dataset
from src.modeling.feature_sets import FEATURE_SET_NAMES, select_feature_columns
from src.modeling.runner import MODEL_NAMES, TARGET_COLUMNS


DIAGNOSTICS_VERSION = "model_diagnostics_v1"


@dataclass(frozen=True)
class DiagnosticArtifactInfo:
    path: Path
    dataset_id: int | None
    created_at: str
    summary: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if pd.isna(value) if value is not None and not isinstance(value, (str, bytes, list, dict, tuple)) else False:
        return None
    return value


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        result = float(value)
    except Exception:
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def _safe_int(value: Any) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _target_columns(frame: pd.DataFrame) -> list[str]:
    preferred = list(TARGET_COLUMNS.values())
    return [column for column in preferred if column in frame.columns]


def _feature_group(column: str) -> str:
    lowered = column.lower()
    if lowered.startswith("sec_"):
        return "sec"
    if lowered.startswith("earnings_") or lowered.startswith("latest_eps_") or lowered.startswith("latest_revenue_") or lowered == "sessions_since_latest_earnings":
        return "earnings"
    if lowered.startswith("llm_") or lowered.startswith("published_llm_") or "catalyst" in lowered:
        return "catalyst_llm"
    if lowered.startswith("regime_") or lowered == "market_regime" or lowered == "market_regime_confidence":
        return "regime"
    if "quality" in lowered or "missing" in lowered or "stale" in lowered or "warning" in lowered:
        return "data_quality"
    return "technical"


def _nonzero_rate(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float((numeric.fillna(0) != 0).mean())
    non_empty = series.fillna("").astype(str).str.strip().ne("")
    return float(non_empty.mean())


def _latest_completed_model_runs(db_path: str | Path, dataset_id: int) -> pd.DataFrame:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        frame = pd.read_sql_query(
            """
            SELECT *
            FROM model_runs
            WHERE dataset_id = ? AND status = 'completed'
            ORDER BY model_run_id DESC
            """,
            conn,
            params=(int(dataset_id),),
        )
    if frame.empty:
        return frame
    return frame.drop_duplicates(["target_column", "feature_set_name", "model_name"], keep="first").reset_index(drop=True)


def _fold_metrics_frame(db_path: str | Path, model_run_ids: list[int]) -> pd.DataFrame:
    if not model_run_ids:
        return pd.DataFrame()
    placeholders = ",".join("?" for _ in model_run_ids)
    with storage.connect(db_path) as conn:
        frame = pd.read_sql_query(
            f"""
            SELECT mr.target_column, mr.target_horizon, mr.feature_set_name, mr.model_name,
                   mr.task, mfm.*
            FROM model_fold_metrics mfm
            JOIN model_runs mr ON mr.model_run_id = mfm.model_run_id
            WHERE mfm.model_run_id IN ({placeholders})
            ORDER BY mr.target_horizon, mr.feature_set_name, mr.model_name, mfm.metric_id
            """,
            conn,
            params=tuple(int(value) for value in model_run_ids),
        )
    if frame.empty:
        return frame
    metrics = frame["metrics_json"].map(lambda value: _json_loads(value, {}))
    metric_frame = pd.json_normalize(metrics).add_prefix("metric_")
    return pd.concat([frame.drop(columns=["metrics_json"]), metric_frame], axis=1)


def _predictions_for_run(db_path: str | Path, model_run_id: int) -> pd.DataFrame:
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM model_predictions
            WHERE model_run_id = ?
            ORDER BY split_name, fold_name, snapshot_date, ticker
            """,
            conn,
            params=(int(model_run_id),),
        )


def target_distribution(frame: pd.DataFrame, target_columns: list[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in target_columns or _target_columns(frame):
        values = pd.to_numeric(frame[column], errors="coerce").dropna()
        if values.empty:
            rows.append({"target": column, "n": 0})
            continue
        positive = values.gt(0)
        abs_values = values.abs()
        top_1pct_cutoff = abs_values.quantile(0.99)
        top_1pct = values[abs_values.ge(top_1pct_cutoff)]
        rows.append(
            {
                "target": column,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "skew": _safe_float(values.skew()),
                "kurtosis": _safe_float(values.kurtosis()),
                "positive_rate": float(positive.mean()),
                "negative_rate": float(values.lt(0).mean()),
                "zero_rate": float(values.eq(0).mean()),
                "p01": float(values.quantile(0.01)),
                "p05": float(values.quantile(0.05)),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "outlier_abs_3sigma_count": int(abs_values.gt(abs_values.mean() + 3 * abs_values.std(ddof=0)).sum()),
                "top_1pct_abs_mean": _safe_float(top_1pct.abs().mean()),
            }
        )
    return rows


def cross_sectional_dispersion(frame: pd.DataFrame, target_columns: list[str] | None = None) -> list[dict[str, Any]]:
    if "trading_date" not in frame.columns:
        return []
    rows: list[dict[str, Any]] = []
    for column in target_columns or _target_columns(frame):
        if column not in frame.columns:
            continue
        grouped = (
            frame[["trading_date", column]]
            .assign(value=lambda item: pd.to_numeric(item[column], errors="coerce"))
            .dropna(subset=["value"])
            .groupby("trading_date")["value"]
        )
        daily_std = grouped.std(ddof=0).dropna()
        daily_range = grouped.apply(lambda series: series.max() - series.min()).dropna()
        rows.append(
            {
                "target": column,
                "dates": int(daily_std.shape[0]),
                "mean_daily_std": _safe_float(daily_std.mean()),
                "median_daily_std": _safe_float(daily_std.median()),
                "p90_daily_std": _safe_float(daily_std.quantile(0.90)) if not daily_std.empty else None,
                "mean_daily_range": _safe_float(daily_range.mean()),
            }
        )
    return rows


def feature_diagnostics(X: pd.DataFrame) -> dict[str, Any]:
    group_rows: list[dict[str, Any]] = []
    detail_rows: list[dict[str, Any]] = []
    for column in X.columns:
        series = X[column]
        missing_rate = float(series.isna().mean())
        non_null = series.dropna()
        unique_count = int(non_null.nunique(dropna=True))
        dominant_rate = None
        if not non_null.empty:
            dominant_rate = float(non_null.astype(str).value_counts(normalize=True, dropna=False).iloc[0])
        detail_rows.append(
            {
                "feature": column,
                "group": _feature_group(column),
                "missing_rate": missing_rate,
                "unique_count": unique_count,
                "dominant_rate": dominant_rate,
                "nonzero_rate": _nonzero_rate(series),
            }
        )
    detail = pd.DataFrame(detail_rows)
    for group, group_frame in detail.groupby("group", dropna=False):
        group_rows.append(
            {
                "group": str(group),
                "feature_count": int(len(group_frame)),
                "mean_missing_rate": float(group_frame["missing_rate"].mean()),
                "max_missing_rate": float(group_frame["missing_rate"].max()),
                "mean_nonzero_rate": float(group_frame["nonzero_rate"].mean()),
                "near_constant_count": int(group_frame["dominant_rate"].fillna(0).ge(0.995).sum()),
            }
        )

    numeric = X.apply(lambda series: pd.to_numeric(series, errors="coerce").astype(float))
    numeric = numeric.loc[:, numeric.notna().any(axis=0)]
    scale_rows: list[dict[str, Any]] = []
    for column in numeric.columns:
        values = numeric[column].dropna()
        if values.empty:
            continue
        std = values.std(ddof=0)
        scale_rows.append(
            {
                "feature": column,
                "group": _feature_group(column),
                "mean": _safe_float(values.mean()),
                "std": _safe_float(std),
                "p01": _safe_float(values.quantile(0.01)),
                "p99": _safe_float(values.quantile(0.99)),
                "abs_z_gt_5_rate": 0.0 if std == 0 or pd.isna(std) else float(((values - values.mean()).abs() / std).gt(5).mean()),
            }
        )

    corr_pairs: list[dict[str, Any]] = []
    varying = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if varying.shape[1] > 1:
        corr = varying.corr().abs()
        columns = corr.columns.tolist()
        for i, left in enumerate(columns):
            for right in columns[i + 1 :]:
                value = corr.loc[left, right]
                if pd.notna(value) and float(value) >= 0.98:
                    corr_pairs.append({"feature_a": left, "feature_b": right, "abs_corr": float(value)})
        corr_pairs = sorted(corr_pairs, key=lambda item: item["abs_corr"], reverse=True)[:50]

    return {
        "group_missingness": sorted(group_rows, key=lambda item: item["group"]),
        "near_constant_features": detail[detail["dominant_rate"].fillna(0).ge(0.995)]
        .sort_values(["group", "feature"])
        .head(100)
        .to_dict("records"),
        "high_missing_features": detail[detail["missing_rate"].gt(0.20)]
        .sort_values("missing_rate", ascending=False)
        .head(100)
        .to_dict("records"),
        "highly_correlated_pairs": corr_pairs,
        "scale_outliers": pd.DataFrame(scale_rows)
        .sort_values("abs_z_gt_5_rate", ascending=False)
        .head(100)
        .to_dict("records")
        if scale_rows
        else [],
    }


def coverage_by_ticker(frame: pd.DataFrame, feature_columns: list[str]) -> list[dict[str, Any]]:
    if "ticker" not in frame.columns:
        return []
    sec_activity_columns = [
        column
        for column in feature_columns
        if column.startswith("sec_")
        and (
            "_event_days_" in column
            or "_present_" in column
            or column.startswith("sec_recent_")
        )
    ]
    earnings_activity_columns = [
        column
        for column in feature_columns
        if column.startswith("earnings_event_present_")
        or column in {"earnings_timing_known", "latest_eps_surprise_percent", "sessions_since_latest_earnings"}
    ]
    rows: list[dict[str, Any]] = []
    for ticker, group in frame.groupby("ticker", dropna=False):
        row = {"ticker": str(ticker), "rows": int(len(group))}
        if sec_activity_columns:
            sec_numeric = group[sec_activity_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
            row["sec_any_activity_rate"] = float(sec_numeric.ne(0).any(axis=1).mean())
        if "sec_metadata_available" in group.columns:
            sec_available = pd.to_numeric(group["sec_metadata_available"], errors="coerce").fillna(0)
            row["sec_metadata_available_rate"] = float(sec_available.ne(0).mean())
        if earnings_activity_columns:
            earnings_numeric = group[earnings_activity_columns].apply(pd.to_numeric, errors="coerce").fillna(0)
            row["earnings_any_activity_rate"] = float(earnings_numeric.ne(0).any(axis=1).mean())
        if "earnings_data_available" in group.columns:
            earnings_available = pd.to_numeric(group["earnings_data_available"], errors="coerce").fillna(0)
            row["earnings_data_available_rate"] = float(earnings_available.ne(0).mean())
        rows.append(row)
    return sorted(rows, key=lambda item: item["ticker"])


def per_ticker_target_diagnostics(frame: pd.DataFrame, target_column: str) -> list[dict[str, Any]]:
    if target_column not in frame.columns or "ticker" not in frame.columns:
        return []
    values = frame[["ticker", target_column]].assign(target=lambda item: pd.to_numeric(item[target_column], errors="coerce"))
    rows: list[dict[str, Any]] = []
    for ticker, group in values.groupby("ticker"):
        target = group["target"].dropna()
        rows.append(
            {
                "ticker": ticker,
                "rows": int(len(group)),
                "label_count": int(len(target)),
                "label_available_rate": float(len(target) / len(group)) if len(group) else 0.0,
                "target_mean": _safe_float(target.mean()) if not target.empty else None,
                "target_std": _safe_float(target.std(ddof=0)) if not target.empty else None,
                "positive_rate": _safe_float(target.gt(0).mean()) if not target.empty else None,
            }
        )
    return sorted(rows, key=lambda item: item["ticker"])


def per_ticker_prediction_error(predictions: pd.DataFrame) -> list[dict[str, Any]]:
    if predictions.empty or not {"ticker", "y_true", "y_pred"}.issubset(predictions.columns):
        return []
    frame = predictions.copy()
    frame["y_true"] = pd.to_numeric(frame["y_true"], errors="coerce")
    frame["y_pred"] = pd.to_numeric(frame["y_pred"], errors="coerce")
    frame = frame.dropna(subset=["y_true", "y_pred"])
    frame["error"] = frame["y_pred"] - frame["y_true"]
    frame["abs_error"] = frame["error"].abs()
    rows: list[dict[str, Any]] = []
    for ticker, group in frame.groupby("ticker"):
        rows.append(
            {
                "ticker": ticker,
                "prediction_rows": int(len(group)),
                "mae": float(group["abs_error"].mean()),
                "rmse": float(np.sqrt(np.mean(group["error"] ** 2))),
                "bias": float(group["error"].mean()),
                "target_std": _safe_float(group["y_true"].std(ddof=0)),
                "directional_accuracy": float((np.sign(group["y_true"]) == np.sign(group["y_pred"])).mean()),
            }
        )
    return sorted(rows, key=lambda item: item["rmse"], reverse=True)


def split_distribution(frame: pd.DataFrame, target_column: str, fold_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    if target_column not in frame.columns or "trading_date" not in frame.columns or fold_metrics.empty:
        return []
    data = frame[["trading_date", target_column]].copy()
    data["date"] = pd.to_datetime(data["trading_date"]).dt.date
    data["target"] = pd.to_numeric(data[target_column], errors="coerce")
    rows: list[dict[str, Any]] = []

    final_rows = fold_metrics[fold_metrics["fold_name"].eq("final_test")]
    if not final_rows.empty:
        final = final_rows.iloc[0]
        periods = [
            ("final_train", final.get("train_start_date"), final.get("train_end_date")),
            ("final_test", final.get("eval_start_date"), final.get("eval_end_date")),
        ]
    else:
        periods = []
    validation = fold_metrics[fold_metrics["split_name"].eq("validation")]
    if not validation.empty:
        periods.append(("validation_eval_all", validation["eval_start_date"].min(), validation["eval_end_date"].max()))

    for name, start, end in periods:
        if pd.isna(start) or pd.isna(end):
            continue
        start_date = pd.to_datetime(start).date()
        end_date = pd.to_datetime(end).date()
        values = data[data["date"].between(start_date, end_date)]["target"].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "period": name,
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "n": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": float(values.std(ddof=0)),
                "positive_rate": float(values.gt(0).mean()),
                "p05": float(values.quantile(0.05)),
                "p95": float(values.quantile(0.95)),
            }
        )
    return rows


def regime_diagnostics(frame: pd.DataFrame, target_column: str, predictions: pd.DataFrame | None = None) -> dict[str, Any]:
    if "market_regime" not in frame.columns or target_column not in frame.columns:
        return {"target_by_regime": [], "prediction_error_by_regime": []}
    rows: list[dict[str, Any]] = []
    data = frame[["snapshot_id", "market_regime", target_column]].copy()
    data["target"] = pd.to_numeric(data[target_column], errors="coerce")
    for regime, group in data.dropna(subset=["target"]).groupby("market_regime", dropna=False):
        target = group["target"]
        rows.append(
            {
                "market_regime": str(regime),
                "n": int(len(target)),
                "mean": float(target.mean()),
                "std": float(target.std(ddof=0)),
                "positive_rate": float(target.gt(0).mean()),
            }
        )

    error_rows: list[dict[str, Any]] = []
    if predictions is not None and not predictions.empty and "snapshot_id" in predictions.columns:
        merged = predictions.merge(data[["snapshot_id", "market_regime"]], on="snapshot_id", how="left")
        merged["y_true"] = pd.to_numeric(merged["y_true"], errors="coerce")
        merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="coerce")
        merged = merged.dropna(subset=["y_true", "y_pred"])
        merged["error"] = merged["y_pred"] - merged["y_true"]
        for regime, group in merged.groupby("market_regime", dropna=False):
            error_rows.append(
                {
                    "market_regime": str(regime),
                    "prediction_rows": int(len(group)),
                    "rmse": float(np.sqrt(np.mean(group["error"] ** 2))),
                    "mae": float(group["error"].abs().mean()),
                    "directional_accuracy": float((np.sign(group["y_true"]) == np.sign(group["y_pred"])).mean()),
                }
            )
    return {"target_by_regime": rows, "prediction_error_by_regime": error_rows}


def ablation_diagnostics(fold_metrics: pd.DataFrame) -> list[dict[str, Any]]:
    if fold_metrics.empty:
        return []
    ridge = fold_metrics[fold_metrics["model_name"].eq("ridge_regression")].copy()
    if ridge.empty:
        return []
    rows: list[dict[str, Any]] = []
    keys = ["target_column", "fold_name", "split_name"]
    base = ridge[ridge["feature_set_name"].eq("technical_only")]
    for feature_set in ["technical_plus_sec", "technical_plus_earnings", "all_model_features"]:
        comp = ridge[ridge["feature_set_name"].eq(feature_set)]
        merged = base.merge(comp, on=keys, suffixes=("_technical", "_comparison"))
        for _, row in merged.iterrows():
            rows.append(
                {
                    "target": row["target_column"],
                    "fold": row["fold_name"],
                    "split": row["split_name"],
                    "comparison_feature_set": feature_set,
                    "rmse_delta_vs_technical": _safe_float(row.get("metric_rmse_comparison") - row.get("metric_rmse_technical")),
                    "r2_delta_vs_technical": _safe_float(
                        row.get("metric_oos_r2_vs_train_mean_comparison")
                        - row.get("metric_oos_r2_vs_train_mean_technical")
                    ),
                    "spearman_delta_vs_technical": _safe_float(
                        row.get("metric_spearman_ic_comparison") - row.get("metric_spearman_ic_technical")
                    ),
                    "directional_accuracy_delta_vs_technical": _safe_float(
                        row.get("metric_directional_accuracy_comparison")
                        - row.get("metric_directional_accuracy_technical")
                    ),
                }
            )
    return rows


def _run_lookup(runs: pd.DataFrame, target_column: str, feature_set: str, model_name: str) -> int | None:
    if runs.empty:
        return None
    matches = runs[
        runs["target_column"].astype(str).eq(target_column)
        & runs["feature_set_name"].astype(str).eq(feature_set)
        & runs["model_name"].astype(str).eq(model_name)
    ]
    if matches.empty:
        return None
    return int(matches.iloc[0]["model_run_id"])


def _diagnostic_summary(
    target_rows: list[dict[str, Any]],
    feature_diag: dict[str, Any],
    ablations: list[dict[str, Any]],
    split_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    primary = next((row for row in target_rows if row.get("target") == TARGET_COLUMNS["5_session"]), {})
    worsened = [row for row in ablations if (row.get("rmse_delta_vs_technical") or 0) > 0]
    improved = [row for row in ablations if (row.get("rmse_delta_vs_technical") or 0) < 0]
    groups = {row["group"]: row for row in feature_diag.get("group_missingness", [])}
    final_train = next((row for row in split_rows if row.get("period") == "final_train"), {})
    final_test = next((row for row in split_rows if row.get("period") == "final_test"), {})
    shift = None
    if final_train and final_test and final_train.get("std"):
        shift = abs((final_test.get("mean") or 0) - (final_train.get("mean") or 0)) / (final_train.get("std") or 1)
    return {
        "primary_target": TARGET_COLUMNS["5_session"],
        "primary_target_std": primary.get("std"),
        "primary_target_positive_rate": primary.get("positive_rate"),
        "rmse_ablation_rows_worse_than_technical": len(worsened),
        "rmse_ablation_rows_better_than_technical": len(improved),
        "sec_mean_nonzero_rate": groups.get("sec", {}).get("mean_nonzero_rate"),
        "earnings_mean_nonzero_rate": groups.get("earnings", {}).get("mean_nonzero_rate"),
        "final_test_mean_shift_in_train_std": shift,
        "likely_failure_modes": [
            item
            for item, active in [
                ("target_noise", (primary.get("std") or 0) > abs(primary.get("mean") or 0) * 5),
                ("alternative_data_sparsity", (groups.get("sec", {}).get("mean_nonzero_rate") or 0) < 0.20
                 or (groups.get("earnings", {}).get("mean_nonzero_rate") or 0) < 0.20),
                ("feature_overload_or_noise", len(worsened) > len(improved)),
                ("possible_regime_shift", shift is not None and shift > 0.25),
                ("small_universe", True),
            ]
            if active
        ],
    }


def build_model_diagnostics(db_path: str | Path, dataset_id: int) -> dict[str, Any]:
    frame = flatten_saved_dataset(db_path, dataset_id)
    if frame.empty:
        raise ValueError(f"Dataset #{dataset_id} has no rows.")

    training = load_training_dataset(db_path, dataset_id, TARGET_COLUMNS["5_session"])
    runs = _latest_completed_model_runs(db_path, dataset_id)
    model_run_ids = runs["model_run_id"].astype(int).tolist() if not runs.empty else []
    fold_metrics = _fold_metrics_frame(db_path, model_run_ids)

    primary_run_id = _run_lookup(runs, TARGET_COLUMNS["5_session"], "technical_only", "ridge_regression")
    primary_predictions = _predictions_for_run(db_path, primary_run_id) if primary_run_id else pd.DataFrame()
    primary_fold_metrics = (
        fold_metrics[
            fold_metrics["model_run_id"].astype(int).eq(primary_run_id)
        ]
        if primary_run_id and not fold_metrics.empty
        else pd.DataFrame()
    )

    target_rows = target_distribution(frame)
    split_rows = split_distribution(frame, TARGET_COLUMNS["5_session"], primary_fold_metrics)
    feature_diag = feature_diagnostics(training.X)
    ablations = ablation_diagnostics(fold_metrics)

    diagnostics = {
        "diagnostics_version": DIAGNOSTICS_VERSION,
        "created_at": _now_iso(),
        "dataset_id": int(dataset_id),
        "dataset_rows": int(len(frame)),
        "model_runs_used": runs[
            [
                "model_run_id",
                "target_column",
                "target_horizon",
                "feature_set_name",
                "model_name",
                "task",
                "status",
                "created_at",
            ]
        ].to_dict("records")
        if not runs.empty
        else [],
        "fold_metrics": fold_metrics.drop(columns=["metrics_json"], errors="ignore").to_dict("records") if not fold_metrics.empty else [],
        "target_distribution": target_rows,
        "cross_sectional_dispersion": cross_sectional_dispersion(frame),
        "split_distribution_5_session": split_rows,
        "per_ticker_target_5_session": per_ticker_target_diagnostics(frame, TARGET_COLUMNS["5_session"]),
        "per_ticker_prediction_error_5_session_technical_ridge": per_ticker_prediction_error(primary_predictions),
        "feature_diagnostics": feature_diag,
        "coverage_by_ticker": coverage_by_ticker(frame, training.feature_columns),
        "regime_diagnostics_5_session": regime_diagnostics(frame, TARGET_COLUMNS["5_session"], primary_predictions),
        "ablation_diagnostics": ablations,
        "summary": _diagnostic_summary(target_rows, feature_diag, ablations, split_rows),
    }
    return _clean_json(diagnostics)


def write_diagnostics_artifact(diagnostics: dict[str, Any], output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_id = int(diagnostics.get("dataset_id", 0) or 0)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = output_dir / f"model_diagnostics_dataset_{dataset_id}_{timestamp}.json"
    path.write_text(json.dumps(_clean_json(diagnostics), indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_diagnostics_artifact(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def list_diagnostic_artifacts(output_dir: str | Path) -> list[DiagnosticArtifactInfo]:
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    infos: list[DiagnosticArtifactInfo] = []
    for path in sorted(output_dir.glob("model_diagnostics_dataset_*.json"), reverse=True):
        try:
            payload = load_diagnostics_artifact(path)
        except Exception:
            continue
        summary = payload.get("summary", {})
        label = ", ".join(summary.get("likely_failure_modes", [])[:4]) if isinstance(summary, dict) else ""
        infos.append(
            DiagnosticArtifactInfo(
                path=path,
                dataset_id=_safe_int(payload.get("dataset_id")),
                created_at=str(payload.get("created_at") or path.stat().st_mtime),
                summary=label,
            )
        )
    return infos
