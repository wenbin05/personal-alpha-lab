from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.training_loader import TrainingDataset, load_training_dataset
from src.modeling.feature_sets import (
    TECHNICAL_PRUNING_EXCLUDE,
    event_feature_columns,
    feature_group,
    is_generic_sec_volume_proxy,
    technical_core_columns,
    technical_feature_columns,
)
from src.modeling.splits import ModelSplit, make_walk_forward_splits
from src.modeling.targets import (
    RAW_TARGET_5_SESSION,
    get_target_definition,
    precompute_target_series,
    target_metadata,
    transform_target_for_split,
)


FEATURE_QUALITY_VERSION = "feature_quality_v1"
CORRELATION_THRESHOLD = 0.98
HIGH_MISSING_THRESHOLD = 0.20
LOW_MISSING_THRESHOLD = 0.10
NEAR_CONSTANT_DOMINANT_THRESHOLD = 0.995
SPARSE_ACTIVE_RATE_THRESHOLD = 0.01
SPARSE_ACTIVE_COUNT_THRESHOLD = 50


@dataclass(frozen=True)
class FeatureQualityArtifactInfo:
    path: Path
    dataset_id: int | None
    created_at: str
    summary: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2)


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
    try:
        if value is not None and not isinstance(value, (str, bytes, list, dict, tuple)) and pd.isna(value):
            return None
    except Exception:
        pass
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


def _safe_date_mask(metadata: pd.DataFrame, dates: list[Any]) -> pd.Series:
    date_set = {pd.to_datetime(value).date() for value in dates}
    return pd.to_datetime(metadata["trading_date"]).dt.date.isin(date_set)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    data = {column: pd.to_numeric(frame[column], errors="coerce").astype(float) for column in frame.columns}
    numeric = pd.DataFrame(data, index=frame.index)
    return numeric.loc[:, numeric.notna().any(axis=0)]


def _nonzero_rate(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.notna().any():
        return float((numeric.fillna(0) != 0).mean())
    non_empty = series.fillna("").astype(str).str.strip().ne("")
    return float(non_empty.mean())


def _dominant_rate(series: pd.Series) -> float | None:
    non_null = series.dropna()
    if non_null.empty:
        return None
    return float(non_null.astype(str).value_counts(normalize=True, dropna=False).iloc[0])


def _series_missing_by_ticker(X: pd.DataFrame, metadata: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if "ticker" not in metadata.columns:
        return rows
    frame = pd.concat([metadata[["ticker"]].reset_index(drop=True), X.isna().reset_index(drop=True)], axis=1)
    feature_columns = X.columns.tolist()
    for ticker, group in frame.groupby("ticker", dropna=False):
        missing_values = group[feature_columns].mean()
        rows.append(
            {
                "ticker": str(ticker),
                "rows": int(len(group)),
                "mean_feature_missing_rate": float(missing_values.mean()),
                "max_feature_missing_rate": float(missing_values.max()) if not missing_values.empty else None,
            }
        )
    return sorted(rows, key=lambda item: item["ticker"])


def _series_missing_by_date(metadata: pd.DataFrame, X: pd.DataFrame, limit: int = 30) -> list[dict[str, Any]]:
    if "trading_date" not in metadata.columns:
        return []
    frame = pd.concat(
        [
            pd.to_datetime(metadata["trading_date"]).dt.date.rename("trading_date").reset_index(drop=True),
            X.isna().reset_index(drop=True),
        ],
        axis=1,
    )
    feature_columns = X.columns.tolist()
    grouped = frame.groupby("trading_date")[feature_columns].mean().mean(axis=1).sort_values(ascending=False)
    return [
        {"trading_date": str(date_value), "mean_feature_missing_rate": float(value)}
        for date_value, value in grouped.head(limit).items()
    ]


def split_feature_missingness(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    folds: list[ModelSplit],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    periods: list[tuple[str, pd.Series]] = []
    for split in folds:
        periods.append((f"{split.fold_name}_{split.split_name}_train", _safe_date_mask(metadata, split.train_dates)))
        periods.append((f"{split.fold_name}_{split.split_name}_eval", _safe_date_mask(metadata, split.eval_dates)))
    seen: set[str] = set()
    for name, mask in periods:
        if name in seen:
            continue
        seen.add(name)
        subset = X.loc[mask]
        if subset.empty:
            continue
        rows.append(
            {
                "period": name,
                "rows": int(len(subset)),
                "mean_missing_rate": float(subset.isna().mean().mean()),
                "max_feature_missing_rate": float(subset.isna().mean().max()),
            }
        )
    return rows


def feature_missingness_detail(
    X: pd.DataFrame,
    development_mask: pd.Series,
    final_test_mask: pd.Series,
) -> list[dict[str, Any]]:
    dev = X.loc[development_mask]
    final = X.loc[final_test_mask]
    rows: list[dict[str, Any]] = []
    for column in X.columns:
        dev_series = dev[column] if column in dev.columns else pd.Series(dtype=float)
        final_series = final[column] if column in final.columns else pd.Series(dtype=float)
        nonzero_rate = _nonzero_rate(dev_series) if not dev_series.empty else 0.0
        active_count = int((pd.to_numeric(dev_series, errors="coerce").fillna(0) != 0).sum()) if not dev_series.empty else 0
        dominant = _dominant_rate(dev_series)
        rows.append(
            {
                "feature": column,
                "group": feature_group(column),
                "development_missing_rate": float(dev_series.isna().mean()) if not dev_series.empty else None,
                "final_test_missing_rate": float(final_series.isna().mean()) if not final_series.empty else None,
                "unique_count_development": int(dev_series.dropna().nunique(dropna=True)) if not dev_series.empty else 0,
                "dominant_rate_development": dominant,
                "nonzero_rate_development": nonzero_rate,
                "active_count_development": active_count,
                "mostly_unavailable_in_development": bool(dev_series.isna().mean() > HIGH_MISSING_THRESHOLD) if not dev_series.empty else True,
                "mostly_unavailable_in_final_test": bool(final_series.isna().mean() > HIGH_MISSING_THRESHOLD) if not final_series.empty else False,
            }
        )
    return rows


def near_constant_features(missingness_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = [
        {
            **row,
            "reason": "zero_variance_or_near_zero_variance",
        }
        for row in missingness_rows
        if int(row.get("unique_count_development") or 0) <= 1
        or float(row.get("dominant_rate_development") or 0.0) >= NEAR_CONSTANT_DOMINANT_THRESHOLD
    ]
    return sorted(rows, key=lambda item: (item.get("group", ""), item.get("feature", "")))


def sparse_event_features(missingness_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    event_groups = {"sec", "earnings", "catalyst_llm"}
    rows = [
        {
            **row,
            "reason": "too_few_active_observations_for_stable_linear_signal",
        }
        for row in missingness_rows
        if row.get("group") in event_groups
        and (
            float(row.get("nonzero_rate_development") or 0.0) < SPARSE_ACTIVE_RATE_THRESHOLD
            or int(row.get("active_count_development") or 0) < SPARSE_ACTIVE_COUNT_THRESHOLD
        )
    ]
    return sorted(rows, key=lambda item: (item.get("group", ""), item.get("feature", "")))


def _correlation_clusters(pairs: list[dict[str, Any]]) -> list[set[str]]:
    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        if parent[item] != item:
            parent[item] = find(parent[item])
        return parent[item]

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    for pair in pairs:
        union(str(pair["feature_a"]), str(pair["feature_b"]))
    clusters: dict[str, set[str]] = {}
    for item in parent:
        clusters.setdefault(find(item), set()).add(item)
    return [cluster for cluster in clusters.values() if len(cluster) > 1]


def _representative_feature(cluster: set[str], missing_lookup: dict[str, dict[str, Any]]) -> str:
    def sort_key(column: str) -> tuple[float, float, int, str]:
        row = missing_lookup.get(column, {})
        missing = float(row.get("development_missing_rate") or 0.0)
        dominant = float(row.get("dominant_rate_development") or 0.0)
        raw_level_penalty = 1 if column in TECHNICAL_PRUNING_EXCLUDE or column.startswith("ma_") else 0
        return (missing, dominant, raw_level_penalty, column)

    return sorted(cluster, key=sort_key)[0]


def correlation_audit(
    X: pd.DataFrame,
    development_mask: pd.Series,
    missingness_rows: list[dict[str, Any]],
    threshold: float = CORRELATION_THRESHOLD,
) -> dict[str, Any]:
    dev_numeric = _numeric_frame(X.loc[development_mask])
    pairs: list[dict[str, Any]] = []
    if dev_numeric.shape[1] > 1:
        varying = dev_numeric.loc[:, dev_numeric.nunique(dropna=True) > 1]
        if varying.shape[1] > 1:
            corr = varying.corr().abs()
            columns = corr.columns.tolist()
            for i, left in enumerate(columns):
                for right in columns[i + 1 :]:
                    value = corr.loc[left, right]
                    if pd.notna(value) and float(value) >= threshold:
                        pairs.append({"feature_a": left, "feature_b": right, "abs_corr": float(value)})
    pairs = sorted(pairs, key=lambda item: (-float(item["abs_corr"]), item["feature_a"], item["feature_b"]))
    missing_lookup = {row["feature"]: row for row in missingness_rows}
    groups: list[dict[str, Any]] = []
    remove: set[str] = set()
    for cluster in _correlation_clusters(pairs):
        representative = _representative_feature(cluster, missing_lookup)
        dropped = sorted(cluster - {representative})
        remove.update(dropped)
        groups.append(
            {
                "representative": representative,
                "redundant_features": dropped,
                "cluster_size": len(cluster),
            }
        )
    return {
        "threshold": threshold,
        "highly_correlated_pairs": pairs[:100],
        "redundant_groups": sorted(groups, key=lambda item: item["representative"]),
        "remove_features": sorted(remove),
    }


def outlier_scale_audit(
    X: pd.DataFrame,
    development_mask: pd.Series,
    final_test_mask: pd.Series,
) -> list[dict[str, Any]]:
    dev_numeric = _numeric_frame(X.loc[development_mask])
    final_numeric = _numeric_frame(X.loc[final_test_mask])
    rows: list[dict[str, Any]] = []
    for column in dev_numeric.columns:
        values = dev_numeric[column].dropna()
        if values.empty:
            continue
        std = values.std(ddof=0)
        mean = values.mean()
        final_values = final_numeric[column].dropna() if column in final_numeric.columns else pd.Series(dtype=float)
        rows.append(
            {
                "feature": column,
                "group": feature_group(column),
                "development_mean": _safe_float(mean),
                "development_std": _safe_float(std),
                "development_p01": _safe_float(values.quantile(0.01)),
                "development_p99": _safe_float(values.quantile(0.99)),
                "development_abs_z_gt_5_rate": 0.0
                if std == 0 or pd.isna(std)
                else float(((values - mean).abs() / std).gt(5).mean()),
                "final_test_outside_development_p01_p99_rate": None
                if final_values.empty
                else float(
                    final_values.lt(values.quantile(0.01)).mean()
                    + final_values.gt(values.quantile(0.99)).mean()
                ),
                "standardization_policy": "fold_safe_preprocessor_fit_on_train_split_only",
            }
        )
    return sorted(rows, key=lambda item: item.get("development_abs_z_gt_5_rate") or 0.0, reverse=True)


def univariate_ic_audit(
    X: pd.DataFrame,
    metadata: pd.DataFrame,
    target_series: pd.Series,
    folds: list[ModelSplit],
    target_name: str = RAW_TARGET_5_SESSION,
) -> list[dict[str, Any]]:
    definition = get_target_definition(target_name)
    fold_values: dict[str, list[float]] = {column: [] for column in X.columns}
    numeric = _numeric_frame(X)
    for split in folds:
        if split.split_name != "validation":
            continue
        train_mask = _safe_date_mask(metadata, split.train_dates)
        eval_mask = _safe_date_mask(metadata, split.eval_dates)
        _, y_eval, _ = transform_target_for_split(definition, target_series, train_mask, eval_mask)
        eval_y = pd.to_numeric(y_eval.dropna(), errors="coerce")
        if eval_y.empty:
            continue
        for column in numeric.columns:
            aligned = pd.DataFrame({"feature": numeric.loc[eval_y.index, column], "target": eval_y}).dropna()
            if len(aligned) < 20 or aligned["feature"].nunique(dropna=True) < 2 or aligned["target"].nunique(dropna=True) < 2:
                continue
            value = aligned["feature"].corr(aligned["target"], method="spearman")
            if pd.notna(value):
                fold_values[column].append(float(value))

    rows: list[dict[str, Any]] = []
    for column, values in fold_values.items():
        if not values:
            rows.append(
                {
                    "feature": column,
                    "group": feature_group(column),
                    "fold_count": 0,
                    "mean_ic": None,
                    "ic_volatility": None,
                    "hit_rate_positive": None,
                }
            )
            continue
        series = pd.Series(values, dtype=float)
        rows.append(
            {
                "feature": column,
                "group": feature_group(column),
                "fold_count": int(len(values)),
                "mean_ic": _safe_float(series.mean()),
                "ic_volatility": _safe_float(series.std(ddof=0)),
                "hit_rate_positive": _safe_float(series.gt(0).mean()),
            }
        )
    return sorted(
        rows,
        key=lambda item: (
            -abs(float(item["mean_ic"]) if item.get("mean_ic") is not None else 0.0),
            item["feature"],
        ),
    )


def feature_group_summary(
    missingness_rows: list[dict[str, Any]],
    ic_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    missing_frame = pd.DataFrame(missingness_rows)
    ic_frame = pd.DataFrame(ic_rows)
    rows: list[dict[str, Any]] = []
    for group, group_frame in missing_frame.groupby("group", dropna=False):
        group_ics = ic_frame[ic_frame["group"].eq(group)] if not ic_frame.empty else pd.DataFrame()
        mean_abs_ic = None
        if not group_ics.empty and "mean_ic" in group_ics:
            mean_abs_ic = _safe_float(pd.to_numeric(group_ics["mean_ic"], errors="coerce").abs().mean())
        rows.append(
            {
                "group": str(group),
                "feature_count": int(len(group_frame)),
                "mean_development_missing_rate": _safe_float(group_frame["development_missing_rate"].mean()),
                "max_development_missing_rate": _safe_float(group_frame["development_missing_rate"].max()),
                "mean_nonzero_rate": _safe_float(group_frame["nonzero_rate_development"].mean()),
                "near_constant_count": int(
                    (
                        group_frame["unique_count_development"].fillna(0).le(1)
                        | group_frame["dominant_rate_development"].fillna(0).ge(NEAR_CONSTANT_DOMINANT_THRESHOLD)
                    ).sum()
                ),
                "sparse_active_count": int(
                    (
                        group_frame["nonzero_rate_development"].fillna(0).lt(SPARSE_ACTIVE_RATE_THRESHOLD)
                        | group_frame["active_count_development"].fillna(0).lt(SPARSE_ACTIVE_COUNT_THRESHOLD)
                    ).sum()
                ),
                "mean_abs_univariate_ic": mean_abs_ic,
            }
        )
    return sorted(rows, key=lambda item: item["group"])


def build_pruned_feature_sets(
    feature_columns: list[str],
    missingness_rows: list[dict[str, Any]],
    correlation: dict[str, Any],
) -> dict[str, list[str]]:
    high_missing = {
        row["feature"]
        for row in missingness_rows
        if (row.get("development_missing_rate") is None or float(row.get("development_missing_rate") or 0.0) > HIGH_MISSING_THRESHOLD)
    }
    low_missing_exclusions = {
        row["feature"]
        for row in missingness_rows
        if (row.get("development_missing_rate") is None or float(row.get("development_missing_rate") or 0.0) > LOW_MISSING_THRESHOLD)
    }
    near_constant = {row["feature"] for row in near_constant_features(missingness_rows)}
    sparse_events = {row["feature"] for row in sparse_event_features(missingness_rows)}
    redundant = set(correlation.get("remove_features", []))

    technical_base = technical_feature_columns(feature_columns)
    technical_core = technical_core_columns(feature_columns)
    event_base = event_feature_columns(feature_columns)
    event_pruned = [
        column
        for column in event_base
        if column not in high_missing and column not in near_constant and column not in sparse_events and column not in redundant
    ]
    technical_pruned = [
        column
        for column in technical_base
        if column not in TECHNICAL_PRUNING_EXCLUDE
        and column not in high_missing
        and column not in near_constant
        and column not in redundant
    ]
    low_missing_low_correlation = [
        column
        for column in feature_columns
        if column not in low_missing_exclusions
        and column not in near_constant
        and column not in redundant
        and column not in sparse_events
        and not is_generic_sec_volume_proxy(column)
    ]
    return {
        "technical_core": technical_core,
        "technical_pruned": technical_pruned,
        "event_features_only": event_pruned,
        "technical_core_plus_events": [column for column in feature_columns if column in set(technical_core).union(event_pruned)],
        "low_missing_low_correlation": low_missing_low_correlation,
    }


def build_feature_quality_audit(
    db_path: str | Path,
    dataset_id: int,
    target_name: str = RAW_TARGET_5_SESSION,
    n_folds: int = 3,
    final_test_fraction: float = 0.20,
) -> dict[str, Any]:
    definition = get_target_definition(target_name)
    training = load_training_dataset(db_path, dataset_id, definition.base_label_column)
    target_series = precompute_target_series(definition, training.y, training.X, training.metadata)
    valid = target_series.notna()
    X = training.X.loc[valid].copy()
    metadata = training.metadata.loc[valid].copy()
    target_series = target_series.loc[valid].astype(float)
    folds = make_walk_forward_splits(
        metadata,
        target_series,
        horizon_sessions=definition.horizon_sessions,
        n_folds=n_folds,
        final_test_fraction=final_test_fraction,
        purge_sessions=definition.horizon_sessions,
        embargo_sessions=definition.horizon_sessions,
    )
    final_test_dates = folds[-1].eval_dates
    final_test_mask = _safe_date_mask(metadata, final_test_dates)
    development_mask = ~final_test_mask

    missingness_rows = feature_missingness_detail(X, development_mask, final_test_mask)
    correlation = correlation_audit(X, development_mask, missingness_rows)
    ic_rows = univariate_ic_audit(X, metadata, target_series, folds, target_name=target_name)
    pruned_sets = build_pruned_feature_sets(training.feature_columns, missingness_rows, correlation)
    near_constant = near_constant_features(missingness_rows)
    sparse_events = sparse_event_features(missingness_rows)
    outliers = outlier_scale_audit(X, development_mask, final_test_mask)
    group_summary = feature_group_summary(missingness_rows, ic_rows)

    pruned_set_metadata = {
        name: {
            "columns": columns,
            "column_count": len(columns),
            "selection_data_scope": "development_folds_only_final_test_labels_not_used",
        }
        for name, columns in pruned_sets.items()
    }
    artifact = {
        "feature_quality_version": FEATURE_QUALITY_VERSION,
        "created_at": _now_iso(),
        "dataset_id": int(dataset_id),
        "target": target_metadata(definition),
        "row_scope": {
            "valid_labeled_rows": int(len(X)),
            "development_rows": int(development_mask.sum()),
            "final_test_rows": int(final_test_mask.sum()),
            "final_test_labels_used_for_feature_selection": False,
        },
        "thresholds": {
            "high_missing": HIGH_MISSING_THRESHOLD,
            "low_missing": LOW_MISSING_THRESHOLD,
            "near_constant_dominant_rate": NEAR_CONSTANT_DOMINANT_THRESHOLD,
            "sparse_active_rate": SPARSE_ACTIVE_RATE_THRESHOLD,
            "sparse_active_count": SPARSE_ACTIVE_COUNT_THRESHOLD,
            "correlation_abs_threshold": CORRELATION_THRESHOLD,
        },
        "feature_missingness": missingness_rows,
        "missingness_by_ticker": _series_missing_by_ticker(X, metadata),
        "missingness_by_date_top": _series_missing_by_date(metadata, X),
        "missingness_by_split": split_feature_missingness(X, metadata, folds),
        "near_constant_features": near_constant,
        "sparse_event_features": sparse_events,
        "correlation_audit": correlation,
        "outlier_scale_audit": outliers[:100],
        "univariate_ic": ic_rows,
        "feature_group_summary": group_summary,
        "pruned_feature_sets": pruned_set_metadata,
        "summary": {
            "feature_count": int(len(training.feature_columns)),
            "near_constant_count": int(len(near_constant)),
            "sparse_event_feature_count": int(len(sparse_events)),
            "high_correlation_pair_count": int(len(correlation.get("highly_correlated_pairs", []))),
            "redundant_removed_count": int(len(correlation.get("remove_features", []))),
            "new_feature_sets": {name: len(columns) for name, columns in pruned_sets.items()},
            "selection_guardrail": "pruned feature sets use development folds only and do not inspect final-test labels",
        },
    }
    return _clean_json(artifact)


def write_feature_quality_artifact(artifact: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dataset_id = artifact.get("dataset_id", "unknown")
    path = output / f"phase2d4_feature_quality_dataset{dataset_id}_{timestamp}.json"
    path.write_text(_json_dumps(artifact), encoding="utf-8")
    return path


def load_feature_quality_artifact(path: str | Path) -> dict[str, Any]:
    return _json_loads(Path(path).read_text(encoding="utf-8"), {}) or {}


def list_feature_quality_artifacts(output_dir: str | Path) -> list[FeatureQualityArtifactInfo]:
    output = Path(output_dir)
    if not output.exists():
        return []
    artifacts: list[FeatureQualityArtifactInfo] = []
    for path in sorted(output.glob("phase2d4_feature_quality_dataset*.json"), reverse=True):
        data = load_feature_quality_artifact(path)
        artifacts.append(
            FeatureQualityArtifactInfo(
                path=path,
                dataset_id=data.get("dataset_id"),
                created_at=str(data.get("created_at") or ""),
                summary=f"{data.get('summary', {}).get('feature_count', 0)} features; "
                f"{data.get('summary', {}).get('near_constant_count', 0)} near-constant",
            )
        )
    return artifacts


def pruned_feature_sets_from_artifact(artifact: dict[str, Any]) -> dict[str, list[str]]:
    rows = artifact.get("pruned_feature_sets", {}) or {}
    return {
        str(name): [str(column) for column in (definition.get("columns", []) or [])]
        for name, definition in rows.items()
    }


def feature_set_quality_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, definition in (artifact.get("pruned_feature_sets", {}) or {}).items():
        rows.append(
            {
                "feature_set": name,
                "column_count": int(definition.get("column_count", 0) or 0),
                "selection_scope": definition.get("selection_data_scope"),
            }
        )
    return sorted(rows, key=lambda item: item["feature_set"])


def validate_feature_set_subset(training: TrainingDataset, columns: list[str]) -> list[str]:
    allowed = set(training.feature_columns)
    selected = [column for column in columns if column in allowed]
    if len(selected) != len(columns):
        missing = sorted(set(columns) - allowed)
        raise ValueError(f"Feature set contains columns outside the model manifest: {', '.join(missing)}")
    return selected
