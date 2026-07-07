from __future__ import annotations

import json
import math
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.datasets.training_loader import TrainingDataset, load_training_dataset
from src.modeling.annotation_features import (
    ALL_ANNOTATION_FEATURE_COLUMNS,
    ANNOTATION_FEATURE_COLUMNS,
    ANNOTATION_FEATURE_VERSION,
    COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS,
    COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS,
    build_annotation_model_frame,
)
from src.modeling.splits import make_walk_forward_splits
from src.modeling.targets import RAW_TARGET_5_SESSION, get_target_definition, precompute_target_series, target_metadata


ANNOTATION_DIAGNOSTICS_VERSION = "annotation_feature_diagnostics_v1"
NEAR_ZERO_ACTIVE_RATE = 0.01
NEAR_ZERO_ACTIVE_COUNT = 50
HIGH_CORRELATION_THRESHOLD = 0.98


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


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


def _json_dumps(value: Any) -> str:
    return json.dumps(_clean_json(value), ensure_ascii=False, sort_keys=True, default=str, indent=2)


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(index=frame.index)
    return frame.apply(lambda series: pd.to_numeric(series, errors="coerce").astype(float))


def _date_mask(metadata: pd.DataFrame, dates: list[Any]) -> pd.Series:
    date_set = {pd.to_datetime(value).date() for value in dates}
    return pd.to_datetime(metadata["trading_date"]).dt.date.isin(date_set)


def _active_rows(frame: pd.DataFrame) -> int:
    if frame.empty:
        return 0
    numeric = _numeric_frame(frame).fillna(0.0)
    return int(numeric.ne(0).any(axis=1).sum())


def _condition_summary(frame: pd.DataFrame) -> dict[str, Any]:
    numeric = _numeric_frame(frame).fillna(0.0)
    varying = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if varying.shape[1] < 2:
        return {
            "feature_count": int(frame.shape[1]),
            "varying_feature_count": int(varying.shape[1]),
            "condition_number": None,
            "ill_conditioned": False,
            "reason": "fewer_than_two_varying_features",
        }
    std = varying.std(ddof=0).replace(0, np.nan)
    scaled = ((varying - varying.mean()) / std).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    singular_values = np.linalg.svd(scaled.to_numpy(dtype=float), compute_uv=False)
    if singular_values.size == 0:
        condition = None
        reason = "empty_svd"
    elif float(singular_values[-1]) <= 1e-10:
        condition = None
        reason = "singular_or_nearly_singular"
    else:
        condition = float(singular_values[0] / singular_values[-1])
        reason = "ok"
    return {
        "feature_count": int(frame.shape[1]),
        "varying_feature_count": int(varying.shape[1]),
        "condition_number": condition,
        "ill_conditioned": bool(condition is None or condition > 1e6),
        "reason": reason,
        "largest_singular_value": float(singular_values[0]) if singular_values.size else None,
        "smallest_singular_value": float(singular_values[-1]) if singular_values.size else None,
    }


def annotation_feature_detail(derived: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in ALL_ANNOTATION_FEATURE_COLUMNS:
        numeric = pd.to_numeric(derived[column], errors="coerce")
        nonzero = numeric.fillna(0.0).ne(0)
        unique_count = int(numeric.dropna().nunique(dropna=True))
        active_count = int(nonzero.sum())
        rows.append(
            {
                "feature": column,
                "missing_rate": float(numeric.isna().mean()),
                "active_count": active_count,
                "active_rate": float(nonzero.mean()),
                "std": None if pd.isna(numeric.std(ddof=0)) else float(numeric.std(ddof=0)),
                "unique_count": unique_count,
                "near_zero_variance": bool(
                    unique_count <= 1 or active_count < NEAR_ZERO_ACTIVE_COUNT or float(nonzero.mean()) < NEAR_ZERO_ACTIVE_RATE
                ),
                "compact_decay": column in COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS,
                "compact_weighted": column in COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS,
                "legacy_broad": column in ANNOTATION_FEATURE_COLUMNS,
            }
        )
    return rows


def highly_correlated_annotation_pairs(derived: pd.DataFrame) -> list[dict[str, Any]]:
    numeric = _numeric_frame(derived[ALL_ANNOTATION_FEATURE_COLUMNS]).fillna(0.0)
    varying = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    if varying.shape[1] < 2:
        return []
    corr = varying.corr().abs()
    rows: list[dict[str, Any]] = []
    columns = corr.columns.tolist()
    for idx, left in enumerate(columns):
        for right in columns[idx + 1 :]:
            value = corr.loc[left, right]
            if pd.notna(value) and float(value) >= HIGH_CORRELATION_THRESHOLD:
                rows.append({"feature_a": left, "feature_b": right, "abs_corr": float(value)})
    return sorted(rows, key=lambda item: item["abs_corr"], reverse=True)[:100]


def fold_activation_coverage(
    derived: pd.DataFrame,
    metadata: pd.DataFrame,
    target_series: pd.Series,
    feature_sets: dict[str, list[str]],
    horizon_sessions: int,
) -> list[dict[str, Any]]:
    folds = make_walk_forward_splits(
        metadata,
        target_series,
        horizon_sessions=horizon_sessions,
        n_folds=3,
        purge_sessions=horizon_sessions,
        embargo_sessions=horizon_sessions,
    )
    feature_set_names = [
        "annotation_features_only",
        "annotation_compact_decay",
        "annotation_compact_weighted",
        "technical_core_plus_annotation_compact_decay",
        "technical_core_plus_annotation_compact_weighted",
    ]
    rows: list[dict[str, Any]] = []
    for split in folds:
        for role, dates in [("train", split.train_dates), ("eval", split.eval_dates)]:
            mask = _date_mask(metadata, dates)
            subset = derived.loc[mask]
            for name in feature_set_names:
                columns = [column for column in feature_sets.get(name, []) if column in subset.columns]
                if not columns:
                    continue
                active = _active_rows(subset[columns])
                rows.append(
                    {
                        "fold_name": split.fold_name,
                        "split_name": split.split_name,
                        "role": role,
                        "feature_set_name": name,
                        "rows": int(len(subset)),
                        "active_rows": active,
                        "active_rate": float(active / len(subset)) if len(subset) else 0.0,
                    }
                )
    return rows


def build_annotation_feature_diagnostics(
    db_path: str | Path,
    dataset_id: int = 49,
    target_name: str = RAW_TARGET_5_SESSION,
) -> dict[str, Any]:
    definition = get_target_definition(target_name)
    training = load_training_dataset(db_path, dataset_id, definition.base_label_column)
    target_series = precompute_target_series(definition, training.y, training.X, training.metadata)
    valid = target_series.notna()
    valid_training = TrainingDataset(
        X=training.X.loc[valid].reset_index(drop=True),
        y=training.y.loc[valid].reset_index(drop=True),
        metadata=training.metadata.loc[valid].reset_index(drop=True),
        audit=training.audit.loc[valid].reset_index(drop=True),
        feature_columns=training.feature_columns,
        label_column=training.label_column,
    )
    model_frame, derived, feature_sets = build_annotation_model_frame(valid_training, db_path)
    target_valid = target_series.loc[valid].reset_index(drop=True)

    compact_sets = {
        "annotation_features_only": [column for column in ANNOTATION_FEATURE_COLUMNS if column in derived.columns],
        "annotation_compact_decay": [column for column in COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS if column in derived.columns],
        "annotation_compact_weighted": [column for column in COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS if column in derived.columns],
        "technical_core_plus_annotations": [column for column in feature_sets["technical_core_plus_annotations"] if column in model_frame.columns],
        "technical_core_plus_annotation_compact_decay": [
            column for column in feature_sets["technical_core_plus_annotation_compact_decay"] if column in model_frame.columns
        ],
        "technical_core_plus_annotation_compact_weighted": [
            column for column in feature_sets["technical_core_plus_annotation_compact_weighted"] if column in model_frame.columns
        ],
    }
    condition_numbers = {
        name: _condition_summary((derived if name.startswith("annotation_") else model_frame)[columns])
        for name, columns in compact_sets.items()
        if columns
    }
    feature_rows = annotation_feature_detail(derived)
    near_zero_rows = [row for row in feature_rows if row["near_zero_variance"]]
    corr_pairs = highly_correlated_annotation_pairs(derived)
    fold_coverage = fold_activation_coverage(
        derived,
        valid_training.metadata,
        target_valid,
        feature_sets,
        horizon_sessions=definition.horizon_sessions,
    )
    artifact = {
        "artifact_type": "annotation_feature_design_diagnostics",
        "diagnostics_version": ANNOTATION_DIAGNOSTICS_VERSION,
        "annotation_feature_version": ANNOTATION_FEATURE_VERSION,
        "created_at": _now_iso(),
        "dataset_id": int(dataset_id),
        "target": target_metadata(definition),
        "row_count": int(len(derived)),
        "annotation_feature_count": int(len(ALL_ANNOTATION_FEATURE_COLUMNS)),
        "compact_decay_columns": list(COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS),
        "compact_weighted_columns": list(COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS),
        "feature_detail": feature_rows,
        "near_zero_variance_features": near_zero_rows,
        "highly_correlated_pairs": corr_pairs,
        "condition_numbers": condition_numbers,
        "fold_activation_coverage": fold_coverage,
        "summary": {
            "near_zero_variance_count": int(len(near_zero_rows)),
            "high_correlation_pair_count": int(len(corr_pairs)),
            "annotation_features_only_condition": condition_numbers.get("annotation_features_only", {}),
            "compact_decay_condition": condition_numbers.get("annotation_compact_decay", {}),
            "compact_weighted_condition": condition_numbers.get("annotation_compact_weighted", {}),
            "compact_decay_active_rows": _active_rows(derived[COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS]),
            "compact_weighted_active_rows": _active_rows(derived[COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS]),
            "scanner_scoring_effect": 0,
            "research_only": True,
        },
    }
    return _clean_json(artifact)


def write_annotation_feature_diagnostics(diagnostics: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    dataset_id = int(diagnostics.get("dataset_id", 0) or 0)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    path = output / f"phase2d6d4_annotation_feature_diagnostics_dataset{dataset_id}_{timestamp}.json"
    path.write_text(_json_dumps(diagnostics), encoding="utf-8")
    return path
