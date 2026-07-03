from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge

from src.data import storage
from src.datasets.repository import list_dataset_builds
from src.datasets.training_loader import TrainingDataset, load_training_dataset
from src.modeling.feature_sets import FEATURE_SET_NAMES, select_feature_columns
from src.modeling.feature_quality import (
    build_feature_quality_audit,
    pruned_feature_sets_from_artifact,
    write_feature_quality_artifact,
)
from src.modeling.metrics import classification_metrics, regression_metrics
from src.modeling.preprocessing import fit_transform_matrices
from src.modeling.repository import (
    complete_model_run,
    insert_final_metric,
    insert_fold_metric,
    insert_model_run,
    insert_predictions,
)
from src.modeling.splits import (
    ModelSplit,
    make_walk_forward_splits,
    split_config_dict,
)
from src.modeling.targets import (
    TARGET_ENGINEERING_TARGETS,
    get_target_definition,
    precompute_target_series,
    target_metadata,
    transform_target_for_split,
)


MODEL_NAMES = [
    "zero_prediction",
    "train_mean",
    "ridge_regression",
    "logistic_regression",
]

TARGET_COLUMNS = {
    "1_session": "label_1_session_excess_return",
    "5_session": "label_5_session_excess_return",
    "20_session": "label_20_session_excess_return",
}


@dataclass(frozen=True)
class ModelRunSummary:
    model_run_id: int
    dataset_id: int
    target_column: str
    feature_set_name: str
    model_name: str
    final_metrics: dict[str, Any]
    warnings: list[str]


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def latest_accepted_dataset_id(db_path: str | Path, preferred_dataset_id: int = 49) -> int:
    """Return Dataset 49 when present, otherwise the latest non-empty build.

    Dataset acceptance is still a human process; this helper avoids declaring a
    newer build accepted unless Dataset 49 is unavailable.
    """
    builds = list_dataset_builds(db_path, limit=200)
    if builds.empty:
        raise ValueError("No dataset builds found.")
    if int(preferred_dataset_id) in set(builds["dataset_id"].astype(int).tolist()):
        row = builds[builds["dataset_id"].astype(int).eq(int(preferred_dataset_id))].iloc[0]
        if int(row.get("row_count", 0) or 0) > 0 and str(row.get("data_hash", "")) != "pending":
            return int(preferred_dataset_id)
    completed = builds[(builds["row_count"].fillna(0).astype(int) > 0) & (builds["data_hash"].astype(str) != "pending")]
    if completed.empty:
        raise ValueError("No completed non-empty dataset builds found.")
    return int(completed.sort_values("dataset_id", ascending=False).iloc[0]["dataset_id"])


def dataset_hash_for_id(db_path: str | Path, dataset_id: int) -> str:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute("SELECT data_hash FROM dataset_builds WHERE dataset_id = ?", (int(dataset_id),)).fetchone()
    if row is None:
        raise ValueError(f"Dataset #{dataset_id} not found.")
    return str(row["data_hash"])


def _date_mask(metadata: pd.DataFrame, dates: list[Any]) -> pd.Series:
    date_set = {pd.to_datetime(value).date() for value in dates}
    return pd.to_datetime(metadata["trading_date"]).dt.date.isin(date_set)


def _prediction_rows(
    model_run_id: int,
    metadata: pd.DataFrame,
    target_horizon: str,
    split: ModelSplit,
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
    feature_set_name: str,
    model_name: str,
    y_score: np.ndarray | None = None,
    y_pred_label: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    y_true_values = np.asarray(y_true, dtype=float)
    for idx, (_, meta_row) in enumerate(metadata.iterrows()):
        rows.append(
            {
                "model_run_id": model_run_id,
                "snapshot_id": int(meta_row["snapshot_id"]),
                "ticker": meta_row["ticker"],
                "snapshot_date": str(pd.to_datetime(meta_row["trading_date"]).date()),
                "target_horizon": target_horizon,
                "split_name": split.split_name,
                "fold_name": split.fold_name,
                "y_true": float(y_true_values[idx]) if pd.notna(y_true_values[idx]) else None,
                "y_pred": float(y_pred[idx]) if pd.notna(y_pred[idx]) else None,
                "y_pred_label": None if y_pred_label is None else int(y_pred_label[idx]),
                "y_score": None if y_score is None else float(y_score[idx]),
                "feature_set_name": feature_set_name,
                "model_name": model_name,
            }
        )
    return rows


def _prediction_metric_frame(metadata: pd.DataFrame, y_true: Any, y_pred: Any) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trading_date": pd.to_datetime(metadata["trading_date"]).dt.date.values,
            "ticker": metadata["ticker"].values,
            "y_true": np.asarray(y_true, dtype=float),
            "y_pred": np.asarray(y_pred, dtype=float),
        }
    )


def _run_regression_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
) -> np.ndarray:
    if model_name == "zero_prediction":
        return np.zeros(len(X_eval), dtype=float)
    if model_name == "train_mean":
        return np.full(len(X_eval), float(y_train.mean()), dtype=float)
    if model_name == "ridge_regression":
        train_matrix, eval_matrix, _ = fit_transform_matrices(X_train, X_eval)
        model = Ridge(alpha=1.0)
        model.fit(train_matrix, y_train.astype(float))
        return np.asarray(model.predict(eval_matrix), dtype=float)
    raise ValueError(f"Unsupported regression model: {model_name}")


def _run_logistic_model(
    X_train: pd.DataFrame,
    y_train_binary: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    train_matrix, eval_matrix, _ = fit_transform_matrices(X_train, X_eval)
    model = LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear", random_state=42)
    model.fit(train_matrix, y_train_binary.astype(int))
    score = np.asarray(model.predict_proba(eval_matrix)[:, 1], dtype=float)
    labels = (score >= 0.5).astype(int)
    return score, labels


def _run_classification_model(
    model_name: str,
    X_train: pd.DataFrame,
    y_train_binary: pd.Series,
    X_eval: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    if model_name == "zero_prediction":
        score = np.zeros(len(X_eval), dtype=float)
        return score, np.zeros(len(X_eval), dtype=int)
    if model_name == "train_mean":
        train_rate = float(y_train_binary.mean())
        score = np.full(len(X_eval), train_rate, dtype=float)
        return score, (score >= 0.5).astype(int)
    if model_name == "logistic_regression":
        return _run_logistic_model(X_train, y_train_binary, X_eval)
    raise ValueError(f"Unsupported classification model: {model_name}")


def run_single_baseline_model(
    db_path: str | Path,
    dataset_id: int,
    target_column: str,
    feature_set_name: str,
    model_name: str,
    n_folds: int = 3,
    final_test_fraction: float = 0.20,
    feature_columns_override: list[str] | None = None,
    feature_frame_override: pd.DataFrame | None = None,
    feature_set_metadata: dict[str, Any] | None = None,
    phase: str | None = None,
) -> ModelRunSummary:
    definition = get_target_definition(target_column)
    if model_name not in definition.allowed_models:
        raise ValueError(f"Model {model_name!r} is not allowed for target {target_column!r}.")
    training = load_training_dataset(db_path, dataset_id, definition.base_label_column)
    dataset_hash = dataset_hash_for_id(db_path, dataset_id)
    if feature_frame_override is not None:
        if not feature_frame_override.index.equals(training.X.index):
            raise ValueError("Feature frame override must use the same row index as the training dataset.")
        forbidden_prefixes = ("label_", "audit_", "raw_", "forward_")
        forbidden_columns = [
            column
            for column in feature_frame_override.columns
            if str(column).startswith(forbidden_prefixes)
            or str(column) in {"snapshot_id", "dataset_id", "ticker", "trading_date", "as_of_timestamp"}
        ]
        if forbidden_columns:
            raise ValueError(f"Feature frame override contains forbidden columns: {', '.join(forbidden_columns)}")
    if feature_columns_override is None:
        selected_features = select_feature_columns(training.feature_columns, feature_set_name)
    else:
        source_columns = feature_frame_override.columns.tolist() if feature_frame_override is not None else training.feature_columns
        allowed = set(source_columns)
        missing = sorted(set(feature_columns_override) - allowed)
        if missing:
            raise ValueError(f"Feature override contains columns outside the selected feature frame: {', '.join(missing)}")
        selected_features = [column for column in source_columns if column in set(feature_columns_override)]
    if not selected_features:
        raise ValueError(f"Feature set {feature_set_name!r} has no columns.")
    horizon_sessions = definition.horizon_sessions
    target_horizon = f"{definition.horizon_sessions}_session"
    purge_sessions = horizon_sessions
    embargo_sessions = horizon_sessions
    split_config = split_config_dict(
        definition.base_label_column,
        n_folds=n_folds,
        final_test_fraction=final_test_fraction,
        purge_sessions=purge_sessions,
        embargo_sessions=embargo_sessions,
    )
    split_config["target_column"] = target_column
    split_config["base_label_column"] = definition.base_label_column
    split_config["target_definition"] = target_metadata(definition)
    task = definition.task
    config = {
        "phase": phase or ("2D-3" if definition.name != definition.base_label_column else "2D-1"),
        "research_only": True,
        "scanner_scoring_effect": 0,
        "no_buy_sell_hold_recommendations": True,
        "model_name": model_name,
        "feature_set_name": feature_set_name,
        "target_definition": target_metadata(definition),
        "feature_set_metadata": feature_set_metadata or {},
        "uses_derived_feature_frame": feature_frame_override is not None,
    }
    model_run_id = insert_model_run(
        db_path,
        dataset_id=dataset_id,
        dataset_hash=dataset_hash,
        target_column=target_column,
        target_horizon=target_horizon,
        task=task,
        feature_set_name=feature_set_name,
        model_name=model_name,
        config=config,
        split_config=split_config,
        feature_columns=selected_features,
    )

    warnings: list[str] = []
    try:
        target_series = precompute_target_series(definition, training.y, training.X, training.metadata)
        valid = target_series.notna()
        source_X = feature_frame_override if feature_frame_override is not None else training.X
        X = source_X.loc[valid, selected_features].copy()
        y = target_series.loc[valid].astype(float)
        metadata = training.metadata.loc[valid].copy()
        folds = make_walk_forward_splits(
            metadata,
            y,
            horizon_sessions=horizon_sessions,
            n_folds=n_folds,
            final_test_fraction=final_test_fraction,
            purge_sessions=purge_sessions,
            embargo_sessions=embargo_sessions,
        )
        final_metrics: dict[str, Any] = {}
        for split in folds:
            train_mask = _date_mask(metadata, split.train_dates)
            eval_mask = _date_mask(metadata, split.eval_dates)
            y_train_all, y_eval_all, target_transform_metadata = transform_target_for_split(definition, y, train_mask, eval_mask)
            train_index = y_train_all.dropna().index
            eval_index = y_eval_all.dropna().index
            X_train = X.loc[train_index]
            y_train = y_train_all.loc[train_index]
            X_eval = X.loc[eval_index]
            y_eval = y_eval_all.loc[eval_index]
            meta_eval = metadata.loc[eval_index]
            if X_train.empty or X_eval.empty:
                warnings.append(f"{split.fold_name}: empty train/eval split skipped.")
                continue

            if task == "classification":
                y_train_binary = y_train.astype(int)
                y_eval_binary = y_eval.astype(int)
                if y_train_binary.nunique() < 2:
                    warnings.append(f"{split.fold_name}: logistic regression skipped because train set has one class.")
                    continue
                y_score, y_pred_label = _run_classification_model(model_name, X_train, y_train_binary, X_eval)
                metrics = classification_metrics(y_eval_binary, y_score, y_pred_label)
                metrics["target_transform_metadata"] = target_transform_metadata
                prediction_rows = _prediction_rows(
                    model_run_id,
                    meta_eval,
                    target_horizon,
                    split,
                    y_eval_binary,
                    y_score,
                    feature_set_name,
                    model_name,
                    y_score=y_score,
                    y_pred_label=y_pred_label,
                )
            else:
                y_pred = _run_regression_model(model_name, X_train, y_train, X_eval)
                metric_frame = _prediction_metric_frame(meta_eval, y_eval, y_pred)
                metrics = regression_metrics(y_eval, y_pred, float(y_train.mean()), metric_frame)
                metrics["target_transform_metadata"] = target_transform_metadata
                prediction_rows = _prediction_rows(
                    model_run_id,
                    meta_eval,
                    target_horizon,
                    split,
                    y_eval,
                    y_pred,
                    feature_set_name,
                    model_name,
                )

            insert_fold_metric(
                db_path,
                model_run_id,
                split.fold_name,
                split.split_name,
                split.train_start,
                split.train_end,
                split.eval_start,
                split.eval_end,
                len(X_train),
                len(X_eval),
                metrics,
            )
            insert_predictions(db_path, prediction_rows)
            if split.fold_name == "final_test":
                final_metrics = metrics
                insert_final_metric(db_path, model_run_id, "test", metrics)

        complete_model_run(db_path, model_run_id, "completed", warnings)
        return ModelRunSummary(
            model_run_id=model_run_id,
            dataset_id=dataset_id,
            target_column=target_column,
            feature_set_name=feature_set_name,
            model_name=model_name,
            final_metrics=final_metrics,
            warnings=warnings,
        )
    except Exception as exc:
        complete_model_run(db_path, model_run_id, "failed", [str(exc)])
        raise


def run_baseline_suite(
    db_path: str | Path,
    dataset_id: int,
    horizons: list[str] | None = None,
    feature_sets: list[str] | None = None,
    model_names: list[str] | None = None,
) -> list[ModelRunSummary]:
    horizons = horizons or ["5_session", "1_session", "20_session"]
    feature_sets = feature_sets or list(FEATURE_SET_NAMES)
    model_names = model_names or list(MODEL_NAMES)
    summaries: list[ModelRunSummary] = []
    for horizon in horizons:
        target_column = TARGET_COLUMNS[horizon]
        for feature_set_name in feature_sets:
            for model_name in model_names:
                summaries.append(
                    run_single_baseline_model(
                        db_path,
                        dataset_id=dataset_id,
                        target_column=target_column,
                        feature_set_name=feature_set_name,
                        model_name=model_name,
                    )
                )
    return summaries


def run_target_engineering_suite(
    db_path: str | Path,
    dataset_id: int,
    target_names: list[str] | None = None,
    feature_sets: list[str] | None = None,
) -> list[ModelRunSummary]:
    target_names = target_names or list(TARGET_ENGINEERING_TARGETS)
    feature_sets = feature_sets or list(FEATURE_SET_NAMES)
    summaries: list[ModelRunSummary] = []
    for target_name in target_names:
        definition = get_target_definition(target_name)
        for feature_set_name in feature_sets:
            for model_name in definition.allowed_models:
                summaries.append(
                    run_single_baseline_model(
                        db_path,
                        dataset_id=dataset_id,
                        target_column=target_name,
                        feature_set_name=feature_set_name,
                        model_name=model_name,
                    )
                )
    return summaries


def run_feature_pruning_suite(
    db_path: str | Path,
    dataset_id: int,
    output_dir: str | Path,
    target_names: list[str] | None = None,
    include_binary: bool = False,
) -> tuple[dict[str, Any], Path, list[ModelRunSummary]]:
    """Create a Phase 2D-4 feature-quality artifact and run simple baselines.

    Feature selection uses the artifact's development-fold-only logic. The final
    test labels remain untouched until each persisted model run evaluates them.
    """
    target_names = target_names or [
        "label_5_session_excess_return",
        "label_5_session_excess_return_winsorized_q01_q99",
    ]
    if include_binary:
        target_names = [*target_names, "label_5_session_excess_return_top_bottom_q20"]
    artifact = build_feature_quality_audit(db_path, dataset_id, target_name="label_5_session_excess_return")
    artifact_path = write_feature_quality_artifact(artifact, output_dir)
    feature_sets = pruned_feature_sets_from_artifact(artifact)
    summaries: list[ModelRunSummary] = []
    for target_name in target_names:
        definition = get_target_definition(target_name)
        for feature_set_name, columns in feature_sets.items():
            for model_name in definition.allowed_models:
                summaries.append(
                    run_single_baseline_model(
                        db_path,
                        dataset_id=dataset_id,
                        target_column=target_name,
                        feature_set_name=feature_set_name,
                        model_name=model_name,
                        feature_columns_override=columns,
                        feature_set_metadata={
                            "source": "phase2d4_feature_quality_artifact",
                            "artifact": str(artifact_path),
                            "column_count": len(columns),
                            "selection_data_scope": "development_folds_only_final_test_labels_not_used",
                        },
                        phase="2D-4",
                    )
                )
    return artifact, artifact_path, summaries
