from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage
from src.datasets.feature_manifest import role_sets_from_frame
from src.datasets.repository import flatten_saved_dataset


@dataclass(frozen=True)
class TrainingDataset:
    X: pd.DataFrame
    y: pd.Series
    metadata: pd.DataFrame
    audit: pd.DataFrame
    feature_columns: list[str]
    label_column: str


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _build_row(db_path: str | Path, dataset_id: int) -> dict[str, Any]:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM dataset_builds WHERE dataset_id = ?", (int(dataset_id),)).fetchone()
    if row is None:
        raise ValueError(f"Dataset #{dataset_id} not found.")
    return dict(row)


def load_training_dataset(db_path: str | Path, dataset_id: int, label_column: str) -> TrainingDataset:
    """Load model features and one explicit target without audit/metadata leakage."""
    frame = flatten_saved_dataset(db_path, dataset_id)
    if frame.empty:
        raise ValueError(f"Dataset #{dataset_id} has no rows.")

    build = _build_row(db_path, dataset_id)
    feature_columns = _json_loads(build.get("feature_columns_json"), []) or []
    label_columns = _json_loads(build.get("label_columns_json"), []) or []
    identifier_columns = _json_loads(build.get("identifier_columns_json"), []) or []
    metadata_columns = _json_loads(build.get("metadata_columns_json"), []) or []
    audit_columns = _json_loads(build.get("audit_columns_json"), []) or []

    if not feature_columns:
        feature_columns = role_sets_from_frame(frame).model_features
    if not label_columns:
        label_columns = role_sets_from_frame(frame).label_columns
    if not identifier_columns or not metadata_columns or not audit_columns:
        role_sets = role_sets_from_frame(frame)
        identifier_columns = identifier_columns or role_sets.identifier_columns
        metadata_columns = metadata_columns or role_sets.metadata_columns
        audit_columns = audit_columns or role_sets.audit_columns

    if label_column not in label_columns or label_column not in frame.columns:
        available = ", ".join(label_columns)
        raise ValueError(f"Label column {label_column!r} is not available. Available labels: {available}")

    forbidden = set(audit_columns) | set(label_columns) | set(identifier_columns) | set(metadata_columns)
    leaked = sorted(set(feature_columns) & forbidden)
    if leaked:
        raise ValueError(f"Dataset feature manifest contains forbidden columns: {', '.join(leaked)}")

    missing_features = [column for column in feature_columns if column not in frame.columns]
    if missing_features:
        raise ValueError(f"Dataset is missing model feature columns: {', '.join(missing_features)}")

    X = frame[feature_columns].copy()
    y = frame[label_column].copy()
    metadata = frame[[column for column in [*identifier_columns, *metadata_columns] if column in frame.columns]].copy()
    audit = frame[[column for column in audit_columns if column in frame.columns]].copy()
    return TrainingDataset(
        X=X,
        y=y,
        metadata=metadata,
        audit=audit,
        feature_columns=feature_columns,
        label_column=label_column,
    )
