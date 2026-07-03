from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.annotations.repository import list_annotations
from src.data import storage
from src.datasets.training_loader import TrainingDataset, load_training_dataset
from src.modeling.feature_sets import technical_core_columns
from src.modeling.splits import make_walk_forward_splits
from src.modeling.targets import RAW_TARGET_5_SESSION, get_target_definition, precompute_target_series, target_metadata
from src.utils.trading_calendar import trading_days_between


ANNOTATION_FEATURE_VERSION = "research_annotation_features_v1"
ANNOTATION_FEATURE_COLUMNS = [
    "recent_positive_annotation_count_20s",
    "recent_negative_annotation_count_20s",
    "days_since_latest_positive_annotation",
    "days_since_latest_negative_annotation",
    "max_recent_annotation_strength",
    "weighted_recent_annotation_sentiment",
    "annotation_coverage_available",
]
ANNOTATION_FEATURE_SET_NAMES = [
    "technical_core",
    "annotation_features_only",
    "technical_core_plus_annotations",
]
SENTIMENT_WEIGHTS = {
    "positive": 1.0,
    "negative": -1.0,
    "mixed": 0.0,
    "neutral": 0.0,
    "unknown": 0.0,
}


@dataclass(frozen=True)
class AnnotationArtifactInfo:
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


def _as_date(value: Any) -> date:
    return pd.to_datetime(value).date()


def _as_utc(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.min.tz_localize(UTC)
    return parsed


def _trading_sessions_between(start: date, end: date) -> int | None:
    if end < start:
        return None
    days = trading_days_between(start, end)
    if not days:
        return None
    return max(0, len(days) - 1)


def _annotation_rows_for_features(db_path: str | Path) -> pd.DataFrame:
    frame = list_annotations(db_path, limit=None)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "event_date_parsed",
                "available_at_parsed",
                "sentiment_label",
                "strength",
                "confidence",
            ]
        )
    output = frame.copy()
    output["ticker"] = output["ticker"].astype(str).str.upper()
    output["event_date_parsed"] = pd.to_datetime(output["event_date"], errors="coerce").dt.date
    output["available_at_parsed"] = pd.to_datetime(output["available_at"], utc=True, errors="coerce")
    output["sentiment_label"] = output["sentiment_label"].fillna("unknown").astype(str).str.lower()
    output["strength"] = pd.to_numeric(output["strength"], errors="coerce").fillna(0).clip(lower=0, upper=10).astype(float)
    output["confidence"] = pd.to_numeric(output["confidence"], errors="coerce").fillna(0).clip(lower=0, upper=1).astype(float)
    output = output.dropna(subset=["event_date_parsed", "available_at_parsed"])
    return output


def derive_annotation_features(db_path: str | Path, metadata: pd.DataFrame) -> pd.DataFrame:
    """Derive point-in-time research-only annotation features for dataset rows.

    An annotation can activate only after both its `available_at` timestamp and
    its event date are known at the snapshot. This prevents future event labels
    from becoming historical model inputs.
    """
    annotations = _annotation_rows_for_features(db_path)
    if annotations.empty:
        return pd.DataFrame(0.0, index=metadata.index, columns=ANNOTATION_FEATURE_COLUMNS)

    by_ticker = {
        ticker: group.sort_values(["available_at_parsed", "event_date_parsed", "annotation_id" if "annotation_id" in group.columns else "ticker"]).copy()
        for ticker, group in annotations.groupby("ticker", dropna=False)
    }
    rows: list[dict[str, float]] = []
    for _, row in metadata.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        snapshot_date = _as_date(row.get("trading_date"))
        as_of = _as_utc(row.get("as_of_timestamp") or snapshot_date)
        ticker_events = by_ticker.get(ticker)
        if ticker_events is None or ticker_events.empty:
            rows.append({column: 0.0 for column in ANNOTATION_FEATURE_COLUMNS})
            continue

        available = ticker_events[
            (ticker_events["available_at_parsed"].le(as_of))
            & (ticker_events["event_date_parsed"].map(lambda value: value <= snapshot_date))
        ].copy()
        if available.empty:
            rows.append({column: 0.0 for column in ANNOTATION_FEATURE_COLUMNS})
            continue

        available["sessions_since"] = available["event_date_parsed"].map(lambda value: _trading_sessions_between(value, snapshot_date))
        available = available.dropna(subset=["sessions_since"])
        recent = available[available["sessions_since"].le(20)].copy()
        positives = available[available["sentiment_label"].eq("positive")]
        negatives = available[available["sentiment_label"].eq("negative")]
        recent_positive = recent[recent["sentiment_label"].eq("positive")]
        recent_negative = recent[recent["sentiment_label"].eq("negative")]

        recent_strength = pd.to_numeric(recent.get("strength"), errors="coerce").fillna(0) if not recent.empty else pd.Series(dtype=float)
        weights = recent["sentiment_label"].map(SENTIMENT_WEIGHTS).fillna(0.0) if not recent.empty else pd.Series(dtype=float)
        weighted_base = (recent_strength * pd.to_numeric(recent.get("confidence"), errors="coerce").fillna(0)).astype(float) if not recent.empty else pd.Series(dtype=float)
        denominator = float(weighted_base.abs().sum()) if not weighted_base.empty else 0.0
        weighted_sentiment = float((weights * weighted_base).sum() / denominator) if denominator > 0 else 0.0

        rows.append(
            {
                "recent_positive_annotation_count_20s": float(len(recent_positive)),
                "recent_negative_annotation_count_20s": float(len(recent_negative)),
                "days_since_latest_positive_annotation": float(positives["sessions_since"].min()) if not positives.empty else np.nan,
                "days_since_latest_negative_annotation": float(negatives["sessions_since"].min()) if not negatives.empty else np.nan,
                "max_recent_annotation_strength": float(recent_strength.max()) if not recent_strength.empty else 0.0,
                "weighted_recent_annotation_sentiment": weighted_sentiment,
                "annotation_coverage_available": 1.0,
            }
        )

    return pd.DataFrame(rows, index=metadata.index, columns=ANNOTATION_FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan)


def build_annotation_model_frame(training: TrainingDataset, db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    derived = derive_annotation_features(db_path, training.metadata)
    technical_core = [column for column in technical_core_columns(training.feature_columns) if column in training.X.columns]
    model_frame = pd.concat([training.X[technical_core].copy(), derived], axis=1)
    feature_sets = {
        "technical_core": technical_core,
        "annotation_features_only": list(ANNOTATION_FEATURE_COLUMNS),
        "technical_core_plus_annotations": [*technical_core, *ANNOTATION_FEATURE_COLUMNS],
    }
    return model_frame, derived, feature_sets


def _date_mask(metadata: pd.DataFrame, dates: list[Any]) -> pd.Series:
    date_set = {pd.to_datetime(value).date() for value in dates}
    return pd.to_datetime(metadata["trading_date"]).dt.date.isin(date_set)


def _annotation_active_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(pd.to_numeric(frame["annotation_coverage_available"], errors="coerce").fillna(0).gt(0).mean())


def _feature_active_counts(derived: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in ANNOTATION_FEATURE_COLUMNS:
        numeric = pd.to_numeric(derived[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "active_count": int(numeric.fillna(0).ne(0).sum()),
                "active_rate": float(numeric.fillna(0).ne(0).mean()),
                "missing_rate": float(numeric.isna().mean()),
                "mean": None if pd.isna(numeric.mean()) else float(numeric.mean()),
            }
        )
    return rows


def _annotation_db_summary(db_path: str | Path) -> dict[str, Any]:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        rows = pd.read_sql_query("SELECT * FROM research_event_annotations", conn)
    if rows.empty:
        return {
            "annotation_count": 0,
            "by_ticker": [],
            "by_event_type": [],
            "sentiment_distribution": [],
            "missing_strength_or_confidence": {"zero_strength": 0, "zero_confidence": 0},
        }
    by_ticker = rows.groupby("ticker").size().reset_index(name="annotation_count").sort_values("ticker").to_dict("records")
    by_event = rows.groupby("event_type").size().reset_index(name="annotation_count").sort_values("event_type").to_dict("records")
    sentiment = rows.groupby("sentiment_label").size().reset_index(name="annotation_count").sort_values("sentiment_label").to_dict("records")
    return {
        "annotation_count": int(len(rows)),
        "by_ticker": by_ticker,
        "by_event_type": by_event,
        "sentiment_distribution": sentiment,
        "missing_strength_or_confidence": {
            "zero_strength": int(pd.to_numeric(rows["strength"], errors="coerce").fillna(0).eq(0).sum()),
            "zero_confidence": int(pd.to_numeric(rows["confidence"], errors="coerce").fillna(0).eq(0).sum()),
        },
    }


def annotation_density_by_fold(
    derived: pd.DataFrame,
    metadata: pd.DataFrame,
    target_series: pd.Series,
    horizon_sessions: int,
    n_folds: int = 3,
) -> list[dict[str, Any]]:
    folds = make_walk_forward_splits(
        metadata,
        target_series,
        horizon_sessions=horizon_sessions,
        n_folds=n_folds,
        purge_sessions=horizon_sessions,
        embargo_sessions=horizon_sessions,
    )
    rows: list[dict[str, Any]] = []
    for split in folds:
        for role, dates in [("train", split.train_dates), ("eval", split.eval_dates)]:
            subset = derived.loc[_date_mask(metadata, dates)]
            rows.append(
                {
                    "fold_name": split.fold_name,
                    "split_name": split.split_name,
                    "role": role,
                    "rows": int(len(subset)),
                    "annotation_active_rate": _annotation_active_rate(subset),
                    "positive_active_rows": int(pd.to_numeric(subset["recent_positive_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                    "negative_active_rows": int(pd.to_numeric(subset["recent_negative_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                }
            )
    return rows


def build_annotation_coverage_audit(
    db_path: str | Path,
    dataset_id: int,
    target_name: str = RAW_TARGET_5_SESSION,
) -> dict[str, Any]:
    definition = get_target_definition(target_name)
    training = load_training_dataset(db_path, dataset_id, definition.base_label_column)
    target_series = precompute_target_series(definition, training.y, training.X, training.metadata)
    valid = target_series.notna()
    metadata = training.metadata.loc[valid].copy()
    derived = derive_annotation_features(db_path, metadata)
    density = annotation_density_by_fold(
        derived,
        metadata,
        target_series.loc[valid],
        horizon_sessions=definition.horizon_sessions,
    )
    joined = pd.concat([metadata[["ticker", "trading_date"]].reset_index(drop=True), derived.reset_index(drop=True)], axis=1)
    by_ticker: list[dict[str, Any]] = []
    for ticker, group in joined.groupby("ticker", dropna=False):
        by_ticker.append(
            {
                "ticker": str(ticker),
                "rows": int(len(group)),
                "annotation_active_rate": _annotation_active_rate(group),
                "positive_active_rows": int(pd.to_numeric(group["recent_positive_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                "negative_active_rows": int(pd.to_numeric(group["recent_negative_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
            }
        )
    sparse_tickers = [row["ticker"] for row in by_ticker if float(row["annotation_active_rate"]) < 0.01]
    artifact = {
        "annotation_feature_version": ANNOTATION_FEATURE_VERSION,
        "artifact_type": "annotation_coverage",
        "created_at": _now_iso(),
        "dataset_id": int(dataset_id),
        "target": target_metadata(definition),
        "row_count": int(len(derived)),
        "derived_feature_count": int(derived.shape[1]),
        "feature_sets": {
            "annotation_features_only": {"columns": list(ANNOTATION_FEATURE_COLUMNS), "column_count": len(ANNOTATION_FEATURE_COLUMNS)},
            "technical_core_plus_annotations": {"columns": [*technical_core_columns(training.feature_columns), *ANNOTATION_FEATURE_COLUMNS]},
        },
        "active_observation_counts": _feature_active_counts(derived),
        "coverage_by_ticker": sorted(by_ticker, key=lambda item: item["ticker"]),
        "coverage_by_fold": density,
        "sparse_tickers": sparse_tickers,
        "annotation_db_summary": _annotation_db_summary(db_path),
        "summary": {
            "annotation_rows": int(_annotation_db_summary(db_path)["annotation_count"]),
            "rows_with_annotation_coverage": int(pd.to_numeric(derived["annotation_coverage_available"], errors="coerce").fillna(0).gt(0).sum()),
            "annotation_active_rate": _annotation_active_rate(derived),
            "scanner_scoring_effect": 0,
            "research_only": True,
        },
    }
    return _clean_json(artifact)


def write_annotation_artifact(artifact: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dataset_id = artifact.get("dataset_id", "unknown")
    path = output / f"phase2d6a_annotation_coverage_dataset{dataset_id}_{timestamp}.json"
    path.write_text(_json_dumps(artifact), encoding="utf-8")
    return path


def load_annotation_artifact(path: str | Path) -> dict[str, Any]:
    return _json_loads(Path(path).read_text(encoding="utf-8"), {}) or {}


def list_annotation_artifacts(output_dir: str | Path) -> list[AnnotationArtifactInfo]:
    output = Path(output_dir)
    if not output.exists():
        return []
    artifacts: list[AnnotationArtifactInfo] = []
    for path in sorted(output.glob("phase2d6a_annotation_coverage_dataset*.json"), reverse=True):
        data = load_annotation_artifact(path)
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        artifacts.append(
            AnnotationArtifactInfo(
                path=path,
                dataset_id=data.get("dataset_id") if isinstance(data, dict) else None,
                created_at=str(data.get("created_at") or "") if isinstance(data, dict) else "",
                summary=f"active_rate={summary.get('annotation_active_rate', 0)} rows={summary.get('rows_with_annotation_coverage', 0)}",
            )
        )
    return artifacts


def run_annotation_baseline_suite(
    db_path: str | Path,
    dataset_id: int,
    output_dir: str | Path,
    target_names: list[str] | None = None,
) -> tuple[dict[str, Any], Path, list[Any]]:
    from src.modeling.runner import run_single_baseline_model

    target_names = target_names or [
        RAW_TARGET_5_SESSION,
        "label_5_session_excess_return_winsorized_q01_q99",
    ]
    coverage = build_annotation_coverage_audit(db_path, dataset_id)
    coverage_path = write_annotation_artifact(coverage, output_dir)
    training = load_training_dataset(db_path, dataset_id, RAW_TARGET_5_SESSION)
    model_frame, _derived, feature_sets = build_annotation_model_frame(training, db_path)
    summaries: list[Any] = []
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
                        feature_frame_override=model_frame,
                        feature_set_metadata={
                            "source": "phase2d6a_research_annotation_features",
                            "coverage_artifact": str(coverage_path),
                            "column_count": len(columns),
                            "research_only": True,
                            "scanner_scoring_effect": 0,
                            "active_catalyst_table_modified": False,
                        },
                        phase="2D-6A",
                    )
                )
    return coverage, coverage_path, summaries

