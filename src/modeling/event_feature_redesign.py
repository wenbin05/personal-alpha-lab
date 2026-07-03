from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data import storage
from src.datasets.training_loader import TrainingDataset, load_training_dataset
from src.modeling.feature_sets import technical_core_columns
from src.modeling.splits import make_walk_forward_splits
from src.modeling.targets import RAW_TARGET_5_SESSION, get_target_definition, precompute_target_series, target_metadata
from src.utils.trading_calendar import trading_days_between


EVENT_REDESIGN_VERSION = "event_feature_redesign_v1"
EVENT_REDESIGN_TARGETS = [
    RAW_TARGET_5_SESSION,
    "label_5_session_excess_return_winsorized_q01_q99",
]
EVENT_FEATURE_SET_NAMES = [
    "event_recency_only",
    "earnings_recency_surprise",
    "sec_recency_categories",
    "technical_core_plus_event_recency",
    "technical_core_plus_earnings_recency",
    "technical_core_plus_sec_recency",
]
SEC_REDESIGN_CATEGORIES = [
    "core_periodic",
    "current_event",
    "ownership",
    "equity_financing",
    "debt_financing",
    "structured_note",
    "registration_or_prospectus_other",
    "amendment",
]
DECAY_HALF_LIFE_SESSIONS = 10.0


@dataclass(frozen=True)
class EventRedesignArtifactInfo:
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


def _numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype(float)


def _bool_numeric(series: pd.Series) -> pd.Series:
    return _numeric(series).fillna(0).ne(0).astype(float)


def _days_series(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype=float)
    return _numeric(frame[column])


def _decay_from_days(days: pd.Series, half_life: float = DECAY_HALF_LIFE_SESSIONS, cap_days: float = 90.0) -> pd.Series:
    clean = pd.to_numeric(days, errors="coerce")
    decay = np.exp(-np.log(2.0) * clean.clip(lower=0, upper=cap_days) / float(half_life))
    return pd.Series(decay, index=days.index).where(clean.notna(), 0.0).astype(float)


def _bucket_columns(days: pd.Series, prefix: str) -> dict[str, pd.Series]:
    clean = pd.to_numeric(days, errors="coerce")
    return {
        f"{prefix}_same_day": clean.eq(0).astype(float),
        f"{prefix}_post_1_5d": clean.between(1, 5, inclusive="both").astype(float),
        f"{prefix}_post_6_20d": clean.between(6, 20, inclusive="both").astype(float),
        f"{prefix}_post_21_90d": clean.between(21, 90, inclusive="both").astype(float),
        f"{prefix}_missing_or_stale": (clean.isna() | clean.gt(90)).astype(float),
    }


def derive_event_features(X: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic event recency features from PIT model columns only."""
    output: dict[str, pd.Series] = {}

    for category in SEC_REDESIGN_CATEGORIES:
        days_col = f"sec_days_since_latest_{category}"
        event_7 = f"sec_{category}_event_days_7s"
        event_30 = f"sec_{category}_event_days_30s"
        event_90 = f"sec_{category}_event_days_90s"
        prefix = f"event_sec_{category}"
        days = _days_series(X, days_col)
        output[f"{prefix}_decay_90d"] = _decay_from_days(days)
        output[f"{prefix}_recent_7s"] = _bool_numeric(X[event_7]) if event_7 in X.columns else pd.Series(0.0, index=X.index)
        output[f"{prefix}_recent_30s"] = _bool_numeric(X[event_30]) if event_30 in X.columns else pd.Series(0.0, index=X.index)
        output[f"{prefix}_recent_90s"] = _bool_numeric(X[event_90]) if event_90 in X.columns else pd.Series(0.0, index=X.index)
        output.update(_bucket_columns(days, f"{prefix}_bucket"))

    for form_name, source_col in [
        ("8k", "sec_days_since_latest_8k"),
        ("10q", "sec_days_since_latest_10q"),
        ("10k", "sec_days_since_latest_10k"),
    ]:
        days = _days_series(X, source_col)
        output[f"event_sec_{form_name}_decay_90d"] = _decay_from_days(days)
        output.update(_bucket_columns(days, f"event_sec_{form_name}_bucket"))

    metadata_available = _bool_numeric(X["sec_metadata_available"]) if "sec_metadata_available" in X.columns else pd.Series(0.0, index=X.index)
    structured_note = (
        _bool_numeric(X["sec_recent_structured_note_flag"])
        if "sec_recent_structured_note_flag" in X.columns
        else pd.Series(0.0, index=X.index)
    )
    output["event_sec_metadata_available"] = metadata_available
    output["event_sec_structured_note_recent_30s"] = structured_note
    # Structured-note overload is diagnostic/control-style; keep it out of default feature sets unless later justified.
    output["event_sec_structured_note_overload_control"] = (
        _numeric(X["sec_structured_note_event_days_30s"]).fillna(0).clip(lower=0, upper=5) / 5.0
        if "sec_structured_note_event_days_30s" in X.columns
        else pd.Series(0.0, index=X.index)
    )

    earnings_days = _days_series(X, "sessions_since_latest_earnings")
    output["event_earnings_data_available"] = (
        _bool_numeric(X["earnings_data_available"]) if "earnings_data_available" in X.columns else pd.Series(0.0, index=X.index)
    )
    output["event_earnings_timing_known"] = (
        _bool_numeric(X["earnings_timing_known"]) if "earnings_timing_known" in X.columns else pd.Series(0.0, index=X.index)
    )
    output["event_earnings_decay_20s"] = _decay_from_days(earnings_days, half_life=5.0, cap_days=20.0)
    output["event_earnings_same_session"] = earnings_days.eq(0).astype(float)
    output["event_earnings_post_1_5s"] = earnings_days.between(1, 5, inclusive="both").astype(float)
    output["event_earnings_post_6_20s"] = earnings_days.between(6, 20, inclusive="both").astype(float)
    output["event_earnings_missing_or_stale"] = (earnings_days.isna() | earnings_days.gt(20)).astype(float)

    eps = _numeric(X["latest_eps_surprise_percent"]) if "latest_eps_surprise_percent" in X.columns else pd.Series(np.nan, index=X.index)
    eps_clipped = eps.clip(lower=-100.0, upper=100.0)
    output["event_earnings_eps_surprise_magnitude_clipped"] = eps_clipped.fillna(0.0) / 100.0
    output["event_earnings_eps_surprise_positive"] = eps.gt(0).astype(float)
    output["event_earnings_eps_surprise_negative"] = eps.lt(0).astype(float)
    output["event_earnings_eps_surprise_missing"] = eps.isna().astype(float)
    output["event_earnings_eps_surprise_large_abs"] = eps.abs().ge(20.0).fillna(False).astype(float)
    output["event_earnings_revenue_surprise_missing"] = (
        _numeric(X["latest_revenue_surprise_percent"]).isna().astype(float)
        if "latest_revenue_surprise_percent" in X.columns
        else pd.Series(1.0, index=X.index)
    )

    output["event_catalyst_active"] = (
        _numeric(X["active_catalyst_count"]).fillna(0).gt(0).astype(float)
        if "active_catalyst_count" in X.columns
        else pd.Series(0.0, index=X.index)
    )
    output["event_llm_supported_active"] = (
        _numeric(X["published_llm_supported_count"]).fillna(0).gt(0).astype(float)
        if "published_llm_supported_count" in X.columns
        else pd.Series(0.0, index=X.index)
    )

    frame = pd.DataFrame(output, index=X.index)
    return frame.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def event_feature_groups(columns: list[str]) -> dict[str, list[str]]:
    sec = [
        column
        for column in columns
        if column.startswith("event_sec_")
        and column != "event_sec_structured_note_overload_control"
        and not column.endswith("_missing_or_stale")
    ]
    earnings = [column for column in columns if column.startswith("event_earnings_")]
    catalyst = [column for column in columns if column.startswith("event_catalyst_") or column.startswith("event_llm_")]
    controls = [column for column in columns if column.endswith("_control")]
    return {
        "sec": sec,
        "earnings": earnings,
        "catalyst_llm": catalyst,
        "controls_audit_only_by_default": controls,
    }


def build_event_model_frame(training: TrainingDataset) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    derived = derive_event_features(training.X)
    technical_core = [column for column in technical_core_columns(training.feature_columns) if column in training.X.columns]
    model_frame = pd.concat([training.X[technical_core].copy(), derived], axis=1)
    groups = event_feature_groups(derived.columns.tolist())
    return model_frame, derived, groups


def redesigned_event_feature_sets(training: TrainingDataset, derived: pd.DataFrame) -> dict[str, list[str]]:
    technical_core = [column for column in technical_core_columns(training.feature_columns) if column in training.X.columns]
    groups = event_feature_groups(derived.columns.tolist())
    def eligible(columns: list[str]) -> list[str]:
        selected: list[str] = []
        for column in columns:
            active_rate = float((pd.to_numeric(derived[column], errors="coerce").fillna(0) != 0).mean())
            if active_rate <= 0.005 or active_rate >= 0.995:
                continue
            selected.append(column)
        return selected

    sec_columns = eligible(groups["sec"])
    earnings_columns = eligible(groups["earnings"])
    catalyst_columns = groups["catalyst_llm"]
    active_catalyst_columns = [
        column
        for column in catalyst_columns
        if float((pd.to_numeric(derived[column], errors="coerce").fillna(0) != 0).mean()) >= 0.01
    ]
    event_columns = [*sec_columns, *earnings_columns, *active_catalyst_columns]
    return {
        "event_recency_only": event_columns,
        "earnings_recency_surprise": earnings_columns,
        "sec_recency_categories": sec_columns,
        "technical_core_plus_event_recency": [*technical_core, *event_columns],
        "technical_core_plus_earnings_recency": [*technical_core, *earnings_columns],
        "technical_core_plus_sec_recency": [*technical_core, *sec_columns],
    }


def _date_mask(metadata: pd.DataFrame, dates: list[Any]) -> pd.Series:
    date_set = {pd.to_datetime(value).date() for value in dates}
    return pd.to_datetime(metadata["trading_date"]).dt.date.isin(date_set)


def _active_rate(frame: pd.DataFrame, columns: list[str]) -> float:
    if frame.empty or not columns:
        return 0.0
    numeric = frame[columns].apply(pd.to_numeric, errors="coerce").fillna(0)
    return float(numeric.ne(0).any(axis=1).mean())


def _feature_active_counts(derived: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    groups = event_feature_groups(derived.columns.tolist())
    group_lookup = {column: group for group, columns in groups.items() for column in columns}
    for column in derived.columns:
        numeric = pd.to_numeric(derived[column], errors="coerce").fillna(0)
        rows.append(
            {
                "feature": column,
                "group": group_lookup.get(column, "event"),
                "active_count": int(numeric.ne(0).sum()),
                "active_rate": float(numeric.ne(0).mean()),
                "mean": _safe_float(numeric.mean()),
                "std": _safe_float(numeric.std(ddof=0)),
            }
        )
    return sorted(rows, key=lambda item: (-item["active_rate"], item["feature"]))


def event_density_by_fold(
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
    groups = event_feature_groups(derived.columns.tolist())
    rows: list[dict[str, Any]] = []
    for split in folds:
        for role, dates in [("train", split.train_dates), ("eval", split.eval_dates)]:
            mask = _date_mask(metadata, dates)
            subset = derived.loc[mask]
            row = {
                "fold_name": split.fold_name,
                "split_name": split.split_name,
                "role": role,
                "rows": int(len(subset)),
            }
            for group, columns in groups.items():
                row[f"{group}_active_rate"] = _active_rate(subset, columns)
            rows.append(row)
    return rows


def coverage_by_ticker_date(
    metadata: pd.DataFrame,
    derived: pd.DataFrame,
    max_date_rows: int = 40,
) -> dict[str, Any]:
    frame = pd.concat(
        [
            metadata[["ticker", "trading_date"]].reset_index(drop=True),
            derived.reset_index(drop=True),
        ],
        axis=1,
    )
    groups = event_feature_groups(derived.columns.tolist())
    ticker_rows: list[dict[str, Any]] = []
    for ticker, group in frame.groupby("ticker", dropna=False):
        row = {"ticker": str(ticker), "rows": int(len(group))}
        for group_name, columns in groups.items():
            row[f"{group_name}_active_rate"] = _active_rate(group, columns)
        ticker_rows.append(row)
    date_activity = frame.copy()
    date_activity["trading_date"] = pd.to_datetime(date_activity["trading_date"]).dt.date.astype(str)
    date_rows: list[dict[str, Any]] = []
    for trading_date, group in date_activity.groupby("trading_date", dropna=False):
        row = {"trading_date": trading_date, "rows": int(len(group))}
        for group_name, columns in groups.items():
            row[f"{group_name}_active_rate"] = _active_rate(group, columns)
        date_rows.append(row)
    date_rows = sorted(date_rows, key=lambda item: max(item.get("sec_active_rate", 0), item.get("earnings_active_rate", 0)), reverse=True)
    return {
        "by_ticker": sorted(ticker_rows, key=lambda item: item["ticker"]),
        "by_date_top_activity": date_rows[:max_date_rows],
    }


def _event_rows_from_db(db_path: str | Path) -> dict[str, pd.DataFrame]:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        sec = pd.read_sql_query(
            """
            SELECT c.id, c.ticker, c.available_at, c.created_at, c.event_date,
                   s.classification, s.feature_eligible, s.form
            FROM catalysts c
            LEFT JOIN sec_filing_classifications s ON s.catalyst_id = c.id
            WHERE c.event_type = 'sec_filing' AND c.source = 'SEC EDGAR'
            ORDER BY c.ticker, datetime(c.available_at), c.id
            """,
            conn,
        )
        earnings = pd.read_sql_query(
            """
            SELECT earnings_event_id, ticker, available_at, announced_at, timing,
                   eps_surprise_percent, revenue_surprise_percent
            FROM earnings_events
            ORDER BY ticker, datetime(available_at), earnings_event_id
            """,
            conn,
        )
        catalysts = pd.read_sql_query(
            """
            SELECT id, ticker, event_type, source, available_at, created_at, sentiment_label,
                   catalyst_strength, confidence
            FROM catalysts
            WHERE NOT (event_type = 'sec_filing' AND source = 'SEC EDGAR')
            ORDER BY ticker, datetime(COALESCE(available_at, created_at)), id
            """,
            conn,
        )
    return {"sec": sec, "earnings": earnings, "catalysts": catalysts}


def _parse_available_date(frame: pd.DataFrame, fallback_column: str | None = None) -> pd.DataFrame:
    if frame.empty:
        return frame
    parsed = pd.to_datetime(frame.get("available_at"), utc=True, errors="coerce")
    if fallback_column and fallback_column in frame:
        fallback = pd.to_datetime(frame[fallback_column], utc=True, errors="coerce")
        parsed = parsed.where(parsed.notna(), fallback)
    output = frame.copy()
    output["available_at_parsed"] = parsed
    output["available_date"] = parsed.dt.date
    return output.dropna(subset=["available_date"]).copy()


def _event_lag_summary(events: pd.DataFrame, metadata: pd.DataFrame, event_kind: str, category_column: str | None = None) -> list[dict[str, Any]]:
    if events.empty:
        return []
    rows: list[dict[str, Any]] = []
    metadata_dates = {
        ticker: sorted(pd.to_datetime(group["trading_date"]).dt.date.unique().tolist())
        for ticker, group in metadata.groupby("ticker", dropna=False)
    }
    for _, event in events.iterrows():
        ticker = str(event.get("ticker") or "").upper()
        available_date = event.get("available_date")
        if not ticker or not available_date or ticker not in metadata_dates:
            continue
        dates = [value for value in metadata_dates[ticker] if value >= available_date]
        for snapshot_date in dates[:80]:
            sessions = max(0, len(trading_days_between(available_date, snapshot_date)) - 1)
            if sessions == 0:
                bucket = "same_session"
            elif sessions <= 5:
                bucket = "post_1_5_sessions"
            elif sessions <= 20:
                bucket = "post_6_20_sessions"
            else:
                bucket = "later_than_20_sessions"
            rows.append(
                {
                    "event_kind": event_kind,
                    "category": str(event.get(category_column) or event_kind) if category_column else event_kind,
                    "bucket": bucket,
                    "ticker": ticker,
                }
            )
    if not rows:
        return []
    frame = pd.DataFrame(rows)
    grouped = frame.groupby(["event_kind", "category", "bucket"], dropna=False).size().reset_index(name="snapshot_observations")
    return grouped.sort_values(["event_kind", "category", "bucket"]).to_dict("records")


def _pre_availability_violations(
    training: TrainingDataset,
    events: pd.DataFrame,
    feature_columns_by_category: dict[str, list[str]],
    category_column: str,
) -> list[dict[str, Any]]:
    if events.empty:
        return []
    violations: list[dict[str, Any]] = []
    metadata = training.metadata.copy()
    metadata["date"] = pd.to_datetime(metadata["trading_date"]).dt.date
    for (ticker, category), group in events.groupby(["ticker", category_column], dropna=False):
        category = str(category or "unknown")
        columns = [column for column in feature_columns_by_category.get(category, []) if column in training.X.columns]
        if not columns:
            continue
        earliest = min(group["available_date"])
        row_mask = metadata["ticker"].astype(str).str.upper().eq(str(ticker).upper()) & metadata["date"].lt(earliest)
        if not bool(row_mask.any()):
            continue
        values = training.X.loc[row_mask, columns].apply(pd.to_numeric, errors="coerce").fillna(0)
        active_rows = int(values.ne(0).any(axis=1).sum())
        if active_rows:
            violations.append(
                {
                    "ticker": str(ticker),
                    "category": category,
                    "earliest_available_date": str(earliest),
                    "pre_availability_active_rows": active_rows,
                    "checked_columns": columns,
                }
            )
    return violations


def build_event_coverage_audit(
    db_path: str | Path,
    dataset_id: int,
    target_name: str = RAW_TARGET_5_SESSION,
) -> dict[str, Any]:
    definition = get_target_definition(target_name)
    training = load_training_dataset(db_path, dataset_id, definition.base_label_column)
    target_series = precompute_target_series(definition, training.y, training.X, training.metadata)
    valid = target_series.notna()
    metadata = training.metadata.loc[valid].copy()
    derived = derive_event_features(training.X.loc[valid])
    feature_sets = redesigned_event_feature_sets(training, derive_event_features(training.X))
    groups = event_feature_groups(derived.columns.tolist())
    density = event_density_by_fold(
        derived,
        metadata,
        target_series.loc[valid],
        horizon_sessions=definition.horizon_sessions,
    )
    coverage = coverage_by_ticker_date(metadata, derived)
    active_counts = _feature_active_counts(derived)
    inactive_groups = [
        group
        for group, columns in groups.items()
        if columns and max((row["active_rate"] for row in active_counts if row["feature"] in columns), default=0.0) < 0.01
    ]
    db_events = _event_rows_from_db(db_path)
    db_counts = {
        name: int(len(frame)) for name, frame in db_events.items()
    }
    artifact = {
        "event_redesign_version": EVENT_REDESIGN_VERSION,
        "artifact_type": "event_coverage",
        "created_at": _now_iso(),
        "dataset_id": int(dataset_id),
        "target": target_metadata(definition),
        "row_count": int(len(derived)),
        "derived_feature_count": int(derived.shape[1]),
        "feature_sets": {name: {"columns": columns, "column_count": len(columns)} for name, columns in feature_sets.items()},
        "feature_groups": {name: {"columns": columns, "column_count": len(columns)} for name, columns in groups.items()},
        "active_observation_counts": active_counts,
        "coverage_by_ticker": coverage["by_ticker"],
        "coverage_by_date_top_activity": coverage["by_date_top_activity"],
        "event_density_by_fold": density,
        "db_event_counts": db_counts,
        "inactive_groups": inactive_groups,
        "summary": {
            "sec_feature_count": len(groups["sec"]),
            "earnings_feature_count": len(groups["earnings"]),
            "catalyst_llm_feature_count": len(groups["catalyst_llm"]),
            "inactive_groups": inactive_groups,
            "catalyst_llm_coverage": "inactive_or_near_zero" if "catalyst_llm" in inactive_groups else "active",
        },
    }
    return _clean_json(artifact)


def build_event_timing_audit(
    db_path: str | Path,
    dataset_id: int,
    target_name: str = RAW_TARGET_5_SESSION,
) -> dict[str, Any]:
    definition = get_target_definition(target_name)
    training = load_training_dataset(db_path, dataset_id, definition.base_label_column)
    db_events = _event_rows_from_db(db_path)
    sec = _parse_available_date(db_events["sec"], fallback_column="created_at")
    earnings = _parse_available_date(db_events["earnings"], fallback_column="announced_at")
    catalysts = _parse_available_date(db_events["catalysts"], fallback_column="created_at")

    sec_feature_map = {
        category: [
            f"sec_{category}_event_days_90s",
            f"sec_days_since_latest_{category}",
        ]
        for category in SEC_REDESIGN_CATEGORIES
    }
    sec_violations = _pre_availability_violations(training, sec, sec_feature_map, "classification")
    earnings_feature_map = {
        "earnings": ["earnings_data_available", "sessions_since_latest_earnings", "earnings_event_present_20s"]
    }
    earnings_for_check = earnings.assign(category="earnings") if not earnings.empty else earnings
    earnings_violations = _pre_availability_violations(training, earnings_for_check, earnings_feature_map, "category")

    lag_summary = [
        *_event_lag_summary(sec, training.metadata, "sec", category_column="classification"),
        *_event_lag_summary(earnings, training.metadata, "earnings"),
        *_event_lag_summary(catalysts, training.metadata, "catalyst"),
    ]
    missing_available = {
        "sec": int(pd.to_datetime(db_events["sec"].get("available_at"), utc=True, errors="coerce").isna().sum())
        if not db_events["sec"].empty
        else 0,
        "earnings": int(pd.to_datetime(db_events["earnings"].get("available_at"), utc=True, errors="coerce").isna().sum())
        if not db_events["earnings"].empty
        else 0,
        "catalysts": int(pd.to_datetime(db_events["catalysts"].get("available_at"), utc=True, errors="coerce").isna().sum())
        if not db_events["catalysts"].empty
        else 0,
    }
    timing_counts = {}
    if not earnings.empty and "timing" in earnings.columns:
        timing_counts = {
            str(key): int(value)
            for key, value in earnings["timing"].fillna("unknown").astype(str).value_counts().sort_index().items()
        }
    artifact = {
        "event_redesign_version": EVENT_REDESIGN_VERSION,
        "artifact_type": "event_timing",
        "created_at": _now_iso(),
        "dataset_id": int(dataset_id),
        "target": target_metadata(definition),
        "available_at_missing_counts": missing_available,
        "earnings_timing_counts": timing_counts,
        "pre_availability_violations": {
            "sec": sec_violations,
            "earnings": earnings_violations,
            "total": len(sec_violations) + len(earnings_violations),
        },
        "lag_bucket_summary": lag_summary,
        "summary": {
            "sec_events_with_available_at": int(len(sec)),
            "earnings_events_with_available_at": int(len(earnings)),
            "catalyst_events_with_available_at": int(len(catalysts)),
            "pre_availability_violation_count": len(sec_violations) + len(earnings_violations),
            "timing_guardrail": "features are derived from point-in-time Dataset 49 columns and audited against event available_at dates",
        },
    }
    return _clean_json(artifact)


def write_event_artifact(artifact: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dataset_id = artifact.get("dataset_id", "unknown")
    artifact_type = str(artifact.get("artifact_type") or "event")
    path = output / f"phase2d5_{artifact_type}_dataset{dataset_id}_{timestamp}.json"
    path.write_text(_json_dumps(artifact), encoding="utf-8")
    return path


def load_event_artifact(path: str | Path) -> dict[str, Any]:
    return _json_loads(Path(path).read_text(encoding="utf-8"), {}) or {}


def list_event_redesign_artifacts(output_dir: str | Path) -> list[EventRedesignArtifactInfo]:
    output = Path(output_dir)
    if not output.exists():
        return []
    artifacts: list[EventRedesignArtifactInfo] = []
    for path in sorted(output.glob("phase2d5_*_dataset*.json"), reverse=True):
        data = load_event_artifact(path)
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        artifacts.append(
            EventRedesignArtifactInfo(
                path=path,
                dataset_id=data.get("dataset_id") if isinstance(data, dict) else None,
                created_at=str(data.get("created_at") or "") if isinstance(data, dict) else "",
                summary=f"{data.get('artifact_type', 'event')} | {summary}",
            )
        )
    return artifacts


def run_event_redesign_suite(
    db_path: str | Path,
    dataset_id: int,
    output_dir: str | Path,
    target_names: list[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], Path, Path, list[Any]]:
    from src.modeling.runner import run_single_baseline_model

    target_names = target_names or list(EVENT_REDESIGN_TARGETS)
    coverage = build_event_coverage_audit(db_path, dataset_id)
    timing = build_event_timing_audit(db_path, dataset_id)
    coverage_path = write_event_artifact(coverage, output_dir)
    timing_path = write_event_artifact(timing, output_dir)

    training = load_training_dataset(db_path, dataset_id, RAW_TARGET_5_SESSION)
    model_frame, derived, _groups = build_event_model_frame(training)
    feature_sets = redesigned_event_feature_sets(training, derived)
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
                            "source": "phase2d5_event_feature_redesign",
                            "coverage_artifact": str(coverage_path),
                            "timing_artifact": str(timing_path),
                            "column_count": len(columns),
                            "derived_from_pit_dataset_features": True,
                            "scanner_scoring_effect": 0,
                        },
                        phase="2D-5",
                    )
                )
    return coverage, timing, coverage_path, timing_path, summaries


def event_feature_set_rows(artifact: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for name, definition in (artifact.get("feature_sets", {}) or {}).items():
        rows.append(
            {
                "feature_set": name,
                "column_count": int(definition.get("column_count", 0) or 0),
            }
        )
    return sorted(rows, key=lambda item: item["feature_set"])


def _assert_no_leaky_columns(columns: list[str]) -> None:
    forbidden = [column for column in columns if column.startswith("label_") or column.startswith("forward_") or column.startswith("raw_")]
    if forbidden:
        raise ValueError(f"Leaky columns are not allowed in event feature sets: {', '.join(forbidden)}")
