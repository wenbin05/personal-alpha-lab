from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

from src.data import storage
from src.modeling.evaluation_regime import get_dataset_evaluation_regime
from src.utils.trading_calendar import next_trading_day, trading_days_between


DEFAULT_HORIZONS = ("1_session", "5_session", "20_session")
HOLDOUT_MATURITY_VERSION = "holdout_maturity_v1"


@dataclass(frozen=True)
class HoldoutMaturityThresholds:
    protocol_min_rows: int = 1
    protocol_min_tickers: int = 1
    sanity_min_rows: int = 250
    sanity_min_tickers: int = 10
    sanity_min_5_session_labeled_dates: int = 20
    sanity_min_5_session_coverage: float = 0.50
    final_min_rows: int = 1_000
    final_min_tickers: int = 20
    final_min_5_session_labeled_dates: int = 60
    final_min_5_session_coverage: float = 0.80
    final_min_20_session_labeled_dates: int = 60
    final_min_20_session_coverage: float = 0.80


DEFAULT_HOLDOUT_THRESHOLDS = HoldoutMaturityThresholds()


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _date_or_none(value: Any) -> date | None:
    if value is None or value == "":
        return None
    try:
        return date.fromisoformat(str(value)[:10])
    except Exception:
        return None


def _pct(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(float(numerator) / float(denominator), 6)


def _manifest_violations(build: dict[str, Any]) -> list[str]:
    feature_columns = _json_loads(build.get("feature_columns_json"), []) or []
    audit_columns = _json_loads(build.get("audit_columns_json"), []) or []
    label_columns = _json_loads(build.get("label_columns_json"), []) or []
    identifier_columns = _json_loads(build.get("identifier_columns_json"), []) or []
    metadata_columns = _json_loads(build.get("metadata_columns_json"), []) or []
    manifest = _json_loads(build.get("feature_manifest_json"), {}) or {}
    forbidden = set(audit_columns) | set(label_columns) | set(identifier_columns) | set(metadata_columns)
    violations: list[str] = []
    leaked = sorted(set(feature_columns) & forbidden)
    if leaked:
        violations.append(f"Feature columns include forbidden columns: {', '.join(leaked)}")
    for column in feature_columns:
        role_payload = manifest.get(column, {})
        role = role_payload.get("role") if isinstance(role_payload, dict) else role_payload
        if role and role != "model_feature":
            violations.append(f"Feature column {column} has manifest role {role!r}.")
    return violations


def _read_dataset_build(db_path: str | Path, dataset_id: int) -> dict[str, Any]:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM dataset_builds WHERE dataset_id = ?", (int(dataset_id),)).fetchone()
    if row is None:
        raise ValueError(f"Dataset #{dataset_id} not found.")
    return dict(row)


def _snapshot_summary(db_path: str | Path, dataset_id: int) -> dict[str, Any]:
    with storage.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS row_count,
                   COUNT(DISTINCT ticker) AS ticker_count,
                   MIN(trading_date) AS start_date,
                   MAX(trading_date) AS end_date
            FROM feature_snapshots
            WHERE dataset_id = ?
            """,
            (int(dataset_id),),
        ).fetchone()
        dates = [
            str(item["trading_date"])
            for item in conn.execute(
                """
                SELECT DISTINCT trading_date
                FROM feature_snapshots
                WHERE dataset_id = ?
                ORDER BY trading_date
                """,
                (int(dataset_id),),
            ).fetchall()
        ]
    return {
        "row_count": int(row["row_count"] or 0),
        "ticker_count": int(row["ticker_count"] or 0),
        "start_date": row["start_date"],
        "end_date": row["end_date"],
        "snapshot_dates": dates,
        "snapshot_date_count": len(dates),
    }


def _label_summary(db_path: str | Path, dataset_id: int, snapshot_dates: list[str]) -> dict[str, Any]:
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT ol.horizon,
                   COUNT(*) AS label_count,
                   COUNT(DISTINCT fs.trading_date) AS labeled_date_count
            FROM outcome_labels ol
            JOIN feature_snapshots fs ON fs.snapshot_id = ol.snapshot_id
            WHERE fs.dataset_id = ?
            GROUP BY ol.horizon
            """,
            (int(dataset_id),),
        ).fetchall()
        labeled_dates_by_horizon = {
            str(row["horizon"]): {
                str(item["trading_date"])
                for item in conn.execute(
                    """
                    SELECT DISTINCT fs.trading_date
                    FROM outcome_labels ol
                    JOIN feature_snapshots fs ON fs.snapshot_id = ol.snapshot_id
                    WHERE fs.dataset_id = ? AND ol.horizon = ?
                    ORDER BY fs.trading_date
                    """,
                    (int(dataset_id), str(row["horizon"])),
                ).fetchall()
            }
            for row in rows
        }
    total_rows = 0
    with storage.connect(db_path) as conn:
        total_rows = int(
            conn.execute("SELECT COUNT(*) AS n FROM feature_snapshots WHERE dataset_id = ?", (int(dataset_id),)).fetchone()["n"] or 0
        )

    by_horizon: dict[str, dict[str, Any]] = {
        horizon: {
            "label_count": 0,
            "labeled_date_count": 0,
            "label_coverage_pct": 0.0,
            "missing_label_dates": list(snapshot_dates),
            "missing_label_date_count": len(snapshot_dates),
        }
        for horizon in DEFAULT_HORIZONS
    }
    for row in rows:
        horizon = str(row["horizon"])
        labeled_dates = sorted(labeled_dates_by_horizon.get(horizon, set()))
        missing_dates = [value for value in snapshot_dates if value not in set(labeled_dates)]
        by_horizon[horizon] = {
            "label_count": int(row["label_count"] or 0),
            "labeled_date_count": int(row["labeled_date_count"] or 0),
            "label_coverage_pct": _pct(int(row["label_count"] or 0), total_rows),
            "missing_label_dates": missing_dates,
            "missing_label_date_count": len(missing_dates),
        }
    return by_horizon


def _readiness(ready: bool, blockers: list[str]) -> dict[str, Any]:
    return {"ready": bool(ready), "blockers": blockers}


def assess_holdout_maturity(
    db_path: str | Path,
    dataset_id: int,
    thresholds: HoldoutMaturityThresholds = DEFAULT_HOLDOUT_THRESHOLDS,
) -> dict[str, Any]:
    """Assess whether a holdout dataset is mature enough for different uses."""
    build = _read_dataset_build(db_path, dataset_id)
    snapshots = _snapshot_summary(db_path, dataset_id)
    labels = _label_summary(db_path, dataset_id, snapshots["snapshot_dates"])
    regime = get_dataset_evaluation_regime(db_path, dataset_id) or {
        "dataset_id": int(dataset_id),
        "evaluation_regime": "unclassified",
        "strategy": "unclassified",
        "rationale": "No dataset_evaluation_regimes row exists yet.",
        "parent_dataset_id": None,
        "metadata": {},
    }
    manifest_violations = _manifest_violations(build)
    row_count = int(snapshots["row_count"])
    ticker_count = int(snapshots["ticker_count"])
    five = labels["5_session"]
    twenty = labels["20_session"]

    protocol_blockers: list[str] = []
    if row_count < thresholds.protocol_min_rows:
        protocol_blockers.append(f"row_count {row_count} < protocol_min_rows {thresholds.protocol_min_rows}")
    if ticker_count < thresholds.protocol_min_tickers:
        protocol_blockers.append(f"ticker_count {ticker_count} < protocol_min_tickers {thresholds.protocol_min_tickers}")
    if manifest_violations:
        protocol_blockers.append("manifest/leakage violations exist")

    sanity_blockers = list(protocol_blockers)
    if row_count < thresholds.sanity_min_rows:
        sanity_blockers.append(f"row_count {row_count} < sanity_min_rows {thresholds.sanity_min_rows}")
    if ticker_count < thresholds.sanity_min_tickers:
        sanity_blockers.append(f"ticker_count {ticker_count} < sanity_min_tickers {thresholds.sanity_min_tickers}")
    if int(five["labeled_date_count"]) < thresholds.sanity_min_5_session_labeled_dates:
        sanity_blockers.append(
            f"5-session labeled dates {five['labeled_date_count']} < sanity_min_5_session_labeled_dates {thresholds.sanity_min_5_session_labeled_dates}"
        )
    if float(five["label_coverage_pct"]) < thresholds.sanity_min_5_session_coverage:
        sanity_blockers.append(
            f"5-session label coverage {five['label_coverage_pct']:.1%} < sanity_min_5_session_coverage {thresholds.sanity_min_5_session_coverage:.1%}"
        )

    final_5_blockers = list(protocol_blockers)
    if regime.get("evaluation_regime") != "holdout_candidate":
        final_5_blockers.append("dataset is not a holdout_candidate")
    if row_count < thresholds.final_min_rows:
        final_5_blockers.append(f"row_count {row_count} < final_min_rows {thresholds.final_min_rows}")
    if ticker_count < thresholds.final_min_tickers:
        final_5_blockers.append(f"ticker_count {ticker_count} < final_min_tickers {thresholds.final_min_tickers}")
    if int(five["labeled_date_count"]) < thresholds.final_min_5_session_labeled_dates:
        final_5_blockers.append(
            f"5-session labeled dates {five['labeled_date_count']} < final_min_5_session_labeled_dates {thresholds.final_min_5_session_labeled_dates}"
        )
    if float(five["label_coverage_pct"]) < thresholds.final_min_5_session_coverage:
        final_5_blockers.append(
            f"5-session label coverage {five['label_coverage_pct']:.1%} < final_min_5_session_coverage {thresholds.final_min_5_session_coverage:.1%}"
        )

    final_20_blockers = list(final_5_blockers)
    if int(twenty["labeled_date_count"]) < thresholds.final_min_20_session_labeled_dates:
        final_20_blockers.append(
            f"20-session labeled dates {twenty['labeled_date_count']} < final_min_20_session_labeled_dates {thresholds.final_min_20_session_labeled_dates}"
        )
    if float(twenty["label_coverage_pct"]) < thresholds.final_min_20_session_coverage:
        final_20_blockers.append(
            f"20-session label coverage {twenty['label_coverage_pct']:.1%} < final_min_20_session_coverage {thresholds.final_min_20_session_coverage:.1%}"
        )

    return {
        "artifact_type": "holdout_maturity_status",
        "version": HOLDOUT_MATURITY_VERSION,
        "created_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "dataset_id": int(dataset_id),
        "evaluation_regime": regime.get("evaluation_regime"),
        "strategy": regime.get("strategy"),
        "rationale": regime.get("rationale"),
        "parent_dataset_id": regime.get("parent_dataset_id"),
        "date_range": {
            "requested_start_date": build.get("requested_start_date"),
            "requested_end_date": build.get("requested_end_date"),
            "actual_start_date": snapshots["start_date"],
            "actual_end_date": snapshots["end_date"],
            "snapshot_date_count": snapshots["snapshot_date_count"],
        },
        "row_count": row_count,
        "ticker_count": ticker_count,
        "data_hash": build.get("data_hash"),
        "label_coverage": labels,
        "thresholds": asdict(thresholds),
        "manifest": {"violation_count": len(manifest_violations), "violations": manifest_violations},
        "readiness": {
            "protocol_validation": _readiness(not protocol_blockers, protocol_blockers),
            "holdout_candidate_sanity_check": _readiness(not sanity_blockers, sanity_blockers),
            "final_holdout_evaluation_5_session": _readiness(not final_5_blockers, final_5_blockers),
            "final_holdout_evaluation_20_session": _readiness(not final_20_blockers, final_20_blockers),
        },
        "promotion": {
            "allowed_for_5_session": not final_5_blockers,
            "allowed_for_20_session": not final_20_blockers,
            "explicit_user_confirmation_required": True,
            "repeated_evaluation_allowed": False,
        },
        "modeling": {
            "blocked": bool(final_5_blockers),
            "reason": "Final holdout evaluation is blocked until maturity thresholds pass."
            if final_5_blockers
            else "5-session final holdout maturity thresholds pass; still require explicit promotion and confirmation.",
        },
    }


def build_holdout_extension_plan(db_path: str | Path, dataset_id: int) -> dict[str, Any]:
    """Plan a cache-only extension without mutating datasets or fetching providers."""
    build = _read_dataset_build(db_path, dataset_id)
    tickers = _json_loads(build.get("ticker_universe_json"), []) or []
    current_end = _date_or_none(build.get("requested_end_date"))
    if current_end is None:
        raise ValueError(f"Dataset #{dataset_id} has no requested_end_date.")
    next_start = next_trading_day(current_end)
    per_ticker: list[dict[str, Any]] = []
    with storage.connect(db_path) as conn:
        for ticker in tickers:
            row = conn.execute(
                "SELECT MIN(date) AS start_date, MAX(date) AS end_date, COUNT(*) AS rows FROM ohlcv_cache WHERE ticker = ?",
                (str(ticker).upper(),),
            ).fetchone()
            per_ticker.append(
                {
                    "ticker": str(ticker).upper(),
                    "cache_start_date": row["start_date"],
                    "cache_end_date": row["end_date"],
                    "cached_rows": int(row["rows"] or 0),
                }
            )
    cache_end_dates = [_date_or_none(row["cache_end_date"]) for row in per_ticker]
    complete_cache_end_dates = [value for value in cache_end_dates if value is not None]
    common_cache_end = min(complete_cache_end_dates) if complete_cache_end_dates and len(complete_cache_end_dates) == len(per_ticker) else None
    extension_dates = trading_days_between(next_start, common_cache_end) if common_cache_end is not None else []
    return {
        "artifact_type": "holdout_extension_plan",
        "version": HOLDOUT_MATURITY_VERSION,
        "dataset_id": int(dataset_id),
        "parent_dataset_id": int(dataset_id),
        "cache_only_default": True,
        "provider_fetch_allowed_by_default": False,
        "current_requested_end_date": current_end.isoformat(),
        "candidate_start_date": next_start.isoformat(),
        "common_cache_end_date": common_cache_end.isoformat() if common_cache_end is not None else None,
        "extension_available": bool(extension_dates),
        "extension_trading_days": [day.isoformat() for day in extension_dates],
        "extension_trading_day_count": len(extension_dates),
        "recommended_action": "create_new_holdout_candidate_dataset"
        if extension_dates
        else "wait_for_more_cached_sessions",
        "recommended_version_suffix": "holdout_candidate_extension_v1",
        "per_ticker_cache_coverage": per_ticker,
        "warnings": []
        if common_cache_end is not None
        else ["At least one dataset ticker has no cached OHLCV coverage; cache-only extension is unavailable."],
    }
