from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from src.datasets.builder import as_of_after_close, precompute_feature_sets_for_dates, precompute_market_regimes_for_dates
from src.modeling.artifacts import check_registered_artifact
from src.utils.trading_calendar import is_trading_day, latest_expected_trading_day


BENCHMARK_TICKERS = ("SPY", "QQQ", "IWM", "^VIX")
MIN_PRELIMINARY_PREDICTION_DATES = 20
MIN_MEANINGFUL_PREDICTION_DATES = 60


class ShadowPredictionError(ValueError):
    pass


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _readonly_connect(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table_name,)
    ).fetchone() is not None


def _artifact_record(db_path: str | Path, artifact_id: str) -> dict[str, Any]:
    with _readonly_connect(db_path) as conn:
        if not _table_exists(conn, "model_artifacts"):
            raise ShadowPredictionError("Model artifact registry is unavailable.")
        row = conn.execute("SELECT * FROM model_artifacts WHERE artifact_id=?", (artifact_id,)).fetchone()
    if row is None:
        raise ShadowPredictionError(f"Artifact {artifact_id} is not registered.")
    return dict(row)


def _load_manifest(record: dict[str, Any]) -> dict[str, Any]:
    try:
        manifest = json.loads(Path(record["manifest_path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        raise ShadowPredictionError(f"Artifact manifest cannot be loaded: {exc}") from exc
    if manifest.get("artifact_id") != record.get("artifact_id"):
        raise ShadowPredictionError("Artifact manifest identity does not match the registry.")
    return manifest


def _load_cached_history(conn: sqlite3.Connection, ticker: str, as_of: date) -> pd.DataFrame:
    rows = conn.execute(
        """
        SELECT date, open, high, low, close, adj_close, volume
        FROM ohlcv_cache
        WHERE ticker=? AND date<=?
        ORDER BY date
        """,
        (ticker, as_of.isoformat()),
    ).fetchall()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame([dict(row) for row in rows])
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    for column in ["open", "high", "low", "close", "adj_close", "volume"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    return frame.dropna(subset=["date", "close"]).set_index("date").sort_index()


def latest_cache_complete_session(db_path: str | Path, artifact_id: str) -> date | None:
    record = _artifact_record(db_path, artifact_id)
    manifest = _load_manifest(record)
    required = sorted(set(str(value) for value in manifest.get("universe", [])) | set(BENCHMARK_TICKERS))
    if not required:
        return None
    latest_allowed = latest_expected_trading_day()
    with _readonly_connect(db_path) as conn:
        common_dates: set[str] | None = None
        for ticker in required:
            rows = conn.execute(
                "SELECT date FROM ohlcv_cache WHERE ticker=? AND date<=? ORDER BY date",
                (ticker, latest_allowed.isoformat()),
            ).fetchall()
            dates = {str(row["date"]) for row in rows}
            common_dates = dates if common_dates is None else common_dates & dates
            if not common_dates:
                return None
    valid = [pd.to_datetime(value).date() for value in common_dates or set() if is_trading_day(value)]
    return max(valid) if valid else None


def _technical_row(feature_set: dict[str, Any], regime: dict[str, Any]) -> dict[str, Any]:
    return {
        "ret_5d": feature_set.get("ret_5d"),
        "ret_20d": feature_set.get("ret_20d"),
        "ret_60d": feature_set.get("ret_60d"),
        "ret_120d": feature_set.get("ret_120d"),
        "daily_return": feature_set.get("daily_return"),
        "volatility_20d": feature_set.get("volatility_20d"),
        "distance_20d_ma": feature_set.get("distance_20d_ma"),
        "distance_50d_ma": feature_set.get("distance_50d_ma"),
        "above_50d_ma": feature_set.get("above_50d_ma"),
        "above_200d_ma": feature_set.get("above_200d_ma"),
        "relative_strength_20d": feature_set.get("relative_strength_20d"),
        "relative_strength_60d": feature_set.get("relative_strength_60d"),
        "relative_strength_score_raw": feature_set.get("relative_strength_score_raw"),
        "volume_ratio_20d": feature_set.get("volume_ratio_20d"),
        "volume_anomaly": feature_set.get("volume_anomaly"),
        "avg_dollar_volume_20d": feature_set.get("avg_dollar_volume_20d"),
        "avg_dollar_volume_ok": feature_set.get("avg_dollar_volume_ok"),
        "liquidity_score_raw": feature_set.get("liquidity_score_raw"),
        "price_ok": feature_set.get("price_ok"),
        "market_regime": regime.get("regime", "Neutral"),
        "market_regime_confidence": regime.get("confidence", "unknown"),
        "regime_qqq_spy_rs_20": regime.get("qqq_spy_rs_20"),
        "regime_iwm_spy_rs_20": regime.get("iwm_spy_rs_20"),
        "regime_vix": regime.get("vix"),
        "regime_vix_elevated": bool(regime.get("vix_elevated", False)),
        "bars_available": int(feature_set.get("bars", 0) or 0),
        "has_data": bool(feature_set.get("has_data", False)),
        "insufficient_history_200d": feature_set.get("ma_200") is None,
        "failed_spy_comparison": feature_set.get("relative_strength_20d") is None,
        "missing_volume": feature_set.get("current_volume") is None,
    }


def _clean_hash_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (bool, str, int)):
        return value
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return None if not np.isfinite(numeric) else numeric


def _rank_predictions(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(rows, key=lambda item: (-float(item["predicted_value"]), str(item["ticker"])))
    total = len(ordered)
    for index, row in enumerate(ordered, start=1):
        row["predicted_rank"] = index
        row["predicted_percentile"] = 1.0 if total == 1 else float((total - index) / (total - 1))
    return ordered


@dataclass(frozen=True)
class ShadowPredictionPlan:
    artifact_id: str
    artifact_checksum: str
    feature_manifest_hash: str
    universe_hash: str
    prediction_date: str
    as_of_timestamp: str
    latest_cache_complete_session: str | None
    eligible_tickers: list[str]
    excluded_tickers: list[dict[str, str]]
    predictions: list[dict[str, Any]]
    warnings: list[str]
    duplicate_run_id: int | None


def build_shadow_prediction_plan(
    db_path: str | Path,
    artifact_id: str,
    as_of: date | str,
) -> ShadowPredictionPlan:
    prediction_date = pd.to_datetime(as_of, errors="raise").date()
    if not is_trading_day(prediction_date):
        raise ShadowPredictionError(f"{prediction_date} is not a U.S. trading session.")
    if prediction_date > latest_expected_trading_day():
        raise ShadowPredictionError("Requested session is newer than the latest completed U.S. session.")
    integrity = check_registered_artifact(db_path, artifact_id)
    if integrity.get("status") != "passed":
        raise ShadowPredictionError(f"Artifact integrity gate failed: {integrity.get('errors', [])}")
    record = _artifact_record(db_path, artifact_id)
    manifest = _load_manifest(record)
    feature_columns = [str(value) for value in manifest.get("feature_columns", [])]
    universe = [str(value).upper() for value in manifest.get("universe", [])]
    if not feature_columns or not universe:
        raise ShadowPredictionError("Artifact feature or universe contract is empty.")
    if manifest.get("dataset_id") != 49:
        raise ShadowPredictionError("Shadow inference permits the frozen Dataset 49 artifact contract only.")
    bundle = joblib.load(record["artifact_path"] + "/model.joblib")
    if list(bundle.feature_columns) != feature_columns:
        raise ShadowPredictionError("Executable artifact and manifest feature contracts differ.")

    required_cache = sorted(set(universe) | set(BENCHMARK_TICKERS))
    histories: dict[str, pd.DataFrame] = {}
    excluded: list[dict[str, str]] = []
    with _readonly_connect(db_path) as conn:
        for ticker in required_cache:
            history = _load_cached_history(conn, ticker, prediction_date)
            histories[ticker] = history
        for benchmark in BENCHMARK_TICKERS:
            history = histories[benchmark]
            if history.empty or pd.Timestamp(prediction_date) not in history.index:
                raise ShadowPredictionError(f"Required benchmark cache is incomplete for {benchmark} on {prediction_date}.")
        for ticker in universe:
            history = histories.get(ticker, pd.DataFrame())
            if history.empty:
                excluded.append({"ticker": ticker, "reason": "no_cached_history"})
            elif pd.Timestamp(prediction_date) not in history.index:
                excluded.append({"ticker": ticker, "reason": "missing_requested_session_bar"})
        duplicate_run_id = None
        if _table_exists(conn, "shadow_prediction_runs"):
            duplicate = conn.execute(
                "SELECT run_id FROM shadow_prediction_runs WHERE prediction_date=? AND artifact_id=?",
                (prediction_date.isoformat(), artifact_id),
            ).fetchone()
            duplicate_run_id = None if duplicate is None else int(duplicate["run_id"])

    eligible = [ticker for ticker in universe if ticker not in {item["ticker"] for item in excluded}]
    regime = precompute_market_regimes_for_dates(histories, [prediction_date]).get(prediction_date, {})
    input_rows: list[dict[str, Any]] = []
    quality_by_ticker: dict[str, dict[str, Any]] = {}
    for ticker in eligible:
        feature_set = precompute_feature_sets_for_dates(
            ticker, histories[ticker], histories["SPY"], [prediction_date]
        ).get(prediction_date)
        if feature_set is None:
            excluded.append({"ticker": ticker, "reason": "feature_assembly_failed"})
            continue
        row = _technical_row(feature_set, regime)
        missing_columns = [column for column in feature_columns if column not in row]
        if missing_columns:
            raise ShadowPredictionError(f"Missing required feature columns for {ticker}: {', '.join(missing_columns)}")
        ordered = {column: row[column] for column in feature_columns}
        missing_values = [column for column, value in ordered.items() if _clean_hash_value(value) is None]
        quality_by_ticker[ticker] = {
            "missing_value_count": len(missing_values),
            "missing_values": missing_values,
            "bars_available": int(row.get("bars_available") or 0),
            "cache_only": True,
            "as_of_enforced": True,
        }
        input_rows.append({"ticker": ticker, "features": ordered})
    if not input_rows:
        raise ShadowPredictionError("No eligible tickers remain after point-in-time feature assembly.")
    frame = pd.DataFrame([item["features"] for item in input_rows], columns=feature_columns)
    predicted = bundle.predict(frame)
    prediction_rows = []
    for item, value in zip(input_rows, predicted, strict=True):
        ticker = item["ticker"]
        hash_payload = {column: _clean_hash_value(item["features"][column]) for column in feature_columns}
        prediction_rows.append(
            {
                "ticker": ticker,
                "prediction_date": prediction_date.isoformat(),
                "predicted_value": float(value),
                "feature_input_hash": _sha256_json(hash_payload),
                "data_quality_flags": quality_by_ticker[ticker],
            }
        )
    ranked = _rank_predictions(prediction_rows)
    warnings = [f"Excluded {item['ticker']}: {item['reason']}" for item in excluded]
    latest_complete = latest_cache_complete_session(db_path, artifact_id)
    return ShadowPredictionPlan(
        artifact_id=artifact_id,
        artifact_checksum=str(record["artifact_checksum"]),
        feature_manifest_hash=str(record["feature_manifest_hash"]),
        universe_hash=str(record["universe_hash"]),
        prediction_date=prediction_date.isoformat(),
        as_of_timestamp=as_of_after_close(prediction_date).isoformat(),
        latest_cache_complete_session=None if latest_complete is None else latest_complete.isoformat(),
        eligible_tickers=[item["ticker"] for item in ranked],
        excluded_tickers=excluded,
        predictions=ranked,
        warnings=warnings,
        duplicate_run_id=duplicate_run_id,
    )


def dry_run_shadow_prediction(db_path: str | Path, artifact_id: str, as_of: date | str) -> dict[str, Any]:
    plan = build_shadow_prediction_plan(db_path, artifact_id, as_of)
    return {
        "status": "duplicate_blocked" if plan.duplicate_run_id is not None else "ready",
        "dry_run": True,
        "database_mutated": False,
        "plan": asdict(plan),
        "prediction_count": len(plan.predictions),
        "research_only": True,
        "scanner_scoring_effect": 0,
    }


def _backup_database(db_path: str | Path) -> Path:
    source = Path(db_path)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    backup = source.with_name(f"{source.stem}_backup_phase3a1a_{stamp}{source.suffix}")
    with sqlite3.connect(source) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)
    return backup


def _create_shadow_schema(conn: sqlite3.Connection) -> None:
    statements = [
        """
        CREATE TABLE IF NOT EXISTS shadow_prediction_runs (
            run_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_date TEXT NOT NULL,
            as_of_timestamp TEXT NOT NULL,
            artifact_id TEXT NOT NULL,
            artifact_checksum TEXT NOT NULL,
            feature_manifest_hash TEXT NOT NULL,
            universe_hash TEXT NOT NULL,
            status TEXT NOT NULL,
            prediction_count INTEGER NOT NULL,
            warnings_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            UNIQUE(prediction_date, artifact_id)
        )
        """,
        """
        CREATE TABLE IF NOT EXISTS shadow_predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            prediction_run_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            prediction_date TEXT NOT NULL,
            predicted_value REAL NOT NULL,
            predicted_rank INTEGER NOT NULL,
            predicted_percentile REAL NOT NULL,
            feature_input_hash TEXT NOT NULL,
            data_quality_flags_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL,
            UNIQUE(prediction_run_id, ticker)
        )
        """,
        "CREATE INDEX IF NOT EXISTS idx_shadow_predictions_date ON shadow_predictions(prediction_date, predicted_rank)",
        """
        CREATE TRIGGER IF NOT EXISTS shadow_prediction_runs_immutable_update
        BEFORE UPDATE ON shadow_prediction_runs WHEN OLD.status='completed'
        BEGIN SELECT RAISE(ABORT, 'completed shadow prediction runs are immutable'); END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS shadow_prediction_runs_immutable_delete
        BEFORE DELETE ON shadow_prediction_runs
        BEGIN SELECT RAISE(ABORT, 'shadow prediction runs are immutable'); END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS shadow_predictions_immutable_update
        BEFORE UPDATE ON shadow_predictions
        BEGIN SELECT RAISE(ABORT, 'shadow predictions are immutable'); END
        """,
        """
        CREATE TRIGGER IF NOT EXISTS shadow_predictions_immutable_delete
        BEFORE DELETE ON shadow_predictions
        BEGIN SELECT RAISE(ABORT, 'shadow predictions are immutable'); END
        """,
    ]
    for statement in statements:
        conn.execute(statement)


def apply_shadow_prediction(db_path: str | Path, artifact_id: str, as_of: date | str) -> dict[str, Any]:
    plan = build_shadow_prediction_plan(db_path, artifact_id, as_of)
    if plan.duplicate_run_id is not None:
        raise ShadowPredictionError(
            f"Shadow run already exists for {plan.prediction_date} and {artifact_id}: run {plan.duplicate_run_id}."
        )
    backup = _backup_database(db_path)
    now = _now_iso()
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("BEGIN IMMEDIATE")
        _create_shadow_schema(conn)
        duplicate = conn.execute(
            "SELECT run_id FROM shadow_prediction_runs WHERE prediction_date=? AND artifact_id=?",
            (plan.prediction_date, artifact_id),
        ).fetchone()
        if duplicate is not None:
            raise ShadowPredictionError(f"Duplicate shadow run blocked: {int(duplicate['run_id'])}.")
        cursor = conn.execute(
            """
            INSERT INTO shadow_prediction_runs (
                prediction_date, as_of_timestamp, artifact_id, artifact_checksum,
                feature_manifest_hash, universe_hash, status, prediction_count,
                warnings_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'completed', ?, ?, ?)
            """,
            (
                plan.prediction_date, plan.as_of_timestamp, artifact_id, plan.artifact_checksum,
                plan.feature_manifest_hash, plan.universe_hash, len(plan.predictions),
                _canonical_json(plan.warnings), now,
            ),
        )
        run_id = int(cursor.lastrowid)
        conn.executemany(
            """
            INSERT INTO shadow_predictions (
                prediction_run_id, ticker, prediction_date, predicted_value,
                predicted_rank, predicted_percentile, feature_input_hash,
                data_quality_flags_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    run_id, row["ticker"], plan.prediction_date, row["predicted_value"],
                    row["predicted_rank"], row["predicted_percentile"], row["feature_input_hash"],
                    _canonical_json(row["data_quality_flags"]), now,
                )
                for row in plan.predictions
            ],
        )
        stored = conn.execute(
            "SELECT COUNT(*) FROM shadow_predictions WHERE prediction_run_id=?", (run_id,)
        ).fetchone()[0]
        if int(stored) != len(plan.predictions):
            raise ShadowPredictionError("Stored prediction count does not match the completed run contract.")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "status": "recorded",
        "run_id": run_id,
        "prediction_date": plan.prediction_date,
        "prediction_count": len(plan.predictions),
        "excluded_tickers": plan.excluded_tickers,
        "database_backup": str(backup),
        "artifact_id": artifact_id,
        "artifact_checksum": plan.artifact_checksum,
        "feature_manifest_hash": plan.feature_manifest_hash,
        "research_only": True,
        "scanner_scoring_effect": 0,
    }


def list_shadow_prediction_runs(db_path: str | Path, limit: int = 100) -> pd.DataFrame:
    with _readonly_connect(db_path) as conn:
        if not _table_exists(conn, "shadow_prediction_runs"):
            return pd.DataFrame()
        return pd.read_sql_query(
            "SELECT * FROM shadow_prediction_runs ORDER BY prediction_date DESC, run_id DESC LIMIT ?",
            conn,
            params=(int(limit),),
        )


def list_shadow_predictions(db_path: str | Path, run_id: int, limit: int = 500) -> pd.DataFrame:
    with _readonly_connect(db_path) as conn:
        if not _table_exists(conn, "shadow_predictions"):
            return pd.DataFrame()
        frame = pd.read_sql_query(
            """
            SELECT * FROM shadow_predictions
            WHERE prediction_run_id=?
            ORDER BY predicted_rank, ticker
            LIMIT ?
            """,
            conn,
            params=(int(run_id), int(limit)),
        )
    if not frame.empty:
        frame["data_quality_flags"] = frame["data_quality_flags_json"].map(json.loads)
    return frame


def shadow_status_report(db_path: str | Path, artifact_id: str | None = None) -> dict[str, Any]:
    with _readonly_connect(db_path) as conn:
        runs = []
        predictions_per_run = []
        violations: list[str] = []
        if _table_exists(conn, "shadow_prediction_runs"):
            runs = [dict(row) for row in conn.execute("SELECT * FROM shadow_prediction_runs ORDER BY prediction_date, run_id")]
            duplicate_runs = conn.execute(
                """
                SELECT prediction_date, artifact_id, COUNT(*) AS count
                FROM shadow_prediction_runs GROUP BY prediction_date, artifact_id HAVING COUNT(*)>1
                """
            ).fetchall()
            violations.extend(f"duplicate_run:{row['prediction_date']}:{row['artifact_id']}" for row in duplicate_runs)
            for run in runs:
                count = conn.execute(
                    "SELECT COUNT(*) FROM shadow_predictions WHERE prediction_run_id=?", (run["run_id"],)
                ).fetchone()[0]
                predictions_per_run.append(
                    {"run_id": int(run["run_id"]), "prediction_date": run["prediction_date"], "prediction_count": int(count)}
                )
                if int(count) != int(run["prediction_count"]):
                    violations.append(f"prediction_count_mismatch:{run['run_id']}")
                artifact = conn.execute(
                    "SELECT artifact_checksum, feature_manifest_hash, universe_hash FROM model_artifacts WHERE artifact_id=?",
                    (run["artifact_id"],),
                ).fetchone()
                if artifact is None:
                    violations.append(f"missing_artifact:{run['artifact_id']}")
                elif any(str(run[field]) != str(artifact[field]) for field in ("artifact_checksum", "feature_manifest_hash", "universe_hash")):
                    violations.append(f"artifact_hash_mismatch:{run['run_id']}")
        selected_artifact = artifact_id or (str(runs[-1]["artifact_id"]) if runs else None)
    integrity = None if selected_artifact is None else check_registered_artifact(db_path, selected_artifact)
    if integrity is not None and integrity.get("status") != "passed":
        violations.append("registered_artifact_integrity_failed")
    prediction_dates = sorted({str(run["prediction_date"]) for run in runs})
    date_count = len(prediction_dates)
    sample_status = (
        "insufficient_forward_sample"
        if date_count < MIN_PRELIMINARY_PREDICTION_DATES
        else "preliminary_monitoring"
        if date_count < MIN_MEANINGFUL_PREDICTION_DATES
        else "meaningful_review_sample"
    )
    latest = runs[-1] if runs else None
    return {
        "status": "passed" if not violations else "failed",
        "artifact_id": selected_artifact,
        "artifact_integrity": None if integrity is None else integrity.get("status"),
        "run_count": len(runs),
        "prediction_date_range": None if not prediction_dates else [prediction_dates[0], prediction_dates[-1]],
        "predictions_per_run": predictions_per_run,
        "prediction_date_count": date_count,
        "sample_status": sample_status,
        "minimum_preliminary_dates": MIN_PRELIMINARY_PREDICTION_DATES,
        "minimum_meaningful_dates": MIN_MEANINGFUL_PREDICTION_DATES,
        "latest_run_warnings": [] if latest is None else json.loads(str(latest["warnings_json"] or "[]")),
        "violations": violations,
        "research_only": True,
        "scanner_scoring_effect": 0,
    }
