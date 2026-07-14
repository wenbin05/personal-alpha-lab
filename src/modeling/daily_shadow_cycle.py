from __future__ import annotations

import fcntl
import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from src.data import market_data, storage
from src.modeling.artifacts import check_registered_artifact
from src.modeling.shadow_predictions import (
    BENCHMARK_TICKERS,
    ShadowPredictionError,
    apply_shadow_outcomes,
    apply_shadow_prediction,
    dry_run_shadow_outcomes,
    dry_run_shadow_prediction,
    latest_cache_complete_session,
    shadow_status_report,
)
from src.utils.trading_calendar import latest_expected_trading_day, next_trading_day, trading_days_between


DEFAULT_SHADOW_ARTIFACT_ID = "shadow_ridge_technical_v1_1ee8071db3f0"


class DailyShadowCycleError(RuntimeError):
    def __init__(self, message: str, exit_code: int = 2):
        super().__init__(message)
        self.exit_code = exit_code


def _readonly_connect(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _artifact_universe(db_path: str | Path, artifact_id: str) -> tuple[list[str], dict[str, Any]]:
    with _readonly_connect(db_path) as conn:
        row = conn.execute(
            "SELECT manifest_path FROM model_artifacts WHERE artifact_id=?", (artifact_id,)
        ).fetchone()
    if row is None:
        raise DailyShadowCycleError(f"Artifact {artifact_id} is not registered.", 2)
    try:
        manifest = json.loads(Path(row["manifest_path"]).read_text(encoding="utf-8"))
    except Exception as exc:
        raise DailyShadowCycleError(f"Artifact manifest cannot be read: {exc}", 2) from exc
    universe = sorted({str(value).upper() for value in manifest.get("universe", [])} | set(BENCHMARK_TICKERS))
    if not universe:
        raise DailyShadowCycleError("Artifact universe is empty.", 2)
    return universe, manifest


def audit_cache_coverage(
    db_path: str | Path,
    artifact_id: str,
    resolved_session: date,
) -> dict[str, Any]:
    universe, _manifest = _artifact_universe(db_path, artifact_id)
    rows: list[dict[str, Any]] = []
    with _readonly_connect(db_path) as conn:
        for ticker in universe:
            cached = conn.execute(
                "SELECT date FROM ohlcv_cache WHERE ticker=? ORDER BY date", (ticker,)
            ).fetchall()
            all_dates = [pd.to_datetime(row["date"]).date() for row in cached]
            usable_dates = [value for value in all_dates if value <= resolved_session]
            latest_usable = max(usable_dates) if usable_dates else None
            missing_dates = (
                trading_days_between(next_trading_day(latest_usable), resolved_session)
                if latest_usable is not None and latest_usable < resolved_session
                else []
            )
            rows.append(
                {
                    "ticker": ticker,
                    "latest_cached_date": None if not all_dates else max(all_dates).isoformat(),
                    "latest_usable_date": None if latest_usable is None else latest_usable.isoformat(),
                    "missing_session_count": len(missing_dates),
                    "missing_dates": [value.isoformat() for value in missing_dates],
                    "future_or_incomplete_bar_count": sum(value > resolved_session for value in all_dates),
                    "complete_through_resolved_session": resolved_session in usable_dates,
                }
            )
    return {
        "resolved_session": resolved_session.isoformat(),
        "required_ticker_count": len(universe),
        "complete_ticker_count": sum(row["complete_through_resolved_session"] for row in rows),
        "incomplete_ticker_count": sum(not row["complete_through_resolved_session"] for row in rows),
        "tickers": rows,
    }


def _compact_cache_status(status: dict[str, Any]) -> dict[str, Any]:
    return {
        "resolved_session": status["resolved_session"],
        "required_ticker_count": status["required_ticker_count"],
        "complete_ticker_count": status["complete_ticker_count"],
        "incomplete_ticker_count": status["incomplete_ticker_count"],
        "incomplete_tickers": [
            row["ticker"] for row in status["tickers"] if not row["complete_through_resolved_session"]
        ],
        "future_or_incomplete_bars": [
            {"ticker": row["ticker"], "count": row["future_or_incomplete_bar_count"]}
            for row in status["tickers"]
            if row["future_or_incomplete_bar_count"]
        ],
    }


def _backup_database(db_path: str | Path) -> Path:
    source = Path(db_path)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    backup = source.with_name(f"{source.stem}_backup_phase3a2b_{stamp}{source.suffix}")
    with sqlite3.connect(source) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)
    return backup


@contextmanager
def cycle_lock(lock_path: str | Path) -> Iterator[Path]:
    path = Path(lock_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(descriptor)
        raise DailyShadowCycleError(f"Daily shadow cycle lock already exists: {path}", 3) from exc
    try:
        os.ftruncate(descriptor, 0)
        os.write(descriptor, f"pid={os.getpid()} created_at={datetime.now(UTC).isoformat()}\n".encode("ascii"))
        yield path
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)
        path.unlink(missing_ok=True)


def _refresh_missing_ranges(
    db_path: str | Path,
    cache_status: dict[str, Any],
    resolved_session: date,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    provider = market_data.get_provider("yfinance")
    refreshed: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for item in cache_status["tickers"]:
        if item["complete_through_resolved_session"]:
            continue
        ticker = str(item["ticker"])
        latest = item["latest_usable_date"]
        start = next_trading_day(pd.to_datetime(latest).date()) if latest else resolved_session - timedelta(days=730)
        try:
            downloaded = provider.download_history(ticker, start=start, end=resolved_session)
            if downloaded.empty:
                raise market_data.MarketDataError("provider returned no rows")
            normalized = storage.normalize_ohlcv(downloaded)
            dates = pd.to_datetime(normalized["date"], errors="coerce").dt.date
            bounded = normalized.loc[(dates >= start) & (dates <= resolved_session)].copy()
            if bounded.empty:
                raise market_data.MarketDataError("provider returned no completed-session rows")
            storage.upsert_ohlcv(db_path, ticker, bounded)
            refreshed.append(
                {
                    "ticker": ticker,
                    "requested_start": start.isoformat(),
                    "requested_end_inclusive": resolved_session.isoformat(),
                    "raw_provider_end_exclusive": (resolved_session + timedelta(days=1)).isoformat(),
                    "downloaded_completed_rows": int(len(bounded)),
                }
            )
        except Exception as exc:
            failures.append({"ticker": ticker, "error": str(exc)})
    return refreshed, failures


def run_daily_shadow_cycle(
    db_path: str | Path,
    *,
    artifact_id: str = DEFAULT_SHADOW_ARTIFACT_ID,
    apply: bool = False,
    refresh_market_data: bool = False,
    reference_time: datetime | date | None = None,
    lock_path: str | Path | None = None,
) -> dict[str, Any]:
    db_path = Path(db_path)
    resolved_session = latest_expected_trading_day(reference_time)
    lock_file = Path(lock_path) if lock_path else db_path.with_suffix(db_path.suffix + ".shadow-cycle.lock")
    report: dict[str, Any] = {
        "status": "completed",
        "mode": "apply" if apply else "dry_run",
        "database_mutated": False,
        "network_authorized": bool(apply and refresh_market_data),
        "resolved_session": resolved_session.isoformat(),
        "artifact_id": artifact_id,
        "database_backup": None,
        "refreshed_tickers": [],
        "refresh_failures": [],
        "warnings": [],
        "errors": [],
    }
    with cycle_lock(lock_file):
        integrity = check_registered_artifact(db_path, artifact_id)
        report["artifact_integrity"] = integrity.get("status")
        if integrity.get("status") != "passed":
            raise DailyShadowCycleError(f"Artifact integrity failed: {integrity.get('errors', [])}", 2)

        before = audit_cache_coverage(db_path, artifact_id, resolved_session)
        report["cache_status_before"] = _compact_cache_status(before)
        backup: Path | None = None

        def ensure_backup() -> Path:
            nonlocal backup
            if backup is None:
                backup = _backup_database(db_path)
                report["database_backup"] = str(backup)
            return backup

        if apply and refresh_market_data and before["incomplete_ticker_count"]:
            ensure_backup()
            refreshed, failures = _refresh_missing_ranges(db_path, before, resolved_session)
            report["refreshed_tickers"] = refreshed
            report["refresh_failures"] = failures
            report["database_mutated"] = bool(refreshed)
            if failures:
                report["status"] = "failed"
                report["errors"].append("One or more authorized OHLCV refreshes failed.")
        elif before["incomplete_ticker_count"]:
            report["warnings"].append(
                "Cache is incomplete; no network refresh occurred without --apply --refresh-market-data."
            )

        after = audit_cache_coverage(db_path, artifact_id, resolved_session)
        report["cache_status_after"] = _compact_cache_status(after)
        if apply and refresh_market_data and after["incomplete_ticker_count"]:
            failed_tickers = {item["ticker"] for item in report["refresh_failures"]}
            for item in after["tickers"]:
                if not item["complete_through_resolved_session"] and item["ticker"] not in failed_tickers:
                    report["refresh_failures"].append(
                        {"ticker": item["ticker"], "error": "cache remains incomplete after refresh"}
                    )
            report["status"] = "failed"
            report["errors"].append("Authorized OHLCV refresh did not complete the required cache.")
        if report["refresh_failures"]:
            report["shadow_status"] = shadow_status_report(db_path, artifact_id)
            return report

        evaluation_time = reference_time if isinstance(reference_time, datetime) else None
        try:
            outcome_dry_run = dry_run_shadow_outcomes(db_path, evaluated_at=evaluation_time)
        except ShadowPredictionError as exc:
            if "records are unavailable" not in str(exc):
                raise
            outcome_dry_run = {
                "status": "no_shadow_predictions",
                "outcomes_planned": 0,
                "pending_outcomes": 0,
                "plan": {"missing_price_cases": []},
            }
        report["outcome_dry_run"] = {
            "status": outcome_dry_run["status"],
            "outcomes_planned": outcome_dry_run["outcomes_planned"],
            "pending_outcomes": outcome_dry_run["pending_outcomes"],
            "missing_price_cases": len(outcome_dry_run["plan"]["missing_price_cases"]),
        }
        report["outcomes_added_by_horizon"] = {"1": 0, "5": 0, "20": 0}
        if apply and outcome_dry_run["outcomes_planned"]:
            ensure_backup()
            outcome_apply = apply_shadow_outcomes(
                db_path, evaluated_at=evaluation_time, create_backup=False
            )
            report["outcome_apply"] = outcome_apply
            report["outcomes_added_by_horizon"] = outcome_apply["outcomes_created_by_horizon"]
            report["database_mutated"] = True

        cache_complete = latest_cache_complete_session(db_path, artifact_id)
        report["latest_cache_complete_session"] = None if cache_complete is None else cache_complete.isoformat()
        report["prediction_run"] = {"status": "skipped", "reason": "no_cache_complete_session"}
        report["prediction_count"] = 0
        if cache_complete is not None:
            prediction_dry_run = dry_run_shadow_prediction(db_path, artifact_id, cache_complete)
            report["prediction_dry_run"] = {
                "status": prediction_dry_run["status"],
                "prediction_date": prediction_dry_run["plan"]["prediction_date"],
                "prediction_count": prediction_dry_run["prediction_count"],
                "duplicate_run_id": prediction_dry_run["plan"]["duplicate_run_id"],
            }
            report["prediction_count"] = prediction_dry_run["prediction_count"]
            if prediction_dry_run["plan"]["duplicate_run_id"] is not None:
                report["prediction_run"] = {
                    "status": "skipped",
                    "reason": "duplicate_date_artifact_run",
                    "run_id": prediction_dry_run["plan"]["duplicate_run_id"],
                }
            elif cache_complete != resolved_session:
                report["prediction_run"] = {
                    "status": "skipped",
                    "reason": "latest_completed_session_not_cache_complete",
                    "cache_complete_session": cache_complete.isoformat(),
                }
            elif apply:
                ensure_backup()
                created = apply_shadow_prediction(
                    db_path, artifact_id, cache_complete, create_backup=False
                )
                report["prediction_run"] = {
                    "status": "created",
                    "run_id": created["run_id"],
                    "prediction_date": created["prediction_date"],
                    "prediction_count": created["prediction_count"],
                }
                report["database_mutated"] = True
            else:
                report["prediction_run"] = {
                    "status": "planned",
                    "prediction_date": cache_complete.isoformat(),
                    "prediction_count": prediction_dry_run["prediction_count"],
                }

        status = shadow_status_report(db_path, artifact_id)
        report["shadow_status"] = status
        report["outcomes_still_pending"] = status.get("pending_outcome_count", 0)
        report["sample_status"] = status.get("sample_status")
        if not report["database_mutated"]:
            report["status"] = "no_op" if apply else "dry_run_complete"
        return report
