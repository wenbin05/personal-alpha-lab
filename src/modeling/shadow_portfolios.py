from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.datasets.builder import as_of_after_close
from src.utils.trading_calendar import latest_expected_trading_day, next_trading_day


POLICY_ID = "top5_equal_weight_5session_v1"
SOURCE_ARTIFACT_ID = "shadow_ridge_technical_v1_1ee8071db3f0"
BENCHMARK_TICKER = "SPY"
SELECTION_COUNT = 5
HOLDING_PERIOD_SESSIONS = 5
TRANSACTION_COST_BPS_PER_SIDE = 10.0
POLICY_STATUS = "frozen_exploratory"
EVALUATION_REGIME = "exploratory_shadow"


class ShadowPortfolioError(ValueError):
    pass


@dataclass(frozen=True)
class PortfolioCohortPlan:
    policy_id: str
    prediction_run_id: int
    prediction_date: str
    entry_date: str
    exit_date: str
    selections: list[dict[str, Any]]
    duplicate_cohort_id: int | None
    warnings: list[str]


@dataclass(frozen=True)
class PortfolioOutcomePlan:
    policy_id: str
    evaluated_at: str
    cohort_count: int
    existing_outcome_count: int
    planned_outcomes: list[dict[str, Any]]
    pending_outcomes: list[dict[str, Any]]
    missing_price_cases: list[dict[str, Any]]
    warnings: list[str]


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True, separators=(",", ":"), default=str)


def _now_utc() -> datetime:
    return datetime.now(UTC)


def _as_utc(value: datetime | None) -> datetime:
    resolved = value or _now_utc()
    if resolved.tzinfo is None:
        resolved = resolved.replace(tzinfo=UTC)
    return resolved.astimezone(UTC)


def _readonly_connect(db_path: str | Path) -> sqlite3.Connection:
    uri = f"file:{Path(db_path).resolve().as_posix()}?mode=ro"
    conn = sqlite3.connect(uri, uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def _backup_database(db_path: str | Path) -> Path:
    source = Path(db_path)
    stamp = _now_utc().strftime("%Y%m%d_%H%M%S_%f")
    backup = source.with_name(f"{source.stem}_backup_phase3c1_{stamp}{source.suffix}")
    with sqlite3.connect(source) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)
    return backup


def _policy_spec() -> dict[str, Any]:
    return {
        "policy_id": POLICY_ID,
        "artifact_id": SOURCE_ARTIFACT_ID,
        "exclude_tickers": [BENCHMARK_TICKER],
        "selection": "top_stored_prediction_rank",
        "selection_count": SELECTION_COUNT,
        "weighting_method": "equal_weight",
        "constituent_weight": 1.0 / SELECTION_COUNT,
        "long_only": True,
        "leverage": 1.0,
        "entry_convention": "next_trading_session_close_after_prediction_date",
        "holding_period_sessions": HOLDING_PERIOD_SESSIONS,
        "exit_convention": "five_sessions_after_entry_close",
        "transaction_cost_bps_per_side": TRANSACTION_COST_BPS_PER_SIDE,
        "transaction_cost_method": "entry_notional_plus_exit_proceeds",
        "benchmark_ticker": BENCHMARK_TICKER,
        "research_only": True,
        "scanner_scoring_effect": 0,
    }


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS shadow_portfolio_policies (
            policy_id TEXT PRIMARY KEY,
            artifact_id TEXT NOT NULL,
            status TEXT NOT NULL,
            evaluation_regime TEXT NOT NULL,
            selection_count INTEGER NOT NULL,
            weighting_method TEXT NOT NULL,
            holding_period_sessions INTEGER NOT NULL,
            transaction_cost_bps_per_side REAL NOT NULL,
            benchmark_ticker TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            eligible_after_prediction_run_id INTEGER NOT NULL,
            policy_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS shadow_portfolio_cohorts (
            cohort_id INTEGER PRIMARY KEY AUTOINCREMENT,
            policy_id TEXT NOT NULL,
            prediction_run_id INTEGER NOT NULL,
            prediction_date TEXT NOT NULL,
            entry_date TEXT NOT NULL,
            exit_date TEXT NOT NULL,
            status TEXT NOT NULL,
            constituent_count INTEGER NOT NULL,
            warnings_json TEXT NOT NULL DEFAULT '[]',
            created_at TEXT NOT NULL,
            UNIQUE(policy_id, prediction_run_id)
        );
        CREATE TABLE IF NOT EXISTS shadow_portfolio_constituents (
            constituent_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cohort_id INTEGER NOT NULL,
            prediction_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            predicted_rank INTEGER NOT NULL,
            weight REAL NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(cohort_id, ticker),
            UNIQUE(cohort_id, prediction_id)
        );
        CREATE TABLE IF NOT EXISTS shadow_portfolio_outcomes (
            outcome_id INTEGER PRIMARY KEY AUTOINCREMENT,
            cohort_id INTEGER NOT NULL UNIQUE,
            entry_date TEXT NOT NULL,
            exit_date TEXT NOT NULL,
            gross_return REAL NOT NULL,
            transaction_cost_return REAL NOT NULL,
            net_return REAL NOT NULL,
            benchmark_return REAL NOT NULL,
            excess_return REAL NOT NULL,
            label_available_at TEXT NOT NULL,
            evaluated_at TEXT NOT NULL,
            constituent_returns_json TEXT NOT NULL,
            data_quality_flags_json TEXT NOT NULL DEFAULT '{}',
            created_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_shadow_portfolio_cohorts_policy_date
            ON shadow_portfolio_cohorts (policy_id, prediction_date);
        CREATE INDEX IF NOT EXISTS idx_shadow_portfolio_constituents_cohort
            ON shadow_portfolio_constituents (cohort_id, predicted_rank, ticker);
        CREATE TRIGGER IF NOT EXISTS shadow_portfolio_policies_immutable_update
        BEFORE UPDATE ON shadow_portfolio_policies
        BEGIN SELECT RAISE(ABORT, 'shadow portfolio policies are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS shadow_portfolio_policies_immutable_delete
        BEFORE DELETE ON shadow_portfolio_policies
        BEGIN SELECT RAISE(ABORT, 'shadow portfolio policies are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS shadow_portfolio_cohorts_immutable_update
        BEFORE UPDATE ON shadow_portfolio_cohorts
        BEGIN SELECT RAISE(ABORT, 'shadow portfolio cohorts are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS shadow_portfolio_cohorts_immutable_delete
        BEFORE DELETE ON shadow_portfolio_cohorts
        BEGIN SELECT RAISE(ABORT, 'shadow portfolio cohorts are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS shadow_portfolio_constituents_immutable_update
        BEFORE UPDATE ON shadow_portfolio_constituents
        BEGIN SELECT RAISE(ABORT, 'shadow portfolio constituents are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS shadow_portfolio_constituents_immutable_delete
        BEFORE DELETE ON shadow_portfolio_constituents
        BEGIN SELECT RAISE(ABORT, 'shadow portfolio constituents are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS shadow_portfolio_outcomes_immutable_update
        BEFORE UPDATE ON shadow_portfolio_outcomes
        BEGIN SELECT RAISE(ABORT, 'shadow portfolio outcomes are immutable'); END;
        CREATE TRIGGER IF NOT EXISTS shadow_portfolio_outcomes_immutable_delete
        BEFORE DELETE ON shadow_portfolio_outcomes
        BEGIN SELECT RAISE(ABORT, 'shadow portfolio outcomes are immutable'); END;
        """
    )


def policy_registration_plan(db_path: str | Path) -> dict[str, Any]:
    with _readonly_connect(db_path) as conn:
        if not _table_exists(conn, "shadow_prediction_runs"):
            raise ShadowPortfolioError("Shadow prediction runs are unavailable.")
        maximum_run_id = int(conn.execute("SELECT COALESCE(MAX(run_id), 0) FROM shadow_prediction_runs").fetchone()[0])
        existing = None
        if _table_exists(conn, "shadow_portfolio_policies"):
            row = conn.execute("SELECT * FROM shadow_portfolio_policies WHERE policy_id=?", (POLICY_ID,)).fetchone()
            existing = None if row is None else dict(row)
    boundary = int(existing["eligible_after_prediction_run_id"]) if existing else maximum_run_id
    return {
        "status": "already_registered" if existing else "ready",
        "dry_run": True,
        "database_mutated": False,
        "policy": _policy_spec(),
        "eligible_after_prediction_run_id": boundary,
        "existing_policy": existing,
        "next_eligible_prediction_run_id_floor": boundary + 1,
    }


def register_policy(db_path: str | Path, *, create_backup: bool = True) -> dict[str, Any]:
    plan = policy_registration_plan(db_path)
    if plan["existing_policy"] is not None:
        existing_spec = json.loads(str(plan["existing_policy"]["policy_json"]))
        if existing_spec != _policy_spec():
            raise ShadowPortfolioError(f"Policy {POLICY_ID} exists with a different immutable specification.")
        return {
            "status": "already_registered",
            "database_mutated": False,
            "policy_id": POLICY_ID,
            "registered_at": plan["existing_policy"]["registered_at"],
            "eligible_after_prediction_run_id": plan["existing_policy"]["eligible_after_prediction_run_id"],
            "database_backup": None,
        }
    backup = _backup_database(db_path) if create_backup else None
    now = _now_utc().isoformat(timespec="seconds")
    with sqlite3.connect(db_path) as conn:
        conn.execute("BEGIN IMMEDIATE")
        _create_schema(conn)
        conn.execute(
            """
            INSERT INTO shadow_portfolio_policies (
                policy_id, artifact_id, status, evaluation_regime, selection_count,
                weighting_method, holding_period_sessions, transaction_cost_bps_per_side,
                benchmark_ticker, registered_at, eligible_after_prediction_run_id,
                policy_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                POLICY_ID,
                SOURCE_ARTIFACT_ID,
                POLICY_STATUS,
                EVALUATION_REGIME,
                SELECTION_COUNT,
                "equal_weight",
                HOLDING_PERIOD_SESSIONS,
                TRANSACTION_COST_BPS_PER_SIDE,
                BENCHMARK_TICKER,
                now,
                int(plan["eligible_after_prediction_run_id"]),
                _canonical_json(_policy_spec()),
                now,
            ),
        )
    return {
        "status": "registered",
        "database_mutated": True,
        "policy_id": POLICY_ID,
        "registered_at": now,
        "eligible_after_prediction_run_id": int(plan["eligible_after_prediction_run_id"]),
        "next_eligible_prediction_run_id_floor": int(plan["eligible_after_prediction_run_id"]) + 1,
        "database_backup": None if backup is None else str(backup),
    }


def _advance_sessions(start: date, count: int) -> date:
    current = start
    for _ in range(count):
        current = next_trading_day(current)
    return current


def _policy_record(conn: sqlite3.Connection, policy_id: str) -> dict[str, Any]:
    if not _table_exists(conn, "shadow_portfolio_policies"):
        raise ShadowPortfolioError("Shadow portfolio policy registry is unavailable.")
    row = conn.execute("SELECT * FROM shadow_portfolio_policies WHERE policy_id=?", (policy_id,)).fetchone()
    if row is None:
        raise ShadowPortfolioError(f"Policy {policy_id} is not registered.")
    record = dict(row)
    if record["artifact_id"] != SOURCE_ARTIFACT_ID or json.loads(record["policy_json"]) != _policy_spec():
        raise ShadowPortfolioError("Registered policy does not match the frozen policy contract.")
    return record


def build_cohort_plan(
    db_path: str | Path,
    prediction_run_id: int,
    *,
    policy_id: str = POLICY_ID,
) -> PortfolioCohortPlan:
    with _readonly_connect(db_path) as conn:
        policy = _policy_record(conn, policy_id)
        run = conn.execute("SELECT * FROM shadow_prediction_runs WHERE run_id=?", (int(prediction_run_id),)).fetchone()
        if run is None:
            raise ShadowPortfolioError(f"Shadow prediction run {prediction_run_id} was not found.")
        run = dict(run)
        boundary = int(policy["eligible_after_prediction_run_id"])
        if int(prediction_run_id) <= boundary:
            raise ShadowPortfolioError(
                f"Prediction run {prediction_run_id} predates policy registration; eligible runs must be greater than {boundary}."
            )
        if run["artifact_id"] != policy["artifact_id"]:
            raise ShadowPortfolioError("Prediction run artifact does not match the frozen portfolio policy.")
        if run["status"] != "completed":
            raise ShadowPortfolioError("Only completed immutable shadow runs are eligible.")
        registered_at = pd.to_datetime(policy["registered_at"], utc=True)
        run_created_at = pd.to_datetime(run["created_at"], utc=True)
        run_as_of = pd.to_datetime(run["as_of_timestamp"], utc=True)
        if run_created_at <= registered_at or run_as_of <= registered_at:
            raise ShadowPortfolioError(
                "Prediction run timestamps predate policy registration; retrospective cohorts are blocked."
            )
        duplicate = conn.execute(
            "SELECT cohort_id FROM shadow_portfolio_cohorts WHERE policy_id=? AND prediction_run_id=?",
            (policy_id, int(prediction_run_id)),
        ).fetchone()
        predictions = [
            dict(row)
            for row in conn.execute(
                """
                SELECT prediction_id, ticker, predicted_rank, predicted_value
                FROM shadow_predictions
                WHERE prediction_run_id=? AND ticker<>?
                ORDER BY predicted_rank ASC, ticker ASC
                LIMIT ?
                """,
                (int(prediction_run_id), BENCHMARK_TICKER, SELECTION_COUNT),
            )
        ]
    if len(predictions) != SELECTION_COUNT:
        raise ShadowPortfolioError(
            f"Run {prediction_run_id} has {len(predictions)} eligible equities; exactly {SELECTION_COUNT} are required."
        )
    prediction_date = pd.to_datetime(run["prediction_date"]).date()
    entry_date = next_trading_day(prediction_date)
    exit_date = _advance_sessions(entry_date, HOLDING_PERIOD_SESSIONS)
    weight = 1.0 / SELECTION_COUNT
    selections = [
        {
            "prediction_id": int(row["prediction_id"]),
            "ticker": str(row["ticker"]),
            "predicted_rank": int(row["predicted_rank"]),
            "predicted_value": float(row["predicted_value"]),
            "weight": weight,
        }
        for row in predictions
    ]
    return PortfolioCohortPlan(
        policy_id=policy_id,
        prediction_run_id=int(prediction_run_id),
        prediction_date=prediction_date.isoformat(),
        entry_date=entry_date.isoformat(),
        exit_date=exit_date.isoformat(),
        selections=selections,
        duplicate_cohort_id=None if duplicate is None else int(duplicate["cohort_id"]),
        warnings=[],
    )


def dry_run_cohort(db_path: str | Path, prediction_run_id: int) -> dict[str, Any]:
    plan = build_cohort_plan(db_path, prediction_run_id)
    return {
        "status": "already_exists" if plan.duplicate_cohort_id is not None else "ready",
        "dry_run": True,
        "database_mutated": False,
        "plan": asdict(plan),
        "research_only": True,
        "scanner_scoring_effect": 0,
    }


def create_cohort(
    db_path: str | Path,
    prediction_run_id: int,
    *,
    create_backup: bool = True,
) -> dict[str, Any]:
    plan = build_cohort_plan(db_path, prediction_run_id)
    if plan.duplicate_cohort_id is not None:
        return {
            "status": "already_exists",
            "database_mutated": False,
            "cohort_id": plan.duplicate_cohort_id,
            "database_backup": None,
        }
    backup = _backup_database(db_path) if create_backup else None
    now = _now_utc().isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _create_schema(conn)
        duplicate = conn.execute(
            "SELECT cohort_id FROM shadow_portfolio_cohorts WHERE policy_id=? AND prediction_run_id=?",
            (plan.policy_id, plan.prediction_run_id),
        ).fetchone()
        if duplicate is not None:
            raise ShadowPortfolioError(f"Duplicate cohort blocked: {int(duplicate[0])}.")
        cursor = conn.execute(
            """
            INSERT INTO shadow_portfolio_cohorts (
                policy_id, prediction_run_id, prediction_date, entry_date, exit_date,
                status, constituent_count, warnings_json, created_at
            ) VALUES (?, ?, ?, ?, ?, 'selected_pending_outcome', ?, ?, ?)
            """,
            (
                plan.policy_id,
                plan.prediction_run_id,
                plan.prediction_date,
                plan.entry_date,
                plan.exit_date,
                len(plan.selections),
                _canonical_json(plan.warnings),
                now,
            ),
        )
        cohort_id = int(cursor.lastrowid)
        conn.executemany(
            """
            INSERT INTO shadow_portfolio_constituents (
                cohort_id, prediction_id, ticker, predicted_rank, weight, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            [
                (cohort_id, row["prediction_id"], row["ticker"], row["predicted_rank"], row["weight"], now)
                for row in plan.selections
            ],
        )
        stored_weight = float(
            conn.execute(
                "SELECT COALESCE(SUM(weight), 0) FROM shadow_portfolio_constituents WHERE cohort_id=?",
                (cohort_id,),
            ).fetchone()[0]
        )
        if abs(stored_weight - 1.0) > 1e-12:
            raise ShadowPortfolioError(f"Stored cohort weights sum to {stored_weight}, not 1.0.")
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "status": "recorded",
        "database_mutated": True,
        "cohort_id": cohort_id,
        "prediction_run_id": plan.prediction_run_id,
        "prediction_date": plan.prediction_date,
        "constituent_count": len(plan.selections),
        "database_backup": None if backup is None else str(backup),
        "research_only": True,
        "scanner_scoring_effect": 0,
    }


def _close_map(conn: sqlite3.Connection, ticker: str, cutoff: date) -> dict[date, float]:
    rows = conn.execute(
        "SELECT date, close FROM ohlcv_cache WHERE ticker=? AND date<=? ORDER BY date",
        (ticker, cutoff.isoformat()),
    ).fetchall()
    return {
        pd.to_datetime(row["date"]).date(): float(row["close"])
        for row in rows
        if row["close"] is not None
    }


def build_outcome_plan(
    db_path: str | Path,
    *,
    policy_id: str = POLICY_ID,
    cohort_id: int | None = None,
    evaluated_at: datetime | None = None,
) -> PortfolioOutcomePlan:
    evaluation_time = _as_utc(evaluated_at)
    cache_cutoff = latest_expected_trading_day(evaluation_time)
    with _readonly_connect(db_path) as conn:
        policy = _policy_record(conn, policy_id)
        params: list[Any] = [policy_id]
        cohort_filter = ""
        if cohort_id is not None:
            cohort_filter = "AND c.cohort_id=?"
            params.append(int(cohort_id))
        cohorts = [
            dict(row)
            for row in conn.execute(
                f"""
                SELECT c.*
                FROM shadow_portfolio_cohorts c
                WHERE c.policy_id=? {cohort_filter}
                ORDER BY c.prediction_date, c.cohort_id
                """,
                tuple(params),
            )
        ]
        if cohort_id is not None and not cohorts:
            raise ShadowPortfolioError(f"Portfolio cohort {cohort_id} was not found.")
        existing = {
            int(row["cohort_id"])
            for row in conn.execute("SELECT cohort_id FROM shadow_portfolio_outcomes")
        } if _table_exists(conn, "shadow_portfolio_outcomes") else set()
        constituent_rows = {
            int(cohort["cohort_id"]): [
                dict(row)
                for row in conn.execute(
                    "SELECT * FROM shadow_portfolio_constituents WHERE cohort_id=? ORDER BY predicted_rank, ticker",
                    (int(cohort["cohort_id"]),),
                )
            ]
            for cohort in cohorts
        }
        tickers = sorted(
            {BENCHMARK_TICKER}
            | {str(row["ticker"]) for rows in constituent_rows.values() for row in rows}
        )
        close_maps = {ticker: _close_map(conn, ticker, cache_cutoff) for ticker in tickers}

    planned: list[dict[str, Any]] = []
    pending: list[dict[str, Any]] = []
    missing_cases: list[dict[str, Any]] = []
    for cohort in cohorts:
        selected_id = int(cohort["cohort_id"])
        if selected_id in existing:
            continue
        entry_date = pd.to_datetime(cohort["entry_date"]).date()
        exit_date = pd.to_datetime(cohort["exit_date"]).date()
        available_at = as_of_after_close(exit_date)
        base = {
            "cohort_id": selected_id,
            "prediction_run_id": int(cohort["prediction_run_id"]),
            "entry_date": entry_date.isoformat(),
            "exit_date": exit_date.isoformat(),
            "label_available_at": available_at.isoformat(),
        }
        if available_at > evaluation_time:
            pending.append({**base, "reason": "not_yet_mature"})
            continue
        rows = constituent_rows[selected_id]
        weight_sum = sum(float(row["weight"]) for row in rows)
        if len(rows) != SELECTION_COUNT or abs(weight_sum - 1.0) > 1e-12:
            raise ShadowPortfolioError(f"Cohort {selected_id} violates the immutable constituent contract.")
        missing: list[str] = []
        returns: list[dict[str, Any]] = []
        for constituent in rows:
            ticker = str(constituent["ticker"])
            closes = close_maps.get(ticker, {})
            if entry_date not in closes:
                missing.append(f"{ticker}:missing_entry_close")
                continue
            if exit_date not in closes:
                missing.append(f"{ticker}:missing_exit_close")
                continue
            entry_price = float(closes[entry_date])
            if entry_price == 0:
                missing.append(f"{ticker}:zero_entry_close")
                continue
            ticker_return = float(closes[exit_date]) / entry_price - 1.0
            returns.append(
                {
                    "ticker": ticker,
                    "weight": float(constituent["weight"]),
                    "entry_price": entry_price,
                    "exit_price": float(closes[exit_date]),
                    "return": ticker_return,
                }
            )
        spy = close_maps.get(BENCHMARK_TICKER, {})
        if entry_date not in spy:
            missing.append("SPY:missing_entry_close")
        if exit_date not in spy:
            missing.append("SPY:missing_exit_close")
        if missing:
            missing_row = {**base, "reason": ",".join(missing)}
            pending.append(missing_row)
            missing_cases.append(missing_row)
            continue
        benchmark_entry = float(spy[entry_date])
        if benchmark_entry == 0:
            missing_row = {**base, "reason": "SPY:zero_entry_close"}
            pending.append(missing_row)
            missing_cases.append(missing_row)
            continue
        gross_return = sum(row["weight"] * row["return"] for row in returns)
        cost_rate = float(policy["transaction_cost_bps_per_side"]) / 10_000.0
        transaction_cost_return = cost_rate + cost_rate * (1.0 + gross_return)
        net_return = gross_return - transaction_cost_return
        benchmark_return = float(spy[exit_date]) / benchmark_entry - 1.0
        planned.append(
            {
                **base,
                "gross_return": gross_return,
                "transaction_cost_return": transaction_cost_return,
                "net_return": net_return,
                "benchmark_return": benchmark_return,
                "excess_return": net_return - benchmark_return,
                "constituent_returns": returns,
                "data_quality_flags": {
                    "cache_only": True,
                    "benchmark": BENCHMARK_TICKER,
                    "transaction_cost_bps_per_side": TRANSACTION_COST_BPS_PER_SIDE,
                    "research_only": True,
                    "scanner_scoring_effect": 0,
                },
            }
        )
    warnings = []
    if missing_cases:
        warnings.append(f"{len(missing_cases)} matured cohorts are blocked by incomplete cached prices.")
    return PortfolioOutcomePlan(
        policy_id=policy_id,
        evaluated_at=evaluation_time.isoformat(timespec="seconds"),
        cohort_count=len(cohorts),
        existing_outcome_count=sum(int(cohort["cohort_id"]) in existing for cohort in cohorts),
        planned_outcomes=planned,
        pending_outcomes=pending,
        missing_price_cases=missing_cases,
        warnings=warnings,
    )


def dry_run_outcomes(
    db_path: str | Path,
    *,
    cohort_id: int | None = None,
    evaluated_at: datetime | None = None,
) -> dict[str, Any]:
    plan = build_outcome_plan(db_path, cohort_id=cohort_id, evaluated_at=evaluated_at)
    return {
        "status": "ready" if plan.planned_outcomes else "no_newly_matured_outcomes",
        "dry_run": True,
        "database_mutated": False,
        "plan": asdict(plan),
        "outcomes_planned": len(plan.planned_outcomes),
        "pending_outcomes": len(plan.pending_outcomes),
        "research_only": True,
        "scanner_scoring_effect": 0,
    }


def apply_outcomes(
    db_path: str | Path,
    *,
    cohort_id: int | None = None,
    evaluated_at: datetime | None = None,
    create_backup: bool = True,
) -> dict[str, Any]:
    plan = build_outcome_plan(db_path, cohort_id=cohort_id, evaluated_at=evaluated_at)
    if not plan.planned_outcomes:
        return {
            "status": "no_changes",
            "database_mutated": False,
            "outcomes_created": 0,
            "pending_outcomes": len(plan.pending_outcomes),
            "database_backup": None,
        }
    backup = _backup_database(db_path) if create_backup else None
    now = _now_utc().isoformat(timespec="seconds")
    conn = sqlite3.connect(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        _create_schema(conn)
        conn.executemany(
            """
            INSERT INTO shadow_portfolio_outcomes (
                cohort_id, entry_date, exit_date, gross_return,
                transaction_cost_return, net_return, benchmark_return,
                excess_return, label_available_at, evaluated_at,
                constituent_returns_json, data_quality_flags_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    row["cohort_id"], row["entry_date"], row["exit_date"], row["gross_return"],
                    row["transaction_cost_return"], row["net_return"], row["benchmark_return"],
                    row["excess_return"], row["label_available_at"], plan.evaluated_at,
                    _canonical_json(row["constituent_returns"]),
                    _canonical_json(row["data_quality_flags"]), now,
                )
                for row in plan.planned_outcomes
            ],
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return {
        "status": "recorded",
        "database_mutated": True,
        "outcomes_created": len(plan.planned_outcomes),
        "pending_outcomes": len(plan.pending_outcomes),
        "missing_price_cases": len(plan.missing_price_cases),
        "database_backup": None if backup is None else str(backup),
        "research_only": True,
        "scanner_scoring_effect": 0,
    }


def _sample_status(matured_count: int) -> str:
    if matured_count < 20:
        return "insufficient_forward_sample"
    if matured_count < 60:
        return "preliminary_only"
    if matured_count < 120:
        return "developing_sample"
    return "eligible_for_formal_review"


def portfolio_shadow_status(db_path: str | Path, *, policy_id: str = POLICY_ID) -> dict[str, Any]:
    with _readonly_connect(db_path) as conn:
        if not _table_exists(conn, "shadow_portfolio_policies"):
            return {
                "status": "passed",
                "policy_id": policy_id,
                "policy_registered": False,
                "cohort_count": 0,
                "matured_cohort_count": 0,
                "pending_cohort_count": 0,
                "sample_status": "insufficient_forward_sample",
                "violations": [],
            }
        policy_row = conn.execute("SELECT * FROM shadow_portfolio_policies WHERE policy_id=?", (policy_id,)).fetchone()
        if policy_row is None:
            raise ShadowPortfolioError(f"Policy {policy_id} is not registered.")
        policy = dict(policy_row)
        cohorts = [
            dict(row)
            for row in conn.execute(
                """
                SELECT c.*, o.outcome_id, o.gross_return, o.transaction_cost_return,
                       o.net_return, o.benchmark_return, o.excess_return, o.label_available_at
                FROM shadow_portfolio_cohorts c
                LEFT JOIN shadow_portfolio_outcomes o ON o.cohort_id=c.cohort_id
                WHERE c.policy_id=?
                ORDER BY c.prediction_date DESC, c.cohort_id DESC
                """,
                (policy_id,),
            )
        ]
        constituents = [
            dict(row)
            for row in conn.execute(
                """
                SELECT x.*, c.prediction_run_id, c.prediction_date
                FROM shadow_portfolio_constituents x
                JOIN shadow_portfolio_cohorts c ON c.cohort_id=x.cohort_id
                WHERE c.policy_id=?
                ORDER BY c.prediction_date DESC, x.predicted_rank, x.ticker
                """,
                (policy_id,),
            )
        ]
        duplicate_cohorts = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT policy_id, prediction_run_id, COUNT(*)
                    FROM shadow_portfolio_cohorts
                    GROUP BY policy_id, prediction_run_id HAVING COUNT(*)>1
                )
                """
            ).fetchone()[0]
        )
        duplicate_constituents = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT cohort_id, ticker, COUNT(*)
                    FROM shadow_portfolio_constituents
                    GROUP BY cohort_id, ticker HAVING COUNT(*)>1
                )
                """
            ).fetchone()[0]
        )
        weight_violations = [
            dict(row)
            for row in conn.execute(
                """
                SELECT cohort_id, SUM(weight) AS weight_sum, COUNT(*) AS constituent_count
                FROM shadow_portfolio_constituents
                GROUP BY cohort_id
                HAVING ABS(SUM(weight)-1.0)>0.000000000001 OR COUNT(*)<>?
                """,
                (SELECTION_COUNT,),
            )
        ]
        next_run = conn.execute(
            """
            SELECT r.run_id, r.prediction_date
            FROM shadow_prediction_runs r
            LEFT JOIN shadow_portfolio_cohorts c
              ON c.prediction_run_id=r.run_id AND c.policy_id=?
            WHERE r.artifact_id=? AND r.run_id>? AND c.cohort_id IS NULL
            ORDER BY r.run_id LIMIT 1
            """,
            (policy_id, SOURCE_ARTIFACT_ID, int(policy["eligible_after_prediction_run_id"])),
        ).fetchone()
    matured = sum(row["outcome_id"] is not None for row in cohorts)
    outcome_plan = build_outcome_plan(db_path, policy_id=policy_id) if cohorts else None
    violations: list[str] = []
    if duplicate_cohorts:
        violations.append("duplicate policy/prediction-run cohorts detected")
    if duplicate_constituents:
        violations.append("duplicate cohort constituents detected")
    if weight_violations:
        violations.append("cohort weight or constituent-count contract violation detected")
    if json.loads(policy["policy_json"]) != _policy_spec():
        violations.append("registered policy specification differs from the frozen contract")
    return {
        "status": "passed" if not violations else "failed",
        "policy_id": policy_id,
        "policy_registered": True,
        "registered_at": policy["registered_at"],
        "eligible_after_prediction_run_id": int(policy["eligible_after_prediction_run_id"]),
        "next_eligible_prediction_run_id_floor": int(policy["eligible_after_prediction_run_id"]) + 1,
        "next_available_prediction_run": None if next_run is None else dict(next_run),
        "cohort_count": len(cohorts),
        "matured_cohort_count": matured,
        "pending_cohort_count": len(cohorts) - matured,
        "sample_status": _sample_status(matured),
        "cohorts": cohorts,
        "constituents": constituents,
        "duplicate_cohort_count": duplicate_cohorts,
        "duplicate_constituent_count": duplicate_constituents,
        "weight_violation_count": len(weight_violations),
        "missing_price_case_count": 0 if outcome_plan is None else len(outcome_plan.missing_price_cases),
        "violations": violations,
        "research_only": True,
        "scanner_scoring_effect": 0,
    }
