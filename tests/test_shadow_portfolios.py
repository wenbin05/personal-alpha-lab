from __future__ import annotations

import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import storage
from src.datasets.builder import as_of_after_close
from src.modeling.shadow_portfolios import (
    POLICY_ID,
    SOURCE_ARTIFACT_ID,
    ShadowPortfolioError,
    _sample_status,
    apply_outcomes,
    build_cohort_plan,
    build_outcome_plan,
    create_cohort,
    dry_run_cohort,
    dry_run_outcomes,
    policy_registration_plan,
    portfolio_shadow_status,
    register_policy,
)
from src.modeling.shadow_predictions import _create_shadow_schema, apply_shadow_prediction
from src.quality.harness import check_portfolio_shadow_status
from src.utils.trading_calendar import next_trading_day
from tests.test_shadow_predictions import AS_OF, _seed_shadow_environment


FUTURE_PREDICTION_DATE = next_trading_day(datetime.now(UTC).date()).isoformat()


def _advance(start, count):
    current = start
    values = []
    for _ in range(count):
        current = next_trading_day(current)
        values.append(current)
    return values


def _seed_future_run(db_path: Path, artifact_id: str) -> int:
    with storage.connect(db_path) as conn:
        base = conn.execute("SELECT * FROM shadow_prediction_runs ORDER BY run_id DESC LIMIT 1").fetchone()
        created = (datetime.now(UTC) + timedelta(minutes=1)).isoformat(timespec="seconds")
        cursor = conn.execute(
            """
            INSERT INTO shadow_prediction_runs (
                prediction_date, as_of_timestamp, artifact_id, artifact_checksum,
                feature_manifest_hash, universe_hash, status, prediction_count,
                warnings_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, 'completed', 6, '[]', ?)
            """,
            (
                FUTURE_PREDICTION_DATE,
                as_of_after_close(pd.Timestamp(FUTURE_PREDICTION_DATE).date()).isoformat(),
                artifact_id,
                base["artifact_checksum"],
                base["feature_manifest_hash"],
                base["universe_hash"],
                created,
            ),
        )
        run_id = int(cursor.lastrowid)
        rows = [
            ("SPY", 1, 0.30),
            ("EEE", 6, 0.10),
            ("CCC", 3, 0.20),
            ("AAA", 2, 0.25),
            ("DDD", 5, 0.15),
            ("BBB", 3, 0.20),
        ]
        conn.executemany(
            """
            INSERT INTO shadow_predictions (
                prediction_run_id, ticker, prediction_date, predicted_value,
                predicted_rank, predicted_percentile, feature_input_hash,
                data_quality_flags_json, created_at
            ) VALUES (?, ?, ?, ?, ?, 0, ?, '{}', ?)
            """,
            [
                (run_id, ticker, FUTURE_PREDICTION_DATE, value, rank, f"hash-{ticker}", created)
                for ticker, rank, value in rows
            ],
        )
    return run_id


def _seed_portfolio_environment(tmp_path: Path) -> tuple[Path, str, int, int]:
    db_path, _artifact_id = _seed_shadow_environment(tmp_path)
    created = datetime.now(UTC).isoformat(timespec="seconds")
    with storage.connect(db_path) as conn:
        _create_shadow_schema(conn)
        cursor = conn.execute(
            """
            INSERT INTO shadow_prediction_runs (
                prediction_date, as_of_timestamp, artifact_id, artifact_checksum,
                feature_manifest_hash, universe_hash, status, prediction_count,
                warnings_json, created_at
            ) VALUES (?, ?, ?, 'artifact-checksum', 'feature-hash', 'universe-hash',
                      'completed', 0, '[]', ?)
            """,
            (AS_OF, as_of_after_close(pd.Timestamp(AS_OF).date()).isoformat(), SOURCE_ARTIFACT_ID, created),
        )
        existing_run_id = int(cursor.lastrowid)
    registered = register_policy(db_path, create_backup=False)
    assert registered["eligible_after_prediction_run_id"] == existing_run_id
    future_run_id = _seed_future_run(db_path, SOURCE_ARTIFACT_ID)
    return db_path, SOURCE_ARTIFACT_ID, existing_run_id, future_run_id


def _seed_prices(db_path: Path, *, omit: set[str] | None = None) -> list:
    omit = omit or set()
    prediction_date = pd.Timestamp(FUTURE_PREDICTION_DATE).date()
    sessions = _advance(prediction_date, 6)
    end_prices = {"AAA": 110.0, "BBB": 100.0, "CCC": 90.0, "DDD": 105.0, "EEE": 95.0, "SPY": 102.0}
    for ticker, end_price in end_prices.items():
        if ticker in omit:
            continue
        close = np.linspace(100.0, end_price, len(sessions))
        frame = pd.DataFrame(
            {
                "open": close,
                "high": close,
                "low": close,
                "close": close,
                "adj_close": close,
                "volume": np.full(len(sessions), 1_000_000.0),
            },
            index=pd.to_datetime(sessions),
        )
        storage.upsert_ohlcv(db_path, ticker, frame)
    return sessions


def test_policy_registration_freezes_existing_run_boundary_and_is_immutable(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF, create_backup=False)["run_id"])
    dry = policy_registration_plan(db_path)

    assert dry["database_mutated"] is False
    assert dry["eligible_after_prediction_run_id"] == run_id
    result = register_policy(db_path, create_backup=False)
    repeat = register_policy(db_path, create_backup=False)
    assert result["status"] == "registered"
    assert repeat["status"] == "already_registered"
    with storage.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE shadow_portfolio_policies SET selection_count=4 WHERE policy_id=?", (POLICY_ID,))


def test_existing_prediction_run_is_permanently_ineligible(tmp_path: Path) -> None:
    db_path, _artifact, existing_run_id, _future = _seed_portfolio_environment(tmp_path)
    with pytest.raises(ShadowPortfolioError, match="predates policy registration"):
        build_cohort_plan(db_path, existing_run_id)


def test_top_five_selection_excludes_spy_and_weights_sum_to_one(tmp_path: Path) -> None:
    db_path, _artifact, _existing, future_run_id = _seed_portfolio_environment(tmp_path)
    plan = build_cohort_plan(db_path, future_run_id)

    assert [row["ticker"] for row in plan.selections] == ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert [row["predicted_rank"] for row in plan.selections] == [2, 3, 3, 5, 6]
    assert sum(row["weight"] for row in plan.selections) == pytest.approx(1.0)
    assert all(row["weight"] == pytest.approx(0.2) for row in plan.selections)
    prediction_date = pd.Timestamp(FUTURE_PREDICTION_DATE).date()
    expected_entry = next_trading_day(prediction_date)
    expected_exit = _advance(expected_entry, 5)[-1]
    assert plan.entry_date == expected_entry.isoformat()
    assert plan.exit_date == expected_exit.isoformat()


def test_cohort_dry_run_is_read_only_and_apply_is_idempotent_and_immutable(tmp_path: Path) -> None:
    db_path, _artifact, _existing, future_run_id = _seed_portfolio_environment(tmp_path)
    before = db_path.read_bytes()
    dry = dry_run_cohort(db_path, future_run_id)
    assert dry["database_mutated"] is False
    assert db_path.read_bytes() == before

    first = create_cohort(db_path, future_run_id, create_backup=False)
    second = create_cohort(db_path, future_run_id, create_backup=False)
    assert first["status"] == "recorded"
    assert second["status"] == "already_exists"
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM shadow_portfolio_cohorts").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM shadow_portfolio_constituents").fetchone()[0] == 5
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE shadow_portfolio_constituents SET weight=1")


def test_outcome_timing_costs_and_idempotency(tmp_path: Path) -> None:
    db_path, _artifact, _existing, future_run_id = _seed_portfolio_environment(tmp_path)
    cohort_id = int(create_cohort(db_path, future_run_id, create_backup=False)["cohort_id"])
    sessions = _seed_prices(db_path)

    premature = build_outcome_plan(db_path, cohort_id=cohort_id, evaluated_at=as_of_after_close(sessions[4]))
    assert premature.planned_outcomes == []
    assert premature.pending_outcomes[0]["reason"] == "not_yet_mature"

    evaluation_time = as_of_after_close(sessions[5]) + timedelta(seconds=1)
    plan = build_outcome_plan(db_path, cohort_id=cohort_id, evaluated_at=evaluation_time)
    outcome = plan.planned_outcomes[0]
    assert outcome["entry_date"] == sessions[0].isoformat()
    assert outcome["exit_date"] == sessions[5].isoformat()
    assert outcome["gross_return"] == pytest.approx(0.0)
    assert outcome["transaction_cost_return"] == pytest.approx(0.002)
    assert outcome["net_return"] == pytest.approx(-0.002)
    assert outcome["benchmark_return"] == pytest.approx(0.02)
    assert outcome["excess_return"] == pytest.approx(-0.022)

    first = apply_outcomes(db_path, cohort_id=cohort_id, evaluated_at=evaluation_time, create_backup=False)
    second = apply_outcomes(db_path, cohort_id=cohort_id, evaluated_at=evaluation_time, create_backup=False)
    assert first["outcomes_created"] == 1
    assert second["status"] == "no_changes"
    with storage.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE shadow_portfolio_outcomes SET net_return=0")


def test_missing_price_is_pending_and_not_fabricated(tmp_path: Path) -> None:
    db_path, _artifact, _existing, future_run_id = _seed_portfolio_environment(tmp_path)
    cohort_id = int(create_cohort(db_path, future_run_id, create_backup=False)["cohort_id"])
    sessions = _seed_prices(db_path, omit={"EEE"})
    evaluation_time = as_of_after_close(sessions[5]) + timedelta(seconds=1)

    dry = dry_run_outcomes(db_path, cohort_id=cohort_id, evaluated_at=evaluation_time)

    assert dry["outcomes_planned"] == 0
    assert dry["plan"]["missing_price_cases"]
    assert "EEE:missing_entry_close" in dry["plan"]["missing_price_cases"][0]["reason"]
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM shadow_portfolio_outcomes").fetchone()[0] == 0


def test_portfolio_status_and_harness_report_sample_governance(tmp_path: Path) -> None:
    db_path, _artifact, _existing, future_run_id = _seed_portfolio_environment(tmp_path)
    create_cohort(db_path, future_run_id, create_backup=False)
    report = portfolio_shadow_status(db_path)
    harness = check_portfolio_shadow_status(db_path)
    assert report["status"] == "passed"
    assert report["cohort_count"] == 1
    assert report["matured_cohort_count"] == 0
    assert report["sample_status"] == "insufficient_forward_sample"
    assert harness.status == "passed"


def test_portfolio_operations_do_not_mutate_scanner_models_datasets_or_predictions(tmp_path: Path) -> None:
    db_path, _artifact, _existing, future_run_id = _seed_portfolio_environment(tmp_path)
    with storage.connect(db_path) as conn:
        before = {
            "scan_results": conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0],
            "model_runs": conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0],
            "dataset_49": conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=49").fetchone()[0],
            "dataset_50": conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=50").fetchone()[0],
            "shadow_runs": conn.execute("SELECT COUNT(*) FROM shadow_prediction_runs").fetchone()[0],
            "predictions": conn.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0],
        }
    create_cohort(db_path, future_run_id, create_backup=False)
    with storage.connect(db_path) as conn:
        after = {
            "scan_results": conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0],
            "model_runs": conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0],
            "dataset_49": conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=49").fetchone()[0],
            "dataset_50": conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=50").fetchone()[0],
            "shadow_runs": conn.execute("SELECT COUNT(*) FROM shadow_prediction_runs").fetchone()[0],
            "predictions": conn.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0],
        }
    assert after == before


@pytest.mark.parametrize(
    ("count", "expected"),
    [
        (0, "insufficient_forward_sample"),
        (19, "insufficient_forward_sample"),
        (20, "preliminary_only"),
        (59, "preliminary_only"),
        (60, "developing_sample"),
        (119, "developing_sample"),
        (120, "eligible_for_formal_review"),
    ],
)
def test_portfolio_sample_status(count: int, expected: str) -> None:
    assert _sample_status(count) == expected
