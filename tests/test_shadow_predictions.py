from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import market_data, storage
from src.modeling.shadow_predictions import (
    ShadowPredictionError,
    _rank_predictions,
    _shadow_sample_status,
    apply_shadow_prediction,
    apply_shadow_outcomes,
    build_shadow_outcome_plan,
    build_shadow_prediction_plan,
    dry_run_shadow_outcomes,
    dry_run_shadow_prediction,
    latest_cache_complete_session,
    list_shadow_prediction_outcomes,
    list_shadow_prediction_runs,
    shadow_status_report,
)
from src.quality.harness import check_shadow_status
from src.datasets.builder import as_of_after_close
from src.utils.trading_calendar import next_trading_day
from tests.test_model_artifacts import _apply_test_artifact


AS_OF = "2024-03-28"


def _history(end: str = AS_OF, rows: int = 280, offset: float = 0.0) -> pd.DataFrame:
    dates = pd.bdate_range(end=end, periods=rows)
    close = 100.0 + offset + np.linspace(0, 20, rows)
    return pd.DataFrame(
        {
            "open": close - 0.2,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "adj_close": close,
            "volume": np.full(rows, 2_000_000.0 + offset * 1000),
        },
        index=dates,
    )


def _seed_shadow_environment(tmp_path: Path) -> tuple[Path, str]:
    db_path, result = _apply_test_artifact(tmp_path)
    for index, ticker in enumerate(["AAA", "BBB", "SPY", "QQQ", "IWM", "^VIX"]):
        storage.upsert_ohlcv(db_path, ticker, _history(offset=float(index)))
    return db_path, str(result["artifact_id"])


def _future_sessions(count: int = 21) -> list[date]:
    current = pd.to_datetime(AS_OF).date()
    sessions = []
    for _ in range(count):
        current = next_trading_day(current)
        sessions.append(current)
    return sessions


def _append_future_prices(db_path: Path, tickers: list[str], sessions: list[date]) -> None:
    for offset, ticker in enumerate(tickers):
        close = 130.0 + offset * 10.0 + np.arange(len(sessions), dtype=float)
        frame = pd.DataFrame(
            {
                "open": close - 0.2,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "adj_close": close,
                "volume": np.full(len(sessions), 2_000_000.0),
            },
            index=pd.to_datetime(sessions),
        )
        storage.upsert_ohlcv(db_path, ticker, frame)


def test_shadow_dry_run_is_read_only_and_cache_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    before = db_path.read_bytes()

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("Network provider must not be called")

    monkeypatch.setattr(market_data, "get_provider", network_forbidden)
    report = dry_run_shadow_prediction(db_path, artifact_id, AS_OF)

    assert report["status"] == "ready"
    assert report["prediction_count"] == 2
    assert report["database_mutated"] is False
    assert db_path.read_bytes() == before
    assert list_shadow_prediction_runs(db_path).empty


def test_apply_persists_immutable_ranked_predictions_and_blocks_duplicate(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    result = apply_shadow_prediction(db_path, artifact_id, AS_OF)

    assert result["prediction_count"] == 2
    runs = list_shadow_prediction_runs(db_path)
    assert len(runs) == 1
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT ticker, predicted_rank, predicted_percentile FROM shadow_predictions ORDER BY predicted_rank"
        ).fetchall()
        assert [int(row["predicted_rank"]) for row in rows] == [1, 2]
        assert [float(row["predicted_percentile"]) for row in rows] == [1.0, 0.0]
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE shadow_predictions SET predicted_value=0")
    with pytest.raises(ShadowPredictionError, match="already exists"):
        apply_shadow_prediction(db_path, artifact_id, AS_OF)


def test_ranking_is_deterministic_with_ticker_tie_break() -> None:
    ranked = _rank_predictions(
        [
            {"ticker": "BBB", "predicted_value": 0.1},
            {"ticker": "AAA", "predicted_value": 0.1},
            {"ticker": "CCC", "predicted_value": -0.2},
        ]
    )
    assert [row["ticker"] for row in ranked] == ["AAA", "BBB", "CCC"]
    assert [row["predicted_rank"] for row in ranked] == [1, 2, 3]
    assert [row["predicted_percentile"] for row in ranked] == [1.0, 0.5, 0.0]


def test_future_cache_rows_do_not_change_prior_as_of_inputs(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    before = build_shadow_prediction_plan(db_path, artifact_id, AS_OF)
    for ticker in ["AAA", "BBB", "SPY", "QQQ", "IWM", "^VIX"]:
        storage.upsert_ohlcv(db_path, ticker, _history(end="2024-04-05", rows=5, offset=10_000.0))
    after = build_shadow_prediction_plan(db_path, artifact_id, AS_OF)
    assert [row["feature_input_hash"] for row in before.predictions] == [
        row["feature_input_hash"] for row in after.predictions
    ]
    assert [row["predicted_value"] for row in before.predictions] == [
        row["predicted_value"] for row in after.predictions
    ]


def test_missing_required_feature_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    from src.modeling import shadow_predictions as module

    original = module._technical_row

    def missing_feature(*args, **kwargs):
        row = original(*args, **kwargs)
        row.pop("ret_5d")
        return row

    monkeypatch.setattr(module, "_technical_row", missing_feature)
    with pytest.raises(ShadowPredictionError, match="Missing required feature"):
        build_shadow_prediction_plan(db_path, artifact_id, AS_OF)


def test_artifact_integrity_gate_blocks_corruption(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    with storage.connect(db_path) as conn:
        artifact_path = Path(conn.execute(
            "SELECT artifact_path FROM model_artifacts WHERE artifact_id=?", (artifact_id,)
        ).fetchone()["artifact_path"])
    (artifact_path / "model.joblib").write_bytes(b"corrupted")
    with pytest.raises(ShadowPredictionError, match="integrity gate failed"):
        build_shadow_prediction_plan(db_path, artifact_id, AS_OF)


def test_inference_does_not_touch_scanner_model_runs_or_dataset_50(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    with storage.connect(db_path) as conn:
        before = {
            "scan_results": conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0],
            "model_runs": conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0],
            "dataset_50": conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=50").fetchone()[0],
        }
    apply_shadow_prediction(db_path, artifact_id, AS_OF)
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0] == before["scan_results"]
        assert conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0] == before["model_runs"]
        assert conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=50").fetchone()[0] == before["dataset_50"]


def test_shadow_status_reports_integrity_and_insufficient_sample(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    apply_shadow_prediction(db_path, artifact_id, AS_OF)
    assert latest_cache_complete_session(db_path, artifact_id).isoformat() == AS_OF
    report = shadow_status_report(db_path, artifact_id)
    harness = check_shadow_status(db_path, artifact_id)
    assert report["status"] == "passed"
    assert report["sample_status"] == "insufficient_forward_sample"
    assert report["run_count"] == 1
    assert report["outcomes_by_horizon"] == {
        "1": {"matured": 0, "pending": 2},
        "5": {"matured": 0, "pending": 2},
        "20": {"matured": 0, "pending": 2},
    }
    assert harness.status == "passed"


def test_shadow_outcome_dry_run_is_read_only_and_cache_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF)["run_id"])
    before = db_path.read_bytes()

    def network_forbidden(*_args, **_kwargs):
        raise AssertionError("Network provider must not be called")

    monkeypatch.setattr(market_data, "get_provider", network_forbidden)
    report = dry_run_shadow_outcomes(db_path, run_id=run_id)

    assert report["database_mutated"] is False
    assert db_path.read_bytes() == before
    assert list_shadow_prediction_outcomes(db_path).empty


def test_shadow_outcome_timing_matches_next_session_close_contract(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF)["run_id"])
    sessions = _future_sessions(21)
    _append_future_prices(db_path, ["AAA", "BBB", "SPY"], sessions)
    evaluation_time = as_of_after_close(sessions[-1]) + timedelta(seconds=1)

    plan = build_shadow_outcome_plan(db_path, run_id=run_id, evaluated_at=evaluation_time)
    rows = {(row["ticker"], row["horizon_sessions"]): row for row in plan.planned_outcomes}

    assert len(rows) == 6
    assert rows[("AAA", 1)]["entry_date"] == sessions[0].isoformat()
    assert rows[("AAA", 1)]["exit_date"] == sessions[1].isoformat()
    assert rows[("AAA", 5)]["exit_date"] == sessions[5].isoformat()
    assert rows[("AAA", 20)]["exit_date"] == sessions[20].isoformat()
    assert rows[("AAA", 1)]["realized_return"] == pytest.approx(131.0 / 130.0 - 1.0)
    assert rows[("AAA", 1)]["benchmark_return"] == pytest.approx(151.0 / 150.0 - 1.0)
    assert rows[("AAA", 1)]["label_available_at"] == as_of_after_close(sessions[1]).isoformat()


def test_shadow_outcomes_do_not_mature_prematurely(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF)["run_id"])
    sessions = _future_sessions(21)
    _append_future_prices(db_path, ["AAA", "BBB", "SPY"], sessions)

    plan = build_shadow_outcome_plan(db_path, run_id=run_id, evaluated_at=as_of_after_close(sessions[0]))

    assert plan.planned_outcomes == []
    assert len(plan.pending_outcomes) == 6
    assert {row["reason"] for row in plan.pending_outcomes} == {"not_yet_mature"}


def test_shadow_outcome_partial_horizon_maturity(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF)["run_id"])
    sessions = _future_sessions(6)
    _append_future_prices(db_path, ["AAA", "BBB", "SPY"], sessions)
    plan = build_shadow_outcome_plan(
        db_path,
        run_id=run_id,
        evaluated_at=as_of_after_close(sessions[-1]) + timedelta(seconds=1),
    )

    assert {row["horizon_sessions"] for row in plan.planned_outcomes} == {1, 5}
    assert len(plan.planned_outcomes) == 4
    assert len([row for row in plan.pending_outcomes if row["horizon_sessions"] == 20]) == 2


def test_shadow_outcome_apply_is_idempotent_and_rows_are_immutable(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF)["run_id"])
    sessions = _future_sessions(21)
    _append_future_prices(db_path, ["AAA", "BBB", "SPY"], sessions)
    evaluation_time = as_of_after_close(sessions[-1]) + timedelta(seconds=1)

    first = apply_shadow_outcomes(db_path, run_id=run_id, evaluated_at=evaluation_time)
    second = apply_shadow_outcomes(db_path, run_id=run_id, evaluated_at=evaluation_time)

    assert first["outcomes_created"] == 6
    assert first["outcomes_created_by_horizon"] == {"1": 2, "5": 2, "20": 2}
    assert second["status"] == "no_changes"
    assert second["database_mutated"] is False
    assert len(list_shadow_prediction_outcomes(db_path, run_id)) == 6
    with storage.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE shadow_prediction_outcomes SET realized_return=0")


def test_shadow_outcome_missing_cache_is_reported_without_fabrication(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF)["run_id"])
    sessions = _future_sessions(6)
    _append_future_prices(db_path, ["SPY"], sessions)
    plan = build_shadow_outcome_plan(
        db_path,
        run_id=run_id,
        evaluated_at=as_of_after_close(sessions[-1]) + timedelta(seconds=1),
    )

    assert plan.planned_outcomes == []
    assert len(plan.missing_price_cases) == 4
    assert all("missing_ticker" in row["reason"] for row in plan.missing_price_cases)


def test_spy_prediction_is_retained_but_marked_as_benchmark_excluded(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF)["run_id"])
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO shadow_predictions (
                prediction_run_id, ticker, prediction_date, predicted_value,
                predicted_rank, predicted_percentile, feature_input_hash,
                data_quality_flags_json, created_at
            ) VALUES (?, 'SPY', ?, 0, 3, 0, 'test', '{}', ?)
            """,
            (run_id, AS_OF, datetime.now(UTC).isoformat()),
        )
    sessions = _future_sessions(6)
    _append_future_prices(db_path, ["SPY"], sessions)
    plan = build_shadow_outcome_plan(
        db_path,
        run_id=run_id,
        evaluated_at=as_of_after_close(sessions[-1]) + timedelta(seconds=1),
    )
    spy_rows = [row for row in plan.planned_outcomes if row["ticker"] == "SPY"]

    assert plan.benchmark_exclusion_count == 1
    assert {row["horizon_sessions"] for row in spy_rows} == {1, 5}
    assert all(row["excess_return"] == pytest.approx(0.0) for row in spy_rows)
    assert all(row["data_quality_flags"]["benchmark_excluded_from_evaluation"] for row in spy_rows)


def test_shadow_outcomes_do_not_touch_scanner_models_or_datasets(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    run_id = int(apply_shadow_prediction(db_path, artifact_id, AS_OF)["run_id"])
    sessions = _future_sessions(21)
    _append_future_prices(db_path, ["AAA", "BBB", "SPY"], sessions)
    with storage.connect(db_path) as conn:
        before = {
            "scan_results": conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0],
            "model_runs": conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0],
            "dataset_49": conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=49").fetchone()[0],
            "dataset_50": conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=50").fetchone()[0],
            "predictions": conn.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0],
        }
    apply_shadow_outcomes(
        db_path,
        run_id=run_id,
        evaluated_at=as_of_after_close(sessions[-1]) + timedelta(seconds=1),
    )
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0] == before["scan_results"]
        assert conn.execute("SELECT COUNT(*) FROM model_runs").fetchone()[0] == before["model_runs"]
        assert conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=49").fetchone()[0] == before["dataset_49"]
        assert conn.execute("SELECT COUNT(*) FROM dataset_builds WHERE dataset_id=50").fetchone()[0] == before["dataset_50"]
        assert conn.execute("SELECT COUNT(*) FROM shadow_predictions").fetchone()[0] == before["predictions"]


@pytest.mark.parametrize(
    ("date_count", "expected"),
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
def test_shadow_sample_status_thresholds(date_count: int, expected: str) -> None:
    assert _shadow_sample_status(date_count) == expected
