from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import market_data, storage
from src.modeling import daily_shadow_cycle as cycle_module
from src.modeling.daily_shadow_cycle import DailyShadowCycleError, cycle_lock, run_daily_shadow_cycle
from src.modeling.shadow_predictions import ShadowPredictionError, apply_shadow_prediction, list_shadow_prediction_runs
from tests.test_shadow_predictions import AS_OF, _append_future_prices, _future_sessions, _seed_shadow_environment


AS_OF_REFERENCE = datetime(2024, 3, 29, 0, 0, tzinfo=UTC)


def _network_forbidden(*_args, **_kwargs):
    raise AssertionError("Network access was not authorized")


def test_default_dry_run_has_zero_database_mutation_and_zero_network_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    before = db_path.read_bytes()
    monkeypatch.setattr(market_data, "get_provider", _network_forbidden)

    report = run_daily_shadow_cycle(
        db_path, artifact_id=artifact_id, reference_time=AS_OF_REFERENCE
    )

    assert report["status"] == "dry_run_complete"
    assert report["database_mutated"] is False
    assert report["network_authorized"] is False
    assert db_path.read_bytes() == before
    assert list_shadow_prediction_runs(db_path).empty


def test_apply_without_refresh_never_calls_provider_and_duplicate_safely_noops(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    existing = apply_shadow_prediction(db_path, artifact_id, AS_OF)
    before = db_path.read_bytes()
    monkeypatch.setattr(market_data, "get_provider", _network_forbidden)

    report = run_daily_shadow_cycle(
        db_path,
        artifact_id=artifact_id,
        apply=True,
        refresh_market_data=False,
        reference_time=AS_OF_REFERENCE,
    )

    assert report["status"] == "no_op"
    assert report["prediction_run"] == {
        "status": "skipped",
        "reason": "duplicate_date_artifact_run",
        "run_id": existing["run_id"],
    }
    assert report["database_backup"] is None
    assert db_path.read_bytes() == before
    assert len(list_shadow_prediction_runs(db_path)) == 1


def test_future_incomplete_bar_is_ignored_by_completed_session_audit(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    apply_shadow_prediction(db_path, artifact_id, AS_OF)
    future = pd.Timestamp("2024-04-01")
    frame = pd.DataFrame(
        {
            "open": [1.0], "high": [1.0], "low": [1.0], "close": [1.0],
            "adj_close": [1.0], "volume": [1.0],
        },
        index=[future],
    )
    storage.upsert_ohlcv(db_path, "AAA", frame)

    report = run_daily_shadow_cycle(
        db_path, artifact_id=artifact_id, apply=True, reference_time=AS_OF_REFERENCE
    )
    future_rows = report["cache_status_before"]["future_or_incomplete_bars"]

    assert report["resolved_session"] == AS_OF
    assert {row["ticker"]: row["count"] for row in future_rows}["AAA"] == 1
    assert report["prediction_run"]["reason"] == "duplicate_date_artifact_run"


def test_authorized_refresh_is_bounded_and_creates_one_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    apply_shadow_prediction(db_path, artifact_id, AS_OF)
    completed = pd.Timestamp("2024-04-01")
    incomplete = pd.Timestamp("2024-04-02")
    backup_calls: list[Path] = []

    class Provider:
        def download_history(self, ticker, period="2y", start=None, end=None):
            values = np.array([120.0, 121.0])
            return pd.DataFrame(
                {
                    "date": [completed.date().isoformat(), incomplete.date().isoformat()],
                    "open": values, "high": values, "low": values, "close": values,
                    "adj_close": values, "volume": np.array([2_000_000.0, 2_000_000.0]),
                }
            )

    monkeypatch.setattr(market_data, "get_provider", lambda _name: Provider())
    monkeypatch.setattr(
        cycle_module,
        "_backup_database",
        lambda path: backup_calls.append(Path(path)) or Path(str(path) + ".backup"),
    )
    reference = datetime(2024, 4, 2, 0, 0, tzinfo=UTC)

    report = run_daily_shadow_cycle(
        db_path,
        artifact_id=artifact_id,
        apply=True,
        refresh_market_data=True,
        reference_time=reference,
    )

    assert report["resolved_session"] == "2024-04-01"
    assert report["prediction_run"]["status"] == "created"
    assert len(backup_calls) == 1
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM ohlcv_cache WHERE date='2024-04-02'").fetchone()[0] == 0
    assert len(list_shadow_prediction_runs(db_path)) == 2


def test_failed_refresh_does_not_create_prediction_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    apply_shadow_prediction(db_path, artifact_id, AS_OF)

    class EmptyProvider:
        def download_history(self, ticker, period="2y", start=None, end=None):
            return pd.DataFrame()

    monkeypatch.setattr(market_data, "get_provider", lambda _name: EmptyProvider())
    monkeypatch.setattr(cycle_module, "_backup_database", lambda path: Path(str(path) + ".backup"))
    reference = datetime(2024, 4, 2, 0, 0, tzinfo=UTC)

    report = run_daily_shadow_cycle(
        db_path,
        artifact_id=artifact_id,
        apply=True,
        refresh_market_data=True,
        reference_time=reference,
    )

    assert report["status"] == "failed"
    assert report["refresh_failures"]
    assert len(list_shadow_prediction_runs(db_path)) == 1


def test_outcomes_are_applied_before_new_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    apply_shadow_prediction(db_path, artifact_id, AS_OF)
    sessions = _future_sessions(6)
    _append_future_prices(db_path, ["AAA", "BBB", "SPY", "QQQ", "IWM", "^VIX"], sessions)
    reference = datetime(2024, 4, 9, 0, 0, tzinfo=UTC)
    calls: list[str] = []
    original_outcomes = cycle_module.apply_shadow_outcomes
    original_prediction = cycle_module.apply_shadow_prediction

    def outcomes(*args, **kwargs):
        calls.append("outcomes")
        return original_outcomes(*args, **kwargs)

    def prediction(*args, **kwargs):
        calls.append("prediction")
        return original_prediction(*args, **kwargs)

    monkeypatch.setattr(cycle_module, "apply_shadow_outcomes", outcomes)
    monkeypatch.setattr(cycle_module, "apply_shadow_prediction", prediction)
    monkeypatch.setattr(cycle_module, "_backup_database", lambda path: Path(str(path) + ".backup"))

    report = run_daily_shadow_cycle(
        db_path, artifact_id=artifact_id, apply=True, reference_time=reference
    )

    assert calls == ["outcomes", "prediction"]
    assert report["outcomes_added_by_horizon"] == {"1": 2, "5": 2, "20": 0}
    assert report["prediction_run"]["status"] == "created"


def test_concurrent_cycle_lock_is_blocked(tmp_path: Path) -> None:
    lock_path = tmp_path / "cycle.lock"
    with cycle_lock(lock_path):
        with pytest.raises(DailyShadowCycleError, match="lock already exists"):
            with cycle_lock(lock_path):
                pass
    assert not lock_path.exists()


def test_prediction_failure_leaves_no_partial_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    apply_shadow_prediction(db_path, artifact_id, AS_OF)
    sessions = _future_sessions(6)
    _append_future_prices(db_path, ["AAA", "BBB", "SPY", "QQQ", "IWM", "^VIX"], sessions)
    reference = datetime(2024, 4, 9, 0, 0, tzinfo=UTC)
    monkeypatch.setattr(cycle_module, "_backup_database", lambda path: Path(str(path) + ".backup"))
    monkeypatch.setattr(
        cycle_module,
        "apply_shadow_prediction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ShadowPredictionError("forced failure")),
    )

    with pytest.raises(ShadowPredictionError, match="forced failure"):
        run_daily_shadow_cycle(db_path, artifact_id=artifact_id, apply=True, reference_time=reference)

    runs = list_shadow_prediction_runs(db_path)
    assert len(runs) == 1
    assert int(runs.iloc[0]["prediction_count"]) == 2


def test_cycle_does_not_mutate_scanner_or_dataset_builds(tmp_path: Path) -> None:
    db_path, artifact_id = _seed_shadow_environment(tmp_path)
    with storage.connect(db_path) as conn:
        before = {
            "scan_results": conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0],
            "dataset_builds": conn.execute("SELECT COUNT(*) FROM dataset_builds").fetchone()[0],
            "dataset_49": conn.execute(
                "SELECT data_hash FROM dataset_builds WHERE dataset_id=49"
            ).fetchone()[0],
        }
    run_daily_shadow_cycle(db_path, artifact_id=artifact_id, reference_time=AS_OF_REFERENCE)
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0] == before["scan_results"]
        assert conn.execute("SELECT COUNT(*) FROM dataset_builds").fetchone()[0] == before["dataset_builds"]
        assert conn.execute(
            "SELECT data_hash FROM dataset_builds WHERE dataset_id=49"
        ).fetchone()[0] == before["dataset_49"]
