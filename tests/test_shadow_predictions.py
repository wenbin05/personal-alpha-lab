from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import market_data, storage
from src.modeling.shadow_predictions import (
    ShadowPredictionError,
    _rank_predictions,
    apply_shadow_prediction,
    build_shadow_prediction_plan,
    dry_run_shadow_prediction,
    latest_cache_complete_session,
    list_shadow_prediction_runs,
    shadow_status_report,
)
from src.quality.harness import check_shadow_status
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
    assert harness.status == "passed"
