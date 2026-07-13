from __future__ import annotations

import json
import sqlite3
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
import pytest

from src.data import storage
from src.datasets.models import DatasetBuild, FeatureSnapshot, OutcomeLabel
from src.datasets.repository import (
    insert_dataset_build,
    insert_feature_snapshots,
    insert_outcome_labels,
    stream_saved_dataset_export_and_hash,
    update_dataset_build_summary,
)
from src.modeling.artifacts import (
    ArtifactContractError,
    ArtifactIntegrityError,
    ExecutableShadowArtifact,
    apply_artifact_build,
    check_registered_artifact,
    dry_run_artifact_build,
    list_model_artifacts,
    validate_artifact_directory,
)
from src.modeling.feature_sets import TECHNICAL_CORE_CANDIDATES


def _seed_dataset_49(db_path: Path) -> tuple[str, int]:
    manifest = {column: {"role": "model_feature"} for column in TECHNICAL_CORE_CANDIDATES}
    manifest.update(
        {
            "snapshot_id": {"role": "identifier"},
            "dataset_id": {"role": "identifier"},
            "ticker": {"role": "identifier"},
            "trading_date": {"role": "metadata"},
            "as_of_timestamp": {"role": "metadata"},
            "label_5_session_excess_return": {"role": "label"},
        }
    )
    build = DatasetBuild(
        version="artifact_test_v1",
        build_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        requested_start_date=date(2024, 1, 2),
        requested_end_date=date(2024, 3, 31),
        ticker_universe=["AAA", "BBB"],
        feature_columns=list(TECHNICAL_CORE_CANDIDATES),
        label_definitions={"5_session": {"target": "excess_return"}},
        row_count=0,
        data_hash="pending",
        audit_columns=[],
        label_columns=["label_5_session_excess_return"],
        identifier_columns=["snapshot_id", "dataset_id", "ticker"],
        metadata_columns=["trading_date", "as_of_timestamp"],
        feature_manifest=manifest,
    )
    dataset_id = insert_dataset_build(db_path, build)
    assert dataset_id == 1
    with storage.connect(db_path) as conn:
        conn.execute("UPDATE dataset_builds SET dataset_id = 49 WHERE dataset_id = 1")

    dates = [value.date() for value in pd.bdate_range("2024-01-02", periods=35)]
    snapshots: list[FeatureSnapshot] = []
    for ticker_index, ticker in enumerate(["AAA", "BBB"]):
        for index, trading_date in enumerate(dates):
            features = {
                column: float((index + ticker_index) % 9) / 10.0
                for column in TECHNICAL_CORE_CANDIDATES
                if column != "market_regime"
            }
            features["market_regime"] = "Risk-On" if index % 2 == 0 else "Neutral"
            if index == 4 and ticker == "BBB":
                features["volume_ratio_20d"] = None
            snapshots.append(
                FeatureSnapshot(
                    ticker=ticker,
                    trading_date=trading_date,
                    as_of_timestamp=datetime.combine(trading_date, datetime.max.time(), tzinfo=UTC),
                    feature_version="artifact_test_v1",
                    market_regime={}, technical={}, relative_strength={}, volume_liquidity={}, catalyst={},
                    llm_supported={}, data_quality={}, features=features,
                )
            )
    ids = insert_feature_snapshots(db_path, 49, snapshots)
    labels = []
    for index, snapshot in enumerate(snapshots):
        value = ((index % 13) - 6) / 100.0
        labels.append(
            OutcomeLabel(
                snapshot_id=ids[(snapshot.ticker, snapshot.trading_date)],
                ticker=snapshot.ticker,
                entry_date=snapshot.trading_date + timedelta(days=1),
                horizon="5_session",
                entry_price=100.0,
                exit_date=snapshot.trading_date + timedelta(days=8),
                exit_price=100.0 * (1 + value),
                forward_return=value,
                spy_forward_return=0.0,
                excess_return=value,
                label_available_at=datetime.combine(snapshot.trading_date + timedelta(days=8), datetime.min.time(), tzinfo=UTC),
            )
        )
    insert_outcome_labels(db_path, labels)
    streamed = stream_saved_dataset_export_and_hash(db_path, 49)
    update_dataset_build_summary(db_path, 49, streamed["row_count"], streamed["data_hash"])
    return str(streamed["data_hash"]), int(streamed["row_count"])


def _apply_test_artifact(tmp_path: Path) -> tuple[Path, dict]:
    db_path = tmp_path / "alpha_lab.db"
    data_hash, row_count = _seed_dataset_49(db_path)
    result = apply_artifact_build(
        db_path,
        tmp_path / "artifacts",
        project_root=tmp_path,
        dataset_id=49,
        expected_hash=data_hash,
        expected_row_count=row_count,
    )
    return db_path, result


def test_dry_run_is_mutation_free_and_wrong_hash_blocks(tmp_path: Path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    data_hash, row_count = _seed_dataset_49(db_path)
    with storage.connect(db_path) as conn:
        conn.execute("DROP TRIGGER model_artifacts_immutable_update")
        conn.execute("DROP TRIGGER model_artifacts_immutable_delete")
        conn.execute("DROP TABLE model_artifacts")
    before = db_path.read_bytes()

    report = dry_run_artifact_build(
        db_path, tmp_path / "artifacts", dataset_id=49, expected_hash=data_hash, expected_row_count=row_count
    )

    assert report["database_mutated"] is False
    assert report["artifact_written"] is False
    assert not (tmp_path / "artifacts").exists()
    assert db_path.read_bytes() == before
    with pytest.raises(ArtifactContractError, match="mismatch"):
        dry_run_artifact_build(
            db_path, tmp_path / "artifacts", dataset_id=49, expected_hash="wrong", expected_row_count=row_count
        )


def test_dataset_50_is_blocked_before_loading(tmp_path: Path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    data_hash, row_count = _seed_dataset_49(db_path)
    with pytest.raises(ArtifactContractError, match="Dataset 49 only"):
        dry_run_artifact_build(
            db_path, tmp_path / "artifacts", dataset_id=50, expected_hash=data_hash, expected_row_count=row_count
        )


def test_artifact_persists_transparent_contract_and_replays(tmp_path: Path) -> None:
    db_path, result = _apply_test_artifact(tmp_path)
    artifact_dir = Path(result["artifact_path"])
    manifest = json.loads((artifact_dir / "model_manifest.json").read_text())
    state = json.loads((artifact_dir / "preprocessing_state.json").read_text())
    bundle = joblib.load(artifact_dir / "model.joblib")

    assert result["status"] == "created"
    assert manifest["derived_from_run_id"] is None
    assert manifest["design_reference_run_ids"] == [145, 283]
    assert manifest["feature_columns"] == TECHNICAL_CORE_CANDIDATES
    assert manifest["target_transformation"]["lower_threshold"] <= manifest["target_transformation"]["upper_threshold"]
    assert state["scaling"] == "none"
    assert state["medians"]
    assert "market_regime" in state["categorical_columns"]
    assert isinstance(bundle, ExecutableShadowArtifact)
    assert (artifact_dir / "coefficients.csv").read_text().count("__INTERCEPT__") == 1
    assert check_registered_artifact(db_path, result["artifact_id"])["status"] == "passed"
    assert list_model_artifacts(db_path).shape[0] == 1


def test_feature_contract_rejects_wrong_order_missing_and_extra(tmp_path: Path) -> None:
    _db_path, result = _apply_test_artifact(tmp_path)
    artifact_dir = Path(result["artifact_path"])
    bundle = joblib.load(artifact_dir / "model.joblib")
    fixture = pd.read_csv(artifact_dir / "replay_fixture.csv")

    with pytest.raises(ArtifactContractError, match="order"):
        bundle.predict(fixture[list(reversed(bundle.feature_columns))])
    with pytest.raises(ArtifactContractError, match="Missing"):
        bundle.predict(fixture[bundle.feature_columns[:-1]])
    with pytest.raises(ArtifactContractError, match="Unexpected"):
        bundle.predict(fixture.assign(extra_feature=1.0))


def test_corrupted_artifact_and_manifest_fail_integrity(tmp_path: Path) -> None:
    _db_path, result = _apply_test_artifact(tmp_path)
    artifact_dir = Path(result["artifact_path"])
    (artifact_dir / "model.joblib").write_bytes(b"corrupted")
    assert validate_artifact_directory(artifact_dir)["status"] == "failed"

    _db_path_2, result_2 = _apply_test_artifact(tmp_path / "second")
    artifact_dir_2 = Path(result_2["artifact_path"])
    (artifact_dir_2 / "model_manifest.json").write_text("{}")
    assert validate_artifact_directory(artifact_dir_2)["status"] == "failed"


def test_registry_and_artifact_are_immutable_and_duplicate_is_blocked(tmp_path: Path) -> None:
    db_path, result = _apply_test_artifact(tmp_path)
    with storage.connect(db_path) as conn:
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE model_artifacts SET status='changed' WHERE artifact_id=?", (result["artifact_id"],))
    with pytest.raises(ArtifactContractError, match="already registered"):
        apply_artifact_build(
            db_path,
            tmp_path / "artifacts",
            project_root=tmp_path,
            dataset_id=49,
            expected_hash=result["contract"]["dataset_hash"],
            expected_row_count=result["contract"]["dataset_row_count"],
        )


def test_artifact_build_does_not_create_model_predictions_or_change_scanner_rows(tmp_path: Path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    data_hash, row_count = _seed_dataset_49(db_path)
    with storage.connect(db_path) as conn:
        before_predictions = conn.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0]
        before_scans = conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0]
    apply_artifact_build(
        db_path,
        tmp_path / "artifacts",
        project_root=tmp_path,
        dataset_id=49,
        expected_hash=data_hash,
        expected_row_count=row_count,
    )
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM model_predictions").fetchone()[0] == before_predictions
        assert conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0] == before_scans


def test_failed_build_leaves_no_registered_artifact(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    db_path = tmp_path / "alpha_lab.db"
    data_hash, row_count = _seed_dataset_49(db_path)

    def fail_validation(*_args, **_kwargs):
        return {"status": "failed", "errors": ["synthetic failure"]}

    monkeypatch.setattr("src.modeling.artifacts.validate_artifact_directory", fail_validation)
    with pytest.raises(ArtifactIntegrityError, match="validation failed"):
        apply_artifact_build(
            db_path,
            tmp_path / "artifacts",
            project_root=tmp_path,
            dataset_id=49,
            expected_hash=data_hash,
            expected_row_count=row_count,
        )
    assert list_model_artifacts(db_path).empty
    assert not list((tmp_path / "artifacts").glob("shadow_ridge_technical_v1_*"))
