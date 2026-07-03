from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from src.annotations.models import ResearchEventAnnotation
from src.annotations.repository import insert_annotation
from src.data import storage
from src.datasets.models import DatasetBuild, FeatureSnapshot, OutcomeLabel
from src.datasets.repository import insert_dataset_build, insert_feature_snapshots, insert_outcome_labels
from src.modeling.repository import insert_final_metric, insert_model_run
from src.quality.harness import (
    check_annotation_coverage,
    check_dataset_manifest,
    compare_model_run_to_baseline,
    compare_scanner_snapshots,
    normalize_scanner_snapshot,
    report_markdown,
    build_final_report,
)


def _seed_dataset(db_path, *, leak_label: bool = False) -> int:
    feature_columns = ["ret_5d", "market_regime"]
    if leak_label:
        feature_columns.append("label_5_session_excess_return")
    feature_manifest = {
        "ret_5d": {"role": "model_feature"},
        "market_regime": {"role": "model_feature"},
        "raw_annotation_count": {"role": "audit"},
        "label_5_session_excess_return": {"role": "label"},
        "snapshot_id": {"role": "identifier"},
        "dataset_id": {"role": "identifier"},
        "ticker": {"role": "identifier"},
        "trading_date": {"role": "metadata"},
        "as_of_timestamp": {"role": "metadata"},
    }
    build = DatasetBuild(
        version="quality_harness_test_v1",
        build_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        requested_start_date=date(2024, 1, 2),
        requested_end_date=date(2024, 3, 29),
        ticker_universe=["AAA", "BBB"],
        feature_columns=feature_columns,
        label_definitions={"5_session": {"target": "excess_return"}},
        row_count=90,
        data_hash="quality_hash",
        audit_columns=["raw_annotation_count"],
        label_columns=["label_5_session_excess_return"],
        identifier_columns=["snapshot_id", "dataset_id", "ticker"],
        metadata_columns=["trading_date", "as_of_timestamp"],
        feature_manifest=feature_manifest,
    )
    dataset_id = insert_dataset_build(db_path, build)
    dates = [pd.Timestamp(value).date() for value in pd.bdate_range("2024-01-02", periods=45)]
    snapshots: list[FeatureSnapshot] = []
    for ticker in ["AAA", "BBB"]:
        for idx, trading_date in enumerate(dates):
            snapshots.append(
                FeatureSnapshot(
                    ticker=ticker,
                    trading_date=trading_date,
                    as_of_timestamp=datetime.combine(trading_date, datetime.max.time(), tzinfo=UTC),
                    feature_version="quality_harness_test_v1",
                    market_regime={},
                    technical={},
                    relative_strength={},
                    volume_liquidity={},
                    catalyst={},
                    llm_supported={},
                    data_quality={},
                    features={"ret_5d": float(idx) / 100.0, "market_regime": "Neutral"},
                )
            )
    ids = insert_feature_snapshots(db_path, dataset_id, snapshots)
    labels: list[OutcomeLabel] = []
    for snapshot in snapshots:
        labels.append(
            OutcomeLabel(
                snapshot_id=ids[(snapshot.ticker, snapshot.trading_date)],
                ticker=snapshot.ticker,
                entry_date=snapshot.trading_date + timedelta(days=1),
                horizon="5_session",
                entry_price=100,
                exit_date=snapshot.trading_date + timedelta(days=8),
                exit_price=101,
                forward_return=0.01,
                spy_forward_return=0.0,
                excess_return=0.01,
                label_available_at=datetime.combine(snapshot.trading_date + timedelta(days=8), datetime.min.time(), tzinfo=UTC),
            )
        )
    insert_outcome_labels(db_path, labels)
    return dataset_id


def test_scanner_snapshot_compare_preserves_zero_scores() -> None:
    snapshot = normalize_scanner_snapshot([{"ticker": "AAA", "score": 0, "label": "Avoid"}])
    assert snapshot["AAA"]["score"] == 0

    same = compare_scanner_snapshots(snapshot, snapshot)
    changed = compare_scanner_snapshots(snapshot, [{"ticker": "AAA", "score": 1, "label": "Avoid"}])

    assert same.status == "passed"
    assert changed.status == "warn"
    assert changed.summary["diff_count"] == 1


def test_dataset_manifest_check_passes_and_detects_label_leak(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_dataset(db_path)
    passed = check_dataset_manifest(db_path, dataset_id, expected_hash="quality_hash", expected_row_count=90)
    assert passed.status == "passed"

    leaky_dataset_id = _seed_dataset(db_path, leak_label=True)
    failed = check_dataset_manifest(db_path, leaky_dataset_id)
    assert failed.status == "failed"
    assert "forbidden" in failed.details["violations"][0]


def test_model_compare_reports_metric_deltas(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    storage.init_db(db_path)
    baseline_id = insert_model_run(
        db_path,
        dataset_id=1,
        dataset_hash="hash",
        target_column="label_5_session_excess_return",
        target_horizon="5_session",
        task="regression",
        feature_set_name="technical_core",
        model_name="ridge_regression",
        config={},
        split_config={},
        feature_columns=["ret_5d"],
    )
    candidate_id = insert_model_run(
        db_path,
        dataset_id=1,
        dataset_hash="hash",
        target_column="label_5_session_excess_return",
        target_horizon="5_session",
        task="regression",
        feature_set_name="technical_core_plus_annotations",
        model_name="ridge_regression",
        config={},
        split_config={},
        feature_columns=["ret_5d", "annotation_coverage_available"],
    )
    insert_final_metric(db_path, baseline_id, "test", {"rmse": 0.10, "oos_r2_vs_train_mean": 0.0, "spearman_ic": 0.01, "directional_accuracy": 0.50})
    insert_final_metric(db_path, candidate_id, "test", {"rmse": 0.09, "oos_r2_vs_train_mean": 0.1, "spearman_ic": 0.02, "directional_accuracy": 0.51})

    result = compare_model_run_to_baseline(db_path, candidate_id, baseline_run_id=baseline_id)

    assert result.status == "passed"
    assert result.summary["improved_metrics"] == 4


def test_annotation_coverage_check_detects_zero_and_active_coverage(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_dataset(db_path)

    empty = check_annotation_coverage(db_path, dataset_id)
    assert empty.status == "passed"
    assert empty.summary["annotation_rows"] == 0
    assert empty.summary["zero_coverage_feature_count"] > 0

    insert_annotation(
        db_path,
        ResearchEventAnnotation(
            ticker="AAA",
            event_date=date(2024, 1, 3),
            available_at=datetime(2024, 1, 4, 14, tzinfo=UTC),
            event_type="news",
            sentiment_label="positive",
            strength=7,
            confidence=0.8,
            source="test",
            title="Synthetic event",
        ),
    )
    covered = check_annotation_coverage(db_path, dataset_id)
    assert covered.status == "passed"
    assert covered.summary["annotation_rows"] == 1
    assert covered.summary["rows_with_annotation_coverage"] > 0
    assert covered.summary["future_availability_violation_count"] == 0


def test_final_report_helper_outputs_expected_markdown() -> None:
    report = build_final_report(
        runtime_status="ok",
        tests="passed",
        files_changed=["src/quality/harness.py"],
        artifacts=["artifact.json"],
        scanner_invariance="passed",
        dataset_model_comparison="n/a",
        decision="accepted",
        next_recommendation="annotation pilot",
    )
    markdown = report_markdown(report)
    assert "Runtime status: ok" in markdown
    assert "src/quality/harness.py" in markdown

