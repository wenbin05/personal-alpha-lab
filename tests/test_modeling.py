from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd
import pytest

from src.data import storage
from src.datasets.models import DatasetBuild, FeatureSnapshot, OutcomeLabel
from src.datasets.repository import insert_dataset_build, insert_feature_snapshots, insert_outcome_labels
from src.datasets.training_loader import load_training_dataset
from src.modeling.feature_sets import select_feature_columns
from src.modeling.feature_quality import (
    build_feature_quality_audit,
    build_pruned_feature_sets,
    correlation_audit,
    feature_missingness_detail,
    pruned_feature_sets_from_artifact,
    sparse_event_features,
    univariate_ic_audit,
)
from src.modeling.event_feature_redesign import (
    build_event_model_frame,
    derive_event_features,
    redesigned_event_feature_sets,
)
from src.modeling.diagnostics import (
    build_model_diagnostics,
    coverage_by_ticker,
    feature_diagnostics,
    load_diagnostics_artifact,
    write_diagnostics_artifact,
)
from src.modeling.evaluation_regime import (
    DATASET_49_INSPECTED_RUN_IDS,
    DatasetEvaluationRegime,
    build_holdout_status_artifact,
    default_annotation_expansion_plan,
    get_dataset_evaluation_regime,
    label_model_runs,
    upsert_dataset_evaluation_regime,
    validate_evaluation_regime,
)
from src.modeling.holdout_maturity import HoldoutMaturityThresholds, assess_holdout_maturity, build_holdout_extension_plan
from src.modeling.metrics import regression_metrics
from src.modeling.preprocessing import fit_transform_matrices
from src.modeling.repository import list_model_final_metrics, list_model_fold_metrics, list_model_predictions
from src.modeling.runner import run_single_baseline_model
from src.modeling.splits import make_walk_forward_splits
from src.modeling.targets import (
    binary_top_bottom_target,
    cross_sectional_rank_target,
    get_target_definition,
    transform_target_for_split,
    volatility_normalized_target,
)
from src.scoring.score_engine import score_ticker_from_features


def _seed_model_dataset(db_path, *, leak_feature: bool = False) -> int:
    feature_columns = [
        "ret_5d",
        "ret_20d",
        "volume_ratio_20d",
        "market_regime",
        "sec_current_event_event_days_30s",
        "earnings_event_present_5s",
        "catalyst_score",
    ]
    label_columns = ["label_5_session_excess_return"]
    if leak_feature:
        feature_columns.append("label_5_session_excess_return")
    build = DatasetBuild(
        version="test_modeling_v1",
        build_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        requested_start_date=date(2024, 1, 2),
        requested_end_date=date(2024, 3, 29),
        ticker_universe=["AAA", "BBB"],
        feature_columns=feature_columns,
        label_definitions={"5_session": {"target": "excess_return"}},
        row_count=0,
        data_hash="test_hash",
        audit_columns=["raw_filing_count"],
        label_columns=label_columns,
        identifier_columns=["snapshot_id", "dataset_id", "ticker", "trading_date"],
        metadata_columns=["as_of_timestamp"],
        feature_manifest={column: "model_feature" for column in feature_columns},
    )
    dataset_id = insert_dataset_build(db_path, build)
    dates = [pd.Timestamp(value).date() for value in pd.bdate_range("2024-01-02", periods=45)]
    snapshots: list[FeatureSnapshot] = []
    for ticker_idx, ticker in enumerate(["AAA", "BBB"]):
        for idx, trading_date in enumerate(dates):
            features = {
                "ret_5d": (idx % 7 - 3) / 100,
                "ret_20d": (idx % 11 - 5) / 100,
                "volume_ratio_20d": 0.8 + (idx % 5) * 0.1,
                "market_regime": "Risk-On" if idx % 2 == 0 else "Neutral",
                "sec_current_event_event_days_30s": idx % 3,
                "earnings_event_present_5s": bool(idx % 13 == 0),
                "catalyst_score": 1.0 if ticker_idx == 0 else 0.0,
            }
            snapshots.append(
                FeatureSnapshot(
                    ticker=ticker,
                    trading_date=trading_date,
                    as_of_timestamp=datetime.combine(trading_date, datetime.min.time(), tzinfo=UTC),
                    feature_version="test_modeling_v1",
                    market_regime={},
                    technical={},
                    relative_strength={},
                    volume_liquidity={},
                    catalyst={},
                    llm_supported={},
                    data_quality={},
                    features=features,
                )
            )
    ids = insert_feature_snapshots(db_path, dataset_id, snapshots)
    labels: list[OutcomeLabel] = []
    for snapshot in snapshots:
        snapshot_id = ids[(snapshot.ticker, snapshot.trading_date)]
        signal = 0.01 if snapshot.features["ret_5d"] > 0 else -0.005
        ticker_offset = 0.002 if snapshot.ticker == "AAA" else -0.001
        y = signal + ticker_offset
        labels.append(
            OutcomeLabel(
                snapshot_id=snapshot_id,
                ticker=snapshot.ticker,
                entry_date=snapshot.trading_date + timedelta(days=1),
                horizon="5_session",
                entry_price=100,
                exit_date=snapshot.trading_date + timedelta(days=8),
                exit_price=100 * (1 + y),
                forward_return=y,
                spy_forward_return=0.0,
                excess_return=y,
                label_available_at=datetime.combine(snapshot.trading_date + timedelta(days=8), datetime.min.time(), tzinfo=UTC),
            )
        )
    insert_outcome_labels(db_path, labels)
    return dataset_id


def test_feature_set_contract_separates_technical_sec_and_earnings() -> None:
    columns = [
        "ret_20d",
        "volume_ratio_20d",
        "sec_feature_eligible_event_days_30s",
        "sec_current_event_event_days_30s",
        "earnings_event_present_5s",
        "latest_eps_surprise_percent",
        "catalyst_score",
        "published_llm_supported_count",
    ]

    technical = select_feature_columns(columns, "technical_only")
    sec = select_feature_columns(columns, "technical_plus_sec")
    earnings = select_feature_columns(columns, "technical_plus_earnings")

    assert technical == ["ret_20d", "volume_ratio_20d"]
    assert "sec_current_event_event_days_30s" in sec
    assert "sec_feature_eligible_event_days_30s" in sec
    assert "sec_feature_eligible_event_days_30s" not in select_feature_columns(columns, "event_features_only")
    assert "earnings_event_present_5s" not in sec
    assert "earnings_event_present_5s" in earnings
    assert "latest_eps_surprise_percent" in earnings
    assert "catalyst_score" not in technical


def test_training_loader_rejects_label_leakage_from_feature_manifest(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path, leak_feature=True)

    with pytest.raises(ValueError, match="forbidden columns"):
        load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")


def test_training_loader_keeps_audit_and_labels_out_of_x(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)

    training = load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")

    assert "raw_filing_count" not in training.X.columns
    assert "label_5_session_excess_return" not in training.X.columns
    assert "raw_filing_count" in training.audit.columns or training.audit.empty
    assert training.label_column == "label_5_session_excess_return"


def test_walk_forward_splits_apply_purge_gap(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    training = load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")

    splits = make_walk_forward_splits(training.metadata, training.y, horizon_sessions=5, n_folds=2)
    ordered_dates = sorted(pd.to_datetime(training.metadata["trading_date"]).dt.date.unique())
    date_pos = {value: idx for idx, value in enumerate(ordered_dates)}

    assert splits[-1].split_name == "test"
    for split in splits:
        assert max(split.train_dates) < min(split.eval_dates)
        assert date_pos[min(split.eval_dates)] - date_pos[max(split.train_dates)] >= 5


def test_regression_metric_computation_includes_no_skill_r2() -> None:
    metrics = regression_metrics([0.1, -0.1, 0.05], [0.08, -0.02, 0.01], train_mean=0.0)

    assert metrics["n"] == 3
    assert metrics["mae"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["oos_r2_vs_train_mean"] is not None
    assert 0 <= metrics["directional_accuracy"] <= 1


def test_preprocessor_uses_train_categories_and_deterministic_columns() -> None:
    train = pd.DataFrame({"numeric": [1.0, None, 3.0], "category": ["a", "b", None]})
    evaluation = pd.DataFrame({"numeric": [None, 5.0], "category": ["b", "future"]})

    train_matrix, eval_matrix, transformer = fit_transform_matrices(train, evaluation)

    assert train_matrix.columns.tolist() == transformer.output_columns
    assert eval_matrix.columns.tolist() == transformer.output_columns
    assert "category=future" not in eval_matrix.columns
    assert eval_matrix.loc[evaluation.index[0], "numeric"] == pytest.approx(2.0)
    assert eval_matrix.loc[evaluation.index[0], "category=b"] == 1.0


def test_model_run_persists_metrics_predictions_and_does_not_change_scoring(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    features = {
        "has_data": True,
        "data_quality": "ok",
        "last_price": 100,
        "ret_20d": 0.05,
        "ret_60d": 0.1,
        "above_50d_ma": True,
        "above_200d_ma": True,
        "relative_strength_20d": 0.02,
        "relative_strength_60d": 0.03,
        "volume_ratio_20d": 1.2,
        "liquidity_score_raw": 1.0,
        "liquidity_label": "Acceptable",
        "avg_dollar_volume_20d": 50_000_000,
        "avg_dollar_volume_ok": True,
        "distance_20d_ma": 0.02,
    }
    regime = {"regime": "Risk-On"}
    before = score_ticker_from_features("AAA", features, regime)

    summary = run_single_baseline_model(
        db_path,
        dataset_id,
        "label_5_session_excess_return",
        "technical_only",
        "ridge_regression",
        n_folds=2,
    )

    after = score_ticker_from_features("AAA", features, regime)
    assert before["score"] == after["score"]
    assert summary.model_run_id > 0
    assert not list_model_fold_metrics(db_path, summary.model_run_id).empty
    assert not list_model_final_metrics(db_path, summary.model_run_id).empty
    assert not list_model_predictions(db_path, summary.model_run_id).empty


def test_evaluation_regime_labels_known_inspected_runs_as_exploratory() -> None:
    labels = label_model_runs([145, 999], inspected_run_ids=DATASET_49_INSPECTED_RUN_IDS)
    by_run = {label.model_run_id: label for label in labels}

    assert by_run[145].evaluation_regime == "exploratory_dev"
    assert by_run[999].evaluation_regime == "holdout_candidate"
    assert "iterative" in by_run[145].reason
    assert validate_evaluation_regime("final_holdout") == "final_holdout"
    with pytest.raises(ValueError, match="Unknown evaluation regime"):
        validate_evaluation_regime("random_split")


def test_holdout_status_artifact_documents_dataset49_and_no_import(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    summary = run_single_baseline_model(
        db_path,
        dataset_id,
        "label_5_session_excess_return",
        "technical_core",
        "ridge_regression",
        n_folds=2,
        phase="test",
    )

    artifact = build_holdout_status_artifact(
        db_path,
        dataset_id=dataset_id,
        inspected_run_ids=(summary.model_run_id,),
    )
    plan = default_annotation_expansion_plan()

    assert artifact["decision"]["dataset_49_future_role"] == "exploratory_dev"
    assert artifact["decision"]["candidate_import_this_phase"] is False
    assert artifact["decision"]["modeling_this_phase"] is False
    assert artifact["inspected_model_runs"][0]["evaluation_regime"] == "exploratory_dev"
    assert plan["scanner_scoring_effect"] == 0
    assert plan["active_catalyst_creation"] is False


def test_dataset_evaluation_regime_metadata_round_trip(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)

    upsert_dataset_evaluation_regime(
        db_path,
        DatasetEvaluationRegime(
            dataset_id=dataset_id,
            evaluation_regime="holdout_candidate",
            parent_dataset_id=49,
            strategy="time_forward",
            rationale="Fresh post-development candidate; do not evaluate as final holdout.",
            metadata={"label_horizons_ready": False},
        ),
    )
    row = get_dataset_evaluation_regime(db_path, dataset_id)

    assert row is not None
    assert row["dataset_id"] == dataset_id
    assert row["evaluation_regime"] == "holdout_candidate"
    assert row["parent_dataset_id"] == 49
    assert row["metadata"]["label_horizons_ready"] is False


def test_holdout_maturity_blocks_immature_candidate(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    upsert_dataset_evaluation_regime(
        db_path,
        DatasetEvaluationRegime(
            dataset_id=dataset_id,
            evaluation_regime="holdout_candidate",
            parent_dataset_id=49,
            strategy="time_forward",
            rationale="Synthetic candidate.",
        ),
    )

    maturity = assess_holdout_maturity(db_path, dataset_id)

    assert maturity["readiness"]["protocol_validation"]["ready"] is True
    assert maturity["readiness"]["holdout_candidate_sanity_check"]["ready"] is False
    assert maturity["readiness"]["final_holdout_evaluation_5_session"]["ready"] is False
    assert maturity["promotion"]["allowed_for_5_session"] is False
    assert "row_count" in maturity["readiness"]["holdout_candidate_sanity_check"]["blockers"][0]


def test_holdout_maturity_can_pass_with_explicit_low_test_thresholds(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    upsert_dataset_evaluation_regime(
        db_path,
        DatasetEvaluationRegime(
            dataset_id=dataset_id,
            evaluation_regime="holdout_candidate",
            parent_dataset_id=49,
            strategy="time_forward",
            rationale="Synthetic candidate.",
        ),
    )

    maturity = assess_holdout_maturity(
        db_path,
        dataset_id,
        thresholds=HoldoutMaturityThresholds(
            sanity_min_rows=10,
            sanity_min_tickers=2,
            sanity_min_5_session_labeled_dates=5,
            sanity_min_5_session_coverage=0.5,
            final_min_rows=10,
            final_min_tickers=2,
            final_min_5_session_labeled_dates=5,
            final_min_5_session_coverage=0.5,
            final_min_20_session_labeled_dates=0,
            final_min_20_session_coverage=0.0,
        ),
    )

    assert maturity["readiness"]["holdout_candidate_sanity_check"]["ready"] is True
    assert maturity["readiness"]["final_holdout_evaluation_5_session"]["ready"] is True
    assert maturity["promotion"]["explicit_user_confirmation_required"] is True


def test_holdout_extension_plan_is_cache_only_and_new_dataset(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    dates = pd.bdate_range("2024-04-01", periods=3)
    frame = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "adj_close": [100.5, 101.5, 102.5],
            "volume": [1_000_000, 1_100_000, 1_200_000],
        }
    )
    storage.upsert_ohlcv(db_path, "AAA", frame)
    storage.upsert_ohlcv(db_path, "BBB", frame)

    plan = build_holdout_extension_plan(db_path, dataset_id)

    assert plan["cache_only_default"] is True
    assert plan["provider_fetch_allowed_by_default"] is False
    assert plan["extension_available"] is True
    assert plan["recommended_action"] == "create_new_holdout_candidate_dataset"
    assert plan["candidate_start_date"] == "2024-04-01"
    assert plan["extension_trading_day_count"] == 3


def test_feature_diagnostics_handles_boolean_and_correlated_features() -> None:
    frame = pd.DataFrame(
        {
            "above_50d_ma": [True, False, True, True],
            "ret_5d": [0.01, 0.02, 0.03, 0.04],
            "ret_5d_duplicate": [0.01, 0.02, 0.03, 0.04],
            "sec_current_event_event_days_30s": [0, 1, 0, 1],
        }
    )

    diagnostics = feature_diagnostics(frame)

    assert any(row["feature"] == "above_50d_ma" for row in diagnostics["scale_outliers"])
    assert any(
        {row["feature_a"], row["feature_b"]} == {"ret_5d", "ret_5d_duplicate"}
        for row in diagnostics["highly_correlated_pairs"]
    )


def test_coverage_by_ticker_separates_availability_from_event_activity() -> None:
    frame = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB", "BBB"],
            "sec_metadata_available": [1, 1, 0, 0],
            "sec_current_event_event_days_30s": [0, 0, 0, 0],
            "earnings_data_available": [1, 1, 1, 1],
            "earnings_event_present_5s": [0, 0, 1, 0],
        }
    )

    rows = {row["ticker"]: row for row in coverage_by_ticker(frame, list(frame.columns))}

    assert rows["AAA"]["sec_metadata_available_rate"] == 1.0
    assert rows["AAA"]["sec_any_activity_rate"] == 0.0
    assert rows["AAA"]["earnings_data_available_rate"] == 1.0
    assert rows["BBB"]["earnings_any_activity_rate"] == 0.5


def test_model_diagnostics_artifact_from_completed_run(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    run_single_baseline_model(
        db_path,
        dataset_id,
        "label_5_session_excess_return",
        "technical_only",
        "ridge_regression",
        n_folds=2,
    )

    diagnostics = build_model_diagnostics(db_path, dataset_id)
    artifact_path = write_diagnostics_artifact(diagnostics, tmp_path)
    reloaded = load_diagnostics_artifact(artifact_path)

    assert reloaded["dataset_id"] == dataset_id
    assert reloaded["summary"]["primary_target"] == "label_5_session_excess_return"
    assert reloaded["fold_metrics"]
    assert reloaded["target_distribution"]


def test_winsorization_uses_train_fold_thresholds_only() -> None:
    definition = get_target_definition("label_5_session_excess_return_winsorized_q01_q99")
    target = pd.Series([0.0, 0.01, 0.02, 10.0, -10.0, 1.0], dtype=float)
    train_mask = pd.Series([True, True, True, False, False, False])
    eval_mask = ~train_mask

    y_train, y_eval, metadata = transform_target_for_split(definition, target, train_mask, eval_mask)

    assert metadata["winsor_threshold_source"] == "train_fold_only"
    assert y_eval.max() <= y_train.max()
    assert y_eval.min() >= y_train.min()


def test_volatility_normalization_uses_snapshot_volatility_only() -> None:
    raw = pd.Series([0.02, 0.04, None], dtype=float)
    X = pd.DataFrame({"volatility_20d": [0.20, 0.40, 0.30]})

    normalized = volatility_normalized_target(raw, X, horizon_sessions=5)

    assert normalized.iloc[0] == pytest.approx(0.02 / (0.20 * (5 / 252) ** 0.5))
    assert normalized.iloc[1] == pytest.approx(0.04 / (0.40 * (5 / 252) ** 0.5))
    assert pd.isna(normalized.iloc[2])


def test_cross_sectional_rank_is_within_snapshot_date_only() -> None:
    raw = pd.Series([0.10, 0.20, -0.10, 0.50], dtype=float)
    metadata = pd.DataFrame(
        {
            "trading_date": ["2024-01-02", "2024-01-02", "2024-01-03", "2024-01-03"],
            "ticker": ["AAA", "BBB", "AAA", "BBB"],
        }
    )

    ranks = cross_sectional_rank_target(raw, metadata)

    assert ranks.tolist() == [0.5, 1.0, 0.5, 1.0]


def test_binary_top_bottom_excludes_middle_bucket() -> None:
    raw = pd.Series([0.0, 0.1, 0.2, 0.3, 0.4], dtype=float)
    metadata = pd.DataFrame({"trading_date": ["2024-01-02"] * 5})

    target = binary_top_bottom_target(raw, metadata, lower_quantile=0.20, upper_quantile=0.80)

    assert target.tolist()[0] == 0.0
    assert target.tolist()[-1] == 1.0
    assert target.isna().sum() == 3


def test_derived_target_model_run_persists_target_metadata(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)

    summary = run_single_baseline_model(
        db_path,
        dataset_id,
        "label_5_session_excess_return_cs_rank_pct",
        "technical_only",
        "ridge_regression",
        n_folds=2,
    )

    assert summary.target_column == "label_5_session_excess_return_cs_rank_pct"
    assert not list_model_final_metrics(db_path, summary.model_run_id).empty


def test_feature_quality_audit_records_final_test_guardrail(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)

    artifact = build_feature_quality_audit(db_path, dataset_id, n_folds=2, final_test_fraction=0.25)

    assert artifact["row_scope"]["final_test_labels_used_for_feature_selection"] is False
    assert artifact["pruned_feature_sets"]["technical_core"]["column_count"] > 0
    selected = pruned_feature_sets_from_artifact(artifact)
    training = load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")
    for columns in selected.values():
        assert set(columns).issubset(set(training.feature_columns))
        assert "label_5_session_excess_return" not in columns
        assert "raw_filing_count" not in columns


def test_feature_quality_flags_sparse_events_and_correlation() -> None:
    feature_columns = [
        "ret_5d",
        "ret_5d_duplicate",
        "sec_current_event_event_days_30s",
        "earnings_event_present_5s",
    ]
    X = pd.DataFrame(
        {
            "ret_5d": [0.01, 0.02, 0.03, 0.04, 0.05],
            "ret_5d_duplicate": [0.01, 0.02, 0.03, 0.04, 0.05],
            "sec_current_event_event_days_30s": [0, 0, 0, 0, 1],
            "earnings_event_present_5s": [0, 0, 0, 0, 0],
        }
    )
    dev_mask = pd.Series([True, True, True, True, True])
    final_mask = ~dev_mask
    missingness = feature_missingness_detail(X, dev_mask, final_mask)

    sparse = sparse_event_features(missingness)
    corr = correlation_audit(X, dev_mask, missingness)
    pruned = build_pruned_feature_sets(feature_columns, missingness, corr)

    assert {row["feature"] for row in sparse} >= {"sec_current_event_event_days_30s", "earnings_event_present_5s"}
    assert corr["remove_features"] == ["ret_5d_duplicate"]
    assert "ret_5d_duplicate" not in pruned["technical_pruned"]


def test_univariate_ic_uses_validation_folds_only(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    training = load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")
    folds = make_walk_forward_splits(training.metadata, training.y, horizon_sessions=5, n_folds=2)

    rows = univariate_ic_audit(training.X, training.metadata, training.y, folds)

    assert rows
    assert all(row["fold_count"] <= 2 for row in rows)


def test_pruned_feature_override_model_run_remains_manifest_safe(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    artifact = build_feature_quality_audit(db_path, dataset_id, n_folds=2)
    selected = pruned_feature_sets_from_artifact(artifact)["technical_core"]

    summary = run_single_baseline_model(
        db_path,
        dataset_id,
        "label_5_session_excess_return",
        "technical_core",
        "ridge_regression",
        n_folds=2,
        feature_columns_override=selected,
        feature_set_metadata={"source": "test_feature_quality"},
        phase="2D-4",
    )

    assert summary.model_run_id > 0
    metrics = list_model_final_metrics(db_path, summary.model_run_id)
    assert not metrics.empty


def test_event_feature_redesign_derives_recency_and_buckets() -> None:
    frame = pd.DataFrame(
        {
            "sec_days_since_latest_current_event": [0, 3, 12, None],
            "sec_current_event_event_days_7s": [1, 1, 0, 0],
            "sec_current_event_event_days_30s": [1, 1, 1, 0],
            "sec_current_event_event_days_90s": [1, 1, 1, 0],
            "sec_metadata_available": [1, 1, 1, 0],
            "sessions_since_latest_earnings": [0, 3, 12, None],
            "latest_eps_surprise_percent": [12.0, -8.0, None, 250.0],
            "earnings_data_available": [1, 1, 1, 0],
            "earnings_timing_known": [1, 0, 1, 0],
            "active_catalyst_count": [0, 0, 0, 0],
            "published_llm_supported_count": [0, 0, 0, 0],
        }
    )

    derived = derive_event_features(frame)

    assert derived.loc[0, "event_sec_current_event_bucket_same_day"] == 1.0
    assert derived.loc[1, "event_sec_current_event_bucket_post_1_5d"] == 1.0
    assert derived.loc[2, "event_sec_current_event_bucket_post_6_20d"] == 1.0
    assert derived.loc[0, "event_earnings_same_session"] == 1.0
    assert derived.loc[1, "event_earnings_post_1_5s"] == 1.0
    assert derived.loc[2, "event_earnings_post_6_20s"] == 1.0
    assert derived.loc[3, "event_earnings_eps_surprise_magnitude_clipped"] == pytest.approx(1.0)
    assert not any(column.startswith("label_") for column in derived.columns)


def test_event_feature_sets_exclude_inactive_catalyst_llm_features(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    training = load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")
    model_frame, derived, groups = build_event_model_frame(training)
    feature_sets = redesigned_event_feature_sets(training, derived)

    assert groups["catalyst_llm"]
    assert "event_catalyst_active" not in feature_sets["event_recency_only"]
    assert all(not column.endswith("_missing_or_stale") for column in feature_sets["sec_recency_categories"])
    assert set(feature_sets["technical_core_plus_event_recency"]).issubset(set(model_frame.columns))
    assert all(not column.startswith("label_") for columns in feature_sets.values() for column in columns)


def test_runner_accepts_safe_derived_event_feature_frame(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_model_dataset(db_path)
    training = load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")
    model_frame, derived, _groups = build_event_model_frame(training)
    feature_sets = redesigned_event_feature_sets(training, derived)

    summary = run_single_baseline_model(
        db_path,
        dataset_id,
        "label_5_session_excess_return",
        "technical_core_plus_event_recency",
        "ridge_regression",
        n_folds=2,
        feature_columns_override=feature_sets["technical_core_plus_event_recency"],
        feature_frame_override=model_frame,
        phase="2D-5",
    )

    assert summary.model_run_id > 0
    assert not list_model_final_metrics(db_path, summary.model_run_id).empty
