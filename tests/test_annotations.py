from __future__ import annotations

from datetime import UTC, date, datetime, timedelta

import pandas as pd

from src.annotations.csv_import import parse_annotation_import_frame
from src.annotations.models import ResearchEventAnnotation
from src.annotations.news_csv_provider import CsvManualNewsEventProvider, parse_candidate_import_frame
from src.annotations.news_events import EmptyNewsEventProvider, ResearchEventAnnotationCandidate
from src.annotations.news_repository import (
    accept_candidate,
    build_candidate_ingestion_artifact,
    import_accepted_candidates,
    list_candidates,
    reject_candidate,
    stage_candidate,
    stage_candidates,
)
from src.annotations.repository import insert_annotation, list_annotations
from src.data import storage
from src.datasets.models import DatasetBuild, FeatureSnapshot, OutcomeLabel
from src.datasets.repository import insert_dataset_build, insert_feature_snapshots, insert_outcome_labels
from src.datasets.training_loader import load_training_dataset
from src.modeling.annotation_features import (
    ANNOTATION_FEATURE_COLUMNS,
    build_annotation_model_frame,
    derive_annotation_features,
)
from src.modeling.runner import run_single_baseline_model
from src.scoring.score_engine import score_ticker_from_features


def _annotation(ticker: str = "AAA", event_date: date = date(2024, 1, 3), available_at: datetime | None = None) -> ResearchEventAnnotation:
    return ResearchEventAnnotation(
        ticker=ticker,
        event_date=event_date,
        available_at=available_at or datetime(2024, 1, 4, 14, tzinfo=UTC),
        event_type="news",
        sentiment_label="positive",
        strength=7,
        confidence=0.8,
        source="test",
        title="Synthetic research-only event",
        summary="Synthetic annotation for model research tests.",
        evidence_text="Company reported a synthetic milestone.",
    )


def _seed_dataset(db_path) -> int:
    build = DatasetBuild(
        version="annotation_test_v1",
        build_timestamp=datetime(2024, 1, 1, tzinfo=UTC),
        requested_start_date=date(2024, 1, 2),
        requested_end_date=date(2024, 3, 29),
        ticker_universe=["AAA", "BBB"],
        feature_columns=["ret_5d", "ret_20d", "volume_ratio_20d", "market_regime"],
        label_definitions={"5_session": {"target": "excess_return"}},
        row_count=0,
        data_hash="annotation_test_hash",
        audit_columns=["raw_event_count"],
        label_columns=["label_5_session_excess_return"],
        identifier_columns=["snapshot_id", "dataset_id", "ticker", "trading_date"],
        metadata_columns=["as_of_timestamp"],
        feature_manifest={
            "ret_5d": "model_feature",
            "ret_20d": "model_feature",
            "volume_ratio_20d": "model_feature",
            "market_regime": "model_feature",
            "raw_event_count": "audit",
            "label_5_session_excess_return": "label",
        },
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
            }
            snapshots.append(
                FeatureSnapshot(
                    ticker=ticker,
                    trading_date=trading_date,
                    as_of_timestamp=datetime.combine(trading_date, datetime.max.time(), tzinfo=UTC),
                    feature_version="annotation_test_v1",
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
        y = 0.01 if snapshot.features["ret_5d"] > 0 else -0.005
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


def test_annotation_table_creation_and_insert_dedupe(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    storage.init_db(db_path)

    with storage.connect(db_path) as conn:
        tables = {row["name"] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "research_event_annotations" in tables

    first = insert_annotation(db_path, _annotation())
    duplicate = insert_annotation(db_path, _annotation())
    frame = list_annotations(db_path)

    assert first.inserted is True
    assert duplicate.inserted is False
    assert duplicate.annotation_id == first.annotation_id
    assert len(frame) == 1
    assert int(frame.iloc[0]["research_only"]) == 1
    assert int(frame.iloc[0]["scanner_scoring_effect"]) == 0


def test_annotation_csv_validation_and_duplicates() -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "event_date": "2024-01-03",
                "available_at": "2024-01-04T14:00:00Z",
                "event_type": "news",
                "sentiment_label": "positive",
                "strength": 7,
                "confidence": 0.8,
                "title": "Same event",
            },
            {
                "ticker": "AAA",
                "event_date": "2024-01-03",
                "available_at": "2024-01-04T14:00:00Z",
                "event_type": "news",
                "sentiment_label": "positive",
                "strength": 7,
                "confidence": 0.8,
                "title": "Same event",
            },
            {
                "ticker": "",
                "event_date": "bad-date",
                "event_type": "not_supported",
            },
        ]
    )

    result = parse_annotation_import_frame(frame)

    assert len(result.annotations) == 1
    assert any("Duplicate" in error.message for error in result.errors)
    assert any("Missing ticker" in error.message for error in result.errors)


def test_annotation_features_are_point_in_time_and_exclude_future_events(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    insert_annotation(db_path, _annotation(event_date=date(2024, 1, 3), available_at=datetime(2024, 1, 4, 14, tzinfo=UTC)))
    insert_annotation(
        db_path,
        _annotation(
            event_date=date(2024, 1, 10),
            available_at=datetime(2024, 1, 4, 14, tzinfo=UTC),
        ),
    )
    metadata = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "AAA"],
            "trading_date": [date(2024, 1, 3), date(2024, 1, 4), date(2024, 1, 10)],
            "as_of_timestamp": [
                datetime(2024, 1, 3, 23, 59, tzinfo=UTC),
                datetime(2024, 1, 4, 23, 59, tzinfo=UTC),
                datetime(2024, 1, 10, 23, 59, tzinfo=UTC),
            ],
        }
    )

    features = derive_annotation_features(db_path, metadata)

    assert features.loc[0, "annotation_coverage_available"] == 0
    assert features.loc[1, "recent_positive_annotation_count_20s"] == 1
    assert features.loc[1, "max_recent_annotation_strength"] == 7
    assert features.loc[2, "recent_positive_annotation_count_20s"] == 2


def test_annotation_model_frame_contains_only_allowed_research_features(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_dataset(db_path)
    insert_annotation(db_path, _annotation("AAA"))
    training = load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")

    model_frame, derived, feature_sets = build_annotation_model_frame(training, db_path)

    assert set(ANNOTATION_FEATURE_COLUMNS).issubset(model_frame.columns)
    assert all(not column.startswith("label_") for column in model_frame.columns)
    assert all(not column.startswith("audit_") for column in model_frame.columns)
    assert feature_sets["annotation_features_only"] == ANNOTATION_FEATURE_COLUMNS
    assert not derived.empty


def test_annotation_baseline_does_not_change_scanner_scoring(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    dataset_id = _seed_dataset(db_path)
    insert_annotation(db_path, _annotation("AAA"))
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

    training = load_training_dataset(db_path, dataset_id, "label_5_session_excess_return")
    model_frame, _derived, feature_sets = build_annotation_model_frame(training, db_path)
    summary = run_single_baseline_model(
        db_path,
        dataset_id,
        "label_5_session_excess_return",
        "annotation_features_only",
        "ridge_regression",
        n_folds=2,
        feature_columns_override=feature_sets["annotation_features_only"],
        feature_frame_override=model_frame,
        phase="2D-6A-test",
    )
    after = score_ticker_from_features("AAA", features, regime)

    assert summary.model_run_id > 0
    assert before["score"] == after["score"]


def test_news_event_provider_interface_and_csv_candidate_parse() -> None:
    empty = EmptyNewsEventProvider()
    assert empty.get_events("AAA", date(2024, 1, 1), date(2024, 1, 31)) == []

    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "event_date": "2024-01-03",
                "available_at": "2024-01-03T14:00:00Z",
                "event_type": "news",
                "title": "Customer expansion announced",
                "summary": "Synthetic candidate.",
                "source": "manual_url",
                "source_url": "https://example.com/news/a",
                "evidence_text": "Company announced a customer expansion.",
                "sentiment_label": "positive",
                "strength": 6,
                "confidence": 0.75,
                "tags": "customer, expansion",
                "provider_metadata_json": '{"source_quality":"manual"}',
            }
        ]
    )

    result = parse_candidate_import_frame(frame)
    provider = CsvManualNewsEventProvider(frame)
    events = provider.get_events("AAA", date(2024, 1, 1), date(2024, 1, 31))

    assert not result.errors
    assert len(result.candidates) == 1
    assert result.candidates[0].provider_metadata["source_quality"] == "manual"
    assert len(events) == 1
    assert events[0].ticker == "AAA"


def test_candidate_staging_detects_duplicates_and_supports_reject(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    candidate = ResearchEventAnnotationCandidate(
        ticker="AAA",
        event_date=date(2024, 1, 3),
        available_at=datetime(2024, 1, 3, 14, tzinfo=UTC),
        event_type="news",
        title="Same candidate",
        source="manual_url",
        source_url="https://example.com/same",
        evidence_text="Same evidence.",
    )

    first = stage_candidate(db_path, candidate)
    duplicate = stage_candidate(
        db_path,
        ResearchEventAnnotationCandidate(
            ticker="AAA",
            event_date=date(2024, 1, 4),
            available_at=datetime(2024, 1, 4, 14, tzinfo=UTC),
            event_type="news",
            title="Different title same URL",
            source="manual_url",
            source_url="https://example.com/same/",
            evidence_text="Different evidence.",
        ),
    )
    reject_candidate(db_path, first.candidate_id, "Not relevant.")
    candidates = list_candidates(db_path, limit=None)

    assert first.status == "staged"
    assert duplicate.status == "duplicate"
    assert duplicate.duplicate_reason == "existing_candidate_source_url"
    assert candidates[candidates["candidate_id"].eq(first.candidate_id)].iloc[0]["status"] == "rejected"


def test_accepted_candidates_import_as_research_only_annotations(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    results = stage_candidates(
        db_path,
        [
            ResearchEventAnnotationCandidate(
                ticker="AAA",
                event_date=date(2024, 1, 3),
                available_at=datetime(2024, 1, 3, 14, tzinfo=UTC),
                event_type="news",
                title="Positive manual event",
                summary="Pipeline test candidate.",
                source="manual_url",
                source_url="https://example.com/positive",
                evidence_text="Positive evidence.",
                sentiment_label="positive",
                strength=7,
                confidence=0.8,
                tags=["pipeline"],
            )
        ],
    )
    accept_candidate(db_path, results[0].candidate_id)
    summary = import_accepted_candidates(db_path)
    annotations = list_annotations(db_path, ticker="AAA", limit=None)
    candidates = list_candidates(db_path, limit=None)

    assert summary.imported_count == 1
    assert summary.skipped_count == 0
    assert int(annotations.iloc[0]["research_only"]) == 1
    assert int(annotations.iloc[0]["scanner_scoring_effect"]) == 0
    assert candidates.iloc[0]["status"] == "imported"
    assert int(candidates.iloc[0]["imported_annotation_id"]) == int(annotations.iloc[0]["annotation_id"])


def test_candidate_duplicates_against_existing_annotations_and_artifact(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    insert_annotation(
        db_path,
        ResearchEventAnnotation(
            ticker="AAA",
            event_date=date(2024, 1, 3),
            available_at=datetime(2024, 1, 3, 14, tzinfo=UTC),
            event_type="news",
            sentiment_label="positive",
            strength=6,
            confidence=0.7,
            source="manual_url",
            source_url="https://example.com/existing",
            title="Existing annotation",
            evidence_text="Existing evidence.",
        ),
    )
    staged = stage_candidate(
        db_path,
        ResearchEventAnnotationCandidate(
            ticker="AAA",
            event_date=date(2024, 1, 4),
            available_at=datetime(2024, 1, 4, 14, tzinfo=UTC),
            event_type="news",
            title="Another title",
            source="manual_url",
            source_url="https://example.com/existing/",
            evidence_text="New evidence.",
        ),
    )
    artifact = build_candidate_ingestion_artifact(db_path)

    assert staged.status == "duplicate"
    assert staged.duplicate_reason == "existing_annotation_source_url"
    assert artifact["scanner_scoring_effect"] == 0
    assert artifact["research_only"] is True
    assert artifact["status_counts"][0]["status"] == "duplicate"
