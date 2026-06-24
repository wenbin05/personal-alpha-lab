from __future__ import annotations

import json
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.catalysts.proposals import create_proposal_from_extraction, set_proposal_status
from src.catalysts.publications import publish_proposal, revert_publication
from src.catalysts.repository import delete_catalyst, insert_catalyst, update_catalyst
from src.catalysts.models import CatalystEvent
from src.catalysts.sec_classification import SEC_FEATURE_POLICY_VERSION, classify_ticker_sec_filings
from src.data import storage
from src.datasets.backfill import (
    CACHE_ONLY_PROVIDER,
    _coverage_warnings,
    create_backfill_run,
    dataset_sufficiency_report,
    list_backfill_items,
    process_backfill_run,
    retry_failed_items,
)
from src.datasets.builder import (
    _publication_overlaps_window,
    _sec_filing_features,
    _sec_metadata_rows_as_of,
    active_catalysts_as_of,
    build_feature_snapshot,
    build_point_in_time_dataset,
    calculate_outcome_labels,
    feature_columns_from_frame,
    precompute_feature_sets_for_dates,
    precompute_sec_features_for_dates,
)
from src.datasets.feature_manifest import FEATURE_CONTRACT_VERSION, MANIFEST_METADATA_KEY, role_sets_from_frame
from src.datasets.repository import list_dataset_builds
from src.datasets.repository import (
    flatten_saved_dataset,
    insert_feature_snapshots,
    stream_saved_dataset_export_and_hash,
)
from src.datasets.models import FeatureSnapshot
from src.datasets.splits import assign_chronological_splits, chronological_split_dates
from src.datasets.training_loader import load_training_dataset
from src.documents.repository import build_source_document, insert_document
from src.extractions.repository import approve_extraction, insert_extraction
from src.features.catalyst import get_catalyst_features
from src.scoring.score_engine import score_ticker_from_features
from src.scoring.score_engine import build_feature_set


def _price_frame(start: str = "2024-01-02", periods: int = 50, base: float = 100.0, step: float = 1.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    close = [base + idx * step for idx in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [value - 0.5 for value in close],
            "high": [value + 1 for value in close],
            "low": [value - 1 for value in close],
            "close": close,
            "adj_close": close,
            "volume": [1_000_000 + idx * 1_000 for idx in range(periods)],
        }
    )


def _seed_prices(db_path, ticker: str = "AAPL") -> None:
    storage.upsert_ohlcv(db_path, ticker, _price_frame(base=100, step=1.0))
    storage.upsert_ohlcv(db_path, "SPY", _price_frame(base=400, step=0.5))
    storage.upsert_ohlcv(db_path, "QQQ", _price_frame(base=300, step=0.4))
    storage.upsert_ohlcv(db_path, "IWM", _price_frame(base=200, step=0.2))
    storage.upsert_ohlcv(db_path, "^VIX", _price_frame(base=15, step=0.0))


def test_precomputed_feature_sets_match_legacy_daily_feature_path(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    storage.upsert_ohlcv(db_path, "AAPL", _price_frame(periods=90, base=100, step=1.2))
    storage.upsert_ohlcv(db_path, "SPY", _price_frame(periods=90, base=400, step=0.4))
    ticker_df = storage.load_ohlcv(db_path, "AAPL")
    spy_df = storage.load_ohlcv(db_path, "SPY")
    sample_dates = [pd.Timestamp("2024-02-15").date(), pd.Timestamp("2024-03-15").date()]

    precomputed = precompute_feature_sets_for_dates("AAPL", ticker_df, spy_df, sample_dates)

    for trading_date in sample_dates:
        expected = build_feature_set(
            "AAPL",
            ticker_df[ticker_df.index <= pd.Timestamp(trading_date)],
            spy_df[spy_df.index <= pd.Timestamp(trading_date)],
        )
        actual = precomputed[trading_date]
        for key, expected_value in expected.items():
            actual_value = actual.get(key)
            if isinstance(expected_value, float):
                assert actual_value == pytest.approx(expected_value)
            else:
                assert actual_value == expected_value


def _document_id(db_path, ticker: str = "AAPL") -> int:
    document = build_source_document(
        ticker,
        "news_article",
        f"{ticker} synthetic earnings note",
        "The company reported record revenue and raised guidance.",
        source="manual",
        source_url=f"https://example.com/{ticker.lower()}/dataset-note",
        published_at=date(2024, 1, 10),
    )
    return insert_document(db_path, document)


def _extraction_payload(document_id: int, ticker: str = "AAPL") -> dict:
    return {
        "document_id": document_id,
        "ticker": ticker,
        "provider": "fallback",
        "model_name": "fallback",
        "extraction_type": "catalyst_analysis",
        "event_type_detected": "earnings",
        "sentiment_label": "positive",
        "catalyst_strength": 7,
        "risk_severity": 2,
        "confidence": 0.8,
        "document_relevance": "relevant",
        "evidence_sufficiency": "sufficient",
        "time_horizon": "short_term",
        "key_positive_points": ["Record revenue"],
        "key_risks": [],
        "evidence_snippets": ["record revenue"],
        "short_summary": "Record revenue and raised guidance.",
        "detailed_summary": "The supplied text says record revenue and raised guidance.",
        "proposed_score_effect": 10,
        "review_status": "pending_review",
    }


def _publish_llm_catalyst(db_path, ticker: str = "AAPL", published_at: str = "2024-01-15T23:59:59+00:00") -> int:
    document_id = _document_id(db_path, ticker)
    extraction_id = insert_extraction(db_path, _extraction_payload(document_id, ticker))
    assert approve_extraction(db_path, extraction_id, "approved")
    proposal = create_proposal_from_extraction(db_path, extraction_id)
    assert proposal.changed
    assert set_proposal_status(db_path, proposal.proposal_id, "reviewed_ready", "ready")
    result = publish_proposal(db_path, proposal.proposal_id, "dataset test publish")
    assert result.changed
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE catalyst_publications
            SET published_at = ?, created_at = ?, updated_at = ?
            WHERE publication_id = ?
            """,
            (published_at, published_at, published_at, result.publication_id),
        )
        conn.execute(
            """
            UPDATE catalysts
            SET event_date = ?, created_at = ?, updated_at = ?
            WHERE id = ?
            """,
            ("2024-01-15", published_at, published_at, result.catalyst_id),
        )
        after_snapshot = dict(conn.execute("SELECT * FROM catalysts WHERE id = ?", (result.catalyst_id,)).fetchone())
        conn.execute(
            """
            UPDATE catalyst_publications
            SET after_snapshot_json = ?, updated_at = ?
            WHERE publication_id = ?
            """,
            (json.dumps(after_snapshot, default=str), published_at, result.publication_id),
        )
    return int(result.publication_id)


def test_snapshot_generation_and_forward_labels(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    histories = {ticker: storage.load_ohlcv(db_path, ticker) for ticker in ["AAPL", "SPY", "QQQ", "IWM", "^VIX"]}
    trading_date = pd.Timestamp(histories["AAPL"].index[5]).date()

    snapshot = build_feature_snapshot(db_path, "AAPL", trading_date, histories)
    labels = calculate_outcome_labels(snapshot, histories["AAPL"], histories["SPY"], horizons=(1, 5))

    assert snapshot is not None
    assert snapshot.features["ret_5d"] == pytest.approx(105 / 100 - 1)
    assert labels[0].entry_date == pd.Timestamp(histories["AAPL"].index[6]).date()
    assert labels[0].exit_date == pd.Timestamp(histories["AAPL"].index[7]).date()
    assert labels[0].forward_return == pytest.approx(107 / 106 - 1)
    assert labels[0].excess_return == pytest.approx((107 / 106 - 1) - (403.5 / 403 - 1))


def test_dataset_build_stores_metadata_and_keeps_labels_out_of_features(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)

    result = build_point_in_time_dataset(
        db_path,
        ["AAPL"],
        date(2024, 1, 10),
        date(2024, 1, 18),
        output_dir=tmp_path,
    )

    assert result.dataset_id > 0
    assert len(result.dataset_frame) > 0
    assert result.export_path is not None
    assert not any(column.startswith("label_") for column in feature_columns_from_frame(result.dataset_frame))
    builds = list_dataset_builds(db_path)
    assert int(builds.iloc[0]["dataset_id"]) == result.dataset_id
    assert int(builds.iloc[0]["row_count"]) == len(result.dataset_frame)


def test_future_manual_catalyst_cannot_leak_backward(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="AAPL",
            event_date=date(2024, 1, 8),
            event_type="news",
            title="Future-entered positive catalyst",
            summary="Created after the early snapshot.",
            source="manual",
            sentiment_label="positive",
            catalyst_strength=8,
            confidence=1.0,
            created_at=datetime(2024, 1, 20, tzinfo=UTC),
            updated_at=datetime(2024, 1, 20, tzinfo=UTC),
        ),
    )

    early = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 10))
    late = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 22))

    assert early.empty
    assert len(late) == 1


def test_backfilled_sec_filing_uses_available_at_not_ingestion_time(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="AAPL",
            event_date=date(2024, 1, 10),
            event_type="sec_filing",
            title="SEC 8-K filing (Needs Review)",
            summary="Metadata only.",
            source="SEC EDGAR",
            sentiment_label="unknown",
            catalyst_strength=0,
            confidence=0.7,
            is_manual=False,
            available_at=datetime(2024, 1, 11, 21, 5, tzinfo=UTC),
            created_at=datetime(2026, 6, 21, tzinfo=UTC),
            updated_at=datetime(2026, 6, 21, tzinfo=UTC),
            raw_payload_json=json.dumps({"form": "8-K", "acceptanceDateTime": "2024-01-11T21:05:00+00:00"}),
        ),
    )

    before = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 10))
    after = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 12))

    assert before.empty
    assert len(after) == 1
    assert pd.to_datetime(after.iloc[0]["available_at"]).date() == date(2024, 1, 11)


def test_sec_filing_snapshot_features_do_not_leak_before_acceptance(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="AAPL",
            event_date=date(2024, 1, 10),
            event_type="sec_filing",
            title="SEC 10-Q/A filing",
            summary="Metadata only.",
            source="SEC EDGAR",
            sentiment_label="unknown",
            catalyst_strength=0,
            confidence=0.7,
            available_at=datetime(2024, 1, 12, 12, 0, tzinfo=UTC),
            created_at=datetime(2026, 6, 21, tzinfo=UTC),
            updated_at=datetime(2026, 6, 21, tzinfo=UTC),
            raw_payload_json=json.dumps({"form": "10-Q/A", "acceptanceDateTime": "2024-01-12T12:00:00+00:00"}),
        ),
    )
    histories = {ticker: storage.load_ohlcv(db_path, ticker) for ticker in ["AAPL", "SPY", "QQQ", "IWM", "^VIX"]}

    before = build_feature_snapshot(db_path, "AAPL", date(2024, 1, 11), histories)
    after = build_feature_snapshot(db_path, "AAPL", date(2024, 1, 12), histories)

    assert before.features["sec_metadata_available"] is False
    assert before.features["sec_feature_eligible_event_days_7s"] == 0
    assert after.features["sec_metadata_available"] is True
    assert after.features["sec_feature_eligible_event_days_7s"] == 1
    assert after.features["sec_amendment_count_30s"] == 1
    assert after.features["sec_amendment_event_days_30s"] == 1


def test_sec_aggregation_by_event_day_and_structured_note_separation(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    for accession in ["0001", "0002"]:
        insert_catalyst(
            db_path,
            CatalystEvent(
                ticker="AAPL",
                event_date=date(2024, 1, 10),
                event_type="sec_filing",
                title=f"SEC 424B2 filing {accession}",
                summary="Metadata only.",
                source="SEC EDGAR",
                source_url=f"https://www.sec.gov/Archives/{accession}.htm",
                sentiment_label="unknown",
                catalyst_strength=0,
                confidence=0.7,
                available_at=datetime(2024, 1, 10, 21, 0, tzinfo=UTC),
                raw_payload_json=json.dumps(
                    {
                        "form": "424B2",
                        "acceptanceDateTime": "2024-01-10T21:00:00+00:00",
                        "accessionNumber": accession,
                        "primaryDocDescription": "Market-linked structured notes",
                    }
                ),
            ),
        )
    histories = {ticker: storage.load_ohlcv(db_path, ticker) for ticker in ["AAPL", "SPY", "QQQ", "IWM", "^VIX"]}
    snapshot = build_feature_snapshot(db_path, "AAPL", date(2024, 1, 11), histories)

    assert snapshot.features["sec_raw_filing_count_7s_audit"] == 2
    assert snapshot.features["sec_structured_note_filing_count_7s_audit"] == 2
    assert snapshot.features["sec_structured_note_event_days_7s"] == 1
    assert snapshot.features["sec_equity_financing_event_days_7s"] == 0
    assert snapshot.features["sec_feature_eligible_event_days_7s"] == 1
    assert snapshot.features["sec_recent_structured_note_flag"] is True
    assert snapshot.features["sec_recent_equity_financing_flag"] is False
    assert "sec_recent_s1_s3_424b_flag" not in snapshot.features


def test_ambiguous_sec_filing_is_excluded_from_curated_features(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="AAPL",
            event_date=date(2024, 1, 10),
            event_type="sec_filing",
            title="SEC 424B2 filing",
            summary="Metadata only.",
            source="SEC EDGAR",
            sentiment_label="unknown",
            catalyst_strength=0,
            confidence=0.7,
            available_at=datetime(2024, 1, 10, 21, 0, tzinfo=UTC),
            raw_payload_json=json.dumps(
                {
                    "form": "424B2",
                    "acceptanceDateTime": "2024-01-10T21:00:00+00:00",
                    "primaryDocDescription": "Prospectus supplement",
                }
            ),
        ),
    )
    histories = {ticker: storage.load_ohlcv(db_path, ticker) for ticker in ["AAPL", "SPY", "QQQ", "IWM", "^VIX"]}
    snapshot = build_feature_snapshot(db_path, "AAPL", date(2024, 1, 11), histories)

    assert snapshot.features["sec_raw_filing_count_7s_audit"] == 1
    assert snapshot.features["sec_unknown_classification_count_30s"] == 1
    assert snapshot.features["sec_feature_excluded_count_30s"] == 1
    assert snapshot.features["sec_feature_eligible_event_days_7s"] == 0


def test_sec_policy_metadata_is_recorded_and_hash_is_deterministic(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="AAPL",
            event_date=date(2024, 1, 10),
            event_type="sec_filing",
            title="SEC 8-K filing",
            summary="Metadata only.",
            source="SEC EDGAR",
            sentiment_label="unknown",
            catalyst_strength=0,
            confidence=0.7,
            available_at=datetime(2024, 1, 10, 21, 0, tzinfo=UTC),
            raw_payload_json=json.dumps({"form": "8-K", "acceptanceDateTime": "2024-01-10T21:00:00+00:00"}),
        ),
    )

    first = build_point_in_time_dataset(db_path, ["AAPL"], date(2024, 1, 8), date(2024, 1, 16), output_dir=tmp_path)
    second = build_point_in_time_dataset(db_path, ["AAPL"], date(2024, 1, 8), date(2024, 1, 16), output_dir=tmp_path)

    assert first.build.label_definitions["sec_feature_policy"]["policy_version"] == SEC_FEATURE_POLICY_VERSION
    assert first.build.data_hash == second.build.data_hash
    assert "label_1_session_forward_return" not in feature_columns_from_frame(first.dataset_frame)


def test_feature_manifest_roles_exclude_audit_counts_and_labels(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="AAPL",
            event_date=date(2024, 1, 10),
            event_type="sec_filing",
            title="SEC 424B2 filing",
            summary="Metadata only.",
            source="SEC EDGAR",
            sentiment_label="unknown",
            catalyst_strength=0,
            confidence=0.7,
            available_at=datetime(2024, 1, 10, 21, 0, tzinfo=UTC),
            raw_payload_json=json.dumps(
                {
                    "form": "424B2",
                    "acceptanceDateTime": "2024-01-10T21:00:00+00:00",
                    "primaryDocDescription": "PRELIMINARY PRICING SUPPLEMENT",
                }
            ),
        ),
    )

    result = build_point_in_time_dataset(db_path, ["AAPL"], date(2024, 1, 8), date(2024, 1, 16), output_dir=tmp_path)
    roles = role_sets_from_frame(result.dataset_frame)
    builds = list_dataset_builds(db_path)
    build = builds[builds["dataset_id"].astype(int).eq(result.dataset_id)].iloc[0]
    stored_features = json.loads(build["feature_columns_json"])
    stored_audit = json.loads(build["audit_columns_json"])
    stored_labels = json.loads(build["label_columns_json"])
    manifest = json.loads(build["feature_manifest_json"])

    assert stored_features == roles.model_features
    assert "sec_feature_eligible_filing_count_30s" in stored_audit
    assert "sec_feature_eligible_filing_count_30s" not in stored_features
    assert "sec_recent_form4_count" not in stored_features
    assert "sec_max_feature_eligible_filings_single_day_30s" not in stored_features
    assert "sec_needs_review_filing_flag" not in stored_features
    assert "sec_recent_structured_note_flag" in stored_features
    assert not any(column.startswith("label_") for column in stored_features)
    assert stored_labels
    assert manifest[MANIFEST_METADATA_KEY]["policy_version"] == FEATURE_CONTRACT_VERSION
    assert manifest["sec_feature_eligible_filing_count_30s"]["role"] == "audit"


def test_default_training_loader_isolates_features_labels_and_audit(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    result = build_point_in_time_dataset(db_path, ["AAPL"], date(2024, 1, 8), date(2024, 1, 18), output_dir=tmp_path)
    target = "label_1_session_forward_return"

    training = load_training_dataset(db_path, result.dataset_id, target)

    assert target == training.label_column
    assert len(training.X) == len(training.y)
    assert target not in training.X.columns
    assert "ticker" not in training.X.columns
    assert "trading_date" not in training.X.columns
    assert not any(column.endswith("_audit") for column in training.X.columns)
    assert "ticker" in training.metadata.columns


def test_vectorized_sec_features_match_reference_point_in_time(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    events = [
        ("8-K", "8-K", "0008k", datetime(2024, 1, 10, 21, 0, tzinfo=UTC)),
        ("424B2", "PRELIMINARY PRICING SUPPLEMENT", "000note1", datetime(2024, 1, 11, 21, 0, tzinfo=UTC)),
        ("424B2", "PRELIMINARY PRICING SUPPLEMENT", "000note2", datetime(2024, 1, 11, 22, 0, tzinfo=UTC)),
        ("4", "", "000form4", datetime(2024, 1, 12, 21, 0, tzinfo=UTC)),
        ("424B3", "FORM 424B3", "000unknown", datetime(2024, 1, 16, 21, 0, tzinfo=UTC)),
    ]
    for form, description, accession, available_at in events:
        insert_catalyst(
            db_path,
            CatalystEvent(
                ticker="AAPL",
                event_date=available_at.date(),
                event_type="sec_filing",
                title=f"SEC {form} filing",
                summary="Metadata only.",
                source="SEC EDGAR",
                source_url=f"https://www.sec.gov/Archives/{accession}.htm",
                sentiment_label="unknown",
                catalyst_strength=0,
                confidence=0.7,
                available_at=available_at,
                raw_payload_json=json.dumps(
                    {
                        "form": form,
                        "accessionNumber": accession,
                        "acceptanceDateTime": available_at.isoformat(),
                        "primaryDocDescription": description,
                    }
                ),
            ),
        )
    dates = [date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 16), date(2024, 1, 17)]
    vectorized = precompute_sec_features_for_dates(db_path, "AAPL", dates)
    for trading_date in dates:
        reference_rows = _sec_metadata_rows_as_of(
            db_path,
            "AAPL",
            datetime.combine(trading_date, datetime.max.time(), tzinfo=UTC),
            trading_date,
        )
        reference = _sec_filing_features(reference_rows, trading_date)
        for key in [
            "sec_feature_eligible_event_days_7s",
            "sec_structured_note_event_days_7s",
            "sec_current_event_event_days_7s",
            "sec_ownership_event_days_7s",
            "sec_unknown_classification_count_30s",
            "sec_recent_structured_note_flag",
            "sec_recent_equity_financing_flag",
            "sec_recent_registration_or_prospectus_other_flag",
        ]:
            assert vectorized[trading_date][key] == reference[key]


def test_late_llm_publication_cannot_affect_earlier_snapshots_and_reverts_exclude(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    publication_id = _publish_llm_catalyst(db_path, published_at="2024-01-15T23:59:59+00:00")

    histories = {ticker: storage.load_ohlcv(db_path, ticker) for ticker in ["AAPL", "SPY", "QQQ", "IWM", "^VIX"]}
    early = build_feature_snapshot(db_path, "AAPL", date(2024, 1, 12), histories)
    late = build_feature_snapshot(db_path, "AAPL", date(2024, 1, 16), histories)

    assert early.features["published_llm_supported_catalyst"] is False
    assert late.features["published_llm_supported_catalyst"] is True
    assert late.features["llm_max_risk_severity"] == 2

    assert revert_publication(db_path, publication_id, "remove from active catalysts").changed
    rebuilt_after_future_reversal = build_feature_snapshot(db_path, "AAPL", date(2024, 1, 16), histories)
    assert rebuilt_after_future_reversal.features["published_llm_supported_catalyst"] is True

    with storage.connect(db_path) as conn:
        conn.execute(
            "UPDATE catalyst_publications SET reverted_at = ?, updated_at = ? WHERE publication_id = ?",
            ("2024-01-17T00:00:00+00:00", "2024-01-17T00:00:00+00:00", publication_id),
        )
    after_reversal_window = build_feature_snapshot(db_path, "AAPL", date(2024, 1, 18), histories)
    assert after_reversal_window.features["published_llm_supported_catalyst"] is False


def test_reverted_publications_do_not_appear_in_active_catalyst_features(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    publication_id = _publish_llm_catalyst(db_path)
    assert revert_publication(db_path, publication_id, "revert").changed
    with storage.connect(db_path) as conn:
        conn.execute(
            "UPDATE catalyst_publications SET reverted_at = ?, updated_at = ? WHERE publication_id = ?",
            ("2024-01-17T00:00:00+00:00", "2024-01-17T00:00:00+00:00", publication_id),
        )

    catalysts = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 20))
    features = get_catalyst_features("AAPL", catalysts, as_of_date=date(2024, 1, 20))

    assert catalysts.empty
    assert features["catalyst_score"] == 0


def test_publication_active_window_before_during_after_and_superseded(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    publication_id = _publish_llm_catalyst(db_path, published_at="2024-01-15T23:59:59+00:00")

    before = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 12))
    during = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 16))
    assert before.empty
    assert len(during) == 1

    with storage.connect(db_path) as conn:
        conn.execute(
            "UPDATE catalyst_publications SET publication_status = 'superseded', updated_at = ? WHERE publication_id = ?",
            ("2024-01-17T00:00:00+00:00", publication_id),
        )
    still_during = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 16))
    after_superseded = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 18))
    assert len(still_during) == 1
    assert after_superseded.empty


def test_zero_duration_publication_window_does_not_trigger_historical_overlap() -> None:
    publication = {
        "publication_status": "reverted",
        "published_at": "2026-06-20T14:56:12+00:00",
        "reverted_at": "2026-06-20T14:56:12+00:00",
        "updated_at": "2026-06-20T14:56:12+00:00",
    }

    assert not _publication_overlaps_window(
        publication,
        datetime(2023, 6, 19, tzinfo=UTC),
        datetime(2026, 6, 23, tzinfo=UTC),
    )


def test_chronological_splits_never_overlap_and_respect_gap(tmp_path) -> None:
    dates = [pd.Timestamp(value).date() for value in pd.bdate_range("2024-01-02", periods=10)]
    splits = chronological_split_dates(dates, dates[3], dates[6], gap_sessions=1)

    assert set(splits["train"]).isdisjoint(splits["validation"])
    assert set(splits["train"]).isdisjoint(splits["test"])
    assert set(splits["validation"]).isdisjoint(splits["test"])
    assert dates[4] not in splits["validation"]
    assert dates[7] not in splits["test"]

    frame = pd.DataFrame({"ticker": ["AAPL"] * len(dates), "trading_date": dates})
    assigned = assign_chronological_splits(frame, dates[3], dates[6], gap_sessions=1)
    assert "gap" in set(assigned["split"])


def test_catalyst_revision_history_replays_create_update_delete(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    catalyst_id = insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="AAPL",
            event_date=date(2024, 1, 10),
            event_type="manual_note",
            title="Original catalyst",
            summary="Original",
            source="manual",
            sentiment_label="neutral",
            catalyst_strength=1,
            confidence=0.5,
            is_manual=True,
            created_at=datetime(2024, 1, 10, tzinfo=UTC),
            updated_at=datetime(2024, 1, 10, tzinfo=UTC),
        ),
    )
    assert update_catalyst(db_path, catalyst_id, {"title": "Updated catalyst", "sentiment_label": "positive"})
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE catalyst_revisions
            SET effective_timestamp = ?
            WHERE catalyst_id = ? AND action = 'update'
            """,
            ("2024-01-15T23:59:59+00:00", catalyst_id),
        )

    before_update = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 12))
    after_update = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 16))
    assert before_update.iloc[0]["title"] == "Original catalyst"
    assert after_update.iloc[0]["title"] == "Updated catalyst"

    assert delete_catalyst(db_path, catalyst_id)
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE catalyst_revisions
            SET effective_timestamp = ?
            WHERE catalyst_id = ? AND action = 'delete'
            """,
            ("2024-01-17T00:00:00+00:00", catalyst_id),
        )
    after_delete = active_catalysts_as_of(db_path, "AAPL", date(2024, 1, 18))
    assert after_delete.empty


def test_backfill_is_resumable_and_idempotent(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path, "AAPL")
    _seed_prices(db_path, "MSFT")
    run_id = create_backfill_run(db_path, ["AAPL", "MSFT"], date(2024, 1, 10), date(2024, 1, 18))

    first = process_backfill_run(db_path, run_id, output_dir=tmp_path, max_tickers=1)
    assert first.processed_tickers == 1
    items_after_first = list_backfill_items(db_path, run_id)
    assert set(items_after_first["status"]) == {"completed", "pending"}

    second = process_backfill_run(db_path, run_id, output_dir=tmp_path)
    assert second.completed_tickers == 2
    with storage.connect(db_path) as conn:
        dataset_id = conn.execute("SELECT dataset_id FROM backfill_runs WHERE run_id = ?", (run_id,)).fetchone()["dataset_id"]
        duplicate_rows = conn.execute(
            """
            SELECT ticker, trading_date, COUNT(*) AS count
            FROM feature_snapshots
            WHERE dataset_id = ?
            GROUP BY ticker, trading_date
            HAVING COUNT(*) > 1
            """,
            (dataset_id,),
        ).fetchall()
        snapshot_count = conn.execute("SELECT COUNT(*) AS count FROM feature_snapshots WHERE dataset_id = ?", (dataset_id,)).fetchone()["count"]
    assert not duplicate_rows

    third = process_backfill_run(db_path, run_id, output_dir=tmp_path)
    with storage.connect(db_path) as conn:
        snapshot_count_after = conn.execute("SELECT COUNT(*) AS count FROM feature_snapshots WHERE dataset_id = ?", (dataset_id,)).fetchone()["count"]
    assert third.processed_tickers == 0
    assert snapshot_count_after == snapshot_count


def test_backfill_per_ticker_provider_failure_and_retry(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path, "AAPL")

    def fail_download(*args, **kwargs):
        raise RuntimeError("provider unavailable")

    monkeypatch.setattr("src.datasets.backfill.market_data.get_history_with_metadata", fail_download)
    run_id = create_backfill_run(db_path, ["AAPL", "FAKE123"], date(2024, 1, 10), date(2024, 1, 18))
    result = process_backfill_run(db_path, run_id, output_dir=tmp_path)
    items = list_backfill_items(db_path, run_id)

    assert result.failed_tickers == 1
    assert items[items["ticker"].eq("AAPL")].iloc[0]["status"] == "completed"
    assert items[items["ticker"].eq("FAKE123")].iloc[0]["status"] == "failed"
    assert retry_failed_items(db_path, run_id) == 1


def test_backfill_expands_cache_without_overwriting_existing_overlap(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path, "SPY")
    _seed_prices(db_path, "QQQ")
    _seed_prices(db_path, "IWM")
    _seed_prices(db_path, "^VIX")

    cached = _price_frame(start="2024-01-08", periods=8, base=100, step=1.0)
    storage.upsert_ohlcv(db_path, "AAPL", cached)
    original_overlap = storage.load_ohlcv(db_path, "AAPL").copy()

    class FakeProvider:
        def download_history(self, ticker: str, period: str = "2y", start=None, end=None) -> pd.DataFrame:
            assert start is not None
            assert end is not None
            return _price_frame(start="2024-01-02", periods=15, base=500, step=10.0)

    monkeypatch.setattr("src.datasets.backfill.market_data.get_provider", lambda name: FakeProvider())

    run_id = create_backfill_run(db_path, ["AAPL"], date(2024, 1, 2), date(2024, 1, 18))
    result = process_backfill_run(db_path, run_id, output_dir=tmp_path)

    after = storage.load_ohlcv(db_path, "AAPL")
    assert result.failed_tickers == 0
    for idx, row in original_overlap.iterrows():
        assert after.loc[idx, "close"] == row["close"]
    assert after.loc[pd.Timestamp("2024-01-02"), "close"] == 500.0
    assert after.loc[pd.Timestamp("2024-01-05"), "close"] == 530.0


def test_backfill_cache_only_mode_does_not_call_provider_for_missing_ranges(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path, "SPY")

    def fail_provider(_name):
        raise AssertionError("provider should not be called in cache-only mode")

    monkeypatch.setattr("src.datasets.backfill.market_data.get_provider", fail_provider)

    run_id = create_backfill_run(
        db_path,
        ["AAPL"],
        date(2024, 1, 10),
        date(2024, 1, 18),
        provider_name=CACHE_ONLY_PROVIDER,
    )
    result = process_backfill_run(db_path, run_id, provider_name=CACHE_ONLY_PROVIDER, output_dir=tmp_path)
    items = list_backfill_items(db_path, run_id)

    assert result.failed_tickers == 1
    assert items.iloc[0]["status"] == "failed"
    assert "no OHLCV data" in str(items.iloc[0]["error"])


def test_backfill_coverage_warnings_use_trading_calendar_dates() -> None:
    weekend_start = _price_frame(start="2024-01-08", periods=3)
    warnings = _coverage_warnings(
        "AAPL",
        weekend_start.set_index("date"),
        weekend_start.set_index("date"),
        date(2024, 1, 6),
        date(2024, 1, 10),
    )
    assert not any("starts after requested start date" in warning for warning in warnings)

    gap = weekend_start[weekend_start["date"].dt.date != date(2024, 1, 9)].set_index("date")
    warnings = _coverage_warnings("AAPL", gap, weekend_start.set_index("date"), date(2024, 1, 8), date(2024, 1, 10))
    assert any("expected trading date(s) missing" in warning for warning in warnings)


def test_backfill_coverage_warnings_skip_exceptional_closure() -> None:
    frame = _price_frame(start="2025-01-08", periods=3)
    # _price_frame uses business days, so drop the exceptional Jan 9 closure and
    # keep Jan 8 / Jan 10 as the surrounding valid sessions.
    frame = frame[frame["date"].dt.date.isin({date(2025, 1, 8), date(2025, 1, 10)})]
    warnings = _coverage_warnings("AAPL", frame.set_index("date"), frame.set_index("date"), date(2025, 1, 8), date(2025, 1, 10))

    assert not any("2025-01-09" in warning for warning in warnings)


def test_backfill_incomplete_label_horizons_and_spy_alignment(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path, "AAPL")
    run_id = create_backfill_run(db_path, ["AAPL"], date(2024, 2, 20), date(2024, 3, 8))
    result = process_backfill_run(db_path, run_id, output_dir=tmp_path)
    items = list_backfill_items(db_path, run_id)

    item = items.iloc[0]
    assert result.completed_tickers == 1
    assert item["completed_labels_20_session"] < item["generated_snapshots"]
    with storage.connect(db_path) as conn:
        dataset_id = conn.execute("SELECT dataset_id FROM backfill_runs WHERE run_id = ?", (run_id,)).fetchone()["dataset_id"]
        spy_labels = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM outcome_labels o
            JOIN feature_snapshots s ON s.snapshot_id = o.snapshot_id
            WHERE s.dataset_id = ? AND o.spy_forward_return IS NOT NULL
            """,
            (dataset_id,),
        ).fetchone()["count"]
    assert spy_labels > 0


def test_sufficiency_report_summarizes_backfill(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path, "AAPL")
    run_id = create_backfill_run(db_path, ["AAPL"], date(2024, 1, 10), date(2024, 1, 18))
    process_backfill_run(db_path, run_id, output_dir=tmp_path)
    with storage.connect(db_path) as conn:
        dataset_id = conn.execute("SELECT dataset_id FROM backfill_runs WHERE run_id = ?", (run_id,)).fetchone()["dataset_id"]

    report = dataset_sufficiency_report(db_path, dataset_id)

    assert report["summary"]["total_rows"] > 0
    assert report["summary"]["tickers"] == 1
    assert not report["label_counts"].empty


def test_dataset_builds_are_deterministic_for_same_inputs(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    kwargs = {
        "tickers": ["AAPL"],
        "start_date": date(2024, 1, 10),
        "end_date": date(2024, 1, 18),
        "output_dir": tmp_path,
    }

    first = build_point_in_time_dataset(db_path, **kwargs)
    second = build_point_in_time_dataset(db_path, **kwargs)

    assert first.build.data_hash == second.build.data_hash
    assert first.dataset_id != second.dataset_id


def test_dataset_hash_ignores_storage_identifiers() -> None:
    from src.datasets.builder import dataset_hash

    first = pd.DataFrame(
        [
            {
                "snapshot_id": 1,
                "dataset_id": 10,
                "ticker": "AAA",
                "trading_date": "2024-01-02",
                "ret_20d": 0.05,
                "label_1_session_forward_return": 0.01,
            }
        ]
    )
    second = first.copy()
    second["snapshot_id"] = 99
    second["dataset_id"] = 11

    assert dataset_hash(first) == dataset_hash(second)


def _snapshot_for_insert(ticker: str, trading_date: date, value: float = 1.0) -> FeatureSnapshot:
    as_of = datetime.combine(trading_date, datetime.min.time(), tzinfo=UTC)
    features = {"ret_20d": value, "above_50d_ma": True}
    return FeatureSnapshot(
        ticker=ticker,
        trading_date=trading_date,
        as_of_timestamp=as_of,
        feature_version="test",
        market_regime={},
        technical={},
        relative_strength={},
        volume_liquidity={},
        catalyst={},
        llm_supported={},
        data_quality={},
        features=features,
    )


def test_batched_snapshot_insert_upserts_without_duplicates(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    storage.init_db(db_path)
    build = build_point_in_time_dataset(db_path, ["FAKE123"], date(2024, 1, 10), date(2024, 1, 18), output_dir=tmp_path).build
    from src.datasets.repository import insert_dataset_build

    dataset_id = insert_dataset_build(db_path, build)
    snapshots = [
        _snapshot_for_insert("AAPL", date(2024, 1, 10), 0.1),
        _snapshot_for_insert("AAPL", date(2024, 1, 11), 0.2),
    ]

    ids = insert_feature_snapshots(db_path, dataset_id, snapshots)
    snapshots[0].features["ret_20d"] = 0.3
    ids_after = insert_feature_snapshots(db_path, dataset_id, snapshots)

    with storage.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM feature_snapshots WHERE dataset_id = ?", (dataset_id,)).fetchone()[0]
        updated = conn.execute(
            "SELECT features_json FROM feature_snapshots WHERE dataset_id = ? AND trading_date = '2024-01-10'",
            (dataset_id,),
        ).fetchone()[0]
    assert count == 2
    assert ids == ids_after
    assert '"ret_20d": 0.3' in updated


def test_batched_snapshot_insert_rolls_back_on_failure(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    storage.init_db(db_path)
    from src.datasets.repository import insert_dataset_build

    build = build_point_in_time_dataset(db_path, ["FAKE123"], date(2024, 1, 10), date(2024, 1, 18), output_dir=tmp_path).build
    dataset_id = insert_dataset_build(db_path, build)
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TRIGGER fail_snapshot_insert
            BEFORE INSERT ON feature_snapshots
            WHEN NEW.ticker = 'FAIL'
            BEGIN
                SELECT RAISE(ABORT, 'forced rollback');
            END;
            """
        )

    with pytest.raises(Exception, match="forced rollback"):
        insert_feature_snapshots(
            db_path,
            dataset_id,
            [
                _snapshot_for_insert("AAPL", date(2024, 1, 10), 0.1),
                _snapshot_for_insert("FAIL", date(2024, 1, 11), 0.2),
            ],
        )

    with storage.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM feature_snapshots WHERE dataset_id = ?", (dataset_id,)).fetchone()[0]
    assert count == 0


def test_streaming_hash_and_csv_match_flattened_dataset(tmp_path) -> None:
    from src.datasets.builder import dataset_hash

    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    result = build_point_in_time_dataset(db_path, ["AAPL"], date(2024, 1, 10), date(2024, 1, 18), output_dir=tmp_path)
    frame = flatten_saved_dataset(db_path, result.dataset_id)
    export_path = tmp_path / "streamed.csv"

    streamed = stream_saved_dataset_export_and_hash(db_path, result.dataset_id, export_path, chunk_size=2)

    assert streamed["data_hash"] == dataset_hash(frame)
    assert export_path.read_text() == frame.to_csv(index=False)
    assert streamed["columns"] == list(frame.columns)


def test_bounded_dataset_preview_does_not_flatten_all_rows(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    result = build_point_in_time_dataset(db_path, ["AAPL"], date(2024, 1, 10), date(2024, 1, 18), output_dir=tmp_path)

    preview = flatten_saved_dataset(db_path, result.dataset_id, limit=2)
    full = flatten_saved_dataset(db_path, result.dataset_id)

    assert len(preview) == 2
    assert len(full) > len(preview)
    assert list(preview.columns) == list(full.columns)


def test_missing_data_fails_gracefully(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    storage.init_db(db_path)

    result = build_point_in_time_dataset(
        db_path,
        ["FAKE123"],
        date(2024, 1, 10),
        date(2024, 1, 18),
        output_dir=tmp_path,
    )

    assert result.dataset_frame.empty
    assert any("no cached OHLCV" in warning for warning in result.warnings)


def test_dataset_build_does_not_change_scanner_score(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    features = {
        "has_data": True,
        "data_quality": "ok",
        "last_price": 50.0,
        "ret_20d": 0.1,
        "ret_60d": 0.2,
        "above_50d_ma": True,
        "above_200d_ma": True,
        "relative_strength_20d": 0.02,
        "relative_strength_60d": 0.03,
        "volume_ratio_20d": 1.2,
        "avg_dollar_volume_20d": 50_000_000,
        "avg_dollar_volume_ok": True,
        "liquidity_score_raw": 1.0,
        "liquidity_label": "Acceptable",
        "distance_20d_ma": 0.03,
        "volatility_20d": 0.2,
    }
    catalyst_features = {"catalyst_score": 0, "catalyst_penalty": 0, "has_catalyst": False}
    before = score_ticker_from_features("AAPL", features, {"regime": "Neutral"}, catalyst_features)

    build_point_in_time_dataset(db_path, ["AAPL"], date(2024, 1, 10), date(2024, 1, 18), output_dir=tmp_path)
    after = score_ticker_from_features("AAPL", features, {"regime": "Neutral"}, catalyst_features)

    assert after["score"] == before["score"]
    assert after["breakdown"] == before["breakdown"]
