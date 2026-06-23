from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime

import pandas as pd

from src.catalysts.models import CatalystEvent
from src.catalysts.proposals import (
    create_proposal_from_extraction,
    get_proposal_by_id,
    set_proposal_status,
    update_proposal,
)
from src.catalysts.publications import (
    build_publication_preview,
    evaluate_publication_eligibility,
    get_publication_by_id,
    list_publications_by_proposal_id,
    publish_proposal,
    revert_publication,
)
from src.catalysts.repository import insert_catalyst, list_catalysts_by_ticker, update_catalyst
from src.data.storage import init_db
from src.documents.repository import build_source_document, insert_document
from src.extractions.repository import approve_extraction, insert_extraction
from src.features.catalyst import get_catalyst_features
from src.scoring.score_engine import score_ticker_from_features


def _features() -> dict:
    return {
        "has_data": True,
        "data_quality": "ok",
        "last_price": 50.0,
        "ret_20d": 0.12,
        "ret_60d": 0.25,
        "above_50d_ma": True,
        "above_200d_ma": True,
        "relative_strength_20d": 0.05,
        "relative_strength_60d": 0.08,
        "volume_ratio_20d": 1.8,
        "avg_dollar_volume_20d": 50_000_000,
        "avg_dollar_volume_ok": True,
        "liquidity_score_raw": 1.0,
        "liquidity_label": "Acceptable",
        "distance_20d_ma": 0.04,
        "volatility_20d": 0.35,
    }


def _document_id(db_path, ticker: str = "AMD") -> int:
    document = build_source_document(
        ticker,
        "news_article",
        "Earnings note",
        "The company reported record revenue and raised guidance for next quarter.",
        source="manual",
        source_url="https://example.com/earnings",
        published_at=datetime.now(UTC).date(),
    )
    return insert_document(db_path, document)


def _payload(document_id: int, ticker: str = "AMD", **overrides) -> dict:
    payload = {
        "document_id": document_id,
        "ticker": ticker,
        "provider": "fallback",
        "model_name": "fallback",
        "extraction_type": "catalyst_analysis",
        "event_type_detected": "earnings",
        "sentiment_label": "positive",
        "catalyst_strength": 7,
        "risk_severity": 1,
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
    payload.update(overrides)
    return payload


def _reviewed_ready_proposal(db_path, ticker: str = "AMD", target_catalyst_id: int | None = None, **payload_overrides) -> int:
    document_id = _document_id(db_path, ticker)
    extraction_id = insert_extraction(db_path, _payload(document_id, ticker, **payload_overrides))
    assert approve_extraction(db_path, extraction_id, "approved")
    result = create_proposal_from_extraction(db_path, extraction_id, target_catalyst_id=target_catalyst_id)
    assert result.changed
    assert set_proposal_status(db_path, result.proposal_id, "reviewed_ready", "ready")
    return result.proposal_id


def _target_catalyst(db_path, ticker: str = "AMD") -> int:
    return insert_catalyst(
        db_path,
        CatalystEvent(
            ticker=ticker,
            event_date=datetime.now(UTC).date(),
            event_type="manual_note",
            title="Original catalyst",
            summary="Original summary",
            source="manual",
            source_url="https://example.com/original",
            sentiment_label="neutral",
            catalyst_strength=1,
            confidence=0.4,
            is_manual=True,
        ),
    )


def test_publication_table_and_migration(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(catalyst_publications)").fetchall()}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(catalyst_publications)").fetchall()}

    assert "publication_id" in columns
    assert "before_snapshot_json" in columns
    assert "catalyst_component_delta" in columns
    assert "idx_catalyst_publications_active_proposal" in indexes


def test_publication_eligibility_rules(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)
    assert evaluate_publication_eligibility(db_path, proposal_id).eligible

    draft_id = _reviewed_ready_proposal(db_path)
    assert set_proposal_status(db_path, draft_id, "draft", "")
    draft = evaluate_publication_eligibility(db_path, draft_id)
    assert not draft.eligible
    assert any("reviewed_ready" in reason for reason in draft.reasons)

    weak_id = _reviewed_ready_proposal(db_path, document_relevance="uncertain")
    weak = evaluate_publication_eligibility(db_path, weak_id)
    assert not weak.eligible
    assert any("Document relevance" in reason for reason in weak.reasons)

    no_evidence_id = _reviewed_ready_proposal(db_path)
    assert update_proposal(db_path, no_evidence_id, {"evidence_snippets": [], "proposed_sentiment": "positive"})
    no_evidence = evaluate_publication_eligibility(db_path, no_evidence_id)
    assert not no_evidence.eligible
    assert any("evidence" in reason.lower() for reason in no_evidence.reasons)


def test_publication_preview_uses_production_catalyst_scoring(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)

    preview = build_publication_preview(db_path, proposal_id)
    production_features = get_catalyst_features("AMD", pd.DataFrame([preview.after_snapshot]))

    assert preview.eligible
    assert preview.catalyst_component_before == 0
    assert preview.catalyst_component_after == 5.6
    assert preview.catalyst_component_after == production_features["catalyst_score"]
    assert preview.field_diff["before"].map(lambda value: isinstance(value, str)).all()
    assert preview.field_diff["after"].map(lambda value: isinstance(value, str)).all()


def test_create_new_publication_with_complete_provenance(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)

    result = publish_proposal(db_path, proposal_id, "publish reviewed catalyst")
    catalysts = list_catalysts_by_ticker(db_path, "AMD")
    publication = get_publication_by_id(db_path, result.publication_id)
    catalyst = catalysts.iloc[0].to_dict()
    payload = json.loads(catalyst["raw_payload_json"])

    assert result.changed
    assert len(catalysts) == 1
    assert catalyst["source"] == "llm_supported"
    assert payload["llm_supported"] is True
    assert payload["latest_publication_id"] == result.publication_id
    latest = payload["llm_publication_history"][-1]
    assert latest["publication_id"] == result.publication_id
    assert latest["proposal_id"] == proposal_id
    assert latest["extraction_id"] == publication["extraction_id"]
    assert latest["document_id"] == publication["document_id"]
    assert "record revenue" in latest["evidence_snippets"]


def test_update_existing_selected_fields_only(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    catalyst_id = _target_catalyst(db_path)
    proposal_id = _reviewed_ready_proposal(db_path, target_catalyst_id=catalyst_id)

    result = publish_proposal(db_path, proposal_id, "update selected fields", selected_update_fields=["title", "sentiment_label"])
    catalyst = list_catalysts_by_ticker(db_path, "AMD").iloc[0].to_dict()

    assert result.changed
    assert catalyst["id"] == catalyst_id
    assert catalyst["title"] != "Original catalyst"
    assert catalyst["sentiment_label"] == "positive"
    assert str(catalyst["event_date"]) == str(datetime.now(UTC).date())
    assert catalyst["source_url"] == "https://example.com/original"
    assert catalyst["summary"] == "Original summary"


def test_publication_transaction_rolls_back_on_failure(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)

    result = publish_proposal(db_path, proposal_id, "simulate failure", _simulate_failure_after_mutation=True)

    assert result.changed is False
    assert list_catalysts_by_ticker(db_path, "AMD").empty
    assert list_publications_by_proposal_id(db_path, proposal_id).empty


def test_duplicate_publication_prevention_and_no_double_counting(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)

    first = publish_proposal(db_path, proposal_id, "first publish")
    second = publish_proposal(db_path, proposal_id, "duplicate publish")
    catalysts = list_catalysts_by_ticker(db_path, "AMD")
    features = get_catalyst_features("AMD", catalysts)

    assert first.changed is True
    assert second.changed is False
    assert len(catalysts) == 1
    assert features["catalyst_score"] == 5.6


def test_limited_evidence_confidence_cap_and_no_direct_score_effect(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path, evidence_sufficiency="limited", confidence=0.95)
    update_proposal(db_path, proposal_id, {"proposed_confidence": 0.95, "evidence_sufficiency": "limited"})

    result = publish_proposal(db_path, proposal_id, "limited evidence publish")
    catalysts = list_catalysts_by_ticker(db_path, "AMD")
    catalyst = catalysts.iloc[0].to_dict()
    catalyst_features = get_catalyst_features("AMD", catalysts)
    score = score_ticker_from_features("AMD", _features(), {"regime": "Risk-On"}, catalyst_features)

    assert result.changed
    assert catalyst["confidence"] == 0.6
    assert catalyst_features["catalyst_score"] == 4.2
    assert score["breakdown"]["catalyst"] == 4.2
    assert score["breakdown"]["catalyst"] != 10


def test_pending_and_proposal_rows_remain_zero_until_publication(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)
    proposal = get_proposal_by_id(db_path, proposal_id)

    before_features = get_catalyst_features("AMD", list_catalysts_by_ticker(db_path, "AMD"))
    assert before_features["catalyst_score"] == 0
    assert proposal["proposal_status"] == "reviewed_ready"

    publish_proposal(db_path, proposal_id, "publish")
    after_features = get_catalyst_features("AMD", list_catalysts_by_ticker(db_path, "AMD"))
    assert after_features["catalyst_score"] == 5.6


def test_successful_reversal_removes_created_catalyst_and_scoring_effect(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)
    result = publish_proposal(db_path, proposal_id, "publish")

    reverted = revert_publication(db_path, result.publication_id, "undo publication")
    publication = get_publication_by_id(db_path, result.publication_id)
    features = get_catalyst_features("AMD", list_catalysts_by_ticker(db_path, "AMD"))

    assert reverted.changed
    assert publication["publication_status"] == "reverted"
    assert list_catalysts_by_ticker(db_path, "AMD").empty
    assert features["catalyst_score"] == 0


def test_reversal_conflict_detection(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)
    result = publish_proposal(db_path, proposal_id, "publish")

    assert update_catalyst(db_path, result.catalyst_id, {"title": "Manual edit after publication"})
    reverted = revert_publication(db_path, result.publication_id, "try revert")

    assert reverted.changed is False
    assert "changed after publication" in reverted.message


def test_alerts_ignore_unpublished_and_reverted_records(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    proposal_id = _reviewed_ready_proposal(db_path)
    features_before = get_catalyst_features("AMD", list_catalysts_by_ticker(db_path, "AMD"))

    result = publish_proposal(db_path, proposal_id, "publish")
    features_after_publish = get_catalyst_features("AMD", list_catalysts_by_ticker(db_path, "AMD"))
    revert_publication(db_path, result.publication_id, "remove from active catalysts")
    features_after_revert = get_catalyst_features("AMD", list_catalysts_by_ticker(db_path, "AMD"))

    assert features_before["catalyst_score"] == 0
    assert features_after_publish["catalyst_score"] == 5.6
    assert features_after_revert["catalyst_score"] == 0
