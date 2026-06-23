from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd

from src.catalysts.models import CatalystEvent
from src.catalysts.proposals import (
    create_proposal_from_extraction,
    get_proposal_by_id,
    link_extraction_to_catalyst,
    list_links_by_catalyst_id,
    list_proposals_by_extraction_id,
    list_proposals_by_ticker,
    map_extraction_to_proposal,
    proposal_display_frame,
    proposal_score_contribution,
    proposal_summary_by_catalyst,
    set_proposal_status,
    unlink_extraction_catalyst_link,
    update_proposal,
)
from src.catalysts.repository import insert_catalyst, list_catalysts_by_ticker
from src.data.storage import init_db
from src.documents.repository import build_source_document, insert_document
from src.extractions.quality import classify_review_readiness
from src.extractions.repository import approve_extraction, get_extraction_by_id, insert_extraction, reject_extraction
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
        "Quarterly update",
        "The company reported record revenue and raised guidance for next quarter.",
        source="manual",
        source_url="https://example.com/update",
        published_at="2026-06-01",
    )
    return insert_document(db_path, document)


def _extraction_payload(document_id: int, ticker: str = "AMD", **overrides) -> dict:
    payload = {
        "document_id": document_id,
        "ticker": ticker,
        "provider": "fallback",
        "extraction_type": "catalyst_analysis",
        "event_type_detected": "earnings",
        "sentiment_label": "positive",
        "catalyst_strength": 7,
        "risk_severity": 1,
        "confidence": 0.78,
        "document_relevance": "relevant",
        "evidence_sufficiency": "sufficient",
        "time_horizon": "short_term",
        "key_positive_points": ["Record revenue"],
        "key_risks": [],
        "evidence_snippets": ["record revenue"],
        "short_summary": "Record revenue and raised guidance.",
        "detailed_summary": "The supplied document directly states record revenue and raised guidance.",
        "proposed_score_effect": 6,
        "review_status": "pending_review",
    }
    payload.update(overrides)
    return payload


def _approved_extraction(db_path, ticker: str = "AMD", **overrides) -> int:
    document_id = _document_id(db_path, ticker)
    extraction_id = insert_extraction(db_path, _extraction_payload(document_id, ticker, **overrides))
    assert approve_extraction(db_path, extraction_id, "approved for proposal testing")
    return extraction_id


def _catalyst_id(db_path, ticker: str = "AMD") -> int:
    return insert_catalyst(
        db_path,
        CatalystEvent(
            ticker=ticker,
            event_date=datetime.now(UTC).date(),
            event_type="manual_note",
            title="Existing catalyst",
            summary="Original active catalyst summary",
            source="manual",
            sentiment_label="neutral",
            catalyst_strength=2,
            confidence=0.5,
            is_manual=True,
        ),
    )


def test_proposal_table_creation_and_migration(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        proposal_columns = {row[1] for row in conn.execute("PRAGMA table_info(catalyst_proposals)").fetchall()}
        link_columns = {row[1] for row in conn.execute("PRAGMA table_info(extraction_catalyst_links)").fetchall()}

    assert "proposal_id" in proposal_columns
    assert "extraction_id" in proposal_columns
    assert "initiated_by" in proposal_columns
    assert "link_id" in link_columns
    assert "unlinked_at" in link_columns


def test_deterministic_extraction_to_proposal_mapping(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = _approved_extraction(db_path)
    extraction = get_extraction_by_id(db_path, extraction_id)
    document = {"title": "Quarterly update", "published_at": "2026-06-01", "source": "manual", "source_url": "https://example.com/update"}

    proposal = map_extraction_to_proposal(extraction, document)

    assert proposal.ticker == "AMD"
    assert proposal.extraction_id == extraction_id
    assert proposal.proposed_event_type == "earnings"
    assert proposal.proposed_event_date.isoformat() == "2026-06-01"
    assert proposal.proposed_sentiment == "positive"
    assert proposal.evidence_snippets == ["record revenue"]


def test_pending_and_rejected_extractions_cannot_create_proposals(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    document_id = _document_id(db_path)
    pending_id = insert_extraction(db_path, _extraction_payload(document_id))

    pending_result = create_proposal_from_extraction(db_path, pending_id)
    assert pending_result.changed is False
    assert list_proposals_by_extraction_id(db_path, pending_id).empty

    rejected_id = insert_extraction(db_path, _extraction_payload(document_id))
    assert reject_extraction(db_path, rejected_id, "not useful")
    rejected_result = create_proposal_from_extraction(db_path, rejected_id)
    assert rejected_result.changed is False
    assert list_proposals_by_extraction_id(db_path, rejected_id).empty


def test_approved_extraction_can_create_proposal(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = _approved_extraction(db_path)

    result = create_proposal_from_extraction(db_path, extraction_id)
    stored = get_proposal_by_id(db_path, result.proposal_id)

    assert result.changed is True
    assert stored["proposal_status"] == "draft"
    assert stored["ticker"] == "AMD"
    assert stored["evidence_snippets"] == ["record revenue"]


def test_weak_readiness_proposal_requires_override_and_note(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = _approved_extraction(
        db_path,
        evidence_snippets=[],
        proposed_score_effect=5,
        evidence_sufficiency="limited",
    )
    extraction = get_extraction_by_id(db_path, extraction_id)
    assert classify_review_readiness(extraction) == "needs_evidence"

    blocked = create_proposal_from_extraction(db_path, extraction_id)
    no_note = create_proposal_from_extraction(db_path, extraction_id, override_weak_readiness=True)
    allowed = create_proposal_from_extraction(
        db_path,
        extraction_id,
        reviewer_note="Reviewer accepts this as a proposal draft only.",
        override_weak_readiness=True,
    )

    assert blocked.changed is False
    assert no_note.changed is False
    assert allowed.changed is True


def test_proposal_editing_and_status_transitions(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = _approved_extraction(db_path)
    proposal_id = create_proposal_from_extraction(db_path, extraction_id).proposal_id
    rejected_id = create_proposal_from_extraction(db_path, extraction_id).proposal_id
    superseded_id = create_proposal_from_extraction(db_path, extraction_id).proposal_id

    assert update_proposal(db_path, proposal_id, {"proposed_title": "Edited title", "proposed_strength": 4})
    edited = get_proposal_by_id(db_path, proposal_id)
    assert edited["proposed_title"] == "Edited title"
    assert edited["proposed_strength"] == 4

    assert set_proposal_status(db_path, proposal_id, "reviewed_ready", "ready for future controlled publication")
    reviewed = get_proposal_by_id(db_path, proposal_id)
    assert reviewed["proposal_status"] == "reviewed_ready"
    assert reviewed["reviewed_at"] is not None
    assert set_proposal_status(db_path, rejected_id, "rejected", "not suitable")
    assert get_proposal_by_id(db_path, rejected_id)["proposal_status"] == "rejected"
    assert set_proposal_status(db_path, superseded_id, "superseded", "newer proposal exists")
    assert get_proposal_by_id(db_path, superseded_id)["proposal_status"] == "superseded"


def test_link_unlink_existing_catalyst_and_preserve_active_fields(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    catalyst_id = _catalyst_id(db_path)
    extraction_id = _approved_extraction(db_path)
    before = list_catalysts_by_ticker(db_path, "AMD").iloc[0].to_dict()

    link_result = link_extraction_to_catalyst(db_path, extraction_id, catalyst_id, "links evidence to active event")
    links = list_links_by_catalyst_id(db_path, catalyst_id)
    after_link = list_catalysts_by_ticker(db_path, "AMD").iloc[0].to_dict()
    unlink_result = unlink_extraction_catalyst_link(db_path, link_result.link_id, "remove audit link")
    links_after = list_links_by_catalyst_id(db_path, catalyst_id)
    after_unlink = list_catalysts_by_ticker(db_path, "AMD").iloc[0].to_dict()

    assert link_result.changed is True
    assert len(links) == 1
    assert links.iloc[0]["link_status"] == "active"
    assert unlink_result.changed is True
    assert links_after.iloc[0]["link_status"] == "unlinked"
    for field in ["title", "summary", "sentiment_label", "catalyst_strength", "confidence", "event_date"]:
        assert after_link[field] == before[field]
        assert after_unlink[field] == before[field]


def test_proposals_stay_out_of_active_catalyst_queries_and_scoring(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    catalyst_id = _catalyst_id(db_path)
    extraction_id = _approved_extraction(db_path)
    create_proposal_from_extraction(db_path, extraction_id, target_catalyst_id=catalyst_id)

    active = list_catalysts_by_ticker(db_path, "AMD")
    proposals = list_proposals_by_ticker(db_path, "AMD")
    summary = proposal_summary_by_catalyst(db_path, [catalyst_id])
    before = score_ticker_from_features(
        "AMD",
        _features(),
        {"regime": "Risk-On"},
        {"catalyst_score": 0, "catalyst_penalty": 0, "catalyst_warnings": []},
    )
    after = score_ticker_from_features(
        "AMD",
        _features(),
        {"regime": "Risk-On"},
        {"catalyst_score": 0, "catalyst_penalty": 0, "catalyst_warnings": []},
    )

    assert len(active) == 1
    assert len(proposals) == 1
    assert summary[catalyst_id]["proposal_count"] == 1
    assert proposal_score_contribution() == 0
    assert after["score"] == before["score"]


def test_ticker_research_and_debug_helpers_distinguish_proposals(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = _approved_extraction(db_path)
    proposal_id = create_proposal_from_extraction(db_path, extraction_id).proposal_id
    proposals = list_proposals_by_ticker(db_path, "AMD")

    display = proposal_display_frame(proposals)

    assert proposal_id in proposals["proposal_id"].tolist()
    assert "proposal_status" in display.columns
    assert "proposed_title" in display.columns
    assert proposal_score_contribution() == 0
