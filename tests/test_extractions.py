from __future__ import annotations

import sqlite3

import pandas as pd

from src.data.storage import init_db
from src.documents.repository import build_source_document, insert_document, list_documents_by_ticker
from src.extractions.fallback_extractor import run_fallback_extraction
from src.extractions.quality import approval_requirements_met, classify_review_readiness
from src.extractions.repository import (
    approve_extraction,
    delete_extraction,
    get_extraction_by_id,
    insert_extraction,
    list_recent_extractions,
    list_extractions_by_document_id,
    list_extractions_by_ticker,
    list_pending_review_extractions,
    list_reviewed_extractions,
    reject_extraction,
    supersede_extraction,
)
from src.extractions.review_workflow import (
    approve_extraction_with_readiness,
    create_fallback_extraction_for_document,
    document_readiness,
    enrich_extractions_with_documents,
    filter_extractions,
)
from src.extractions.validation import extraction_from_payload, normalize_extraction_payload
from src.scoring.score_engine import score_ticker_from_features


def sample_payload(document_id: int = 1) -> dict:
    return {
        "document_id": document_id,
        "ticker": "aapl",
        "provider": "fallback",
        "extraction_type": "general_document_review",
        "event_type_detected": "unknown",
        "sentiment_label": "unknown",
        "catalyst_strength": 0,
        "risk_severity": 0,
        "confidence": 0.2,
        "time_horizon": "unknown",
        "key_positive_points": ["No strong positive keyword found."],
        "key_risks": [],
        "evidence_snippets": [],
        "short_summary": "Neutral fallback extraction.",
        "detailed_summary": "No real LLM analysis.",
        "proposed_score_effect": 0,
        "review_status": "approved",
    }


def test_extraction_table_insert_list_get_delete(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = insert_extraction(db_path, sample_payload())

    stored = get_extraction_by_id(db_path, extraction_id)
    by_ticker = list_extractions_by_ticker(db_path, "AAPL")
    by_document = list_extractions_by_document_id(db_path, 1)

    assert extraction_id > 0
    assert stored is not None
    assert stored["ticker"] == "AAPL"
    assert stored["review_status"] == "pending_review"
    assert stored["document_relevance"] == "unknown"
    assert stored["evidence_sufficiency"] == "unknown"
    assert stored["key_positive_points"] == ["No strong positive keyword found."]
    assert len(by_ticker) == 1
    assert len(by_document) == 1
    assert delete_extraction(db_path, extraction_id) is True
    assert get_extraction_by_id(db_path, extraction_id) is None


def test_extraction_defaults_to_pending_review(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = insert_extraction(db_path, sample_payload())
    pending = list_pending_review_extractions(db_path)

    assert get_extraction_by_id(db_path, extraction_id)["review_status"] == "pending_review"
    assert len(pending) == 1


def test_approve_reject_and_supersede_extraction(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    approve_id = insert_extraction(db_path, sample_payload())

    assert approve_extraction(db_path, approve_id, "looks reasonable") is True
    approved = get_extraction_by_id(db_path, approve_id)
    assert approved["review_status"] == "approved"
    assert approved["reviewer_note"] == "looks reasonable"
    assert approved["reviewed_at"] is not None

    reject_id = insert_extraction(db_path, sample_payload())
    assert reject_extraction(db_path, reject_id, "not useful") is True
    rejected = get_extraction_by_id(db_path, reject_id)
    assert rejected["review_status"] == "rejected"
    assert rejected["reviewer_note"] == "not useful"
    assert rejected["reviewed_at"] is not None

    supersede_id = insert_extraction(db_path, sample_payload())
    assert supersede_extraction(db_path, supersede_id, "newer extraction exists") is True
    superseded = get_extraction_by_id(db_path, supersede_id)
    assert superseded["review_status"] == "superseded"
    assert superseded["reviewer_note"] == "newer extraction exists"
    assert superseded["reviewed_at"] is not None


def test_reviewed_extraction_cannot_be_reviewed_twice(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = insert_extraction(db_path, sample_payload())

    assert approve_extraction(db_path, extraction_id, "approved first") is True
    assert reject_extraction(db_path, extraction_id, "stale reject") is False
    assert supersede_extraction(db_path, extraction_id, "stale supersede") is False

    stored = get_extraction_by_id(db_path, extraction_id)
    assert stored["review_status"] == "approved"
    assert stored["reviewer_note"] == "approved first"


def test_schema_validation_clamps_and_defaults() -> None:
    payload = {
        "document_id": "5",
        "ticker": " nvda ",
        "provider": "bad_provider",
        "sentiment_label": "bad_sentiment",
        "event_type_detected": "bad_event",
        "catalyst_strength": 99,
        "risk_severity": -4,
        "confidence": 2.5,
        "proposed_score_effect": -99,
        "key_positive_points": '["record revenue"]',
        "key_risks": {"unexpected": "dict"},
        "document_relevance": "bad_relevance",
        "evidence_sufficiency": "bad_sufficiency",
        "review_status": "bad_status",
    }

    normalized = normalize_extraction_payload(payload)
    extraction = extraction_from_payload(payload)

    assert normalized["ticker"] == "NVDA"
    assert normalized["provider"] == "fallback"
    assert normalized["sentiment_label"] == "unknown"
    assert normalized["event_type_detected"] == "unknown"
    assert normalized["catalyst_strength"] == 10
    assert normalized["risk_severity"] == 0
    assert normalized["confidence"] == 1.0
    assert normalized["document_relevance"] == "unknown"
    assert normalized["evidence_sufficiency"] == "unknown"
    assert normalized["proposed_score_effect"] == -15
    assert normalized["review_status"] == "pending_review"
    assert extraction.key_positive_points == ["record revenue"]
    assert extraction.key_risks


def test_malformed_payload_handling_does_not_crash() -> None:
    extraction = extraction_from_payload({"document_id": "nope", "catalyst_id": "bad", "key_risks": "{bad json"})

    assert extraction.document_id == 0
    assert extraction.catalyst_id is None
    assert extraction.ticker == "UNKNOWN"
    assert extraction.review_status == "pending_review"
    assert extraction.key_risks == ["{bad json"]


def test_fallback_extractor_detects_risk_and_positive_keywords(tmp_path) -> None:
    document = build_source_document(
        "AMD",
        "sec_filing",
        "SEC excerpt",
        "The company raises guidance after record revenue, but also announced an offering with dilution risk.",
        source="manual",
    )
    document_id = insert_document(tmp_path / "alpha_lab.db", document)
    document = document.model_copy(update={"document_id": document_id})

    extraction = run_fallback_extraction(document)

    assert extraction.provider == "fallback"
    assert extraction.review_status == "pending_review"
    assert extraction.sentiment_label == "mixed"
    assert extraction.catalyst_strength > 0
    assert extraction.risk_severity > 0
    assert extraction.document_relevance == "uncertain"
    assert extraction.evidence_sufficiency == "limited"
    assert extraction.confidence <= 0.39
    assert extraction.proposed_score_effect <= 10
    assert "Fallback keyword extraction only" in extraction.extraction_warnings


def test_ui_workflow_created_fallback_extraction_stays_pending_and_can_supersede(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    document = build_source_document(
        "AMD",
        "news_article",
        "News note",
        "The company beat expectations and reported record revenue.",
        source="manual",
    )
    document_id = insert_document(db_path, document)

    first = create_fallback_extraction_for_document(db_path, document_id, "news_review")
    assert first.extraction_id is not None
    assert get_extraction_by_id(db_path, first.extraction_id)["review_status"] == "pending_review"

    blocked = create_fallback_extraction_for_document(db_path, document_id, "news_review")
    assert blocked.blocked is True
    assert blocked.extraction_id is None

    second = create_fallback_extraction_for_document(db_path, document_id, "news_review", supersede_existing=True)
    assert second.extraction_id is not None
    assert first.extraction_id in second.superseded_ids
    assert get_extraction_by_id(db_path, first.extraction_id)["review_status"] == "superseded"
    assert get_extraction_by_id(db_path, second.extraction_id)["review_status"] == "pending_review"


def test_fallback_extractor_neutral_without_keywords() -> None:
    document = build_source_document(
        "MSFT",
        "manual_text",
        "Neutral note",
        "This is a general background document with no clear catalyst language.",
        source="manual",
    )

    extraction = run_fallback_extraction(document)

    assert extraction.sentiment_label == "unknown"
    assert extraction.catalyst_strength == 0
    assert extraction.risk_severity == 0
    assert extraction.confidence < 0.2
    assert extraction.evidence_sufficiency == "insufficient"
    assert extraction.proposed_score_effect == 0
    assert extraction.evidence_snippets == []


def test_backward_compatible_sqlite_migration_adds_quality_fields(tmp_path) -> None:
    db_path = tmp_path / "legacy.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE llm_extractions (
                extraction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                catalyst_id INTEGER,
                ticker TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_name TEXT,
                extraction_type TEXT NOT NULL,
                event_type_detected TEXT NOT NULL,
                sentiment_label TEXT NOT NULL,
                catalyst_strength INTEGER NOT NULL,
                risk_severity INTEGER NOT NULL,
                confidence REAL NOT NULL,
                time_horizon TEXT NOT NULL,
                key_positive_points TEXT NOT NULL,
                key_risks TEXT NOT NULL,
                evidence_snippets TEXT NOT NULL,
                short_summary TEXT,
                detailed_summary TEXT,
                proposed_score_effect INTEGER NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'pending_review',
                reviewer_note TEXT,
                reviewed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                raw_llm_response_json TEXT,
                prompt_version TEXT,
                extraction_warnings TEXT
            )
            """
        )

    init_db(db_path)
    extraction_id = insert_extraction(db_path, sample_payload())
    stored = get_extraction_by_id(db_path, extraction_id)

    assert stored["document_relevance"] == "unknown"
    assert stored["evidence_sufficiency"] == "unknown"


def test_approval_override_requires_checkbox_and_note(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    extraction_id = insert_extraction(
        db_path,
        {
            **sample_payload(),
            "proposed_score_effect": 5,
            "evidence_snippets": [],
        },
    )
    extraction = get_extraction_by_id(db_path, extraction_id)

    assert classify_review_readiness(extraction) == "needs_evidence"
    assert approval_requirements_met(extraction, "", override_not_ready=False)[0] is False
    assert approval_requirements_met(extraction, "", override_not_ready=True)[0] is False

    blocked = approve_extraction_with_readiness(db_path, extraction_id, "", override_not_ready=True)
    assert blocked.changed is False
    assert get_extraction_by_id(db_path, extraction_id)["review_status"] == "pending_review"

    approved = approve_extraction_with_readiness(
        db_path,
        extraction_id,
        "Reviewer accepts missing evidence for tracking only.",
        override_not_ready=True,
    )
    assert approved.changed is True
    assert get_extraction_by_id(db_path, extraction_id)["review_status"] == "approved"


def test_missing_and_empty_document_handling(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    missing = document_readiness(None)
    assert missing.can_run is False
    assert "missing" in missing.warnings[0]

    empty_document = build_source_document("MSFT", "manual_text", "Empty", "", source="manual")
    document_id = insert_document(db_path, empty_document)
    result = create_fallback_extraction_for_document(db_path, document_id)

    assert result.blocked is True
    assert result.extraction_id is None
    assert result.warnings


def test_queue_and_history_filtering_helpers(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    doc_a = build_source_document("AAPL", "manual_text", "AAPL doc", "record revenue and beat expectations", source="manual")
    doc_b = build_source_document("MSFT", "manual_text", "MSFT doc", "offering and dilution risk", source="manual")
    doc_a_id = insert_document(db_path, doc_a)
    doc_b_id = insert_document(db_path, doc_b)
    a_id = create_fallback_extraction_for_document(db_path, doc_a_id, "catalyst_analysis").extraction_id
    b_id = create_fallback_extraction_for_document(db_path, doc_b_id, "risk_analysis").extraction_id
    approve_extraction(db_path, a_id, "approved")
    reject_extraction(db_path, b_id, "rejected")

    docs = list_extractions_by_document_id(db_path, doc_a_id)
    all_extractions = list_recent_extractions(db_path)
    reviewed = list_reviewed_extractions(db_path)
    document_rows = pd.concat(
        [list_documents_by_ticker(db_path, "AAPL"), list_documents_by_ticker(db_path, "MSFT")],
        ignore_index=True,
    )
    enriched = enrich_extractions_with_documents(all_extractions, document_rows)

    assert len(docs) == 1
    assert len(reviewed) == 2
    assert len(filter_extractions(all_extractions, ticker="AAPL")) == 1
    assert len(filter_extractions(all_extractions, statuses=["approved"])) == 1
    assert len(filter_extractions(all_extractions, extraction_types=["risk_analysis"])) == 1
    assert "document_title" in enriched.columns


def base_features() -> dict:
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


def test_scanner_score_unchanged_after_extraction_approval(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    features = base_features()
    catalyst_features = {"catalyst_score": 0, "catalyst_penalty": 0, "catalyst_warnings": []}
    before = score_ticker_from_features("AAPL", features, {"regime": "Risk-On"}, catalyst_features)

    document = build_source_document("AAPL", "manual_text", "Review doc", "record revenue and beat expectations", source="manual")
    document_id = insert_document(db_path, document)
    extraction_id = create_fallback_extraction_for_document(db_path, document_id).extraction_id
    approve_extraction(db_path, extraction_id, "approved but not scoring")

    after = score_ticker_from_features("AAPL", features, {"regime": "Risk-On"}, catalyst_features)
    assert after["score"] == before["score"]
    assert after["breakdown"] == before["breakdown"]


def test_app_storage_initializes_without_api_keys(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    db_path = tmp_path / "alpha_lab.db"

    init_db(db_path)
    extraction_id = insert_extraction(db_path, sample_payload())

    assert extraction_id > 0
