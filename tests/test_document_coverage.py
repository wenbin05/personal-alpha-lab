from __future__ import annotations

import socket
import sqlite3
from datetime import UTC, date, datetime, timedelta

from src.annotations.document_coverage import (
    COVERAGE_STATUSES,
    QUEUE_COLUMNS,
    build_document_coverage_audit,
    enrichment_priority_score,
)
from src.annotations.models import ResearchEventAnnotation
from src.annotations.news_events import ResearchEventAnnotationCandidate
from src.annotations.news_repository import stage_candidate
from src.annotations.repository import insert_annotation
from src.data import storage
from src.documents.repository import build_source_document, insert_document
from src.quality.harness import check_document_coverage


PROVIDER = "company_ir_press_release"
REVIEWABLE_TEXT = "This company investor-relations release contains enough exact source text for manual review."


def _candidate(db_path, index: int, *, status: str = "staged", **overrides) -> int:
    event_date = date(2024, 1, 2) + timedelta(days=index)
    candidate = ResearchEventAnnotationCandidate(
        ticker=overrides.get("ticker", "AAA"),
        event_date=event_date,
        available_at=datetime(2024, 1, 2, 14, tzinfo=UTC) + timedelta(days=index),
        event_type=overrides.get("event_type", "news"),
        title=overrides.get("title", f"Candidate {index}"),
        source="company_ir",
        source_url=f"https://example.com/ir/{index}",
        evidence_text=f"Evidence for candidate {index}.",
        sentiment_label=overrides.get("sentiment", "neutral"),
        strength=overrides.get("strength", 5),
        confidence=0.8,
        tags=[f"informativeness:{overrides.get('informativeness', 'material_medium')}", "source_quality:official_company"],
        provider=PROVIDER,
        provider_metadata={
            "provider_event_id": f"event-{index}",
            "review_note": f"Review candidate {index} manually.",
            "source_quality": "official_company",
            "informativeness": overrides.get("informativeness", "material_medium"),
            "network_calls_would_occur": False,
        },
        document_type=PROVIDER,
        published_at=datetime(2024, 1, 2, 13, tzinfo=UTC) + timedelta(days=index),
    )
    result = stage_candidate(db_path, candidate)
    with storage.connect(db_path) as connection:
        connection.execute(
            "UPDATE research_event_candidates SET status = ? WHERE candidate_id = ?",
            (status, result.candidate_id),
        )
    return result.candidate_id


def _link_candidate(db_path, candidate_id: int, document_id: int) -> None:
    with storage.connect(db_path) as connection:
        connection.execute(
            "UPDATE research_event_candidates SET source_document_id = ? WHERE candidate_id = ?",
            (document_id, candidate_id),
        )


def _seed_all_statuses(db_path) -> list[int]:
    storage.init_db(db_path)
    complete_document = insert_document(
        db_path,
        build_source_document("AAA", PROVIDER, "Complete", REVIEWABLE_TEXT, source=PROVIDER),
    )
    partial_document = insert_document(
        db_path,
        build_source_document(
            "AAA",
            PROVIDER,
            "Partial",
            REVIEWABLE_TEXT + " Partial.",
            source=PROVIDER,
            parsing_status="partial",
        ),
    )
    missing_text_document = insert_document(
        db_path,
        build_source_document(
            "AAA",
            PROVIDER,
            "Missing text",
            "",
            source=PROVIDER,
            source_url="https://example.com/document/missing-text",
        ),
    )
    reused_document = insert_document(
        db_path,
        build_source_document("AAA", PROVIDER, "Reused", REVIEWABLE_TEXT + " Reused.", source=PROVIDER),
    )

    complete = _candidate(
        db_path,
        1,
        status="imported",
        event_type="legal_regulatory",
        sentiment="negative",
        informativeness="material_high",
    )
    partial = _candidate(db_path, 2, status="accepted", sentiment="mixed", informativeness="material_high")
    missing_text = _candidate(db_path, 3, status="accepted")
    missing = _candidate(db_path, 4, status="rejected", informativeness="routine_low")
    broken = _candidate(db_path, 5, status="accepted", event_type="financing", sentiment="negative", informativeness="material_high")
    reused_a = _candidate(db_path, 6, status="imported")
    reused_b = _candidate(db_path, 7, status="staged")

    annotation = insert_annotation(
        db_path,
        ResearchEventAnnotation(
            ticker="AAA",
            event_date=date(2024, 1, 3),
            available_at=datetime(2024, 1, 3, 14, tzinfo=UTC),
            event_type="legal_regulatory",
            sentiment_label="negative",
            strength=8,
            confidence=0.9,
            source="company_ir",
            source_url="https://example.com/annotation/complete",
            title="Imported annotation",
            source_document_id=complete_document,
        ),
    )
    with storage.connect(db_path) as connection:
        connection.execute(
            "UPDATE research_event_candidates SET imported_annotation_id = ? WHERE candidate_id = ?",
            (annotation.annotation_id, complete),
        )

    _link_candidate(db_path, complete, complete_document)
    _link_candidate(db_path, partial, partial_document)
    _link_candidate(db_path, missing_text, missing_text_document)
    _link_candidate(db_path, broken, 999_999)
    _link_candidate(db_path, reused_a, reused_document)
    _link_candidate(db_path, reused_b, reused_document)
    return [complete, partial, missing_text, missing, broken, reused_a, reused_b]


def test_document_coverage_classifies_every_status_and_detects_reuse(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_all_statuses(db_path)

    audit = build_document_coverage_audit(db_path)
    statuses = set(audit.rows["coverage_status"])

    assert statuses == set(COVERAGE_STATUSES)
    assert audit.summary["total_company_ir_candidates"] == 7
    assert audit.summary["candidates_with_annotations"] == 1
    assert audit.summary["candidates_with_linked_documents"] == 5
    assert audit.summary["complete_documents"] == 3
    assert audit.summary["partial_documents"] == 1
    assert audit.summary["missing_text_documents"] == 1
    assert audit.summary["missing_documents"] == 1
    assert audit.summary["broken_linkages"] == 1
    assert audit.summary["reused_documents"] == 1
    assert audit.summary["reused_document_candidates"] == 2
    assert set(audit.rows.loc[audit.rows["document_reused"], "coverage_status"]) == {"duplicate_document_reused"}


def test_enrichment_queue_fields_and_priority_are_deterministic(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_all_statuses(db_path)

    first = build_document_coverage_audit(db_path)
    second = build_document_coverage_audit(db_path)

    assert tuple(first.queue.columns) == QUEUE_COLUMNS
    assert len(first.queue) == 4
    assert first.queue["candidate_id"].tolist() == second.queue["candidate_id"].tolist()
    assert first.queue.iloc[0]["coverage_status"] == "broken_linkage"
    assert first.queue.iloc[0]["enrichment_priority"] > first.queue.iloc[1]["enrichment_priority"]
    assert first.queue["raw_text"].fillna("").eq("").all()
    assert first.queue["cleaned_text"].fillna("").eq("").all()
    assert first.queue["provider_event_id"].str.startswith("event-").all()

    imported = enrichment_priority_score(
        {
            "document_quality_status": "missing_document",
            "informativeness": "material_high",
            "sentiment": "negative",
            "event_type": "legal_regulatory",
            "candidate_status": "imported",
        }
    )
    staged = enrichment_priority_score(
        {
            "document_quality_status": "missing_document",
            "informativeness": "material_high",
            "sentiment": "negative",
            "event_type": "legal_regulatory",
            "candidate_status": "staged",
        }
    )
    assert imported > staged


def test_document_coverage_handles_empty_database(tmp_path) -> None:
    db_path = tmp_path / "empty.db"
    sqlite3.connect(db_path).close()

    audit = build_document_coverage_audit(db_path)

    assert audit.summary["total_company_ir_candidates"] == 0
    assert audit.summary["linked_document_pct"] == 0.0
    assert audit.queue.empty
    assert audit.warnings


def test_document_coverage_is_read_only_and_non_networking(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_all_statuses(db_path)

    def fail_network(*_args, **_kwargs):
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    with sqlite3.connect(db_path) as connection:
        before = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM research_event_candidates),
                (SELECT COUNT(*) FROM research_event_annotations),
                (SELECT COUNT(*) FROM source_documents),
                (SELECT COUNT(*) FROM llm_extractions),
                (SELECT COUNT(*) FROM catalysts)
            """
        ).fetchone()

    result = check_document_coverage(db_path)

    with sqlite3.connect(db_path) as connection:
        after = connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM research_event_candidates),
                (SELECT COUNT(*) FROM research_event_annotations),
                (SELECT COUNT(*) FROM source_documents),
                (SELECT COUNT(*) FROM llm_extractions),
                (SELECT COUNT(*) FROM catalysts)
            """
        ).fetchone()
    assert result.status == "passed"
    assert result.summary["read_only"] is True
    assert result.summary["network_calls_would_occur"] is False
    assert result.summary["scanner_scoring_effect"] == 0
    assert before == after
