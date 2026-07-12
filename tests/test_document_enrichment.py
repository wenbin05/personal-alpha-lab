from __future__ import annotations

import socket
import sqlite3
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.annotations.document_enrichment import (
    apply_company_ir_document_enrichment,
    plan_company_ir_document_enrichment,
    run_company_ir_document_enrichment,
)
from src.annotations.models import ResearchEventAnnotation
from src.annotations.news_events import ResearchEventAnnotationCandidate
from src.annotations.news_repository import stage_candidate
from src.annotations.repository import insert_annotation
from src.data import storage
from src.documents.repository import build_source_document, insert_document


PROVIDER = "company_ir_press_release"
TEXT = "The company announced a material product release with exact manually supplied source text for review."


def _seed_candidate(
    db_path,
    *,
    provider: str = PROVIDER,
    status: str = "imported",
    candidate_index: int = 1,
) -> tuple[int, int]:
    event_date = date(2024, 1, candidate_index + 1)
    source_url = f"https://investors.example.com/release-{candidate_index}"
    staged = stage_candidate(
        db_path,
        ResearchEventAnnotationCandidate(
            ticker="AAA",
            event_date=event_date,
            available_at=datetime(2024, 1, candidate_index + 1, 14, tzinfo=UTC),
            event_type="product_launch",
            title=f"Material release {candidate_index}",
            source="Company IR",
            source_url=source_url,
            sentiment_label="mixed",
            strength=6,
            confidence=0.8,
            provider=provider,
            document_type=PROVIDER if provider == PROVIDER else None,
            published_at=datetime(2024, 1, candidate_index + 1, 13, tzinfo=UTC),
        ),
    )
    annotation = insert_annotation(
        db_path,
        ResearchEventAnnotation(
            ticker="AAA",
            event_date=event_date,
            available_at=datetime(2024, 1, candidate_index + 1, 14, tzinfo=UTC),
            event_type="product_launch",
            sentiment_label="mixed",
            strength=6,
            confidence=0.8,
            source="Company IR",
            source_url=source_url,
            title=f"Material release {candidate_index}",
        ),
    )
    with storage.connect(db_path) as connection:
        connection.execute(
            """
            UPDATE research_event_candidates
            SET status = ?, imported_annotation_id = ?
            WHERE candidate_id = ?
            """,
            (status, annotation.annotation_id, staged.candidate_id),
        )
    return staged.candidate_id, annotation.annotation_id


def _frame(candidate_id: int, annotation_id: int | None = None, **overrides) -> pd.DataFrame:
    row = {
        "candidate_id": candidate_id,
        "annotation_id": annotation_id or "",
        "raw_text": TEXT,
        "cleaned_text": "",
        "text": "",
        "text_completeness": "complete",
        "review_note": "Manually verified against the supplied company IR release.",
    }
    row.update(overrides)
    return pd.DataFrame([row])


def _counts(db_path) -> tuple[int, ...]:
    with sqlite3.connect(db_path) as connection:
        return connection.execute(
            """
            SELECT
                (SELECT COUNT(*) FROM research_event_candidates),
                (SELECT COUNT(*) FROM research_event_annotations),
                (SELECT COUNT(*) FROM source_documents),
                (SELECT COUNT(*) FROM catalysts),
                (SELECT COUNT(*) FROM model_runs)
            """
        ).fetchone()


def test_dry_run_causes_zero_mutations_and_reports_projection(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    candidate_id, annotation_id = _seed_candidate(db_path)
    before = db_path.read_bytes()

    report = plan_company_ir_document_enrichment(db_path, _frame(candidate_id, annotation_id))

    assert db_path.read_bytes() == before
    assert report["read_only"] is True
    assert report["summary"]["documents_planned_for_creation"] == 1
    assert report["summary"]["candidate_links_planned"] == 1
    assert report["summary"]["annotation_links_planned"] == 1
    assert report["projected_coverage"]["candidates_with_linked_documents"] == 1


def test_apply_creates_complete_document_and_preserves_candidate_annotation_content(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    candidate_id, annotation_id = _seed_candidate(db_path)
    with storage.connect(db_path) as connection:
        candidate_before = dict(connection.execute("SELECT * FROM research_event_candidates").fetchone())
        annotation_before = dict(connection.execute("SELECT * FROM research_event_annotations").fetchone())

    report = apply_company_ir_document_enrichment(
        db_path,
        _frame(candidate_id, annotation_id),
        backup_dir=tmp_path / "backups",
    )

    with storage.connect(db_path) as connection:
        document = dict(connection.execute("SELECT * FROM source_documents").fetchone())
        candidate_after = dict(connection.execute("SELECT * FROM research_event_candidates").fetchone())
        annotation_after = dict(connection.execute("SELECT * FROM research_event_annotations").fetchone())
    assert report["mode"] == "apply"
    assert report["backup_path"]
    assert document["parsing_status"] == "success"
    assert document["source"] == PROVIDER
    assert document["created_at"] >= candidate_before["created_at"]
    assert candidate_after["source_document_id"] == document["document_id"]
    assert annotation_after["source_document_id"] == document["document_id"]
    for field in ("ticker", "event_type", "sentiment_label", "title", "source_url"):
        assert candidate_after[field] == candidate_before[field]
        assert annotation_after[field] == annotation_before[field]


def test_evidence_only_document_is_partial_with_explicit_warning(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    candidate_id, annotation_id = _seed_candidate(db_path)

    apply_company_ir_document_enrichment(
        db_path,
        _frame(candidate_id, annotation_id, text_completeness="evidence_only"),
        backup_dir=tmp_path,
    )

    with storage.connect(db_path) as connection:
        document = connection.execute("SELECT parsing_status, warnings FROM source_documents").fetchone()
    assert document["parsing_status"] == "partial"
    assert "evidence text only" in document["warnings"].lower()


@pytest.mark.parametrize("match_type", ["url", "text_hash"])
def test_document_reuse_by_url_or_text_hash(tmp_path, match_type: str) -> None:
    db_path = tmp_path / "alpha_lab.db"
    candidate_id, annotation_id = _seed_candidate(db_path)
    existing_url = "https://investors.example.com/release-1" if match_type == "url" else "https://other.example.com/release"
    existing_id = insert_document(
        db_path,
        build_source_document(
            "AAA",
            PROVIDER,
            "Existing release",
            TEXT,
            source=PROVIDER,
            source_url=existing_url,
            published_at="2024-01-02",
        ),
    )

    report = apply_company_ir_document_enrichment(
        db_path,
        _frame(candidate_id, annotation_id),
        backup_dir=tmp_path / "backups",
    )

    assert _counts(db_path)[2] == 1
    assert report["summary"]["documents_planned_for_reuse"] == 1
    with storage.connect(db_path) as connection:
        linked = connection.execute(
            "SELECT source_document_id FROM research_event_candidates WHERE candidate_id = ?", (candidate_id,)
        ).fetchone()[0]
    assert linked == existing_id


def test_repeated_apply_is_idempotent(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    candidate_id, annotation_id = _seed_candidate(db_path)
    frame = _frame(candidate_id, annotation_id)
    apply_company_ir_document_enrichment(db_path, frame, backup_dir=tmp_path / "backups")
    first_counts = _counts(db_path)

    second = apply_company_ir_document_enrichment(db_path, frame, backup_dir=tmp_path / "backups")

    assert _counts(db_path) == first_counts
    assert second["summary"]["skipped_existing_links"] == 1


@pytest.mark.parametrize(
    ("setup", "overrides", "error_fragment"),
    [
        ("missing_candidate", {}, "does not exist"),
        ("wrong_provider", {}, "not a company_ir_press_release"),
        ("annotation_mismatch", {"annotation_id": 999999}, "does not match"),
        ("rejected", {}, "rejected"),
        ("missing_text", {"raw_text": "", "cleaned_text": "", "text": ""}, "usable text"),
    ],
)
def test_invalid_enrichment_rows_are_blocked(tmp_path, setup: str, overrides: dict, error_fragment: str) -> None:
    db_path = tmp_path / "alpha_lab.db"
    if setup == "missing_candidate":
        storage.init_db(db_path)
        candidate_id, annotation_id = 999999, None
    elif setup == "wrong_provider":
        candidate_id, annotation_id = _seed_candidate(db_path, provider="manual_csv")
    elif setup == "rejected":
        candidate_id, annotation_id = _seed_candidate(db_path, status="rejected")
    else:
        candidate_id, annotation_id = _seed_candidate(db_path)
    before = _counts(db_path)

    frame = _frame(candidate_id, annotation_id)
    for key, value in overrides.items():
        frame.loc[0, key] = value
    report = plan_company_ir_document_enrichment(db_path, frame)

    assert report["summary"]["invalid_rows"] == 1
    assert error_fragment in report["planned_actions"][0]["error"]
    assert _counts(db_path) == before


def test_csv_dry_run_makes_no_network_or_llm_calls(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "alpha_lab.db"
    candidate_id, annotation_id = _seed_candidate(db_path)
    csv_path = tmp_path / "enrichment.csv"
    _frame(candidate_id, annotation_id).to_csv(csv_path, index=False)
    before = _counts(db_path)

    def fail_network(*_args, **_kwargs):
        raise AssertionError("network access is not allowed")

    monkeypatch.setattr(socket, "create_connection", fail_network)
    report = run_company_ir_document_enrichment(db_path, csv_path)

    assert report["network_calls_would_occur"] is False
    assert report["llm_calls_would_occur"] is False
    assert _counts(db_path) == before


def test_apply_does_not_change_catalysts_models_or_scanner_rows(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    candidate_id, annotation_id = _seed_candidate(db_path)
    with storage.connect(db_path) as connection:
        connection.execute(
            "INSERT INTO scan_results (run_id, run_at, ticker, score, label, payload_json) VALUES (?, ?, ?, ?, ?, ?)",
            ("baseline", "2024-01-01T00:00:00+00:00", "AAA", 50.0, "Neutral", "{}"),
        )
        scanner_before = connection.execute("SELECT * FROM scan_results").fetchall()
    before = _counts(db_path)

    report = apply_company_ir_document_enrichment(
        db_path,
        _frame(candidate_id, annotation_id),
        backup_dir=tmp_path,
    )

    with storage.connect(db_path) as connection:
        scanner_after = connection.execute("SELECT * FROM scan_results").fetchall()
    after = _counts(db_path)
    assert before[3:] == after[3:]
    assert [tuple(row) for row in scanner_before] == [tuple(row) for row in scanner_after]
    assert report["active_catalysts_unchanged"] is True
    assert report["model_runs_unchanged"] is True
    assert report["scanner_scoring_effect"] == 0
