from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.annotations.document_coverage import build_document_coverage_audit
from src.documents.repository import build_source_document
from src.documents.text_cleaning import MIN_USEFUL_TEXT_CHARS, clean_text


COMPANY_IR_PROVIDER = "company_ir_press_release"
TEXT_COMPLETENESS_VALUES = {"complete", "partial", "evidence_only"}


@dataclass(frozen=True)
class EnrichmentPlanRow:
    row_number: int
    candidate_id: int | None
    annotation_id: int | None
    valid: bool
    planned_action: str
    document_action: str
    candidate_link_action: str
    annotation_link_action: str
    matched_by: str | None
    source_document_id: int | None
    text_completeness: str | None
    error: str | None
    ticker: str = ""
    title: str = ""
    source_url: str = ""


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _read_only_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value) or str(value).strip() == "":
        return None
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    return str(value).strip()


def _date_iso(value: Any) -> str | None:
    text = _text(value)
    if not text:
        return None
    return pd.to_datetime(text, utc=True).date().isoformat()


def _normalized_url(value: Any) -> str:
    return _text(value).rstrip("/").lower()


def _resolved_annotation_id(candidate: sqlite3.Row) -> int | None:
    return _optional_int(candidate["imported_annotation_id"]) or _optional_int(candidate["duplicate_of_annotation_id"])


def _source_text(row: pd.Series) -> tuple[str, str]:
    raw_text = _text(row.get("raw_text"))
    cleaned_text = _text(row.get("cleaned_text"))
    generic_text = _text(row.get("text"))
    if raw_text:
        return raw_text, cleaned_text or clean_text(raw_text)
    if cleaned_text:
        return cleaned_text, clean_text(cleaned_text)
    return generic_text, clean_text(generic_text)


def _find_document_match(
    connection: sqlite3.Connection,
    *,
    ticker: str,
    source_url: str,
    text_hash: str,
    title: str,
    published_at: str | None,
) -> tuple[int | None, str | None]:
    if source_url:
        row = connection.execute(
            """
            SELECT document_id FROM source_documents
            WHERE ticker = ? AND LOWER(RTRIM(source_url, '/')) = LOWER(RTRIM(?, '/'))
            ORDER BY document_id LIMIT 1
            """,
            (ticker, source_url),
        ).fetchone()
        if row is not None:
            return int(row["document_id"]), "ticker_source_url"
    row = connection.execute(
        "SELECT document_id FROM source_documents WHERE ticker = ? AND text_hash = ? ORDER BY document_id LIMIT 1",
        (ticker, text_hash),
    ).fetchone()
    if row is not None:
        return int(row["document_id"]), "ticker_text_hash"
    if published_at:
        row = connection.execute(
            """
            SELECT document_id FROM source_documents
            WHERE ticker = ? AND published_at = ? AND LOWER(TRIM(title)) = LOWER(TRIM(?))
            ORDER BY document_id LIMIT 1
            """,
            (ticker, published_at, title),
        ).fetchone()
        if row is not None:
            return int(row["document_id"]), "ticker_title_date"
    return None, None


def _content_fingerprint(connection: sqlite3.Connection) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for table, id_column in (
        ("research_event_candidates", "candidate_id"),
        ("research_event_annotations", "annotation_id"),
        ("source_documents", "document_id"),
        ("catalysts", "id"),
        ("model_runs", "model_run_id"),
    ):
        row = connection.execute(
            f"SELECT COUNT(*) AS count, COALESCE(MAX({id_column}), 0) AS max_id FROM {table}"
        ).fetchone()
        result[table] = {"count": int(row["count"]), "max_id": int(row["max_id"])}
    return result


def _projected_coverage(current: dict[str, Any], rows: list[dict[str, Any]]) -> dict[str, Any]:
    projected = dict(current)
    candidate_ids = {row["candidate_id"] for row in rows if row["valid"] and row["planned_action"] != "skip"}
    complete_ids = {
        row["candidate_id"]
        for row in rows
        if row["valid"] and row["planned_action"] != "skip" and row["text_completeness"] == "complete"
    }
    partial_ids = candidate_ids - complete_ids
    newly_linked = len(candidate_ids)
    projected["candidates_with_linked_documents"] = min(
        int(current.get("total_company_ir_candidates", 0)),
        int(current.get("candidates_with_linked_documents", 0)) + newly_linked,
    )
    projected["complete_documents"] = int(current.get("complete_documents", 0)) + len(complete_ids)
    projected["partial_documents"] = int(current.get("partial_documents", 0)) + len(partial_ids)
    projected["missing_documents"] = max(0, int(current.get("missing_documents", 0)) - newly_linked)
    total = int(current.get("total_company_ir_candidates", 0))
    projected["linked_document_pct"] = (
        round(100.0 * projected["candidates_with_linked_documents"] / total, 2) if total else 0.0
    )
    projected["complete_document_pct"] = round(100.0 * projected["complete_documents"] / total, 2) if total else 0.0
    projected["missing_document_pct"] = round(100.0 * projected["missing_documents"] / total, 2) if total else 0.0
    return projected


def plan_company_ir_document_enrichment(
    db_path: str | Path,
    frame: pd.DataFrame,
) -> dict[str, Any]:
    current_coverage = build_document_coverage_audit(db_path, provider=COMPANY_IR_PROVIDER).summary
    plans: list[dict[str, Any]] = []
    planned_matches: dict[tuple[str, str], int] = {}

    with _read_only_connection(db_path) as connection:
        before_fingerprint = _content_fingerprint(connection)
        for index, input_row in frame.fillna("").iterrows():
            row_number = int(index) + 2
            candidate_id = _optional_int(input_row.get("candidate_id"))
            error: str | None = None
            candidate = None
            annotation = None
            supplied_annotation_id = _optional_int(input_row.get("annotation_id"))
            supplied_document_id = _optional_int(input_row.get("source_document_id"))
            raw_text, cleaned_text = _source_text(input_row)
            completeness = _text(input_row.get("text_completeness")).lower() or "partial"

            if candidate_id is None:
                error = "candidate_id is required and must be an integer."
            else:
                candidate = connection.execute(
                    "SELECT * FROM research_event_candidates WHERE candidate_id = ?", (candidate_id,)
                ).fetchone()
                if candidate is None:
                    error = f"Candidate {candidate_id} does not exist."
                elif str(candidate["provider"]) != COMPANY_IR_PROVIDER:
                    error = f"Candidate {candidate_id} is not a company_ir_press_release candidate."
                elif str(candidate["status"]) == "rejected":
                    error = f"Candidate {candidate_id} is rejected and cannot be enriched."

            annotation_id = _resolved_annotation_id(candidate) if candidate is not None else None
            if error is None and annotation_id is None:
                error = f"Candidate {candidate_id} does not resolve to an existing annotation."
            if error is None:
                annotation = connection.execute(
                    "SELECT * FROM research_event_annotations WHERE annotation_id = ?", (annotation_id,)
                ).fetchone()
                if annotation is None:
                    error = f"Resolved annotation {annotation_id} does not exist."
            if error is None and supplied_annotation_id is not None and supplied_annotation_id != annotation_id:
                error = (
                    f"Supplied annotation_id {supplied_annotation_id} does not match candidate linkage {annotation_id}."
                )
            if error is None and completeness not in TEXT_COMPLETENESS_VALUES:
                error = "text_completeness must be complete, partial, or evidence_only."
            if error is None and len(cleaned_text) < MIN_USEFUL_TEXT_CHARS:
                error = f"At least {MIN_USEFUL_TEXT_CHARS} characters of usable text are required."

            document_type = _text(input_row.get("document_type")) or COMPANY_IR_PROVIDER
            if error is None and document_type != COMPANY_IR_PROVIDER:
                error = "document_type must be company_ir_press_release."

            title = _text(input_row.get("title")) or (_text(candidate["title"]) if candidate is not None else "")
            source_url = _text(input_row.get("source_url")) or (
                _text(candidate["source_url"]) if candidate is not None else ""
            )
            published_at = _date_iso(input_row.get("published_at")) or (
                _date_iso(candidate["published_at"] or candidate["available_at"]) if candidate is not None else None
            )
            available_at = _text(input_row.get("available_at")) or (
                _text(candidate["available_at"]) if candidate is not None else ""
            )

            document = None
            matched_document_id = None
            matched_by = None
            candidate_link_action = "none"
            annotation_link_action = "none"
            document_action = "none"
            planned_action = "error" if error else "link"
            if error is None and candidate is not None and annotation is not None:
                candidate_link = _optional_int(candidate["source_document_id"])
                annotation_link = _optional_int(annotation["source_document_id"])
                if candidate_link and annotation_link and candidate_link != annotation_link:
                    error = "Candidate and annotation point to different SourceDocuments."
                existing_link = candidate_link or annotation_link
                if supplied_document_id and existing_link and supplied_document_id != existing_link:
                    error = "Supplied source_document_id conflicts with the existing linkage."
                if error is None and supplied_document_id:
                    supplied_document = connection.execute(
                        "SELECT * FROM source_documents WHERE document_id = ?", (supplied_document_id,)
                    ).fetchone()
                    if supplied_document is None:
                        error = f"Supplied SourceDocument {supplied_document_id} does not exist."
                    elif str(supplied_document["ticker"]).upper() != str(candidate["ticker"]).upper():
                        error = "Supplied SourceDocument ticker does not match the candidate ticker."
                    else:
                        matched_document_id = supplied_document_id
                        matched_by = "supplied_source_document_id"
                elif error is None and existing_link:
                    existing_document = connection.execute(
                        "SELECT * FROM source_documents WHERE document_id = ?", (existing_link,)
                    ).fetchone()
                    if existing_document is None:
                        error = f"Linked SourceDocument {existing_link} does not exist."
                    else:
                        matched_document_id = existing_link
                        matched_by = "existing_link"

                if error is None and matched_document_id is None:
                    warnings: list[str] = []
                    parsing_status = "success" if completeness == "complete" else "partial"
                    if completeness == "partial":
                        warnings.append("Manually supplied text is partial; it is not represented as the complete press release.")
                    elif completeness == "evidence_only":
                        warnings.append("Document contains evidence text only; full source text was not provided.")
                    document = build_source_document(
                        ticker=str(candidate["ticker"]),
                        document_type=COMPANY_IR_PROVIDER,
                        title=title,
                        raw_text=raw_text,
                        source=COMPANY_IR_PROVIDER,
                        source_url=source_url or None,
                        published_at=published_at,
                        parsing_status=parsing_status,
                        warnings=warnings,
                        raw_payload_json=json.dumps(
                            {
                                "ingestion_method": "manual_document_enrichment",
                                "candidate_id": candidate_id,
                                "annotation_id": annotation_id,
                                "available_at": available_at,
                                "text_completeness": completeness,
                                "review_note": _text(input_row.get("review_note")),
                                "network_calls_would_occur": False,
                                "llm_calls_would_occur": False,
                            },
                            sort_keys=True,
                        ),
                    )
                    matched_document_id, matched_by = _find_document_match(
                        connection,
                        ticker=document.ticker,
                        source_url=document.source_url or "",
                        text_hash=document.text_hash,
                        title=document.title,
                        published_at=published_at,
                    )
                    virtual_keys = [
                        (document.ticker, f"url:{_normalized_url(document.source_url)}") if document.source_url else None,
                        (document.ticker, f"hash:{document.text_hash}"),
                        (document.ticker, f"title:{document.title.lower()}|{published_at}") if published_at else None,
                    ]
                    for key in [value for value in virtual_keys if value is not None]:
                        if matched_document_id is None and key in planned_matches:
                            matched_document_id = planned_matches[key]
                            matched_by = "planned_batch_duplicate"
                    if matched_document_id is None:
                        document_action = "create"
                        virtual_document_id = -(row_number)
                        matched_document_id = virtual_document_id
                        for key in [value for value in virtual_keys if value is not None]:
                            planned_matches[key] = virtual_document_id
                    else:
                        document_action = "reuse"
                elif error is None:
                    document_action = "reuse"

                if error is None:
                    candidate_link_action = "skip" if candidate_link == matched_document_id else "link"
                    annotation_link_action = "skip" if annotation_link == matched_document_id else "link"
                    if candidate_link_action == "skip" and annotation_link_action == "skip":
                        planned_action = "skip"
                    else:
                        planned_action = "create_and_link" if document_action == "create" else "reuse_and_link"

            if error is not None:
                planned_action = "error"
                document_action = "none"
                candidate_link_action = "none"
                annotation_link_action = "none"

            plan = EnrichmentPlanRow(
                row_number=row_number,
                candidate_id=candidate_id,
                annotation_id=annotation_id,
                valid=error is None,
                planned_action=planned_action,
                document_action=document_action,
                candidate_link_action=candidate_link_action,
                annotation_link_action=annotation_link_action,
                matched_by=matched_by,
                source_document_id=matched_document_id if matched_document_id and matched_document_id > 0 else None,
                text_completeness=completeness if error is None else None,
                error=error,
                ticker=_text(candidate["ticker"]) if candidate is not None else "",
                title=title,
                source_url=source_url,
            )
            payload = asdict(plan)
            payload["_document"] = document
            payload["_published_at"] = published_at
            plans.append(payload)

    public_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in plans]
    valid_rows = [row for row in public_rows if row["valid"]]
    summary = {
        "rows_received": len(public_rows),
        "valid_rows": len(valid_rows),
        "invalid_rows": len(public_rows) - len(valid_rows),
        "documents_planned_for_creation": sum(row["document_action"] == "create" for row in valid_rows),
        "documents_planned_for_reuse": sum(row["document_action"] == "reuse" for row in valid_rows),
        "candidate_links_planned": sum(row["candidate_link_action"] == "link" for row in valid_rows),
        "annotation_links_planned": sum(row["annotation_link_action"] == "link" for row in valid_rows),
        "skipped_existing_links": sum(row["planned_action"] == "skip" for row in valid_rows),
        "duplicates": sum(row["matched_by"] in {"ticker_source_url", "ticker_text_hash", "ticker_title_date", "planned_batch_duplicate"} for row in valid_rows),
        "errors": len(public_rows) - len(valid_rows),
    }
    return {
        "mode": "dry_run",
        "provider": COMPANY_IR_PROVIDER,
        "read_only": True,
        "network_calls_would_occur": False,
        "llm_calls_would_occur": False,
        "summary": summary,
        "before_coverage": current_coverage,
        "projected_coverage": _projected_coverage(current_coverage, public_rows),
        "planned_actions": public_rows,
        "_internal_rows": plans,
        "before_fingerprint": before_fingerprint,
    }


def _backup_database(db_path: str | Path, backup_dir: str | Path | None = None) -> Path:
    source = Path(db_path).expanduser().resolve()
    target_dir = Path(backup_dir).expanduser().resolve() if backup_dir else source.parent
    target_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    target = target_dir / f"{source.stem}_backup_phase2e5b1_enrichment_{timestamp}{source.suffix}"
    with sqlite3.connect(source) as source_connection, sqlite3.connect(target) as target_connection:
        source_connection.backup(target_connection)
    return target


def _insert_document(connection: sqlite3.Connection, document: Any, published_at: str | None) -> int:
    now = _now_iso()
    connection.execute(
        """
        INSERT INTO source_documents (
            ticker, catalyst_id, document_type, source, source_url, accession_number,
            filing_type, title, published_at, raw_text, cleaned_text, text_hash,
            parsing_status, warnings, created_at, updated_at, raw_payload_json
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            document.ticker,
            document.catalyst_id,
            document.document_type,
            document.source,
            document.source_url,
            document.accession_number,
            document.filing_type,
            document.title,
            published_at,
            document.raw_text,
            document.cleaned_text,
            document.text_hash,
            document.parsing_status,
            document.warnings,
            now,
            now,
            document.raw_payload_json,
        ),
    )
    return int(connection.execute("SELECT last_insert_rowid()").fetchone()[0])


def apply_company_ir_document_enrichment(
    db_path: str | Path,
    frame: pd.DataFrame,
    *,
    backup_dir: str | Path | None = None,
) -> dict[str, Any]:
    plan = plan_company_ir_document_enrichment(db_path, frame)
    backup_path = _backup_database(db_path, backup_dir)
    applied_rows: list[dict[str, Any]] = []

    connection = sqlite3.connect(Path(db_path))
    connection.row_factory = sqlite3.Row
    try:
        connection.execute("BEGIN IMMEDIATE")
        for row in plan["_internal_rows"]:
            public = {key: value for key, value in row.items() if not key.startswith("_")}
            if not row["valid"] or row["planned_action"] == "skip":
                applied_rows.append(public)
                continue
            document = row["_document"]
            document_id = row["source_document_id"]
            matched_by = row["matched_by"]
            if document_id is None:
                document_id, matched_by = _find_document_match(
                    connection,
                    ticker=document.ticker,
                    source_url=document.source_url or "",
                    text_hash=document.text_hash,
                    title=document.title,
                    published_at=row["_published_at"],
                )
            if document_id is None:
                document_id = _insert_document(connection, document, row["_published_at"])
                matched_by = None
            now = _now_iso()
            connection.execute(
                "UPDATE research_event_candidates SET source_document_id = ?, updated_at = ? WHERE candidate_id = ?",
                (document_id, now, row["candidate_id"]),
            )
            connection.execute(
                "UPDATE research_event_annotations SET source_document_id = ?, updated_at = ? WHERE annotation_id = ?",
                (document_id, now, row["annotation_id"]),
            )
            public["source_document_id"] = document_id
            public["matched_by"] = matched_by
            applied_rows.append(public)
        connection.commit()
    except Exception:
        connection.rollback()
        raise
    finally:
        connection.close()

    after_coverage = build_document_coverage_audit(db_path, provider=COMPANY_IR_PROVIDER).summary
    with _read_only_connection(db_path) as readonly:
        after_fingerprint = _content_fingerprint(readonly)
    public_plan = {key: value for key, value in plan.items() if not key.startswith("_")}
    public_plan.update(
        {
            "mode": "apply",
            "read_only": False,
            "backup_path": str(backup_path),
            "applied_actions": applied_rows,
            "after_coverage": after_coverage,
            "after_fingerprint": after_fingerprint,
            "candidate_annotation_content_preserved": True,
            "active_catalysts_unchanged": (
                plan["before_fingerprint"]["catalysts"] == after_fingerprint["catalysts"]
            ),
            "model_runs_unchanged": (
                plan["before_fingerprint"]["model_runs"] == after_fingerprint["model_runs"]
            ),
            "scanner_scoring_effect": 0,
        }
    )
    return public_plan


def run_company_ir_document_enrichment(
    db_path: str | Path,
    input_path: str | Path,
    *,
    apply: bool = False,
    backup_dir: str | Path | None = None,
) -> dict[str, Any]:
    frame = pd.read_csv(input_path, dtype=object, keep_default_na=False)
    if apply:
        return apply_company_ir_document_enrichment(db_path, frame, backup_dir=backup_dir)
    plan = plan_company_ir_document_enrichment(db_path, frame)
    return {key: value for key, value in plan.items() if not key.startswith("_")}
