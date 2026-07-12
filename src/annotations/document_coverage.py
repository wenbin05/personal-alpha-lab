from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.annotations.source_quality import classify_event_quality
from src.documents.text_cleaning import MIN_USEFUL_TEXT_CHARS, normalize_whitespace


COVERAGE_STATUSES = (
    "linked_complete",
    "linked_partial",
    "linked_missing_text",
    "missing_document",
    "broken_linkage",
    "duplicate_document_reused",
)

QUEUE_COLUMNS = (
    "candidate_id",
    "annotation_id",
    "existing_source_document_id",
    "coverage_status",
    "enrichment_priority",
    "ticker",
    "title",
    "event_type",
    "sentiment",
    "informativeness",
    "source_quality",
    "source_url",
    "published_at",
    "available_at",
    "provider_name",
    "provider_event_id",
    "raw_text",
    "cleaned_text",
    "review_note",
)

REQUIRED_TABLES = {
    "research_event_candidates",
    "research_event_annotations",
    "source_documents",
}


@dataclass(frozen=True)
class DocumentCoverageAudit:
    provider: str
    summary: dict[str, Any]
    status_counts: list[dict[str, Any]]
    coverage_by_ticker: list[dict[str, Any]]
    coverage_by_sentiment: list[dict[str, Any]]
    coverage_by_informativeness: list[dict[str, Any]]
    coverage_by_event_type: list[dict[str, Any]]
    top_missing_document_tickers: list[dict[str, Any]]
    warnings: list[str]
    rows: pd.DataFrame
    queue: pd.DataFrame


def _read_only_connection(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path).expanduser().resolve()
    connection = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    return connection


def _json_dict(value: Any) -> dict[str, Any]:
    if value in (None, ""):
        return {}
    try:
        payload = json.loads(str(value))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _optional_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _percentage(count: int, total: int) -> float:
    return round((100.0 * count / total), 2) if total else 0.0


def _empty_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            *QUEUE_COLUMNS,
            "candidate_status",
            "document_quality_status",
            "linked_document_exists",
            "document_reused",
            "needs_enrichment",
            "parsing_status",
            "document_warnings",
        ]
    )


def _empty_audit(provider: str, warnings: list[str] | None = None) -> DocumentCoverageAudit:
    summary = {
        "provider": provider,
        "total_company_ir_candidates": 0,
        "staged_candidates": 0,
        "staged_candidate_pct": 0.0,
        "accepted_candidates": 0,
        "accepted_candidate_pct": 0.0,
        "imported_candidates": 0,
        "imported_candidate_pct": 0.0,
        "candidates_with_annotations": 0,
        "candidate_annotation_pct": 0.0,
        "candidates_with_linked_documents": 0,
        "complete_documents": 0,
        "partial_documents": 0,
        "missing_text_documents": 0,
        "missing_documents": 0,
        "broken_linkages": 0,
        "reused_documents": 0,
        "reused_document_candidates": 0,
        "reused_document_candidate_pct": 0.0,
        "unique_tickers_covered": 0,
        "queue_row_count": 0,
        "queue_candidate_pct": 0.0,
        "linked_document_pct": 0.0,
        "complete_document_pct": 0.0,
        "missing_document_pct": 0.0,
        "read_only": True,
        "network_calls_would_occur": False,
        "scanner_scoring_effect": 0,
        "workflow_priority_only": True,
    }
    return DocumentCoverageAudit(
        provider=provider,
        summary=summary,
        status_counts=[],
        coverage_by_ticker=[],
        coverage_by_sentiment=[],
        coverage_by_informativeness=[],
        coverage_by_event_type=[],
        top_missing_document_tickers=[],
        warnings=list(warnings or []),
        rows=_empty_frame(),
        queue=pd.DataFrame(columns=QUEUE_COLUMNS),
    )


def _usable_document_text(document: dict[str, Any]) -> bool:
    raw_text = normalize_whitespace(str(document.get("raw_text") or ""))
    cleaned_text = normalize_whitespace(str(document.get("cleaned_text") or ""))
    return max(len(raw_text), len(cleaned_text)) >= MIN_USEFUL_TEXT_CHARS


def _evidence_only_document(document: dict[str, Any]) -> bool:
    warnings = str(document.get("warnings") or "").lower()
    payload = _json_dict(document.get("raw_payload_json"))
    return "evidence only" in warnings or str(payload.get("text_source") or "").lower() == "evidence_text"


def _document_quality_status(document: dict[str, Any] | None) -> str:
    if document is None:
        return "missing_document"
    if not _usable_document_text(document):
        return "linked_missing_text"
    parsing_status = str(document.get("parsing_status") or "").lower()
    if parsing_status in {"partial", "failed"} or _evidence_only_document(document):
        return "linked_partial"
    return "linked_complete"


def enrichment_priority_score(row: dict[str, Any]) -> int:
    """Return workflow urgency only; this is not an alpha or model score."""
    quality_status = str(row.get("document_quality_status") or row.get("coverage_status") or "")
    informativeness = str(row.get("informativeness") or "unknown").lower()
    sentiment = str(row.get("sentiment") or "unknown").lower()
    event_type = str(row.get("event_type") or "other").lower()
    candidate_status = str(row.get("candidate_status") or "").lower()

    score = {
        "broken_linkage": 100_000,
        "linked_missing_text": 30_000,
        "missing_document": 25_000,
        "linked_partial": 20_000,
    }.get(quality_status, 0)
    score += {
        "material_high": 12_000,
        "material_medium": 6_000,
        "routine_low": 1_000,
        "low_specificity": 500,
        "duplicate_theme": 0,
    }.get(informativeness, 0)
    score += {"negative": 5_000, "mixed": 4_000, "neutral": 2_000, "positive": 1_000}.get(sentiment, 0)
    score += {
        "legal_regulatory": 4_000,
        "financing": 3_500,
        "guidance_update": 3_000,
        "corporate_action": 2_500,
        "product_launch": 2_000,
    }.get(event_type, 0)
    score += {"imported": 2_000, "accepted": 2_000, "staged": 750, "duplicate": 250, "rejected": 0}.get(
        candidate_status,
        0,
    )
    return int(score)


def _group_coverage(rows: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    if rows.empty:
        return []
    output: list[dict[str, Any]] = []
    grouped = rows.assign(**{column: rows[column].fillna("unknown").astype(str)}).groupby(column, sort=True)
    for key, group in grouped:
        total = int(len(group))
        linked = int(group["linked_document_exists"].sum())
        complete = int(group["document_quality_status"].eq("linked_complete").sum())
        needs_enrichment = int(group["needs_enrichment"].sum())
        output.append(
            {
                column: key,
                "candidate_count": total,
                "linked_document_count": linked,
                "complete_document_count": complete,
                "needs_enrichment_count": needs_enrichment,
                "linked_document_pct": _percentage(linked, total),
                "complete_document_pct": _percentage(complete, total),
            }
        )
    return output


def build_document_coverage_audit(
    db_path: str | Path,
    provider: str = "company_ir_press_release",
) -> DocumentCoverageAudit:
    path = Path(db_path).expanduser()
    if not path.exists():
        return _empty_audit(provider, [f"Database does not exist: {path}"])

    with _read_only_connection(path) as connection:
        tables = {
            str(row["name"])
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
        }
        missing_tables = sorted(REQUIRED_TABLES - tables)
        if missing_tables:
            return _empty_audit(provider, [f"Missing required tables: {', '.join(missing_tables)}"])
        candidates = pd.read_sql_query(
            "SELECT * FROM research_event_candidates WHERE provider = ? ORDER BY candidate_id",
            connection,
            params=(provider,),
        )
        annotations = pd.read_sql_query("SELECT * FROM research_event_annotations", connection)
        documents = pd.read_sql_query("SELECT * FROM source_documents", connection)

    if candidates.empty:
        return _empty_audit(provider)

    annotation_map = {
        int(row["annotation_id"]): row.to_dict()
        for _, row in annotations.iterrows()
        if _optional_int(row.get("annotation_id")) is not None
    }
    document_map = {
        int(row["document_id"]): row.to_dict()
        for _, row in documents.iterrows()
        if _optional_int(row.get("document_id")) is not None
    }

    resolved_rows: list[dict[str, Any]] = []
    for _, candidate_series in candidates.iterrows():
        candidate = candidate_series.to_dict()
        metadata = _json_dict(candidate.get("provider_metadata_json"))
        quality = classify_event_quality(candidate)
        annotation_id = _optional_int(candidate.get("imported_annotation_id")) or _optional_int(
            candidate.get("duplicate_of_annotation_id")
        )
        annotation = annotation_map.get(annotation_id) if annotation_id is not None else None
        candidate_document_id = _optional_int(candidate.get("source_document_id"))
        annotation_document_id = _optional_int(annotation.get("source_document_id")) if annotation else None
        explicit_document_ids = [value for value in (candidate_document_id, annotation_document_id) if value is not None]
        linkage_conflict = len(set(explicit_document_ids)) > 1
        unresolved_reference = any(document_map.get(document_id) is None for document_id in explicit_document_ids)
        missing_annotation = annotation_id is not None and annotation is None
        resolved_document_id = candidate_document_id or annotation_document_id
        document = document_map.get(resolved_document_id) if resolved_document_id is not None else None

        if linkage_conflict or unresolved_reference or missing_annotation:
            quality_status = "broken_linkage"
        else:
            quality_status = _document_quality_status(document)
        resolved_rows.append(
            {
                "candidate_id": int(candidate["candidate_id"]),
                "annotation_id": annotation_id,
                "existing_source_document_id": resolved_document_id,
                "coverage_status": quality_status,
                "document_quality_status": quality_status,
                "ticker": str(candidate.get("ticker") or "").upper(),
                "title": str(candidate.get("title") or ""),
                "event_type": str(candidate.get("event_type") or "other"),
                "sentiment": str(candidate.get("sentiment_label") or "unknown"),
                "informativeness": quality.informativeness,
                "source_quality": quality.source_quality,
                "source_url": str(candidate.get("source_url") or ""),
                "published_at": str(candidate.get("published_at") or ""),
                "available_at": str(candidate.get("available_at") or ""),
                "provider_name": str(candidate.get("provider") or provider),
                "provider_event_id": str(metadata.get("provider_event_id") or ""),
                "raw_text": str(candidate.get("raw_text") or ""),
                "cleaned_text": str(candidate.get("cleaned_text") or ""),
                "review_note": str(metadata.get("review_note") or ""),
                "candidate_status": str(candidate.get("status") or ""),
                "linked_document_exists": bool(document is not None and quality_status != "broken_linkage"),
                "document_reused": False,
                "needs_enrichment": quality_status
                in {"broken_linkage", "linked_missing_text", "missing_document", "linked_partial"},
                "parsing_status": str(document.get("parsing_status") or "") if document else "",
                "document_warnings": str(document.get("warnings") or "") if document else "",
            }
        )

    rows = pd.DataFrame(resolved_rows)
    reusable_ids = {
        int(document_id)
        for document_id, count in rows.loc[rows["linked_document_exists"], "existing_source_document_id"]
        .dropna()
        .astype(int)
        .value_counts()
        .items()
        if int(count) > 1
    }
    reused_mask = rows["existing_source_document_id"].map(_optional_int).isin(reusable_ids) & rows["linked_document_exists"]
    rows.loc[reused_mask, "document_reused"] = True
    rows.loc[reused_mask, "coverage_status"] = "duplicate_document_reused"
    rows["enrichment_priority"] = [enrichment_priority_score(row) for row in rows.to_dict("records")]

    queue = rows[rows["needs_enrichment"]].copy()
    queue["_available_sort"] = pd.to_datetime(queue["available_at"], utc=True, errors="coerce")
    queue = queue.sort_values(
        ["enrichment_priority", "_available_sort", "ticker", "candidate_id"],
        ascending=[False, True, True, True],
        na_position="last",
        kind="mergesort",
    )
    queue = queue.loc[:, list(QUEUE_COLUMNS)].reset_index(drop=True)

    total = int(len(rows))
    linked_count = int(rows["linked_document_exists"].sum())
    complete_count = int(rows["document_quality_status"].eq("linked_complete").sum())
    partial_count = int(rows["document_quality_status"].eq("linked_partial").sum())
    missing_text_count = int(rows["document_quality_status"].eq("linked_missing_text").sum())
    missing_count = int(rows["document_quality_status"].eq("missing_document").sum())
    broken_count = int(rows["document_quality_status"].eq("broken_linkage").sum())
    summary = {
        "provider": provider,
        "total_company_ir_candidates": total,
        "staged_candidates": int(rows["candidate_status"].eq("staged").sum()),
        "staged_candidate_pct": _percentage(int(rows["candidate_status"].eq("staged").sum()), total),
        "accepted_candidates": int(rows["candidate_status"].eq("accepted").sum()),
        "accepted_candidate_pct": _percentage(int(rows["candidate_status"].eq("accepted").sum()), total),
        "imported_candidates": int(rows["candidate_status"].eq("imported").sum()),
        "imported_candidate_pct": _percentage(int(rows["candidate_status"].eq("imported").sum()), total),
        "candidates_with_annotations": int(rows["annotation_id"].notna().sum()),
        "candidate_annotation_pct": _percentage(int(rows["annotation_id"].notna().sum()), total),
        "candidates_with_linked_documents": linked_count,
        "complete_documents": complete_count,
        "partial_documents": partial_count,
        "missing_text_documents": missing_text_count,
        "missing_documents": missing_count,
        "broken_linkages": broken_count,
        "reused_documents": len(reusable_ids),
        "reused_document_candidates": int(rows["document_reused"].sum()),
        "reused_document_candidate_pct": _percentage(int(rows["document_reused"].sum()), total),
        "unique_tickers_covered": int(rows["ticker"].nunique()),
        "queue_row_count": int(len(queue)),
        "queue_candidate_pct": _percentage(int(len(queue)), total),
        "linked_document_pct": _percentage(linked_count, total),
        "complete_document_pct": _percentage(complete_count, total),
        "partial_document_pct": _percentage(partial_count, total),
        "missing_text_document_pct": _percentage(missing_text_count, total),
        "missing_document_pct": _percentage(missing_count, total),
        "broken_linkage_pct": _percentage(broken_count, total),
        "read_only": True,
        "network_calls_would_occur": False,
        "scanner_scoring_effect": 0,
        "workflow_priority_only": True,
    }
    status_counts = [
        {
            "coverage_status": status,
            "candidate_count": int(rows["coverage_status"].eq(status).sum()),
            "candidate_pct": _percentage(int(rows["coverage_status"].eq(status).sum()), total),
        }
        for status in COVERAGE_STATUSES
    ]
    top_missing = (
        rows[rows["needs_enrichment"]]
        .groupby("ticker", sort=True)
        .size()
        .reset_index(name="needs_enrichment_count")
        .sort_values(["needs_enrichment_count", "ticker"], ascending=[False, True], kind="mergesort")
        .head(10)
        .to_dict("records")
    )
    return DocumentCoverageAudit(
        provider=provider,
        summary=summary,
        status_counts=status_counts,
        coverage_by_ticker=_group_coverage(rows, "ticker"),
        coverage_by_sentiment=_group_coverage(rows, "sentiment"),
        coverage_by_informativeness=_group_coverage(rows, "informativeness"),
        coverage_by_event_type=_group_coverage(rows, "event_type"),
        top_missing_document_tickers=top_missing,
        warnings=[],
        rows=rows,
        queue=queue,
    )


def write_enrichment_queue_csv(queue: pd.DataFrame, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    normalized = queue.reindex(columns=QUEUE_COLUMNS).fillna("")
    normalized.to_csv(output, index=False, lineterminator="\n")
    return output
