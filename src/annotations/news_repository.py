from __future__ import annotations

import json
from dataclasses import asdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.annotations.models import ResearchEventAnnotation, json_dumps as annotation_json_dumps, normalize_tags
from src.annotations.news_events import (
    ResearchEventAnnotationCandidate,
    candidate_dedupe_key,
    json_dumps,
    json_loads,
    normalize_source_url,
    normalize_title,
    stable_text_hash,
)
from src.annotations.repository import insert_annotation
from src.annotations.source_quality import classify_event_quality, enrich_quality_frame, quality_distribution
from src.data import storage


@dataclass(frozen=True)
class CandidateStageResult:
    candidate_id: int
    inserted: bool
    status: str
    dedupe_key: str
    duplicate_reason: str | None = None


@dataclass(frozen=True)
class CandidateImportSummary:
    imported_count: int
    skipped_count: int
    imported_annotation_ids: list[int]
    warnings: list[str]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dt_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds")


def create_candidate_tables(db_path: str | Path) -> None:
    storage.init_db(db_path)


def _candidate_from_row(row: sqlite_row | dict[str, Any]) -> ResearchEventAnnotationCandidate:
    data = dict(row)
    return ResearchEventAnnotationCandidate(
        candidate_id=int(data["candidate_id"]),
        ticker=data["ticker"],
        event_date=pd.to_datetime(data["event_date"]).date(),
        available_at=pd.to_datetime(data["available_at"], utc=True).to_pydatetime(),
        event_type=data["event_type"],
        title=data.get("title") or "",
        summary=data.get("summary") or "",
        source=data.get("source") or "csv_manual",
        source_url=data.get("source_url"),
        evidence_text=data.get("evidence_text") or "",
        sentiment_label=data.get("sentiment_label") or "unknown",
        strength=int(data.get("strength") or 0),
        confidence=float(data.get("confidence") or 0.0),
        tags=normalize_tags(json_loads(data.get("tags_json"), [])),
        provider=data.get("provider") or "csv_manual",
        provider_metadata=json_loads(data.get("provider_metadata_json"), {}) or {},
        status=data.get("status") or "staged",
        duplicate_of_annotation_id=data.get("duplicate_of_annotation_id"),
        duplicate_of_candidate_id=data.get("duplicate_of_candidate_id"),
        duplicate_reason=data.get("duplicate_reason"),
        rejection_reason=data.get("rejection_reason"),
        created_at=pd.to_datetime(data.get("created_at"), utc=True).to_pydatetime(),
        updated_at=pd.to_datetime(data.get("updated_at"), utc=True).to_pydatetime(),
        reviewed_at=None if not data.get("reviewed_at") else pd.to_datetime(data.get("reviewed_at"), utc=True).to_pydatetime(),
        imported_annotation_id=data.get("imported_annotation_id"),
    )


# Type alias is only for readability; sqlite3.Row is not imported to avoid exposing sqlite internals.
sqlite_row = Any


def _find_duplicate(db_path: str | Path, candidate: ResearchEventAnnotationCandidate) -> tuple[str | None, int | None, int | None]:
    item = candidate.normalized()
    normalized_url = normalize_source_url(item.source_url)
    normalized = normalize_title(item.title)
    text_hash = stable_text_hash(item.evidence_text)
    with storage.connect(db_path) as conn:
        if normalized_url:
            row = conn.execute(
                """
                SELECT annotation_id
                FROM research_event_annotations
                WHERE LOWER(TRIM(source_url, '/')) = ?
                LIMIT 1
                """,
                (normalized_url,),
            ).fetchone()
            if row is not None:
                return "existing_annotation_source_url", int(row["annotation_id"]), None
            row = conn.execute(
                """
                SELECT candidate_id
                FROM research_event_candidates
                WHERE LOWER(TRIM(source_url, '/')) = ? AND status != 'rejected'
                LIMIT 1
                """,
                (normalized_url,),
            ).fetchone()
            if row is not None:
                return "existing_candidate_source_url", None, int(row["candidate_id"])
        row = conn.execute(
            """
            SELECT annotation_id
            FROM research_event_annotations
            WHERE ticker = ? AND event_date = ?
            """,
            (item.ticker, item.event_date.isoformat()),
        ).fetchall()
        for candidate_row in row:
            annotation = conn.execute(
                "SELECT title, evidence_text FROM research_event_annotations WHERE annotation_id = ?",
                (int(candidate_row["annotation_id"]),),
            ).fetchone()
            if annotation is not None:
                same_title = normalized and normalize_title(annotation["title"]) == normalized
                same_text = text_hash and stable_text_hash(annotation["evidence_text"]) == text_hash
                if same_title or same_text:
                    reason = "existing_annotation_title" if same_title else "existing_annotation_text_hash"
                    return reason, int(candidate_row["annotation_id"]), None
        row = conn.execute(
            """
            SELECT candidate_id
            FROM research_event_candidates
            WHERE ticker = ? AND event_date = ? AND status != 'rejected'
              AND (normalized_title = ? OR (evidence_text_hash IS NOT NULL AND evidence_text_hash = ?))
            LIMIT 1
            """,
            (item.ticker, item.event_date.isoformat(), normalized, text_hash),
        ).fetchone()
        if row is not None:
            return "existing_candidate_title_or_text_hash", None, int(row["candidate_id"])
    return None, None, None


def stage_candidate(db_path: str | Path, candidate: ResearchEventAnnotationCandidate) -> CandidateStageResult:
    create_candidate_tables(db_path)
    item = candidate.normalized()
    duplicate_reason, annotation_id, candidate_id = _find_duplicate(db_path, item)
    status = "duplicate" if duplicate_reason else "staged"
    now = _now_iso()
    dedupe_key = candidate_dedupe_key(item)
    payload = (
        item.ticker,
        item.event_date.isoformat(),
        _dt_iso(item.available_at),
        item.event_type,
        item.title,
        item.summary,
        item.source,
        item.source_url,
        item.evidence_text,
        item.sentiment_label,
        item.strength,
        item.confidence,
        annotation_json_dumps(normalize_tags(item.tags)),
        item.provider,
        json_dumps(item.provider_metadata),
        status,
        annotation_id,
        candidate_id,
        duplicate_reason,
        normalize_title(item.title),
        stable_text_hash(item.evidence_text),
        dedupe_key,
        _dt_iso(item.created_at),
        now,
    )
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO research_event_candidates (
                ticker, event_date, available_at, event_type, title, summary,
                source, source_url, evidence_text, sentiment_label, strength,
                confidence, tags_json, provider, provider_metadata_json, status,
                duplicate_of_annotation_id, duplicate_of_candidate_id,
                duplicate_reason, normalized_title, evidence_text_hash, dedupe_key,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        inserted = cursor.rowcount == 1
        row = conn.execute(
            "SELECT candidate_id, status, duplicate_reason FROM research_event_candidates WHERE dedupe_key = ?",
            (dedupe_key,),
        ).fetchone()
    if row is None:
        raise RuntimeError("Candidate stage failed and no duplicate row was found.")
    return CandidateStageResult(
        candidate_id=int(row["candidate_id"]),
        inserted=inserted,
        status=str(row["status"]),
        dedupe_key=dedupe_key,
        duplicate_reason=row["duplicate_reason"],
    )


def stage_candidates(db_path: str | Path, candidates: list[ResearchEventAnnotationCandidate]) -> list[CandidateStageResult]:
    return [stage_candidate(db_path, candidate) for candidate in candidates]


def list_candidates(
    db_path: str | Path,
    status: str | None = None,
    ticker: str | None = None,
    limit: int | None = 500,
) -> pd.DataFrame:
    create_candidate_tables(db_path)
    clauses: list[str] = []
    params: list[Any] = []
    if status:
        clauses.append("status = ?")
        params.append(status)
    if ticker:
        clauses.append("ticker = ?")
        params.append(ticker.strip().upper())
    where = "WHERE " + " AND ".join(clauses) if clauses else ""
    limit_sql = "" if limit is None else "LIMIT ?"
    if limit is not None:
        params.append(int(limit))
    with storage.connect(db_path) as conn:
        frame = pd.read_sql_query(
            f"""
            SELECT *
            FROM research_event_candidates
            {where}
            ORDER BY datetime(created_at) DESC, candidate_id DESC
            {limit_sql}
            """,
            conn,
            params=params,
        )
    if not frame.empty:
        frame["tags"] = frame["tags_json"].map(lambda value: json_loads(value, []))
        frame["provider_metadata"] = frame["provider_metadata_json"].map(lambda value: json_loads(value, {}))
        frame = enrich_quality_frame(frame)
    return frame


def candidate_counts_by_status(db_path: str | Path) -> pd.DataFrame:
    create_candidate_tables(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT status, COUNT(*) AS candidate_count
            FROM research_event_candidates
            GROUP BY status
            ORDER BY status
            """,
            conn,
        )


def set_candidate_status(db_path: str | Path, candidate_id: int, status: str, reason: str | None = None) -> None:
    if status not in {"accepted", "rejected"}:
        raise ValueError("Only accepted/rejected review status changes are supported.")
    create_candidate_tables(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        row = conn.execute(
            "SELECT status FROM research_event_candidates WHERE candidate_id = ?",
            (int(candidate_id),),
        ).fetchone()
        if row is None:
            raise ValueError(f"Candidate {candidate_id} not found.")
        if row["status"] in {"imported", "duplicate"}:
            raise ValueError(f"Candidate {candidate_id} cannot be reviewed from status {row['status']}.")
        conn.execute(
            """
            UPDATE research_event_candidates
            SET status = ?, rejection_reason = ?, reviewed_at = ?, updated_at = ?
            WHERE candidate_id = ?
            """,
            (status, reason if status == "rejected" else None, now, now, int(candidate_id)),
        )


def accept_candidate(db_path: str | Path, candidate_id: int) -> None:
    set_candidate_status(db_path, candidate_id, "accepted")


def reject_candidate(db_path: str | Path, candidate_id: int, reason: str | None = None) -> None:
    set_candidate_status(db_path, candidate_id, "rejected", reason=reason)


def _candidate_to_annotation(candidate: ResearchEventAnnotationCandidate) -> ResearchEventAnnotation:
    quality = classify_event_quality(asdict(candidate))
    quality_tags = [
        f"source_quality:{quality.source_quality}",
        f"informativeness:{quality.informativeness}",
    ]
    if quality.duplicate_theme_key:
        quality_tags.append(f"duplicate_theme_key:{quality.duplicate_theme_key}")
    return ResearchEventAnnotation(
        ticker=candidate.ticker,
        event_date=candidate.event_date,
        available_at=candidate.available_at,
        event_type=candidate.event_type,
        sentiment_label=candidate.sentiment_label,
        strength=candidate.strength,
        confidence=candidate.confidence,
        source=candidate.source,
        source_url=candidate.source_url,
        title=candidate.title,
        summary=candidate.summary,
        evidence_text=candidate.evidence_text,
        tags=normalize_tags([*candidate.tags, "candidate_import", *quality_tags]),
    )


def import_accepted_candidates(db_path: str | Path, candidate_ids: list[int] | None = None) -> CandidateImportSummary:
    create_candidate_tables(db_path)
    params: list[Any] = []
    where = "status = 'accepted'"
    if candidate_ids:
        placeholders = ",".join(["?"] * len(candidate_ids))
        where += f" AND candidate_id IN ({placeholders})"
        params.extend(int(value) for value in candidate_ids)
    with storage.connect(db_path) as conn:
        rows = conn.execute(f"SELECT * FROM research_event_candidates WHERE {where} ORDER BY candidate_id", params).fetchall()

    imported_ids: list[int] = []
    warnings: list[str] = []
    skipped = 0
    for row in rows:
        candidate = _candidate_from_row(row)
        result = insert_annotation(db_path, _candidate_to_annotation(candidate))
        now = _now_iso()
        with storage.connect(db_path) as conn:
            conn.execute(
                """
                UPDATE research_event_candidates
                SET status = 'imported', imported_annotation_id = ?, updated_at = ?
                WHERE candidate_id = ?
                """,
                (result.annotation_id, now, int(candidate.candidate_id or 0)),
            )
        if result.inserted:
            imported_ids.append(result.annotation_id)
        else:
            skipped += 1
            warnings.append(f"Candidate {candidate.candidate_id} mapped to existing annotation {result.annotation_id}.")
    return CandidateImportSummary(
        imported_count=len(imported_ids),
        skipped_count=skipped,
        imported_annotation_ids=imported_ids,
        warnings=warnings,
    )


def build_candidate_ingestion_artifact(db_path: str | Path) -> dict[str, Any]:
    create_candidate_tables(db_path)
    candidates = list_candidates(db_path, limit=None)
    if candidates.empty:
        return {
            "artifact_type": "news_event_candidate_ingestion",
            "created_at": _now_iso(),
            "candidate_count": 0,
            "status_counts": [],
            "by_event_type": [],
            "by_sentiment": [],
            "by_source": [],
            "source_quality_distribution": [],
            "informativeness_distribution": [],
            "scanner_scoring_effect": 0,
            "research_only": True,
        }
    quality = quality_distribution(candidates)
    return {
        "artifact_type": "news_event_candidate_ingestion",
        "created_at": _now_iso(),
        "candidate_count": int(len(candidates)),
        "status_counts": candidates.groupby("status").size().reset_index(name="count").sort_values("status").to_dict("records"),
        "by_event_type": candidates.groupby("event_type").size().reset_index(name="count").sort_values("event_type").to_dict("records"),
        "by_sentiment": candidates.groupby("sentiment_label").size().reset_index(name="count").sort_values("sentiment_label").to_dict("records"),
        "by_source": candidates.groupby("source").size().reset_index(name="count").sort_values("source").to_dict("records"),
        "source_quality_distribution": quality["source_quality_distribution"],
        "informativeness_distribution": quality["informativeness_distribution"],
        "quality_summary": quality,
        "duplicates": candidates[candidates["status"].eq("duplicate")][
            [
                "candidate_id",
                "ticker",
                "event_date",
                "event_type",
                "title",
                "duplicate_reason",
                "duplicate_of_annotation_id",
                "duplicate_of_candidate_id",
            ]
        ].to_dict("records"),
        "scanner_scoring_effect": 0,
        "research_only": True,
    }
