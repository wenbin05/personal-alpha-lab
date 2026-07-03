from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.annotations.models import ResearchEventAnnotation, annotation_dedupe_key, json_dumps, normalize_tags
from src.data import storage


@dataclass(frozen=True)
class AnnotationInsertResult:
    annotation_id: int
    inserted: bool
    dedupe_key: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dt_iso(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=UTC)
    return value.astimezone(UTC).isoformat(timespec="seconds")


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def create_annotation_tables(db_path: str | Path) -> None:
    storage.init_db(db_path)


def insert_annotation(db_path: str | Path, annotation: ResearchEventAnnotation) -> AnnotationInsertResult:
    """Insert one research-only annotation, skipping exact duplicate keys."""
    create_annotation_tables(db_path)
    item = annotation.normalized()
    now = _now_iso()
    created_at = _dt_iso(item.created_at) if item.created_at else now
    updated_at = now
    dedupe_key = annotation_dedupe_key(item)
    payload = (
        item.ticker,
        item.event_date.isoformat(),
        _dt_iso(item.available_at),
        item.event_type,
        item.sentiment_label,
        item.strength,
        item.confidence,
        item.source,
        item.source_url,
        item.title,
        item.summary,
        item.evidence_text,
        json_dumps(normalize_tags(item.tags)),
        1,
        0,
        dedupe_key,
        created_at,
        updated_at,
    )
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO research_event_annotations (
                ticker, event_date, available_at, event_type, sentiment_label,
                strength, confidence, source, source_url, title, summary,
                evidence_text, tags_json, research_only, scanner_scoring_effect,
                dedupe_key, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            payload,
        )
        inserted = cursor.rowcount == 1
        row = conn.execute(
            "SELECT annotation_id FROM research_event_annotations WHERE dedupe_key = ?",
            (dedupe_key,),
        ).fetchone()
    if row is None:
        raise RuntimeError("Annotation insert failed and no duplicate row was found.")
    return AnnotationInsertResult(annotation_id=int(row["annotation_id"]), inserted=inserted, dedupe_key=dedupe_key)


def bulk_insert_annotations(db_path: str | Path, annotations: list[ResearchEventAnnotation]) -> list[AnnotationInsertResult]:
    return [insert_annotation(db_path, annotation) for annotation in annotations]


def list_annotations(
    db_path: str | Path,
    ticker: str | None = None,
    limit: int | None = 500,
) -> pd.DataFrame:
    create_annotation_tables(db_path)
    where = ""
    params: list[Any] = []
    if ticker:
        where = "WHERE ticker = ?"
        params.append(ticker.strip().upper())
    limit_sql = "" if limit is None else "LIMIT ?"
    if limit is not None:
        params.append(int(limit))
    with storage.connect(db_path) as conn:
        frame = pd.read_sql_query(
            f"""
            SELECT *
            FROM research_event_annotations
            {where}
            ORDER BY datetime(available_at) DESC, annotation_id DESC
            {limit_sql}
            """,
            conn,
            params=params,
        )
    if not frame.empty:
        frame["tags"] = frame["tags_json"].map(lambda value: _json_loads(value, []))
    return frame


def annotation_counts_by_ticker(db_path: str | Path) -> pd.DataFrame:
    create_annotation_tables(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT ticker, COUNT(*) AS annotation_count,
                   SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) AS positive_count,
                   SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) AS negative_count,
                   MIN(event_date) AS first_event_date,
                   MAX(event_date) AS last_event_date
            FROM research_event_annotations
            GROUP BY ticker
            ORDER BY ticker
            """,
            conn,
        )

