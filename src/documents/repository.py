from __future__ import annotations

import sqlite3
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage
from src.documents.models import SourceDocument
from src.documents.text_cleaning import clean_text, compute_text_hash, join_warnings, text_quality_warnings


DOCUMENT_COLUMNS = [
    "document_id",
    "ticker",
    "catalyst_id",
    "document_type",
    "source",
    "source_url",
    "accession_number",
    "filing_type",
    "title",
    "published_at",
    "raw_text",
    "cleaned_text",
    "text_hash",
    "parsing_status",
    "warnings",
    "created_at",
    "updated_at",
    "raw_payload_json",
]


def create_documents_table(db_path: str | Path) -> None:
    storage.init_db(db_path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _date_iso(value: date | datetime | str | None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    return pd.to_datetime(value).date().isoformat()


def build_source_document(
    ticker: str,
    document_type: str,
    title: str,
    raw_text: str,
    source: str = "manual",
    catalyst_id: int | None = None,
    source_url: str | None = None,
    accession_number: str | None = None,
    filing_type: str | None = None,
    published_at: date | datetime | str | None = None,
    parsing_status: str | None = None,
    warnings: str | list[str] | None = None,
    raw_payload_json: str | None = None,
) -> SourceDocument:
    cleaned = clean_text(raw_text)
    quality_warnings = text_quality_warnings(raw_text, cleaned)
    status = parsing_status or ("success" if cleaned else "failed")
    if status == "success" and quality_warnings:
        status = "partial"
    fallback_parts = [source_url, accession_number, ticker, document_type, title, _date_iso(published_at)]
    text_hash = compute_text_hash(cleaned, fallback_parts)
    return SourceDocument(
        ticker=ticker,
        catalyst_id=catalyst_id,
        document_type=document_type,
        source=source,
        source_url=source_url,
        accession_number=accession_number,
        filing_type=filing_type,
        title=title,
        published_at=published_at,
        raw_text=raw_text or "",
        cleaned_text=cleaned,
        text_hash=text_hash,
        parsing_status=status,
        warnings=join_warnings(warnings, quality_warnings),
        raw_payload_json=raw_payload_json,
    )


def _document_row(document: SourceDocument) -> dict[str, Any]:
    now = _now_iso()
    created_at = document.created_at.isoformat(timespec="seconds") if isinstance(document.created_at, datetime) else now
    updated_at = document.updated_at.isoformat(timespec="seconds") if isinstance(document.updated_at, datetime) else now
    cleaned = document.cleaned_text or clean_text(document.raw_text)
    fallback_parts = [
        document.source_url,
        document.accession_number,
        document.ticker,
        document.document_type,
        document.title,
        _date_iso(document.published_at),
    ]
    text_hash = document.text_hash or compute_text_hash(cleaned, fallback_parts)
    warnings = join_warnings(document.warnings, text_quality_warnings(document.raw_text, cleaned))
    return {
        "ticker": document.ticker.upper(),
        "catalyst_id": document.catalyst_id,
        "document_type": document.document_type,
        "source": document.source,
        "source_url": document.source_url,
        "accession_number": document.accession_number,
        "filing_type": document.filing_type,
        "title": document.title.strip(),
        "published_at": _date_iso(document.published_at),
        "raw_text": document.raw_text,
        "cleaned_text": cleaned,
        "text_hash": text_hash,
        "parsing_status": document.parsing_status,
        "warnings": warnings,
        "created_at": created_at,
        "updated_at": updated_at,
        "raw_payload_json": document.raw_payload_json,
    }


def _empty_documents_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=DOCUMENT_COLUMNS)


def insert_document(db_path: str | Path, document: SourceDocument) -> int:
    create_documents_table(db_path)
    row = _document_row(document)
    with storage.connect(db_path) as conn:
        if row["source_url"]:
            existing = conn.execute(
                "SELECT document_id FROM source_documents WHERE source_url = ? ORDER BY document_id LIMIT 1",
                (row["source_url"],),
            ).fetchone()
            if existing is not None:
                return int(existing["document_id"])

        existing = conn.execute(
            "SELECT document_id FROM source_documents WHERE text_hash = ? ORDER BY document_id LIMIT 1",
            (row["text_hash"],),
        ).fetchone()
        if existing is not None:
            return int(existing["document_id"])

        try:
            conn.execute(
                """
                INSERT INTO source_documents (
                    ticker, catalyst_id, document_type, source, source_url, accession_number,
                    filing_type, title, published_at, raw_text, cleaned_text, text_hash,
                    parsing_status, warnings, created_at, updated_at, raw_payload_json
                )
                VALUES (
                    :ticker, :catalyst_id, :document_type, :source, :source_url, :accession_number,
                    :filing_type, :title, :published_at, :raw_text, :cleaned_text, :text_hash,
                    :parsing_status, :warnings, :created_at, :updated_at, :raw_payload_json
                )
                """,
                row,
            )
        except sqlite3.IntegrityError:
            existing = conn.execute(
                "SELECT document_id FROM source_documents WHERE text_hash = ? ORDER BY document_id LIMIT 1",
                (row["text_hash"],),
            ).fetchone()
            if existing is None:
                raise
            return int(existing["document_id"])
        inserted = conn.execute(
            "SELECT document_id FROM source_documents WHERE text_hash = ? ORDER BY document_id LIMIT 1",
            (row["text_hash"],),
        ).fetchone()
        return int(inserted["document_id"])


def update_document(db_path: str | Path, document_id: int, updates: dict[str, Any]) -> bool:
    create_documents_table(db_path)
    allowed = {
        "ticker",
        "catalyst_id",
        "document_type",
        "source",
        "source_url",
        "accession_number",
        "filing_type",
        "title",
        "published_at",
        "raw_text",
        "cleaned_text",
        "text_hash",
        "parsing_status",
        "warnings",
        "raw_payload_json",
    }
    cleaned_updates = {key: value for key, value in updates.items() if key in allowed}
    if not cleaned_updates:
        return False

    if "ticker" in cleaned_updates and cleaned_updates["ticker"]:
        cleaned_updates["ticker"] = str(cleaned_updates["ticker"]).upper().strip()
    if "published_at" in cleaned_updates:
        cleaned_updates["published_at"] = _date_iso(cleaned_updates["published_at"])
    if "raw_text" in cleaned_updates and "cleaned_text" not in cleaned_updates:
        cleaned_updates["cleaned_text"] = clean_text(str(cleaned_updates["raw_text"] or ""))
    if ("raw_text" in cleaned_updates or "cleaned_text" in cleaned_updates) and "text_hash" not in cleaned_updates:
        current = get_document_by_id(db_path, document_id) or {}
        fallback_parts = [
            cleaned_updates.get("source_url", current.get("source_url")),
            cleaned_updates.get("accession_number", current.get("accession_number")),
            cleaned_updates.get("ticker", current.get("ticker")),
            cleaned_updates.get("document_type", current.get("document_type")),
            cleaned_updates.get("title", current.get("title")),
            cleaned_updates.get("published_at", current.get("published_at")),
        ]
        cleaned_updates["text_hash"] = compute_text_hash(str(cleaned_updates.get("cleaned_text") or ""), fallback_parts)
    cleaned_updates["updated_at"] = _now_iso()
    cleaned_updates["document_id"] = document_id
    assignments = ", ".join(f"{key} = :{key}" for key in cleaned_updates if key != "document_id")
    with storage.connect(db_path) as conn:
        result = conn.execute(f"UPDATE source_documents SET {assignments} WHERE document_id = :document_id", cleaned_updates)
        return result.rowcount > 0


def delete_document(db_path: str | Path, document_id: int) -> bool:
    create_documents_table(db_path)
    with storage.connect(db_path) as conn:
        result = conn.execute("DELETE FROM source_documents WHERE document_id = ?", (document_id,))
        return result.rowcount > 0


def get_document_by_id(db_path: str | Path, document_id: int) -> dict[str, Any] | None:
    create_documents_table(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute(
            f"SELECT {', '.join(DOCUMENT_COLUMNS)} FROM source_documents WHERE document_id = ?",
            (document_id,),
        ).fetchone()
    return dict(row) if row is not None else None


def _query_documents(
    db_path: str | Path,
    where: str = "",
    params: tuple[Any, ...] = (),
    limit: int | None = None,
) -> pd.DataFrame:
    create_documents_table(db_path)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    sql = f"""
        SELECT {", ".join(DOCUMENT_COLUMNS)}
        FROM source_documents
        {where}
        ORDER BY COALESCE(published_at, created_at) DESC, document_id DESC
        {limit_clause}
    """
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return _empty_documents_frame()
    return df


def list_documents_by_ticker(db_path: str | Path, ticker: str, limit: int | None = 100) -> pd.DataFrame:
    return _query_documents(db_path, "WHERE ticker = ?", (ticker.upper().strip(),), limit)


def list_documents_by_catalyst_id(db_path: str | Path, catalyst_id: int, limit: int | None = 100) -> pd.DataFrame:
    return _query_documents(db_path, "WHERE catalyst_id = ?", (int(catalyst_id),), limit)


def list_recent_documents(db_path: str | Path, limit: int | None = 200) -> pd.DataFrame:
    return _query_documents(db_path, limit=limit)


def link_document_to_catalyst(db_path: str | Path, document_id: int, catalyst_id: int) -> bool:
    return update_document(db_path, document_id, {"catalyst_id": int(catalyst_id)})


def unlink_document_from_catalyst(db_path: str | Path, document_id: int) -> bool:
    return update_document(db_path, document_id, {"catalyst_id": None})


def document_counts_by_catalyst(db_path: str | Path, catalyst_ids: list[int]) -> dict[int, int]:
    create_documents_table(db_path)
    ids = [int(catalyst_id) for catalyst_id in catalyst_ids if pd.notna(catalyst_id)]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT catalyst_id, COUNT(*) AS document_count
            FROM source_documents
            WHERE catalyst_id IN ({placeholders})
            GROUP BY catalyst_id
            """,
            ids,
        ).fetchall()
    return {int(row["catalyst_id"]): int(row["document_count"]) for row in rows}


def document_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "date",
        "ticker",
        "document type",
        "title",
        "source",
        "linked catalyst?",
        "parsing status",
        "text length",
        "warnings",
        "created_at",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    display = df.copy()
    display["date"] = display["published_at"].fillna("")
    display["document type"] = display["document_type"]
    display["linked catalyst?"] = display["catalyst_id"].notna().map({True: "yes", False: "no"})
    display["parsing status"] = display["parsing_status"]
    display["text length"] = display["cleaned_text"].fillna("").str.len()
    return display[
        [
            "date",
            "ticker",
            "document type",
            "title",
            "source",
            "linked catalyst?",
            "parsing status",
            "text length",
            "warnings",
            "created_at",
        ]
    ]

