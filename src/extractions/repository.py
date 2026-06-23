from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage
from src.extractions.models import LLMExtraction
from src.extractions.validation import json_list_dumps, normalize_extraction_payload, safe_json_list


EXTRACTION_COLUMNS = [
    "extraction_id",
    "document_id",
    "catalyst_id",
    "ticker",
    "provider",
    "model_name",
    "extraction_type",
    "event_type_detected",
    "sentiment_label",
    "catalyst_strength",
    "risk_severity",
    "confidence",
    "document_relevance",
    "evidence_sufficiency",
    "time_horizon",
    "key_positive_points",
    "key_risks",
    "evidence_snippets",
    "short_summary",
    "detailed_summary",
    "proposed_score_effect",
    "review_status",
    "reviewer_note",
    "reviewed_at",
    "created_at",
    "updated_at",
    "raw_llm_response_json",
    "prompt_version",
    "extraction_warnings",
]


def create_extraction_table(db_path: str | Path) -> None:
    storage.init_db(db_path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dt_iso(value: datetime | str | None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    try:
        return pd.to_datetime(value).to_pydatetime().isoformat(timespec="seconds")
    except Exception:
        return None


def _row_from_extraction(extraction: LLMExtraction | dict[str, Any]) -> dict[str, Any]:
    if isinstance(extraction, LLMExtraction):
        normalized = normalize_extraction_payload(extraction.model_dump())
    else:
        normalized = normalize_extraction_payload(extraction)

    return {
        "document_id": normalized["document_id"],
        "catalyst_id": normalized["catalyst_id"],
        "ticker": normalized["ticker"],
        "provider": normalized["provider"],
        "model_name": normalized["model_name"],
        "extraction_type": normalized["extraction_type"],
        "event_type_detected": normalized["event_type_detected"],
        "sentiment_label": normalized["sentiment_label"],
        "catalyst_strength": normalized["catalyst_strength"],
        "risk_severity": normalized["risk_severity"],
        "confidence": normalized["confidence"],
        "document_relevance": normalized["document_relevance"],
        "evidence_sufficiency": normalized["evidence_sufficiency"],
        "time_horizon": normalized["time_horizon"],
        "key_positive_points": json_list_dumps(normalized["key_positive_points"]),
        "key_risks": json_list_dumps(normalized["key_risks"]),
        "evidence_snippets": json_list_dumps(normalized["evidence_snippets"]),
        "short_summary": normalized["short_summary"],
        "detailed_summary": normalized["detailed_summary"],
        "proposed_score_effect": normalized["proposed_score_effect"],
        "review_status": normalized["review_status"],
        "reviewer_note": normalized["reviewer_note"],
        "reviewed_at": _dt_iso(normalized["reviewed_at"]),
        "created_at": _dt_iso(normalized["created_at"]) or _now_iso(),
        "updated_at": _dt_iso(normalized["updated_at"]) or _now_iso(),
        "raw_llm_response_json": normalized["raw_llm_response_json"],
        "prompt_version": normalized["prompt_version"],
        "extraction_warnings": normalized["extraction_warnings"],
    }


def _empty_extractions_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=EXTRACTION_COLUMNS)


def _decode_json_list_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    decoded = df.copy()
    for column in ["key_positive_points", "key_risks", "evidence_snippets"]:
        decoded[column] = decoded[column].apply(safe_json_list)
    return decoded


def insert_extraction(db_path: str | Path, extraction: LLMExtraction | dict[str, Any]) -> int:
    create_extraction_table(db_path)
    row = _row_from_extraction(extraction)
    row["review_status"] = "pending_review"
    row["reviewed_at"] = None
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO llm_extractions (
                document_id, catalyst_id, ticker, provider, model_name, extraction_type,
                event_type_detected, sentiment_label, catalyst_strength, risk_severity,
                confidence, document_relevance, evidence_sufficiency, time_horizon,
                key_positive_points, key_risks, evidence_snippets,
                short_summary, detailed_summary, proposed_score_effect, review_status,
                reviewer_note, reviewed_at, created_at, updated_at, raw_llm_response_json,
                prompt_version, extraction_warnings
            )
            VALUES (
                :document_id, :catalyst_id, :ticker, :provider, :model_name, :extraction_type,
                :event_type_detected, :sentiment_label, :catalyst_strength, :risk_severity,
                :confidence, :document_relevance, :evidence_sufficiency, :time_horizon,
                :key_positive_points, :key_risks, :evidence_snippets,
                :short_summary, :detailed_summary, :proposed_score_effect, :review_status,
                :reviewer_note, :reviewed_at, :created_at, :updated_at, :raw_llm_response_json,
                :prompt_version, :extraction_warnings
            )
            """,
            row,
        )
        inserted = conn.execute("SELECT last_insert_rowid() AS extraction_id").fetchone()
    return int(inserted["extraction_id"])


def update_extraction(db_path: str | Path, extraction_id: int, updates: dict[str, Any]) -> bool:
    create_extraction_table(db_path)
    allowed = set(EXTRACTION_COLUMNS) - {"extraction_id", "created_at"}
    cleaned = {key: value for key, value in updates.items() if key in allowed}
    if not cleaned:
        return False

    current = get_extraction_by_id(db_path, extraction_id) or {}
    normalized = normalize_extraction_payload({**current, **cleaned})
    row = _row_from_extraction(normalized)
    row["updated_at"] = _now_iso()
    if normalized["review_status"] == "pending_review":
        row["reviewed_at"] = None
    elif not row.get("reviewed_at"):
        row["reviewed_at"] = _now_iso()

    assignments = ", ".join(f"{key} = :{key}" for key in row)
    row["extraction_id"] = int(extraction_id)
    with storage.connect(db_path) as conn:
        result = conn.execute(f"UPDATE llm_extractions SET {assignments} WHERE extraction_id = :extraction_id", row)
        return result.rowcount > 0


def get_extraction_by_id(db_path: str | Path, extraction_id: int) -> dict[str, Any] | None:
    create_extraction_table(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute(
            f"SELECT {', '.join(EXTRACTION_COLUMNS)} FROM llm_extractions WHERE extraction_id = ?",
            (int(extraction_id),),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    for column in ["key_positive_points", "key_risks", "evidence_snippets"]:
        data[column] = safe_json_list(data.get(column))
    return data


def _query_extractions(
    db_path: str | Path,
    where: str = "",
    params: tuple[Any, ...] = (),
    limit: int | None = None,
) -> pd.DataFrame:
    create_extraction_table(db_path)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    sql = f"""
        SELECT {", ".join(EXTRACTION_COLUMNS)}
        FROM llm_extractions
        {where}
        ORDER BY created_at DESC, extraction_id DESC
        {limit_clause}
    """
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return _empty_extractions_frame()
    return _decode_json_list_columns(df)


def list_extractions_by_ticker(db_path: str | Path, ticker: str, limit: int | None = 100) -> pd.DataFrame:
    return _query_extractions(db_path, "WHERE ticker = ?", (ticker.upper().strip(),), limit)


def list_extractions_by_document_id(db_path: str | Path, document_id: int, limit: int | None = 100) -> pd.DataFrame:
    return _query_extractions(db_path, "WHERE document_id = ?", (int(document_id),), limit)


def list_pending_review_extractions(db_path: str | Path, limit: int | None = 200) -> pd.DataFrame:
    return _query_extractions(db_path, "WHERE review_status = 'pending_review'", limit=limit)


def list_recent_extractions(db_path: str | Path, limit: int | None = 500) -> pd.DataFrame:
    return _query_extractions(db_path, limit=limit)


def list_reviewed_extractions(db_path: str | Path, limit: int | None = 500) -> pd.DataFrame:
    return _query_extractions(
        db_path,
        "WHERE review_status IN ('approved', 'rejected', 'superseded')",
        limit=limit,
    )


def _set_review_status(db_path: str | Path, extraction_id: int, status: str, reviewer_note: str = "") -> bool:
    current = get_extraction_by_id(db_path, extraction_id)
    if current is None or current.get("review_status") != "pending_review":
        return False
    return update_extraction(
        db_path,
        extraction_id,
        {
            "review_status": status,
            "reviewer_note": reviewer_note,
            "reviewed_at": _now_iso(),
        },
    )


def approve_extraction(db_path: str | Path, extraction_id: int, reviewer_note: str = "") -> bool:
    return _set_review_status(db_path, extraction_id, "approved", reviewer_note)


def reject_extraction(db_path: str | Path, extraction_id: int, reviewer_note: str = "") -> bool:
    return _set_review_status(db_path, extraction_id, "rejected", reviewer_note)


def supersede_extraction(db_path: str | Path, extraction_id: int, reviewer_note: str = "") -> bool:
    return _set_review_status(db_path, extraction_id, "superseded", reviewer_note)


def delete_extraction(db_path: str | Path, extraction_id: int) -> bool:
    create_extraction_table(db_path)
    with storage.connect(db_path) as conn:
        result = conn.execute("DELETE FROM llm_extractions WHERE extraction_id = ?", (int(extraction_id),))
        return result.rowcount > 0


def extraction_summary_by_document(db_path: str | Path, document_ids: list[int]) -> dict[int, dict[str, Any]]:
    create_extraction_table(db_path)
    ids = [int(document_id) for document_id in document_ids if pd.notna(document_id)]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    with storage.connect(db_path) as conn:
        count_rows = conn.execute(
            f"""
            SELECT document_id,
                   COUNT(*) AS extraction_count,
                   SUM(CASE WHEN review_status = 'pending_review' THEN 1 ELSE 0 END) AS pending_count
            FROM llm_extractions
            WHERE document_id IN ({placeholders})
            GROUP BY document_id
            """,
            ids,
        ).fetchall()
        latest_rows = conn.execute(
            f"""
            SELECT extraction_id, document_id, review_status, created_at
            FROM llm_extractions
            WHERE document_id IN ({placeholders})
            ORDER BY datetime(created_at) DESC, extraction_id DESC
            """,
            ids,
        ).fetchall()

    summary = {
        int(row["document_id"]): {
            "extraction_count": int(row["extraction_count"] or 0),
            "pending_count": int(row["pending_count"] or 0),
            "latest_review_status": None,
            "latest_extraction_id": None,
        }
        for row in count_rows
    }
    for row in latest_rows:
        document_id = int(row["document_id"])
        if document_id not in summary:
            continue
        if summary[document_id]["latest_extraction_id"] is None:
            summary[document_id]["latest_review_status"] = row["review_status"]
            summary[document_id]["latest_extraction_id"] = int(row["extraction_id"])
    return summary


def extraction_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "created_at",
        "ticker",
        "document_id",
        "provider",
        "extraction_type",
        "event_type_detected",
        "sentiment_label",
        "catalyst_strength",
        "risk_severity",
        "confidence",
        "document_relevance",
        "evidence_sufficiency",
        "proposed_score_effect",
        "review_status",
        "short_summary",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    return df[columns].copy()
