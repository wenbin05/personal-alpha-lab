from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from src.catalysts.models import CatalystEvent
from src.data import storage


CATALYST_COLUMNS = [
    "id",
    "ticker",
    "event_date",
    "event_time",
    "event_type",
    "title",
    "summary",
    "source",
    "source_url",
    "sentiment_label",
    "catalyst_strength",
    "confidence",
    "is_manual",
    "available_at",
    "created_at",
    "updated_at",
    "raw_payload_json",
    "dedupe_key",
]


def create_catalyst_table(db_path: str | Path) -> None:
    storage.init_db(db_path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _normalize_event_date(value: Any) -> str:
    return pd.to_datetime(value).date().isoformat()


def _iso_datetime_or_none(value: Any) -> str | None:
    if value is None:
        return None
    try:
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime().isoformat(timespec="seconds")
    except Exception:
        return None


def make_dedupe_key(event: CatalystEvent) -> str:
    source_url = (event.source_url or "").strip().lower()
    parts = [
        event.ticker.upper(),
        event.event_date.isoformat(),
        event.event_type,
        event.title.strip().lower(),
        event.source.strip().lower(),
        source_url,
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _event_row(event: CatalystEvent, dedupe_key: str | None = None) -> dict[str, Any]:
    now = _now_iso()
    created_at = event.created_at.isoformat(timespec="seconds") if isinstance(event.created_at, datetime) else now
    updated_at = event.updated_at.isoformat(timespec="seconds") if isinstance(event.updated_at, datetime) else now
    available_at = _iso_datetime_or_none(event.available_at) or created_at
    return {
        "ticker": event.ticker.upper(),
        "event_date": event.event_date.isoformat(),
        "event_time": event.event_time,
        "event_type": event.event_type,
        "title": event.title.strip(),
        "summary": event.summary.strip(),
        "source": event.source.strip() or "unknown",
        "source_url": event.source_url,
        "sentiment_label": event.sentiment_label,
        "catalyst_strength": int(max(0, min(10, event.catalyst_strength))),
        "confidence": float(max(0.0, min(1.0, event.confidence))),
        "is_manual": 1 if event.is_manual else 0,
        "available_at": available_at,
        "created_at": created_at,
        "updated_at": updated_at,
        "raw_payload_json": event.raw_payload_json,
        "dedupe_key": dedupe_key or make_dedupe_key(event),
    }


def insert_catalyst(db_path: str | Path, event: CatalystEvent) -> int:
    catalyst_id, _ = insert_catalyst_with_status(db_path, event)
    return catalyst_id


def insert_catalyst_with_status(db_path: str | Path, event: CatalystEvent) -> tuple[int, bool]:
    create_catalyst_table(db_path)
    row = _event_row(event)
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO catalysts (
                ticker, event_date, event_time, event_type, title, summary, source, source_url,
                sentiment_label, catalyst_strength, confidence, is_manual, available_at, created_at, updated_at,
                raw_payload_json, dedupe_key
            )
            VALUES (
                :ticker, :event_date, :event_time, :event_type, :title, :summary, :source, :source_url,
                :sentiment_label, :catalyst_strength, :confidence, :is_manual, :available_at, :created_at, :updated_at,
                :raw_payload_json, :dedupe_key
            )
            """,
            row,
        )
        existing = conn.execute("SELECT id FROM catalysts WHERE dedupe_key = ?", (row["dedupe_key"],)).fetchone()
        catalyst_id = int(existing["id"])
        if cursor.rowcount > 0:
            after = dict(conn.execute(f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?", (catalyst_id,)).fetchone())
            _record_catalyst_revision(
                conn,
                catalyst_id,
                after["ticker"],
                "create",
                None,
                after,
                after.get("available_at") or after.get("created_at"),
            )
        return catalyst_id, bool(cursor.rowcount > 0)


def bulk_insert_catalysts(db_path: str | Path, events: list[CatalystEvent]) -> list[int]:
    return [insert_catalyst(db_path, event) for event in events]


def update_catalyst(db_path: str | Path, catalyst_id: int, updates: dict[str, Any]) -> bool:
    create_catalyst_table(db_path)
    allowed = {
        "event_date",
        "event_time",
        "event_type",
        "title",
        "summary",
        "source",
        "source_url",
        "sentiment_label",
        "catalyst_strength",
        "confidence",
        "available_at",
        "raw_payload_json",
    }
    cleaned = {key: value for key, value in updates.items() if key in allowed}
    if not cleaned:
        return False
    if "event_date" in cleaned:
        cleaned["event_date"] = _normalize_event_date(cleaned["event_date"])
    if "available_at" in cleaned:
        cleaned["available_at"] = _iso_datetime_or_none(cleaned["available_at"])
    cleaned["updated_at"] = _now_iso()
    assignments = ", ".join(f"{key} = :{key}" for key in cleaned)
    cleaned["id"] = catalyst_id
    with storage.connect(db_path) as conn:
        before_row = conn.execute(f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?", (catalyst_id,)).fetchone()
        if before_row is None:
            return False
        before = dict(before_row)
        result = conn.execute(f"UPDATE catalysts SET {assignments} WHERE id = :id", cleaned)
        if result.rowcount <= 0:
            return False
        after = dict(conn.execute(f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?", (catalyst_id,)).fetchone())
        _record_catalyst_revision(conn, catalyst_id, after["ticker"], "update", before, after, after.get("updated_at"))
        return True


def delete_catalyst(db_path: str | Path, catalyst_id: int) -> bool:
    create_catalyst_table(db_path)
    with storage.connect(db_path) as conn:
        before_row = conn.execute(f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?", (catalyst_id,)).fetchone()
        if before_row is None:
            return False
        before = dict(before_row)
        result = conn.execute("DELETE FROM catalysts WHERE id = ?", (catalyst_id,))
        if result.rowcount <= 0:
            return False
        _record_catalyst_revision(conn, catalyst_id, before["ticker"], "delete", before, None, _now_iso())
        return True


def _record_catalyst_revision(
    conn,
    catalyst_id: int,
    ticker: str,
    action: str,
    before_snapshot: dict[str, Any] | None,
    after_snapshot: dict[str, Any] | None,
    effective_timestamp: Any | None = None,
) -> None:
    recorded = _now_iso()
    effective = str(effective_timestamp or recorded)
    conn.execute(
        """
        INSERT INTO catalyst_revisions (
            catalyst_id, ticker, action, before_snapshot_json, after_snapshot_json,
            effective_timestamp, recorded_timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(catalyst_id),
            ticker.upper(),
            action,
            None if before_snapshot is None else _json_dumps(before_snapshot),
            None if after_snapshot is None else _json_dumps(after_snapshot),
            effective,
            recorded,
        ),
    )


def _query_catalysts(db_path: str | Path, where: str = "", params: tuple[Any, ...] = (), limit: int | None = None) -> pd.DataFrame:
    create_catalyst_table(db_path)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    sql = f"""
        SELECT {", ".join(CATALYST_COLUMNS)}
        FROM catalysts
        {where}
        ORDER BY event_date DESC, id DESC
        {limit_clause}
    """
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return pd.DataFrame(columns=CATALYST_COLUMNS)
    df["is_manual"] = df["is_manual"].astype(bool)
    df["event_date"] = pd.to_datetime(df["event_date"]).dt.date
    return df


def list_catalysts_by_ticker(db_path: str | Path, ticker: str, limit: int | None = 100) -> pd.DataFrame:
    return _query_catalysts(db_path, "WHERE ticker = ?", (ticker.upper(),), limit)


def list_catalysts_by_date_range(
    db_path: str | Path,
    start_date: date,
    end_date: date,
    limit: int | None = 500,
) -> pd.DataFrame:
    return _query_catalysts(
        db_path,
        "WHERE event_date BETWEEN ? AND ?",
        (start_date.isoformat(), end_date.isoformat()),
        limit,
    )


def list_recent_catalysts(db_path: str | Path, days: int = 90, limit: int | None = 500) -> pd.DataFrame:
    end = datetime.now(UTC).date() + timedelta(days=45)
    start = datetime.now(UTC).date() - timedelta(days=days)
    return list_catalysts_by_date_range(db_path, start, end, limit)


def get_latest_catalyst_score_for_ticker(db_path: str | Path, ticker: str) -> dict[str, Any]:
    from src.features.catalyst import get_catalyst_features

    events = list_catalysts_by_ticker(db_path, ticker, limit=50)
    return get_catalyst_features(ticker, events)


def catalyst_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "event type",
                "title",
                "sentiment",
                "strength",
                "confidence",
                "source",
                "manual/system",
                "created_at",
            ]
        )
    display = df.copy()
    display["date"] = display["event_date"].astype(str)
    display["event type"] = display["event_type"]
    display["sentiment"] = display["sentiment_label"]
    display["strength"] = display["catalyst_strength"]
    display["manual/system"] = display["is_manual"].map({True: "manual", False: "system"})
    if "available_at" not in display.columns:
        display["available_at"] = display["created_at"]
    return display[
        [
            "date",
            "ticker",
            "event type",
            "title",
            "sentiment",
            "strength",
            "confidence",
            "source",
            "manual/system",
            "available_at",
            "created_at",
        ]
    ]
