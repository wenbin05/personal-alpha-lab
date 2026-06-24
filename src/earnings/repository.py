from __future__ import annotations

import hashlib
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage
from src.earnings.models import EarningsEvent


EARNINGS_COLUMNS = [
    "earnings_event_id",
    "ticker",
    "fiscal_period_end",
    "announced_at",
    "available_at",
    "timing",
    "eps_estimate",
    "eps_actual",
    "eps_surprise",
    "eps_surprise_percent",
    "revenue_estimate",
    "revenue_actual",
    "revenue_surprise_percent",
    "currency",
    "provider",
    "provider_event_id",
    "fetched_at",
    "raw_payload_json",
    "raw_payload_hash",
    "data_quality_status",
    "warnings",
    "created_at",
    "updated_at",
    "dedupe_key",
]


def create_earnings_tables(db_path: str | Path) -> None:
    storage.init_db(db_path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _iso_datetime(value: Any | None) -> str | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.to_pydatetime().isoformat(timespec="seconds")


def _iso_date(value: Any | None) -> str | None:
    if value is None:
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def raw_payload_hash(raw_payload_json: str | None) -> str:
    payload = raw_payload_json or "{}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def make_earnings_dedupe_key(event: EarningsEvent) -> str:
    provider_id = (event.provider_event_id or "").strip().lower()
    if provider_id:
        return hashlib.sha256(
            "|".join([event.ticker.upper(), event.provider.strip().lower(), provider_id]).encode("utf-8")
        ).hexdigest()
    parts = [
        event.ticker.upper(),
        event.provider.strip().lower(),
        event.announced_at.isoformat(timespec="seconds") if event.announced_at else "",
        event.fiscal_period_end.isoformat() if event.fiscal_period_end else "",
        "" if event.eps_estimate is None else f"{event.eps_estimate:.8g}",
        "" if event.eps_actual is None else f"{event.eps_actual:.8g}",
    ]
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _event_row(event: EarningsEvent, dedupe_key: str | None = None) -> dict[str, Any]:
    now = _now_iso()
    raw_hash = event.raw_payload_hash or raw_payload_hash(event.raw_payload_json)
    return {
        "ticker": event.ticker.upper(),
        "fiscal_period_end": _iso_date(event.fiscal_period_end),
        "announced_at": _iso_datetime(event.announced_at),
        "available_at": _iso_datetime(event.available_at) or now,
        "timing": event.timing,
        "eps_estimate": event.eps_estimate,
        "eps_actual": event.eps_actual,
        "eps_surprise": event.eps_surprise,
        "eps_surprise_percent": event.eps_surprise_percent,
        "revenue_estimate": event.revenue_estimate,
        "revenue_actual": event.revenue_actual,
        "revenue_surprise_percent": event.revenue_surprise_percent,
        "currency": event.currency,
        "provider": event.provider,
        "provider_event_id": event.provider_event_id,
        "fetched_at": _iso_datetime(event.fetched_at) or now,
        "raw_payload_json": event.raw_payload_json,
        "raw_payload_hash": raw_hash,
        "data_quality_status": event.data_quality_status,
        "warnings": _json_dumps(event.warnings),
        "created_at": _iso_datetime(event.created_at) or now,
        "updated_at": _iso_datetime(event.updated_at) or now,
        "dedupe_key": dedupe_key or make_earnings_dedupe_key(event),
    }


def _record_revision(
    conn: Any,
    earnings_event_id: int,
    ticker: str,
    action: str,
    before_snapshot: dict[str, Any] | None,
    after_snapshot: dict[str, Any] | None,
    effective_timestamp: Any | None,
) -> None:
    recorded = _now_iso()
    conn.execute(
        """
        INSERT INTO earnings_event_revisions (
            earnings_event_id, ticker, action, before_snapshot_json, after_snapshot_json,
            effective_timestamp, recorded_timestamp
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (
            int(earnings_event_id),
            ticker.upper(),
            action,
            None if before_snapshot is None else _json_dumps(before_snapshot),
            None if after_snapshot is None else _json_dumps(after_snapshot),
            str(effective_timestamp or recorded),
            recorded,
        ),
    )


def _revision_after_hash_exists(conn: Any, earnings_event_id: int, raw_hash: str) -> bool:
    rows = conn.execute(
        """
        SELECT after_snapshot_json
        FROM earnings_event_revisions
        WHERE earnings_event_id = ? AND action = 'update'
        """,
        (int(earnings_event_id),),
    ).fetchall()
    for row in rows:
        snapshot = _json_loads(row["after_snapshot_json"], {})
        if isinstance(snapshot, dict) and snapshot.get("raw_payload_hash") == raw_hash:
            return True
    return False


def insert_earnings_event(db_path: str | Path, event: EarningsEvent) -> tuple[int, str]:
    """Insert an earnings event, dedupe by stable key, and record revisions.

    Returns ``(earnings_event_id, status)`` where status is one of
    ``inserted``, ``duplicate``, or ``updated``.
    """
    create_earnings_tables(db_path)
    row = _event_row(event)
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT OR IGNORE INTO earnings_events (
                ticker, fiscal_period_end, announced_at, available_at, timing,
                eps_estimate, eps_actual, eps_surprise, eps_surprise_percent,
                revenue_estimate, revenue_actual, revenue_surprise_percent, currency,
                provider, provider_event_id, fetched_at, raw_payload_json, raw_payload_hash,
                data_quality_status, warnings, created_at, updated_at, dedupe_key
            )
            VALUES (
                :ticker, :fiscal_period_end, :announced_at, :available_at, :timing,
                :eps_estimate, :eps_actual, :eps_surprise, :eps_surprise_percent,
                :revenue_estimate, :revenue_actual, :revenue_surprise_percent, :currency,
                :provider, :provider_event_id, :fetched_at, :raw_payload_json, :raw_payload_hash,
                :data_quality_status, :warnings, :created_at, :updated_at, :dedupe_key
            )
            """,
            row,
        )
        existing_row = conn.execute(
            f"SELECT {', '.join(EARNINGS_COLUMNS)} FROM earnings_events WHERE dedupe_key = ?",
            (row["dedupe_key"],),
        ).fetchone()
        existing = dict(existing_row)
        earnings_event_id = int(existing["earnings_event_id"])
        if cursor.rowcount > 0:
            _record_revision(
                conn,
                earnings_event_id,
                row["ticker"],
                "create",
                None,
                existing,
                row["available_at"],
            )
            return earnings_event_id, "inserted"

        if existing.get("raw_payload_hash") == row["raw_payload_hash"]:
            return earnings_event_id, "duplicate"

        if _revision_after_hash_exists(conn, earnings_event_id, row["raw_payload_hash"]):
            return earnings_event_id, "duplicate"

        before = existing
        update_row = dict(row)
        update_row["earnings_event_id"] = earnings_event_id
        update_row["created_at"] = before.get("created_at")
        update_row["updated_at"] = _now_iso()
        # Keep the first-seen event row canonical for point-in-time features.
        # Later provider revisions are stored for audit but are not allowed to
        # overwrite historical model-visible values.
        _record_revision(conn, earnings_event_id, row["ticker"], "update", before, update_row, row["available_at"])
        return earnings_event_id, "updated"


def bulk_insert_earnings_events(db_path: str | Path, events: list[EarningsEvent]) -> dict[str, int]:
    counts = {"inserted": 0, "updated": 0, "duplicate": 0}
    for event in events:
        _event_id, status = insert_earnings_event(db_path, event)
        counts[status] = counts.get(status, 0) + 1
    return counts


def list_earnings_by_ticker(
    db_path: str | Path,
    ticker: str,
    start_available_at: datetime | None = None,
    end_available_at: datetime | None = None,
    limit: int | None = None,
) -> pd.DataFrame:
    create_earnings_tables(db_path)
    clauses = ["ticker = ?"]
    params: list[Any] = [ticker.upper()]
    if start_available_at is not None:
        clauses.append("datetime(available_at) >= datetime(?)")
        params.append(_iso_datetime(start_available_at))
    if end_available_at is not None:
        clauses.append("datetime(available_at) <= datetime(?)")
        params.append(_iso_datetime(end_available_at))
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    with storage.connect(db_path) as conn:
        frame = pd.read_sql_query(
            f"""
            SELECT {", ".join(EARNINGS_COLUMNS)}
            FROM earnings_events
            WHERE {" AND ".join(clauses)}
            ORDER BY datetime(available_at) DESC, earnings_event_id DESC
            {limit_clause}
            """,
            conn,
            params=params,
        )
    return frame


def list_recent_earnings_events(db_path: str | Path, limit: int = 200) -> pd.DataFrame:
    create_earnings_tables(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            f"""
            SELECT {", ".join(EARNINGS_COLUMNS)}
            FROM earnings_events
            ORDER BY datetime(available_at) DESC, earnings_event_id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def earnings_coverage_summary(db_path: str | Path, tickers: list[str] | None = None) -> pd.DataFrame:
    create_earnings_tables(db_path)
    params: list[Any] = []
    where = ""
    if tickers:
        clean = sorted({ticker.upper() for ticker in tickers})
        placeholders = ",".join(["?"] * len(clean))
        where = f"WHERE ticker IN ({placeholders})"
        params.extend(clean)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            f"""
            SELECT ticker,
                   COUNT(*) AS events,
                   MIN(available_at) AS first_available_at,
                   MAX(available_at) AS latest_available_at,
                   SUM(CASE WHEN eps_actual IS NULL THEN 1 ELSE 0 END) AS missing_eps_actual,
                   SUM(CASE WHEN eps_estimate IS NULL THEN 1 ELSE 0 END) AS missing_eps_estimate,
                   SUM(CASE WHEN revenue_actual IS NULL THEN 1 ELSE 0 END) AS missing_revenue_actual,
                   SUM(CASE WHEN revenue_estimate IS NULL THEN 1 ELSE 0 END) AS missing_revenue_estimate,
                   SUM(CASE WHEN data_quality_status != 'ok' THEN 1 ELSE 0 END) AS partial_or_warning_events
            FROM earnings_events
            {where}
            GROUP BY ticker
            ORDER BY ticker
            """,
            conn,
            params=params,
        )


def _listing_start_date(db_path: str | Path, ticker: str, requested_start: date) -> date:
    try:
        with storage.connect(db_path) as conn:
            row = conn.execute(
                "SELECT MIN(date) AS first_date FROM ohlcv_cache WHERE ticker = ?",
                (ticker.upper(),),
            ).fetchone()
        first_date = pd.to_datetime(row["first_date"], errors="coerce").date() if row and row["first_date"] else None
    except Exception:
        first_date = None
    if first_date is None:
        return requested_start
    return max(requested_start, first_date)


def expected_quarterly_events(start_date: date, end_date: date) -> int:
    if start_date > end_date:
        return 0
    days = max(1, (end_date - start_date).days + 1)
    return max(1, int(round(days / 91.25)))


def classify_earnings_coverage(event_count: int, expected_events: int) -> str:
    if event_count <= 0:
        return "unavailable"
    if expected_events <= 0:
        return "partial"
    ratio = float(event_count) / float(expected_events)
    if ratio >= 0.8:
        return "complete"
    if ratio >= 0.5:
        return "partial"
    return "sparse"


def earnings_coverage_report(
    db_path: str | Path,
    tickers: list[str],
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Return per-ticker earnings coverage with deterministic coverage buckets."""
    create_earnings_tables(db_path)
    rows: list[dict[str, Any]] = []
    for ticker in sorted({value.upper().strip() for value in tickers if value and value.strip()}):
        listing_start = _listing_start_date(db_path, ticker, start_date)
        expected = expected_quarterly_events(listing_start, end_date)
        frame = list_earnings_by_ticker(
            db_path,
            ticker,
            start_available_at=datetime.combine(start_date, datetime.min.time(), tzinfo=UTC),
            end_available_at=datetime.combine(end_date, datetime.max.time(), tzinfo=UTC),
            limit=None,
        )
        event_count = int(len(frame))
        provider_paths: list[str] = []
        warning_texts: list[str] = []
        if not frame.empty:
            for provider_event_id in frame["provider_event_id"].dropna().astype(str).tolist():
                if provider_event_id.startswith("earnings_history:"):
                    provider_paths.append("earnings_history")
                elif provider_event_id.startswith("earnings_dates:"):
                    provider_paths.append("earnings_dates")
            for warning_value in frame["warnings"].dropna().tolist():
                parsed = _json_loads(warning_value, [])
                if isinstance(parsed, list):
                    warning_texts.extend(str(item) for item in parsed if item)
                elif parsed:
                    warning_texts.append(str(parsed))
        rows.append(
            {
                "ticker": ticker,
                "listing_adjusted_start": listing_start.isoformat(),
                "expected_quarterly_events": expected,
                "event_count": event_count,
                "coverage_classification": classify_earnings_coverage(event_count, expected),
                "earliest_event": None if frame.empty else frame["available_at"].min(),
                "latest_event": None if frame.empty else frame["available_at"].max(),
                "provider_path": ", ".join(sorted(set(provider_paths))) if provider_paths else None,
                "missing_eps_estimate": 0 if frame.empty else int(frame["eps_estimate"].isna().sum()),
                "missing_eps_actual": 0 if frame.empty else int(frame["eps_actual"].isna().sum()),
                "missing_revenue_estimate": 0 if frame.empty else int(frame["revenue_estimate"].isna().sum()),
                "missing_revenue_actual": 0 if frame.empty else int(frame["revenue_actual"].isna().sum()),
                "unknown_timing_count": 0 if frame.empty else int(frame["timing"].astype(str).eq("unknown").sum()),
                "warnings": "; ".join(sorted(set(warning_texts)))[:500],
            }
        )
    return pd.DataFrame(rows)


def cache_provider_response(
    db_path: str | Path,
    cache_key: str,
    provider: str,
    ticker: str,
    response_json: str | None,
    status: str,
    error: str | None = None,
) -> None:
    create_earnings_tables(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO earnings_response_cache (
                cache_key, provider, ticker, response_json, status, error, fetched_at, created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_key) DO UPDATE SET
                response_json = excluded.response_json,
                status = excluded.status,
                error = excluded.error,
                fetched_at = excluded.fetched_at,
                updated_at = excluded.updated_at
            """,
            (cache_key, provider, ticker.upper(), response_json, status, error, now, now, now),
        )


def get_cached_provider_response(db_path: str | Path, cache_key: str) -> dict[str, Any] | None:
    create_earnings_tables(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT cache_key, provider, ticker, response_json, status, error, fetched_at, created_at, updated_at
            FROM earnings_response_cache
            WHERE cache_key = ?
            """,
            (cache_key,),
        ).fetchone()
    return None if row is None else dict(row)
