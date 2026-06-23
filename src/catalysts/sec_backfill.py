from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.catalysts.repository import insert_catalyst_with_status
from src.catalysts.sec_adapter import SecFilingsProvider
from src.data import storage


@dataclass
class SecBackfillResult:
    sec_run_id: int
    processed_tickers: int = 0
    completed_tickers: int = 0
    failed_tickers: int = 0
    events_inserted: int = 0
    duplicates_skipped: int = 0
    warnings: list[str] = field(default_factory=list)


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


def _event_accession(event: Any) -> str | None:
    payload = _json_loads(getattr(event, "raw_payload_json", None), {}) or {}
    accession = payload.get("accessionNumber") if isinstance(payload, dict) else None
    accession = str(accession or "").strip()
    return accession or None


def _existing_sec_accessions(db_path: str | Path, ticker: str) -> set[str]:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT raw_payload_json
            FROM catalysts
            WHERE ticker = ? AND source = 'SEC EDGAR'
            """,
            (ticker.upper(),),
        ).fetchall()
    accessions: set[str] = set()
    for row in rows:
        payload = _json_loads(row["raw_payload_json"], {}) or {}
        accession = payload.get("accessionNumber") if isinstance(payload, dict) else None
        accession = str(accession or "").strip()
        if accession:
            accessions.add(accession)
    return accessions


def create_sec_backfill_run(
    db_path: str | Path,
    tickers: list[str],
    start_date: date,
    end_date: date,
    provider_name: str = "sec_edgar_submissions",
) -> int:
    storage.init_db(db_path)
    clean_tickers = sorted({ticker.upper().strip() for ticker in tickers if ticker and ticker.strip()})
    now = _now_iso()
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO sec_backfill_runs (
                universe_snapshot_json, requested_start_date, requested_end_date, status,
                started_at, total_tickers, provider, warnings_json, created_at, updated_at
            )
            VALUES (?, ?, ?, 'running', ?, ?, ?, '[]', ?, ?)
            """,
            (
                _json_dumps(clean_tickers),
                start_date.isoformat(),
                end_date.isoformat(),
                now,
                len(clean_tickers),
                provider_name,
                now,
                now,
            ),
        )
        sec_run_id = int(cursor.lastrowid)
        conn.executemany(
            """
            INSERT OR IGNORE INTO sec_backfill_items (
                sec_run_id, ticker, status, created_at, updated_at
            )
            VALUES (?, ?, 'pending', ?, ?)
            """,
            [(sec_run_id, ticker, now, now) for ticker in clean_tickers],
        )
    return sec_run_id


def get_sec_backfill_run(db_path: str | Path, sec_run_id: int) -> dict[str, Any] | None:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM sec_backfill_runs WHERE sec_run_id = ?", (int(sec_run_id),)).fetchone()
    return None if row is None else dict(row)


def list_sec_backfill_runs(db_path: str | Path, limit: int = 20) -> pd.DataFrame:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM sec_backfill_runs
            ORDER BY datetime(started_at) DESC, sec_run_id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def list_sec_backfill_items(db_path: str | Path, sec_run_id: int) -> pd.DataFrame:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM sec_backfill_items
            WHERE sec_run_id = ?
            ORDER BY ticker
            """,
            conn,
            params=(int(sec_run_id),),
        )


def retry_failed_sec_items(db_path: str | Path, sec_run_id: int) -> int:
    storage.init_db(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        result = conn.execute(
            """
            UPDATE sec_backfill_items
            SET status = 'pending', error = NULL, warning = NULL, updated_at = ?
            WHERE sec_run_id = ? AND status = 'failed'
            """,
            (now, int(sec_run_id)),
        )
        conn.execute(
            """
            UPDATE sec_backfill_runs
            SET status = 'running', completed_at = NULL, updated_at = ?
            WHERE sec_run_id = ?
            """,
            (now, int(sec_run_id)),
        )
        return int(result.rowcount)


def _mark_item(db_path: str | Path, sec_run_id: int, ticker: str, status: str, **updates: Any) -> None:
    assignments = ["status = ?", "updated_at = ?"]
    values: list[Any] = [status, _now_iso()]
    for key, value in updates.items():
        assignments.append(f"{key} = ?")
        values.append(value)
    values.extend([int(sec_run_id), ticker.upper()])
    with storage.connect(db_path) as conn:
        conn.execute(
            f"""
            UPDATE sec_backfill_items
            SET {", ".join(assignments)}
            WHERE sec_run_id = ? AND ticker = ?
            """,
            values,
        )


def _update_run_progress(db_path: str | Path, sec_run_id: int, warnings: list[str] | None = None) -> None:
    with storage.connect(db_path) as conn:
        stats = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                SUM(events_inserted) AS events_inserted,
                SUM(duplicates_skipped) AS duplicates_skipped,
                COUNT(*) AS total
            FROM sec_backfill_items
            WHERE sec_run_id = ?
            """,
            (int(sec_run_id),),
        ).fetchone()
        completed = int(stats["completed"] or 0)
        failed = int(stats["failed"] or 0)
        total = int(stats["total"] or 0)
        next_status = "completed_with_errors" if completed + failed >= total and failed else "completed" if completed + failed >= total else "running"
        completed_at = _now_iso() if completed + failed >= total else None
        conn.execute(
            """
            UPDATE sec_backfill_runs
            SET status = ?, completed_tickers = ?, failed_tickers = ?, events_inserted = ?,
                duplicates_skipped = ?, completed_at = COALESCE(?, completed_at),
                warnings_json = ?, updated_at = ?
            WHERE sec_run_id = ?
            """,
            (
                next_status,
                completed,
                failed,
                int(stats["events_inserted"] or 0),
                int(stats["duplicates_skipped"] or 0),
                completed_at,
                _json_dumps(list(dict.fromkeys(warnings or []))),
                _now_iso(),
                int(sec_run_id),
            ),
        )


def process_sec_backfill_run(
    db_path: str | Path,
    sec_run_id: int,
    max_tickers: int | None = None,
    retry_failed: bool = False,
    provider: SecFilingsProvider | None = None,
) -> SecBackfillResult:
    run = get_sec_backfill_run(db_path, sec_run_id)
    if not run:
        raise ValueError(f"SEC backfill run #{sec_run_id} not found.")
    if retry_failed:
        retry_failed_sec_items(db_path, sec_run_id)
    start_date = pd.to_datetime(run["requested_start_date"]).date()
    end_date = pd.to_datetime(run["requested_end_date"]).date()
    items = list_sec_backfill_items(db_path, sec_run_id)
    pending = items[items["status"].isin(["pending", "running"])]
    if max_tickers is not None:
        pending = pending.head(int(max_tickers))

    sec_provider = provider or SecFilingsProvider(db_path=db_path)
    warnings: list[str] = []
    processed = 0
    for _, item in pending.iterrows():
        ticker = str(item["ticker"]).upper()
        _mark_item(db_path, sec_run_id, ticker, "running", started_at=_now_iso(), error=None)
        try:
            result = sec_provider.fetch_historical_filing_events(ticker, start_date, end_date)
            existing_accessions = _existing_sec_accessions(db_path, ticker)
            inserted = 0
            duplicates = 0
            first_acceptance = None
            last_acceptance = None
            forms: dict[str, int] = {}
            for event in result.events:
                accession = _event_accession(event)
                if accession and accession in existing_accessions:
                    duplicates += 1
                    payload = _json_loads(event.raw_payload_json, {}) or {}
                    form = str(payload.get("form") or "unknown")
                    forms[form] = forms.get(form, 0) + 1
                    accepted = payload.get("acceptanceDateTime")
                    if accepted:
                        first_acceptance = min(first_acceptance, accepted) if first_acceptance else accepted
                        last_acceptance = max(last_acceptance, accepted) if last_acceptance else accepted
                    continue
                _, was_inserted = insert_catalyst_with_status(db_path, event)
                inserted += 1 if was_inserted else 0
                duplicates += 0 if was_inserted else 1
                if was_inserted and accession:
                    existing_accessions.add(accession)
                payload = _json_loads(event.raw_payload_json, {}) or {}
                form = str(payload.get("form") or "unknown")
                forms[form] = forms.get(form, 0) + 1
                accepted = payload.get("acceptanceDateTime")
                if accepted:
                    first_acceptance = min(first_acceptance, accepted) if first_acceptance else accepted
                    last_acceptance = max(last_acceptance, accepted) if last_acceptance else accepted
            warnings.extend(result.warnings)
            metadata = {
                **result.metadata,
                "forms": forms,
                "provider": sec_provider.name,
                "warnings": result.warnings,
            }
            _mark_item(
                db_path,
                sec_run_id,
                ticker,
                "completed",
                filings_seen=len(result.events),
                events_inserted=inserted,
                duplicates_skipped=duplicates,
                first_acceptance_at=first_acceptance,
                last_acceptance_at=last_acceptance,
                warning="; ".join(result.warnings[:5]),
                metadata_json=_json_dumps(metadata),
                completed_at=_now_iso(),
            )
        except Exception as exc:
            warning = f"{ticker}: {exc}"
            warnings.append(warning)
            _mark_item(
                db_path,
                sec_run_id,
                ticker,
                "failed",
                error=str(exc),
                warning=warning,
                completed_at=_now_iso(),
            )
        processed += 1

    _update_run_progress(db_path, sec_run_id, warnings)
    final_run = get_sec_backfill_run(db_path, sec_run_id) or {}
    return SecBackfillResult(
        sec_run_id=sec_run_id,
        processed_tickers=processed,
        completed_tickers=int(final_run.get("completed_tickers", 0) or 0),
        failed_tickers=int(final_run.get("failed_tickers", 0) or 0),
        events_inserted=int(final_run.get("events_inserted", 0) or 0),
        duplicates_skipped=int(final_run.get("duplicates_skipped", 0) or 0),
        warnings=warnings,
    )
