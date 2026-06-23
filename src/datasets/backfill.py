from __future__ import annotations

import json
import resource
import time
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import market_data, storage
from src.datasets.builder import (
    DEFAULT_HORIZONS,
    FEATURE_VERSION,
    build_feature_snapshot,
    calculate_outcome_labels,
    dataset_hash,
    label_definitions,
    precompute_catalyst_overrides_for_dates,
    precompute_market_regimes_for_dates,
    precompute_sec_features_for_dates,
)
from src.datasets.feature_manifest import role_sets_from_frame
from src.datasets.models import DatasetBuild, OutcomeLabel
from src.datasets.repository import (
    flatten_saved_dataset,
    insert_dataset_build,
    insert_feature_snapshots,
    insert_outcome_labels,
    load_outcome_labels,
    list_dataset_builds,
    stream_saved_dataset_export_and_hash,
    update_dataset_build_summary,
)
from src.utils.trading_calendar import next_trading_day, trading_days_between


PRICE_ADJUSTMENT_CONVENTION = "yfinance daily OHLCV with auto_adjust=False; close and adj_close stored separately"


@dataclass
class BackfillRunResult:
    run_id: int
    dataset_id: int
    processed_tickers: int = 0
    completed_tickers: int = 0
    failed_tickers: int = 0
    generated_rows: int = 0
    warnings: list[str] = field(default_factory=list)
    export_path: str | None = None


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


def _peak_rss_mb() -> float | None:
    try:
        value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    except Exception:
        return None
    # macOS reports bytes; Linux commonly reports KiB.
    if value > 10_000_000:
        return round(value / 1024 / 1024, 2)
    return round(value / 1024, 2)


def _period_for_range(start_date: date, end_date: date) -> str:
    years = max(1, int((end_date - start_date).days / 365) + 2)
    return f"{years}y"


def _cached_covers_range(df: pd.DataFrame, start_date: date, end_date: date) -> bool:
    if df is None or df.empty:
        return False
    dates = pd.to_datetime(df.index).date
    return bool(min(dates) <= start_date and max(dates) >= end_date)


def _missing_trading_ranges(cached: pd.DataFrame, start_date: date, end_date: date) -> list[tuple[date, date]]:
    expected = trading_days_between(start_date, end_date)
    if not expected:
        return []
    cached_dates = set()
    if cached is not None and not cached.empty:
        cached_dates = {
            pd.Timestamp(idx).date()
            for idx in cached.index
            if start_date <= pd.Timestamp(idx).date() <= end_date
        }
    missing = [value for value in expected if value not in cached_dates]
    if not missing:
        return []

    ranges: list[tuple[date, date]] = []
    start = previous = missing[0]
    for current in missing[1:]:
        if current == next_trading_day(previous):
            previous = current
            continue
        ranges.append((start, previous))
        start = previous = current
    ranges.append((start, previous))
    return ranges


def _download_missing_range(
    ticker: str,
    provider_name: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    provider = market_data.get_provider(provider_name)
    period = _period_for_range(start_date, end_date)
    try:
        return provider.download_history(ticker, period=period, start=start_date, end=end_date)
    except TypeError:
        # Test doubles and future providers may only implement period-based
        # downloads. Filter the result before writing so cached overlap remains
        # untouched even when the provider cannot do range requests.
        return provider.download_history(ticker, period=period)


def _trading_dates(df: pd.DataFrame, start_date: date, end_date: date) -> list[date]:
    if df is None or df.empty:
        return []
    return [
        pd.Timestamp(idx).date()
        for idx in df.index
        if start_date <= pd.Timestamp(idx).date() <= end_date
    ]


def _history_with_cache_first(
    ticker: str,
    db_path: str | Path,
    provider_name: str,
    start_date: date,
    end_date: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    cached = storage.load_ohlcv(db_path, ticker)
    metadata = {
        "ticker": ticker.upper(),
        "provider": provider_name,
        "fetch_timestamp": _now_iso(),
        "price_adjustment": PRICE_ADJUSTMENT_CONVENTION,
        "source": "cache",
        "cached_rows_before": int(len(cached)),
        "download_error": None,
    }
    if _cached_covers_range(cached, start_date, end_date):
        return cached, metadata

    missing_ranges = _missing_trading_ranges(cached, start_date, end_date)
    missing_dates = {value.isoformat() for start, end in missing_ranges for value in trading_days_between(start, end)}
    downloaded_rows = 0
    download_errors: list[str] = []
    range_metadata: list[dict[str, Any]] = []

    for range_start, range_end in missing_ranges:
        try:
            downloaded = _download_missing_range(ticker, provider_name, range_start, range_end)
        except Exception as exc:
            message = f"{range_start} to {range_end}: {exc}"
            download_errors.append(message)
            range_metadata.append({"start": range_start.isoformat(), "end": range_end.isoformat(), "error": str(exc)})
            continue

        normalized = storage.normalize_ohlcv(downloaded)
        if not normalized.empty:
            missing_only = normalized[
                normalized["date"].isin(missing_dates)
                & (normalized["date"] >= range_start.isoformat())
                & (normalized["date"] <= range_end.isoformat())
            ]
            downloaded_rows += int(len(missing_only))
            storage.upsert_ohlcv(db_path, ticker, missing_only)
        range_metadata.append(
            {
                "start": range_start.isoformat(),
                "end": range_end.isoformat(),
                "downloaded_rows": int(len(normalized)),
                "inserted_missing_rows": int(len(normalized[normalized["date"].isin(missing_dates)])) if not normalized.empty else 0,
            }
        )

    fresh = storage.load_ohlcv(db_path, ticker)
    source = "cache_plus_range_download" if _cached_covers_range(fresh, start_date, end_date) else "partial_cache_range_download"
    return fresh, {
        **metadata,
        "source": source,
        "fetch_timestamp": _now_iso(),
        "price_adjustment": PRICE_ADJUSTMENT_CONVENTION,
        "missing_ranges_requested": [
            {"start": start.isoformat(), "end": end.isoformat()} for start, end in missing_ranges
        ],
        "downloaded_missing_rows": downloaded_rows,
        "download_errors": download_errors,
        "range_downloads": range_metadata,
        "cached_rows_after": int(len(fresh)),
    }


def create_backfill_run(
    db_path: str | Path,
    tickers: list[str],
    start_date: date,
    end_date: date,
    version: str = FEATURE_VERSION,
    provider_name: str = "yfinance",
) -> int:
    storage.init_db(db_path)
    clean_tickers = sorted({ticker.upper().strip() for ticker in tickers if ticker and ticker.strip()})
    build = DatasetBuild(
        version=version,
        build_timestamp=datetime.now(UTC),
        requested_start_date=start_date,
        requested_end_date=end_date,
        ticker_universe=clean_tickers,
        feature_columns=[],
        label_definitions=label_definitions(DEFAULT_HORIZONS),
        row_count=0,
        data_hash="pending",
        warnings=[],
    )
    dataset_id = insert_dataset_build(db_path, build)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO backfill_runs (
                dataset_id, version, universe_snapshot_json, requested_start_date, requested_end_date,
                status, started_at, total_tickers, provider, price_adjustment, warnings_json,
                created_at, updated_at
            )
            VALUES (?, ?, ?, ?, ?, 'running', ?, ?, ?, ?, '[]', ?, ?)
            """,
            (
                dataset_id,
                version,
                _json_dumps(clean_tickers),
                start_date.isoformat(),
                end_date.isoformat(),
                now,
                len(clean_tickers),
                provider_name,
                PRICE_ADJUSTMENT_CONVENTION,
                now,
                now,
            ),
        )
        run_id = int(cursor.lastrowid)
        conn.executemany(
            """
            INSERT OR IGNORE INTO backfill_items (
                run_id, ticker, status, created_at, updated_at
            )
            VALUES (?, ?, 'pending', ?, ?)
            """,
            [(run_id, ticker, now, now) for ticker in clean_tickers],
        )
    return run_id


def get_backfill_run(db_path: str | Path, run_id: int) -> dict[str, Any] | None:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute("SELECT * FROM backfill_runs WHERE run_id = ?", (int(run_id),)).fetchone()
    return None if row is None else dict(row)


def list_backfill_runs(db_path: str | Path, limit: int = 20) -> pd.DataFrame:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT run_id, dataset_id, version, requested_start_date, requested_end_date, status,
                   started_at, completed_at, total_tickers, completed_tickers, failed_tickers,
                   generated_rows, provider, warnings_json
            FROM backfill_runs
            ORDER BY datetime(started_at) DESC, run_id DESC
            LIMIT ?
            """,
            conn,
            params=(int(limit),),
        )


def list_backfill_items(db_path: str | Path, run_id: int) -> pd.DataFrame:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM backfill_items
            WHERE run_id = ?
            ORDER BY ticker
            """,
            conn,
            params=(int(run_id),),
        )


def retry_failed_items(db_path: str | Path, run_id: int) -> int:
    storage.init_db(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        result = conn.execute(
            """
            UPDATE backfill_items
            SET status = 'pending', error = NULL, warning = NULL, updated_at = ?
            WHERE run_id = ? AND status = 'failed'
            """,
            (now, int(run_id)),
        )
        conn.execute(
            """
            UPDATE backfill_runs
            SET status = 'running', completed_at = NULL, updated_at = ?
            WHERE run_id = ?
            """,
            (now, int(run_id)),
        )
        return int(result.rowcount)


def _mark_item(
    db_path: str | Path,
    run_id: int,
    ticker: str,
    status: str,
    **updates: Any,
) -> None:
    assignments = ["status = ?", "updated_at = ?"]
    values: list[Any] = [status, _now_iso()]
    for key, value in updates.items():
        assignments.append(f"{key} = ?")
        values.append(value)
    values.extend([int(run_id), ticker.upper()])
    with storage.connect(db_path) as conn:
        conn.execute(
            f"""
            UPDATE backfill_items
            SET {", ".join(assignments)}
            WHERE run_id = ? AND ticker = ?
            """,
            values,
        )


def _update_run_progress(db_path: str | Path, run_id: int, status: str | None = None) -> None:
    with storage.connect(db_path) as conn:
        stats = conn.execute(
            """
            SELECT
                SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed,
                SUM(CASE WHEN status = 'failed' THEN 1 ELSE 0 END) AS failed,
                SUM(rows_generated) AS rows_generated,
                COUNT(*) AS total
            FROM backfill_items
            WHERE run_id = ?
            """,
            (int(run_id),),
        ).fetchone()
        completed = int(stats["completed"] or 0)
        failed = int(stats["failed"] or 0)
        generated = int(stats["rows_generated"] or 0)
        total = int(stats["total"] or 0)
        next_status = status
        completed_at = None
        if next_status is None:
            if completed + failed >= total:
                next_status = "completed_with_errors" if failed else "completed"
                completed_at = _now_iso()
            else:
                next_status = "running"
        conn.execute(
            """
            UPDATE backfill_runs
            SET status = ?, completed_tickers = ?, failed_tickers = ?, generated_rows = ?,
                completed_at = COALESCE(?, completed_at), updated_at = ?
            WHERE run_id = ?
            """,
            (next_status, completed, failed, generated, completed_at, _now_iso(), int(run_id)),
        )


def _coverage_warnings(ticker: str, df: pd.DataFrame, spy_df: pd.DataFrame, start_date: date, end_date: date) -> list[str]:
    warnings: list[str] = []
    if df.empty:
        return [f"{ticker}: no OHLCV rows available."]
    if df.index.has_duplicates:
        warnings.append(f"{ticker}: duplicate trading dates detected after normalization.")
    dates = [pd.Timestamp(idx).date() for idx in df.index]
    expected_dates = trading_days_between(start_date, end_date)
    expected_first = min(expected_dates) if expected_dates else start_date
    expected_last = max(expected_dates) if expected_dates else end_date
    ticker_dates = {value for value in dates if start_date <= value <= end_date}
    if min(dates) > expected_first:
        warnings.append(f"{ticker}: cached history starts after requested start date.")
    if max(dates) < expected_last:
        warnings.append(f"{ticker}: cached history ends before requested end date.")
    missing_expected = sorted(set(expected_dates) - ticker_dates)
    if missing_expected:
        sample = ", ".join(value.isoformat() for value in missing_expected[:5])
        suffix = "" if len(missing_expected) <= 5 else f", +{len(missing_expected) - 5} more"
        warnings.append(f"{ticker}: {len(missing_expected)} expected trading date(s) missing from OHLCV ({sample}{suffix}).")
    if not spy_df.empty:
        spy_dates = {pd.Timestamp(idx).date() for idx in spy_df.index}
        missing_spy = sorted(ticker_dates - spy_dates)
        if missing_spy:
            warnings.append(f"{ticker}: {len(missing_spy)} snapshot date(s) lack matching SPY data.")
    return warnings


def _process_ticker(
    db_path: str | Path,
    dataset_id: int,
    ticker: str,
    start_date: date,
    end_date: date,
    provider_name: str,
    regime_feature_map: dict[date, dict[str, Any]] | None = None,
) -> tuple[int, dict[str, Any], list[str]]:
    ticker_start = time.perf_counter()
    required = sorted({ticker.upper(), "SPY", "QQQ", "IWM", "^VIX"})
    histories: dict[str, pd.DataFrame] = {}
    metadata: dict[str, Any] = {"provider": provider_name, "fetches": {}}
    for symbol in required:
        data, fetch_meta = _history_with_cache_first(symbol, db_path, provider_name, start_date, end_date)
        histories[symbol] = data
        metadata["fetches"][symbol] = fetch_meta

    ticker_df = histories.get(ticker.upper(), pd.DataFrame())
    spy_df = histories.get("SPY", pd.DataFrame())
    warnings = _coverage_warnings(ticker.upper(), ticker_df, spy_df, start_date, end_date)
    if ticker_df.empty:
        raise RuntimeError(f"{ticker}: no OHLCV data available.")

    snapshot_dates = _trading_dates(ticker_df, start_date, end_date)
    sec_start = time.perf_counter()
    sec_feature_map = precompute_sec_features_for_dates(db_path, ticker, snapshot_dates)
    sec_seconds = time.perf_counter() - sec_start
    catalyst_override_map = precompute_catalyst_overrides_for_dates(db_path, ticker, snapshot_dates)
    if regime_feature_map is None:
        regime_feature_map = precompute_market_regimes_for_dates(histories, snapshot_dates)
    snapshots: list[Any] = []
    labels_by_key: dict[tuple[str, date], list[OutcomeLabel]] = {}
    label_counts = {"1_session": 0, "5_session": 0, "20_session": 0}
    snapshot_write_seconds = 0.0
    for trading_date in snapshot_dates:
        snapshot = build_feature_snapshot(
            db_path,
            ticker,
            trading_date,
            histories,
            sec_features_override=sec_feature_map.get(trading_date),
            regime_override=regime_feature_map.get(trading_date),
            catalyst_features_override=catalyst_override_map.get(trading_date),
        )
        if snapshot is None:
            warnings.append(f"{ticker} {trading_date}: snapshot unavailable.")
            continue
        snapshots.append(snapshot)
        labels = calculate_outcome_labels(snapshot, ticker_df, spy_df, DEFAULT_HORIZONS)
        labels_by_key[(snapshot.ticker, snapshot.trading_date)] = labels
        for label in labels:
            label_counts[label.horizon] = label_counts.get(label.horizon, 0) + 1
        if len(labels) < len(DEFAULT_HORIZONS):
            warnings.append(f"{ticker} {trading_date}: one or more forward label horizons unavailable.")

    write_start = time.perf_counter()
    snapshot_ids = insert_feature_snapshots(db_path, dataset_id, snapshots)
    snapshot_write_seconds += time.perf_counter() - write_start

    labels_to_insert: list[OutcomeLabel] = []
    for snapshot in snapshots:
        snapshot_id = snapshot_ids.get((snapshot.ticker.upper(), snapshot.trading_date))
        if snapshot_id is None:
            continue
        snapshot.snapshot_id = snapshot_id
        for label in labels_by_key.get((snapshot.ticker, snapshot.trading_date), []):
            label.snapshot_id = snapshot_id
            labels_to_insert.append(label)
    insert_outcome_labels(db_path, labels_to_insert)
    metadata["price_adjustment"] = PRICE_ADJUSTMENT_CONVENTION
    metadata["expected_snapshots"] = len(snapshot_dates)
    metadata["label_counts"] = label_counts
    metadata["generated_at"] = _now_iso()
    metadata["performance"] = {
        "total_seconds": round(time.perf_counter() - ticker_start, 3),
        "sec_aggregation_seconds": round(sec_seconds, 3),
        "snapshot_write_seconds": round(snapshot_write_seconds, 3),
        "peak_rss_mb": _peak_rss_mb(),
    }
    first_date = min(snapshot_dates).isoformat() if snapshot_dates else None
    last_date = max(snapshot_dates).isoformat() if snapshot_dates else None
    metadata["first_date"] = first_date
    metadata["last_date"] = last_date
    return len(snapshots), metadata, warnings


def _shared_regime_feature_map(
    db_path: str | Path,
    provider_name: str,
    start_date: date,
    end_date: date,
) -> dict[date, dict[str, Any]]:
    histories: dict[str, pd.DataFrame] = {}
    for symbol in ["SPY", "QQQ", "IWM", "^VIX"]:
        data, _metadata = _history_with_cache_first(symbol, db_path, provider_name, start_date, end_date)
        histories[symbol] = data
    dates = trading_days_between(start_date, end_date)
    return precompute_market_regimes_for_dates(histories, dates)


def _finalize_dataset(db_path: str | Path, dataset_id: int, output_dir: str | Path, warnings: list[str] | None = None) -> str | None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provisional_export_path = output / f"dataset_{dataset_id}_pending.csv"
    stream_result = stream_saved_dataset_export_and_hash(db_path, dataset_id, provisional_export_path)
    hash_value = str(stream_result["data_hash"])
    row_count = int(stream_result["row_count"])
    export_path = None
    if row_count > 0:
        final_export_path = output / f"dataset_{dataset_id}_{hash_value[:8]}.csv"
        if final_export_path.exists():
            final_export_path.unlink()
        Path(provisional_export_path).replace(final_export_path)
        export_path = str(final_export_path)
    elif provisional_export_path.exists():
        provisional_export_path.unlink()
    role_sets = role_sets_from_frame(pd.DataFrame(columns=stream_result.get("columns", [])))
    feature_columns = role_sets.model_features
    build_rows = list_dataset_builds(db_path, limit=500)
    existing = build_rows[build_rows["dataset_id"].eq(dataset_id)] if not build_rows.empty else pd.DataFrame()
    build_warnings = []
    if not existing.empty:
        build_warnings = _json_loads(existing.iloc[0].get("warnings_json"), [])
    merged_warnings = list(dict.fromkeys([*(build_warnings or []), *(warnings or [])]))
    update_dataset_build_summary(
        db_path,
        dataset_id,
        row_count=row_count,
        data_hash=hash_value,
        warnings=merged_warnings,
        feature_columns=feature_columns,
        audit_columns=role_sets.audit_columns,
        label_columns=role_sets.label_columns,
        identifier_columns=role_sets.identifier_columns,
        metadata_columns=role_sets.metadata_columns,
        feature_manifest=role_sets.manifest,
        export_path=export_path,
    )
    return export_path


def process_backfill_run(
    db_path: str | Path,
    run_id: int,
    provider_name: str = "yfinance",
    output_dir: str | Path = "data/processed",
    max_tickers: int | None = None,
    retry_failed: bool = False,
) -> BackfillRunResult:
    run = get_backfill_run(db_path, run_id)
    if not run:
        raise ValueError(f"Backfill run #{run_id} not found.")
    if retry_failed:
        retry_failed_items(db_path, run_id)
    dataset_id = int(run["dataset_id"])
    start_date = pd.to_datetime(run["requested_start_date"]).date()
    end_date = pd.to_datetime(run["requested_end_date"]).date()
    items = list_backfill_items(db_path, run_id)
    if items.empty:
        _update_run_progress(db_path, run_id, "completed")
        return BackfillRunResult(run_id=run_id, dataset_id=dataset_id)

    pending = items[items["status"].isin(["pending", "running"])]
    if retry_failed:
        pending = list_backfill_items(db_path, run_id)
        pending = pending[pending["status"].eq("pending")]
    if max_tickers is not None:
        pending = pending.head(int(max_tickers))

    processed = 0
    warnings: list[str] = []
    regime_feature_map = _shared_regime_feature_map(db_path, provider_name, start_date, end_date) if not pending.empty else {}
    for _, item in pending.iterrows():
        ticker = str(item["ticker"]).upper()
        _mark_item(db_path, run_id, ticker, "running", started_at=_now_iso(), error=None)
        try:
            rows_generated, metadata, item_warnings = _process_ticker(
                db_path,
                dataset_id,
                ticker,
                start_date,
                end_date,
                provider_name,
                regime_feature_map=regime_feature_map,
            )
            warnings.extend(item_warnings)
            _mark_item(
                db_path,
                run_id,
                ticker,
                "completed",
                rows_generated=int(rows_generated),
                first_date=metadata.get("first_date"),
                last_date=metadata.get("last_date"),
                expected_snapshots=int(metadata.get("expected_snapshots", 0) or 0),
                generated_snapshots=int(rows_generated),
                completed_labels_1_session=int(metadata.get("label_counts", {}).get("1_session", 0)),
                completed_labels_5_session=int(metadata.get("label_counts", {}).get("5_session", 0)),
                completed_labels_20_session=int(metadata.get("label_counts", {}).get("20_session", 0)),
                warning="; ".join(item_warnings[:5]),
                metadata_json=_json_dumps(metadata),
                completed_at=_now_iso(),
            )
        except Exception as exc:
            warning = f"{ticker}: {exc}"
            warnings.append(warning)
            _mark_item(
                db_path,
                run_id,
                ticker,
                "failed",
                error=str(exc),
                warning=warning,
                completed_at=_now_iso(),
            )
        processed += 1

    _update_run_progress(db_path, run_id)
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            UPDATE backfill_runs
            SET warnings_json = ?, updated_at = ?
            WHERE run_id = ?
            """,
            (_json_dumps(list(dict.fromkeys(warnings))), _now_iso(), int(run_id)),
        )
    export_path = _finalize_dataset(db_path, dataset_id, output_dir, warnings)
    final_run = get_backfill_run(db_path, run_id) or {}
    return BackfillRunResult(
        run_id=run_id,
        dataset_id=dataset_id,
        processed_tickers=processed,
        completed_tickers=int(final_run.get("completed_tickers", 0) or 0),
        failed_tickers=int(final_run.get("failed_tickers", 0) or 0),
        generated_rows=int(final_run.get("generated_rows", 0) or 0),
        warnings=warnings,
        export_path=export_path,
    )


def dataset_sufficiency_report(db_path: str | Path, dataset_id: int) -> dict[str, Any]:
    frame = flatten_saved_dataset(db_path, dataset_id)
    labels = load_outcome_labels(db_path, dataset_id)
    if frame.empty:
        return {
            "summary": {"dataset_id": dataset_id, "total_rows": 0, "tickers": 0, "years_covered": 0},
            "per_ticker": pd.DataFrame(),
            "missingness": pd.DataFrame(),
            "label_counts": pd.DataFrame(),
            "return_distribution": pd.DataFrame(),
            "warnings": ["Dataset has no rows."],
        }
    dates = pd.to_datetime(frame["trading_date"]).dt.date
    years = (max(dates) - min(dates)).days / 365.25 if len(dates) else 0
    label_counts = labels.groupby("horizon").size().reset_index(name="labels") if not labels.empty else pd.DataFrame(columns=["horizon", "labels"])
    return_columns = [column for column in frame.columns if column.endswith("_forward_return") or column.endswith("_excess_return")]
    return_distribution = (
        frame[return_columns].describe().transpose().reset_index().rename(columns={"index": "return_column"})
        if return_columns
        else pd.DataFrame()
    )
    per_ticker = (
        frame.groupby("ticker")
        .agg(
            first_date=("trading_date", "min"),
            last_date=("trading_date", "max"),
            generated_snapshots=("trading_date", "count"),
            catalyst_rows=("active_catalyst_count", "sum"),
            positive_catalyst_rows=("positive_catalyst_count_45d", "sum"),
            negative_catalyst_rows=("negative_catalyst_count_45d", "sum"),
            llm_supported_rows=("published_llm_supported_count", "sum"),
        )
        .reset_index()
    )
    missingness = (
        frame.isna()
        .mean()
        .reset_index()
        .rename(columns={"index": "column", 0: "missing_fraction"})
        .sort_values("missing_fraction", ascending=False)
    )
    warnings: list[str] = []
    if "catalyst_revision_history_unavailable" in frame.columns and frame["catalyst_revision_history_unavailable"].astype(bool).any():
        warnings.append("Some catalyst periods use current catalyst rows because immutable revision history is unavailable for older records.")
    return {
        "summary": {
            "dataset_id": dataset_id,
            "total_rows": int(len(frame)),
            "tickers": int(frame["ticker"].nunique()),
            "years_covered": round(years, 2),
            "published_llm_supported_catalyst_count": int(frame.get("published_llm_supported_count", pd.Series(dtype=float)).fillna(0).sum()),
            "positive_catalyst_count": int(frame.get("positive_catalyst_count_45d", pd.Series(dtype=float)).fillna(0).sum()),
            "negative_catalyst_count": int(frame.get("negative_catalyst_count_45d", pd.Series(dtype=float)).fillna(0).sum()),
        },
        "per_ticker": per_ticker,
        "missingness": missingness,
        "label_counts": label_counts,
        "return_distribution": return_distribution,
        "warnings": warnings,
    }
