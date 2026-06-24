from __future__ import annotations

import hashlib
import json
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from src.catalysts.repository import CATALYST_COLUMNS
from src.catalysts.sec_classification import (
    SEC_FEATURE_POLICY,
    SEC_FEATURE_POLICY_VERSION,
    classify_ticker_sec_filings_safe,
)
from src.data import storage
from src.datasets.models import DatasetBuild, FeatureSnapshot, OutcomeLabel
from src.datasets.feature_manifest import role_sets_from_frame
from src.datasets.repository import (
    insert_dataset_build,
    insert_feature_snapshots,
    insert_outcome_labels,
    update_dataset_export_path,
)
from src.earnings.features import earnings_feature_base, precompute_earnings_features_for_dates
from src.features.catalyst import get_catalyst_features
from src.features.momentum import add_technical_columns
from src.features.regime import classify_market_regime
from src.scoring.score_engine import build_feature_set
from src.utils.trading_calendar import previous_trading_day


FEATURE_VERSION = "pit_research_v1_sec_policy_v3_earnings_v1"
DEFAULT_HORIZONS = (1, 5, 20)
LABEL_PREFIX = "label_"
METADATA_COLUMNS = {"snapshot_id", "dataset_id", "ticker", "trading_date", "as_of_timestamp"}
_LABEL_PRICE_CACHE: dict[tuple[int, int, str, str, str, str], dict[str, Any]] = {}

SEC_FEATURE_CATEGORIES = [
    "core_periodic",
    "current_event",
    "ownership",
    "equity_financing",
    "debt_financing",
    "structured_note",
    "registration_or_prospectus_other",
    "amendment",
    "unknown",
]
SEC_FEATURE_WINDOWS = (7, 30, 90)


@dataclass
class DatasetBuildResult:
    dataset_id: int
    dataset_frame: pd.DataFrame
    build: DatasetBuild
    warnings: list[str] = field(default_factory=list)
    export_path: str | None = None


def _json_loads(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return None


def _as_date(value: Any) -> date | None:
    try:
        if value is None or pd.isna(value):
            return None
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _as_utc_datetime(value: Any) -> datetime | None:
    try:
        if value is None or pd.isna(value):
            return None
        parsed = pd.to_datetime(value, utc=True)
        if pd.isna(parsed):
            return None
        return parsed.to_pydatetime()
    except Exception:
        return None


def as_of_after_close(trading_date: date) -> datetime:
    """Conservative after-close timestamp for point-in-time availability checks."""
    return datetime.combine(trading_date, time(23, 59, 59), tzinfo=UTC)


def _sorted_ohlcv(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    ordered = df.copy()
    ordered.index = pd.to_datetime(ordered.index).tz_localize(None)
    return ordered.sort_index()


def _slice_through(df: pd.DataFrame, trading_date: date) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    return df[df.index <= pd.Timestamp(trading_date)]


def _safe_feature_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _feature_bool(value: Any) -> bool:
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return bool(value)


def _trading_dates(df: pd.DataFrame, start_date: date, end_date: date) -> list[date]:
    if df is None or df.empty:
        return []
    dates = [pd.Timestamp(idx).date() for idx in df.index]
    return [value for value in dates if start_date <= value <= end_date]


def _history_map_as_of(histories: dict[str, pd.DataFrame], trading_date: date) -> dict[str, pd.DataFrame]:
    return {ticker: _slice_through(df, trading_date) for ticker, df in histories.items()}


def precompute_feature_sets_for_dates(
    ticker: str,
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    snapshot_dates: list[date],
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
) -> dict[date, dict[str, Any]]:
    """Precompute scanner-compatible market feature sets for dataset snapshots.

    The legacy dataset path called ``build_feature_set`` for every ticker/date,
    which repeatedly sliced the same OHLCV frames and rebuilt rolling columns.
    This function preserves those formulas while doing the rolling work once per
    ticker and once for the aligned SPY comparison.
    """
    del min_price, min_avg_dollar_volume
    dates = sorted(set(snapshot_dates))
    if not dates or ticker_df is None or ticker_df.empty or "close" not in ticker_df.columns:
        return {}

    ordered = _sorted_ohlcv(ticker_df)
    enriched = add_technical_columns(ordered)
    volume = ordered["volume"] if "volume" in ordered.columns else pd.Series(index=ordered.index, dtype=float)
    close = ordered["close"] if "close" in ordered.columns else pd.Series(index=ordered.index, dtype=float)
    avg_volume_20 = volume.shift(1).rolling(20).mean()
    volume_ratio_20 = volume / avg_volume_20
    avg_dollar_volume_20 = avg_volume_20 * close

    rs_20 = pd.Series(index=ordered.index, dtype=float)
    rs_60 = pd.Series(index=ordered.index, dtype=float)
    rs_score = pd.Series(index=ordered.index, dtype=float)
    if spy_df is not None and not spy_df.empty and "close" in spy_df.columns:
        aligned = pd.concat(
            [
                ordered["close"].rename("ticker"),
                _sorted_ohlcv(spy_df)["close"].rename("spy"),
            ],
            axis=1,
            join="inner",
        ).dropna()
        if not aligned.empty:
            aligned_ticker_ret_20 = aligned["ticker"] / aligned["ticker"].shift(20) - 1
            aligned_spy_ret_20 = aligned["spy"] / aligned["spy"].shift(20) - 1
            aligned_ticker_ret_60 = aligned["ticker"] / aligned["ticker"].shift(60) - 1
            aligned_spy_ret_60 = aligned["spy"] / aligned["spy"].shift(60) - 1
            aligned_rs_20 = aligned_ticker_ret_20 - aligned_spy_ret_20
            aligned_rs_60 = aligned_ticker_ret_60 - aligned_spy_ret_60
            aligned_score = ((aligned_rs_20 * 1.5 + aligned_rs_60) / 0.20).clip(-1, 1)
            rs_20 = aligned_rs_20.reindex(ordered.index, method="ffill")
            rs_60 = aligned_rs_60.reindex(ordered.index, method="ffill")
            rs_score = aligned_score.reindex(ordered.index, method="ffill")

    positions = {pd.Timestamp(idx).date(): pos for pos, idx in enumerate(enriched.index)}
    outputs: dict[date, dict[str, Any]] = {}
    for trading_date in dates:
        pos = positions.get(trading_date)
        if pos is None:
            continue
        row = enriched.iloc[pos]
        bars = int(pos + 1)
        last_price = _safe_feature_float(row.get("close"))
        avg_volume = _safe_feature_float(avg_volume_20.iloc[pos]) if len(avg_volume_20) > pos else None
        last_volume = _safe_feature_float(volume.iloc[pos]) if len(volume) > pos else None
        volume_ratio = _safe_feature_float(volume_ratio_20.iloc[pos]) if len(volume_ratio_20) > pos else None
        avg_dollar_volume = _safe_feature_float(avg_dollar_volume_20.iloc[pos]) if len(avg_dollar_volume_20) > pos else None
        price_ok = last_price is not None and last_price >= 5.0
        adv_ok = avg_dollar_volume is not None and avg_dollar_volume >= 10_000_000
        if price_ok and adv_ok:
            liquidity_score = 1.0
            liquidity_label = "Acceptable"
        elif price_ok and avg_dollar_volume is not None and avg_dollar_volume >= 3_500_000:
            liquidity_score = 0.55
            liquidity_label = "Thin"
        else:
            liquidity_score = 0.15
            liquidity_label = "Liquidity Too Low"
        data_quality = "ok" if bars >= 200 else "limited_history"
        if bars < 50:
            data_quality = "not_enough_history"
        outputs[trading_date] = {
            "ticker": ticker.upper(),
            "has_data": True,
            "data_quality": data_quality,
            "bars": bars,
            "last_price": last_price,
            "daily_return": _safe_feature_float(row.get("daily_return")),
            "ret_5d": _safe_feature_float(row.get("ret_5d")),
            "ret_20d": _safe_feature_float(row.get("ret_20d")),
            "ret_60d": _safe_feature_float(row.get("ret_60d")),
            "ret_120d": _safe_feature_float(row.get("ret_120d")),
            "volatility_20d": _safe_feature_float(row.get("volatility_20d")),
            "ma_20": _safe_feature_float(row.get("ma_20")),
            "ma_50": _safe_feature_float(row.get("ma_50")),
            "ma_200": _safe_feature_float(row.get("ma_200")),
            "distance_20d_ma": _safe_feature_float(row.get("distance_20d_ma")),
            "distance_50d_ma": _safe_feature_float(row.get("distance_50d_ma")),
            "above_50d_ma": _feature_bool(row.get("above_50d_ma")),
            "above_200d_ma": _feature_bool(row.get("above_200d_ma")),
            "current_volume": last_volume,
            "avg_volume_20d": avg_volume,
            "volume_ratio_20d": volume_ratio,
            "avg_dollar_volume_20d": avg_dollar_volume,
            "volume_anomaly": bool(volume_ratio is not None and volume_ratio > 1.5),
            "price_ok": bool(price_ok),
            "avg_dollar_volume_ok": bool(adv_ok),
            "liquidity_score_raw": liquidity_score,
            "liquidity_label": liquidity_label,
            "relative_strength_20d": _safe_feature_float(rs_20.loc[pd.Timestamp(trading_date)])
            if pd.Timestamp(trading_date) in rs_20.index
            else None,
            "relative_strength_60d": _safe_feature_float(rs_60.loc[pd.Timestamp(trading_date)])
            if pd.Timestamp(trading_date) in rs_60.index
            else None,
            "relative_strength_score_raw": _safe_feature_float(rs_score.loc[pd.Timestamp(trading_date)])
            if pd.Timestamp(trading_date) in rs_score.index
            else None,
        }
    return outputs


def _publication_row(db_path: str | Path, publication_id: int) -> dict[str, Any] | None:
    with storage.connect(db_key) as conn:
        row = conn.execute(
            """
            SELECT publication_id, publication_status, published_at, reverted_at, updated_at,
                   proposal_id, extraction_id, document_id, after_snapshot_json
            FROM catalyst_publications
            WHERE publication_id = ?
            """,
            (int(publication_id),),
        ).fetchone()
    return None if row is None else dict(row)


def _publication_rows_for_ticker(db_path: str | Path, ticker: str) -> list[dict[str, Any]]:
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT p.publication_id, p.publication_status, p.published_at, p.reverted_at, p.updated_at,
                   p.proposal_id, p.extraction_id, p.document_id, p.after_snapshot_json
            FROM catalyst_publications p
            JOIN catalyst_proposals cp ON cp.proposal_id = p.proposal_id
            WHERE cp.ticker = ?
            ORDER BY datetime(p.published_at), p.publication_id
            """,
            (ticker.upper(),),
        ).fetchall()
    return [dict(row) for row in rows]


def _extraction_row(db_path: str | Path, extraction_id: int) -> dict[str, Any] | None:
    with storage.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT extraction_id, risk_severity, confidence, document_relevance, evidence_sufficiency, provider, model_name
            FROM llm_extractions
            WHERE extraction_id = ?
            """,
            (int(extraction_id),),
        ).fetchone()
    return None if row is None else dict(row)


def _latest_publication_ref(payload: dict[str, Any]) -> dict[str, Any]:
    history = payload.get("llm_publication_history")
    if isinstance(history, list) and history:
        latest = history[-1]
        return latest if isinstance(latest, dict) else {}
    return {}


def _row_available_at(row: pd.Series | dict[str, Any]) -> datetime | None:
    return _as_utc_datetime(row.get("available_at")) or _as_utc_datetime(row.get("created_at"))


def _publication_end_timestamp(publication: dict[str, Any]) -> datetime | None:
    reverted_at = _as_utc_datetime(publication.get("reverted_at"))
    if reverted_at is not None:
        return reverted_at
    if publication.get("publication_status") == "superseded":
        updated_at = _as_utc_datetime(publication.get("updated_at"))
        published_at = _as_utc_datetime(publication.get("published_at"))
        if updated_at is not None and (published_at is None or updated_at > published_at):
            return updated_at
    return None


def _publication_active_as_of(publication: dict[str, Any], as_of_timestamp: datetime) -> bool:
    published_at = _as_utc_datetime(publication.get("published_at"))
    if published_at is None or published_at > as_of_timestamp:
        return False
    end_timestamp = _publication_end_timestamp(publication)
    return bool(end_timestamp is None or as_of_timestamp < end_timestamp)


def _publication_snapshots_as_of(db_path: str | Path, ticker: str, as_of_timestamp: datetime) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for publication in _publication_rows_for_ticker(db_path, ticker):
        if not _publication_active_as_of(publication, as_of_timestamp):
            continue
        snapshot = _json_loads(publication.get("after_snapshot_json")) or {}
        if not isinstance(snapshot, dict):
            continue
        snapshot = dict(snapshot)
        snapshot["ticker"] = ticker.upper()
        snapshot.setdefault("source", "llm_supported")
        snapshot.setdefault("created_at", publication.get("published_at"))
        snapshot.setdefault("updated_at", publication.get("published_at"))
        rows.append(snapshot)
    return rows


def _catalyst_has_revision_history(db_path: str | Path, catalyst_id: int) -> bool:
    with storage.connect(db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM catalyst_revisions WHERE catalyst_id = ? LIMIT 1",
            (int(catalyst_id),),
        ).fetchone()
    return row is not None


def _revision_snapshots_as_of(
    db_path: str | Path,
    ticker: str,
    as_of_timestamp: datetime,
    exclude_sec_edgar: bool = False,
) -> tuple[list[dict[str, Any]], set[int]]:
    db_key = str(Path(db_path).resolve())
    ticker_key = ticker.upper()
    sec_filter = """
              AND (
                  c.id IS NULL
                  OR NOT (c.event_type = 'sec_filing' AND c.source = 'SEC EDGAR')
              )
            """ if exclude_sec_edgar else ""
    with storage.connect(db_path) as conn:
        signature = conn.execute(
            f"""
            SELECT COUNT(*) AS row_count, MAX(recorded_timestamp) AS max_recorded_at
            FROM catalyst_revisions cr
            LEFT JOIN catalysts c ON c.id = cr.catalyst_id
            WHERE cr.ticker = ?
            {sec_filter}
            """,
            (ticker_key,),
        ).fetchone()
    changed_ids = _changed_revision_ids_cached(
        db_key,
        ticker_key,
        int(signature["row_count"] or 0),
        str(signature["max_recorded_at"] or ""),
        exclude_sec_edgar,
    )
    with storage.connect(db_path) as conn:
        if not changed_ids:
            return [], set()
        placeholders = ",".join(["?"] * len(changed_ids))
        revisions = conn.execute(
            f"""
            SELECT cr.catalyst_id, cr.action, cr.before_snapshot_json, cr.after_snapshot_json, cr.effective_timestamp
            FROM catalyst_revisions cr
            LEFT JOIN catalysts c ON c.id = cr.catalyst_id
            WHERE cr.ticker = ?
              AND cr.catalyst_id IN ({placeholders})
              AND datetime(cr.effective_timestamp) <= datetime(?)
              {sec_filter}
            ORDER BY datetime(cr.effective_timestamp), cr.revision_id
            """,
            (ticker.upper(), *sorted(changed_ids), as_of_timestamp.isoformat(timespec="seconds")),
        ).fetchall()
    latest: dict[int, dict[str, Any]] = {}
    ids_with_history: set[int] = set()
    for revision in revisions:
        catalyst_id = int(revision["catalyst_id"])
        ids_with_history.add(catalyst_id)
        if revision["action"] == "delete":
            latest.pop(catalyst_id, None)
            continue
        snapshot = _json_loads(revision["after_snapshot_json"]) or {}
        if isinstance(snapshot, dict) and snapshot:
            latest[catalyst_id] = dict(snapshot)
    return list(latest.values()), ids_with_history


@lru_cache(maxsize=256)
def _changed_revision_ids_cached(
    db_key: str,
    ticker: str,
    row_count: int,
    max_recorded_at: str,
    exclude_sec_edgar: bool,
) -> set[int]:
    del row_count, max_recorded_at
    sec_filter = """
                  AND (
                      c.id IS NULL
                      OR NOT (c.event_type = 'sec_filing' AND c.source = 'SEC EDGAR')
                  )
                """ if exclude_sec_edgar else ""
    with storage.connect(db_key) as conn:
        return {
            int(row["catalyst_id"])
            for row in conn.execute(
                f"""
                SELECT DISTINCT cr.catalyst_id
                FROM catalyst_revisions cr
                LEFT JOIN catalysts c ON c.id = cr.catalyst_id
                WHERE cr.ticker = ? AND cr.action <> 'create'
                {sec_filter}
                """,
                (ticker,),
            ).fetchall()
        }


def _current_catalyst_rows_as_of(
    db_path: str | Path,
    ticker: str,
    as_of_timestamp: datetime,
    trading_date: date | None = None,
    include_all: bool = True,
    exclude_sec_edgar: bool = False,
) -> pd.DataFrame:
    storage.init_db(db_path)
    db_key = str(Path(db_path).resolve())
    ticker_key = ticker.upper()
    with storage.connect(db_path) as conn:
        signature = conn.execute(
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at,
                   MAX(available_at) AS max_available_at
            FROM catalysts
            WHERE ticker = ?
            """,
            (ticker_key,),
        ).fetchone()
        sec_signature = conn.execute(
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at,
                   MAX(id) AS max_id
            FROM catalysts
            WHERE ticker = ?
              AND event_type = 'sec_filing'
              AND source = 'SEC EDGAR'
            """,
            (ticker_key,),
        ).fetchone()
    row_count = int(signature["row_count"] or 0)
    if row_count == 0:
        return pd.DataFrame(columns=CATALYST_COLUMNS)
    if exclude_sec_edgar:
        class_row_count = 0
        class_updated_at = ""
    else:
        _ensure_sec_classifications_current(
            db_key,
            ticker_key,
            int(sec_signature["row_count"] or 0),
            str(sec_signature["max_updated_at"] or ""),
            int(sec_signature["max_id"] or 0),
        )
        with storage.connect(db_path) as conn:
            class_signature = conn.execute(
                """
                SELECT COUNT(*) AS row_count, MAX(updated_at) AS max_updated_at
                FROM sec_filing_classifications
                WHERE ticker = ?
                """,
                (ticker_key,),
            ).fetchone()
        class_row_count = int(class_signature["row_count"] or 0)
        class_updated_at = str(class_signature["max_updated_at"] or "")
    current = _cached_current_catalyst_rows(
        db_key,
        ticker_key,
        row_count,
        str(signature["max_updated_at"] or ""),
        str(signature["max_available_at"] or ""),
        class_row_count,
        class_updated_at,
        exclude_sec_edgar,
    )
    if current.empty:
        return current
    as_of = pd.Timestamp(as_of_timestamp)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    filtered = current[current["_available_at_parsed"].notna() & (current["_available_at_parsed"] <= as_of)].copy()
    if exclude_sec_edgar and not filtered.empty and "event_type" in filtered.columns:
        source = filtered.get("source", pd.Series("", index=filtered.index)).astype(str)
        sec_edgar = filtered["event_type"].astype(str).eq("sec_filing") & source.eq("SEC EDGAR")
        filtered = filtered[~sec_edgar].copy()
    if not include_all and trading_date is not None:
        filtered = _snapshot_relevant_catalyst_rows(filtered, trading_date)
    if include_all:
        return filtered.drop(columns=["_available_at_parsed", "_event_date_parsed", "_sec_form"], errors="ignore")
    return filtered


@lru_cache(maxsize=128)
def _ensure_sec_classifications_current(
    db_key: str,
    ticker: str,
    sec_row_count: int,
    sec_max_updated_at: str,
    sec_max_id: int,
) -> tuple[str, int]:
    del sec_max_updated_at, sec_max_id
    if sec_row_count <= 0:
        return (ticker, 0)
    result = classify_ticker_sec_filings_safe(db_key, ticker)
    return (ticker, int(result.get("classified", 0) or 0))


@lru_cache(maxsize=256)
def _cached_current_catalyst_rows(
    db_key: str,
    ticker: str,
    row_count: int,
    max_updated_at: str,
    max_available_at: str,
    classification_count: int,
    classification_updated_at: str,
    exclude_sec_edgar: bool,
) -> pd.DataFrame:
    del row_count, max_updated_at, max_available_at, classification_count, classification_updated_at
    sec_filter = "AND NOT (event_type = 'sec_filing' AND source = 'SEC EDGAR')" if exclude_sec_edgar else ""
    with storage.connect(db_key) as conn:
        frame = pd.read_sql_query(
            f"""
            SELECT {", ".join(CATALYST_COLUMNS)}
            FROM catalysts
            WHERE ticker = ?
              {sec_filter}
            ORDER BY event_date DESC, id DESC
            """,
            conn,
            params=(ticker,),
        )
        if exclude_sec_edgar:
            classifications = pd.DataFrame()
        else:
            classifications = pd.read_sql_query(
                """
                SELECT catalyst_id, accession_number, form AS sec_classification_form, classification AS sec_classification,
                       classification_reason AS sec_classification_reason, classifier_version AS sec_classifier_version,
                       feature_eligible AS sec_feature_eligible, exclusion_reason AS sec_exclusion_reason,
                       classified_at AS sec_classified_at
                FROM sec_filing_classifications
                WHERE ticker = ?
                """,
                conn,
                params=(ticker,),
            )
    if frame.empty:
        frame["_available_at_parsed"] = pd.Series(dtype="datetime64[ns, UTC]")
        return frame
    if not classifications.empty and not exclude_sec_edgar:
        frame = frame.merge(classifications, how="left", left_on="id", right_on="catalyst_id")
    elif not exclude_sec_edgar:
        frame["sec_classification"] = None
        frame["sec_feature_eligible"] = 0
        frame["sec_exclusion_reason"] = None
    availability = frame["available_at"].where(frame["available_at"].notna() & frame["available_at"].astype(str).ne(""), frame["created_at"])
    frame["_available_at_parsed"] = pd.to_datetime(availability, utc=True, errors="coerce")
    frame["_event_date_parsed"] = pd.to_datetime(frame["event_date"], errors="coerce")
    frame["_sec_form"] = frame["raw_payload_json"].map(_raw_sec_form)
    return frame


def _raw_sec_form(value: Any) -> str:
    payload = _json_loads(value) or {}
    if isinstance(payload, dict):
        return str(payload.get("form") or "").upper().strip()
    return ""


def _sec_metadata_rows_as_of(
    db_path: str | Path,
    ticker: str,
    as_of_timestamp: datetime,
    trading_date: date,
) -> pd.DataFrame:
    storage.init_db(db_path)
    db_key = str(Path(db_path).resolve())
    ticker_key = ticker.upper().strip()
    with storage.connect(db_path) as conn:
        signature = conn.execute(
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at,
                   MAX(id) AS max_id
            FROM catalysts
            WHERE ticker = ?
              AND event_type = 'sec_filing'
              AND source = 'SEC EDGAR'
            """,
            (ticker_key,),
        ).fetchone()
    sec_row_count = int(signature["row_count"] or 0)
    if sec_row_count <= 0:
        return pd.DataFrame()
    _ensure_sec_classifications_current(
        db_key,
        ticker_key,
        sec_row_count,
        str(signature["max_updated_at"] or ""),
        int(signature["max_id"] or 0),
    )
    with storage.connect(db_path) as conn:
        class_signature = conn.execute(
            """
            SELECT COUNT(*) AS row_count, MAX(updated_at) AS max_updated_at
            FROM sec_filing_classifications
            WHERE ticker = ?
            """,
            (ticker_key,),
        ).fetchone()
    frame = _cached_sec_metadata_rows(
        db_key,
        ticker_key,
        sec_row_count,
        str(signature["max_updated_at"] or ""),
        int(signature["max_id"] or 0),
        int(class_signature["row_count"] or 0),
        str(class_signature["max_updated_at"] or ""),
    )
    if frame.empty:
        return frame
    as_of = pd.Timestamp(as_of_timestamp)
    if as_of.tzinfo is None:
        as_of = as_of.tz_localize("UTC")
    else:
        as_of = as_of.tz_convert("UTC")
    filtered = frame[frame["_available_at_parsed"].notna() & (frame["_available_at_parsed"] <= as_of)].copy()
    return _snapshot_relevant_catalyst_rows(filtered, trading_date)


@lru_cache(maxsize=128)
def _cached_sec_metadata_rows(
    db_key: str,
    ticker: str,
    sec_row_count: int,
    sec_max_updated_at: str,
    sec_max_id: int,
    classification_count: int,
    classification_updated_at: str,
) -> pd.DataFrame:
    del sec_row_count, sec_max_updated_at, sec_max_id, classification_count, classification_updated_at
    with storage.connect(db_key) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                c.id, c.ticker, c.event_date, c.event_time, c.event_type,
                c.title, c.summary, c.source, c.source_url, c.sentiment_label,
                c.catalyst_strength, c.confidence, c.is_manual, c.available_at,
                c.created_at, c.updated_at,
                s.form AS _sec_form,
                s.classification AS sec_classification,
                s.classification_reason AS sec_classification_reason,
                s.classifier_version AS sec_classifier_version,
                s.feature_eligible AS sec_feature_eligible,
                s.exclusion_reason AS sec_exclusion_reason,
                s.classified_at AS sec_classified_at
            FROM catalysts c
            LEFT JOIN sec_filing_classifications s ON s.catalyst_id = c.id
            WHERE c.ticker = ?
              AND c.event_type = 'sec_filing'
              AND c.source = 'SEC EDGAR'
            ORDER BY c.available_at, c.event_date, c.id
            """,
            conn,
            params=(ticker,),
        )
    if frame.empty:
        return frame
    availability = frame["available_at"].where(frame["available_at"].notna() & frame["available_at"].astype(str).ne(""), frame["created_at"])
    frame["_available_at_parsed"] = pd.to_datetime(availability, utc=True, errors="coerce")
    frame["_event_date_parsed"] = pd.to_datetime(frame["event_date"], errors="coerce")
    return frame


def _snapshot_relevant_catalyst_rows(current: pd.DataFrame, trading_date: date) -> pd.DataFrame:
    if current.empty or "event_type" not in current.columns:
        return current
    event_type = current["event_type"].astype(str)
    non_sec = ~event_type.eq("sec_filing")
    availability_dates = pd.to_datetime(current.get("_available_at_parsed"), utc=True, errors="coerce")
    recent_start = pd.Timestamp(trading_date - timedelta(days=140))
    recent_end = pd.Timestamp(datetime.combine(trading_date, time.max), tz="UTC")
    recent_sec = availability_dates.notna() & (availability_dates >= recent_start.tz_localize("UTC")) & (availability_dates <= recent_end)
    forms = current.get("_sec_form", pd.Series("", index=current.index)).astype(str).str.upper()
    core_sec = forms.str.startswith(("8-K", "10-Q", "10-K"))
    result = current[non_sec | recent_sec | core_sec].copy()
    # Raw SEC payloads stay in SQLite. The dataset path only needs parsed form,
    # availability, and classification columns, so avoid copying large JSON into
    # every historical snapshot for high-volume issuers.
    source = result.get("source", pd.Series("", index=result.index)).astype(str)
    sec_edgar = result["event_type"].astype(str).eq("sec_filing") & source.eq("SEC EDGAR")
    if "raw_payload_json" in result.columns and bool(sec_edgar.any()):
        result.loc[sec_edgar, "raw_payload_json"] = None
    return result


def _dataset_scoring_catalysts(catalysts: pd.DataFrame) -> pd.DataFrame:
    """Exclude raw neutral SEC metadata from generic catalyst scoring/count features.

    SEC rows are represented by the curated sec_* feature policy below. Published
    LLM-supported rows are kept because they are active reviewed catalysts.
    """
    if catalysts is None or catalysts.empty or "event_type" not in catalysts.columns:
        return catalysts
    event_type = catalysts["event_type"].astype(str)
    source = catalysts.get("source", pd.Series("", index=catalysts.index)).astype(str)
    llm_supported = source.eq("llm_supported")
    return catalysts[~event_type.eq("sec_filing") | llm_supported].copy()


def _is_llm_supported_row(row: pd.Series | dict[str, Any]) -> bool:
    source = str(row.get("source") or "")
    if source == "llm_supported":
        return True
    if source == "SEC EDGAR":
        return False
    raw_payload = row.get("raw_payload_json")
    if not raw_payload or '"llm_supported"' not in str(raw_payload).lower():
        return False
    payload = _json_loads(row.get("raw_payload_json")) or {}
    return bool(payload.get("llm_supported"))


def _catalyst_is_available(
    db_path: str | Path,
    row: pd.Series,
    as_of_timestamp: datetime,
) -> bool:
    available_at = _row_available_at(row)
    if available_at is None or available_at > as_of_timestamp:
        return False

    payload = _json_loads(row.get("raw_payload_json")) or {}
    is_llm_supported = bool(payload.get("llm_supported")) or str(row.get("source") or "") == "llm_supported"
    if not is_llm_supported:
        return True

    publication_id = payload.get("latest_publication_id")
    if publication_id is None:
        publication_id = _latest_publication_ref(payload).get("publication_id")
    if publication_id is None:
        return False

    publication = _publication_row(db_path, int(publication_id))
    if not publication:
        return False
    return _publication_active_as_of(publication, as_of_timestamp)


def active_catalysts_as_of(
    db_path: str | Path,
    ticker: str,
    trading_date: date,
    as_of_timestamp: datetime | None = None,
    include_all: bool = True,
    exclude_sec_edgar: bool = False,
) -> pd.DataFrame:
    as_of_timestamp = as_of_timestamp or as_of_after_close(trading_date)
    current = _current_catalyst_rows_as_of(
        db_path,
        ticker,
        as_of_timestamp,
        trading_date,
        include_all=include_all,
        exclude_sec_edgar=exclude_sec_edgar,
    )
    revision_rows, ids_with_history = _revision_snapshots_as_of(
        db_path,
        ticker,
        as_of_timestamp,
        exclude_sec_edgar=exclude_sec_edgar,
    )
    rows_by_id: dict[int, dict[str, Any]] = {
        int(row.get("id")): dict(row)
        for row in revision_rows
        if row.get("id") is not None
    }
    revision_warnings: list[str] = []

    if not current.empty:
        for _, row in current.iterrows():
            catalyst_id = int(row.get("id")) if pd.notna(row.get("id")) else None
            if catalyst_id is None or catalyst_id in ids_with_history or _is_llm_supported_row(row):
                continue
            rows_by_id[catalyst_id] = dict(row)
            if not ids_with_history:
                revision_warnings = [
                    "Current catalyst rows without update/delete revisions are reconstructed from available_at timestamps."
                ]

    for snapshot in _publication_snapshots_as_of(db_path, ticker, as_of_timestamp):
        catalyst_id = snapshot.get("id")
        if catalyst_id is None:
            catalyst_id = snapshot.get("catalyst_id")
        if catalyst_id is None:
            continue
        rows_by_id[int(catalyst_id)] = snapshot

    available = pd.DataFrame(list(rows_by_id.values()))
    if available.empty:
        available.attrs["revision_history_warnings"] = revision_warnings
        return available
    available.attrs["revision_history_warnings"] = revision_warnings
    if "event_date" in available.columns:
        if "_event_date_parsed" in available.columns:
            parsed_event_dates = pd.to_datetime(available["_event_date_parsed"], errors="coerce")
            parsed_dates = pd.Series(parsed_event_dates.dt.date, index=available.index)
            missing_dates = parsed_event_dates.isna()
            if bool(missing_dates.any()):
                parsed_dates.loc[missing_dates] = available.loc[missing_dates, "event_date"].map(_as_date)
            available["event_date"] = parsed_dates
        else:
            available["event_date"] = available["event_date"].map(_as_date)
    return available.sort_values(["event_date", "id"], ascending=[False, False])


def _recent_catalyst_counts(catalysts: pd.DataFrame, trading_date: date, window_days: int = 45) -> dict[str, int]:
    if catalysts is None or catalysts.empty:
        return {"recent_catalyst_count_45d": 0, "positive_catalyst_count_45d": 0, "negative_catalyst_count_45d": 0}
    events = catalysts.copy()
    if "_event_date_parsed" in events.columns:
        events["event_date_parsed"] = pd.to_datetime(events["_event_date_parsed"], errors="coerce").dt.date
    else:
        events["event_date_parsed"] = events["event_date"].map(_as_date)
    start = trading_date - timedelta(days=window_days)
    recent = events[
        events["event_date_parsed"].notna()
        & (events["event_date_parsed"] >= start)
        & (events["event_date_parsed"] <= trading_date)
    ]
    sentiment = recent["sentiment_label"].astype(str).str.lower() if not recent.empty else pd.Series(dtype=str)
    return {
        "recent_catalyst_count_45d": int(len(recent)),
        "positive_catalyst_count_45d": int(sentiment.eq("positive").sum()),
        "negative_catalyst_count_45d": int(sentiment.eq("negative").sum()),
    }


def precompute_market_regimes_for_dates(
    histories: dict[str, pd.DataFrame],
    trading_dates: list[date],
) -> dict[date, dict[str, Any]]:
    """Precompute market regime once per date instead of once per ticker/date."""
    market_histories = {
        symbol: histories.get(symbol, pd.DataFrame())
        for symbol in ["SPY", "QQQ", "IWM", "^VIX"]
    }
    output: dict[date, dict[str, Any]] = {}
    for trading_date in sorted(set(trading_dates)):
        output[trading_date] = classify_market_regime(_history_map_as_of(market_histories, trading_date))
    return output


def _neutral_catalyst_override() -> dict[str, Any]:
    catalyst_features = get_catalyst_features("", pd.DataFrame())
    return {
        "catalyst_features": catalyst_features,
        "catalyst_counts": {
            "recent_catalyst_count_45d": 0,
            "positive_catalyst_count_45d": 0,
            "negative_catalyst_count_45d": 0,
        },
        "llm_features": _empty_llm_supported_features(),
        "revision_history_warnings": [],
        "active_catalyst_count": 0,
        "available_catalyst_ids": [],
    }


def _publication_overlaps_window(publication: dict[str, Any], start_timestamp: datetime, end_timestamp: datetime) -> bool:
    published_at = _as_utc_datetime(publication.get("published_at"))
    if published_at is None or published_at > end_timestamp:
        return False
    end_at = _publication_end_timestamp(publication)
    if end_at is not None and end_at <= published_at:
        return False
    return bool(end_at is None or end_at > start_timestamp)


def _requires_per_date_catalyst_reconstruction(
    db_path: str | Path,
    ticker: str,
    trading_dates: list[date],
) -> bool:
    storage.init_db(db_path)
    ticker_key = ticker.upper().strip()
    start_timestamp = as_of_after_close(min(trading_dates)) if trading_dates else datetime.min.replace(tzinfo=UTC)
    end_timestamp = as_of_after_close(max(trading_dates)) if trading_dates else datetime.min.replace(tzinfo=UTC)
    with storage.connect(db_path) as conn:
        non_sec = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM catalysts
            WHERE ticker = ?
              AND NOT (event_type = 'sec_filing' AND source = 'SEC EDGAR')
            """,
            (ticker_key,),
        ).fetchone()
        changed_revisions = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM catalyst_revisions cr
            LEFT JOIN catalysts c ON c.id = cr.catalyst_id
            WHERE cr.ticker = ? AND cr.action <> 'create'
              AND (
                  c.id IS NULL
                  OR NOT (c.event_type = 'sec_filing' AND c.source = 'SEC EDGAR')
              )
            """,
            (ticker_key,),
        ).fetchone()
        publication_rows = conn.execute(
            """
            SELECT p.publication_status, p.published_at, p.reverted_at, p.updated_at
            FROM catalyst_publications p
            JOIN catalyst_proposals cp ON cp.proposal_id = p.proposal_id
            WHERE cp.ticker = ?
            """,
            (ticker_key,),
        ).fetchall()
    overlapping_publication = any(
        _publication_overlaps_window(dict(row), start_timestamp, end_timestamp) for row in publication_rows
    )
    return bool(
        int(non_sec["count"] or 0) > 0
        or int(changed_revisions["count"] or 0) > 0
        or overlapping_publication
    )


def precompute_catalyst_overrides_for_dates(
    db_path: str | Path,
    ticker: str,
    trading_dates: list[date],
) -> dict[date, dict[str, Any]]:
    """Precompute non-SEC catalyst inputs while preserving exact fallback semantics.

    SEC EDGAR metadata is handled by the SEC feature layer. If a ticker has no
    manual/system/LLM publication state that can affect catalyst features, the
    output is the same neutral shape for every date without repeatedly querying
    historical catalyst reconstruction tables.
    """
    if not trading_dates:
        return {}
    if not _requires_per_date_catalyst_reconstruction(db_path, ticker, trading_dates):
        neutral = _neutral_catalyst_override()
        return {trading_date: neutral for trading_date in trading_dates}

    output: dict[date, dict[str, Any]] = {}
    for trading_date in trading_dates:
        as_of_timestamp = as_of_after_close(trading_date)
        catalysts = active_catalysts_as_of(
            db_path,
            ticker,
            trading_date,
            as_of_timestamp,
            include_all=False,
            exclude_sec_edgar=True,
        )
        scoring_catalysts = _dataset_scoring_catalysts(catalysts)
        catalyst_features = get_catalyst_features(ticker, scoring_catalysts, as_of_date=trading_date)
        output[trading_date] = {
            "catalyst_features": catalyst_features,
            "catalyst_counts": _recent_catalyst_counts(scoring_catalysts, trading_date),
            "llm_features": _llm_supported_features(db_path, scoring_catalysts),
            "revision_history_warnings": catalysts.attrs.get("revision_history_warnings", []),
            "active_catalyst_count": int(len(scoring_catalysts)),
            "available_catalyst_ids": [
                int(value) for value in scoring_catalysts.get("id", pd.Series(dtype=int)).dropna().tolist()
            ],
        }
    return output


def _sessions_ago(trading_date: date, sessions: int) -> date:
    current = trading_date
    for _ in range(max(0, int(sessions))):
        current = previous_trading_day(current)
    return current


def _form_family(form: Any) -> str:
    value = str(form or "").upper().strip()
    return value.replace("/A", "-A")


def _sec_feature_base() -> dict[str, Any]:
    base = {
        "sec_feature_policy_version": SEC_FEATURE_POLICY_VERSION,
        "sec_metadata_available": False,
        "sec_raw_filing_count_7s_audit": 0,
        "sec_raw_filing_count_30s_audit": 0,
        "sec_raw_filing_count_90s_audit": 0,
        "sec_feature_eligible_filing_count_7s": 0,
        "sec_feature_eligible_filing_count_30s": 0,
        "sec_feature_eligible_filing_count_90s": 0,
        "sec_feature_eligible_event_days_7s": 0,
        "sec_feature_eligible_event_days_30s": 0,
        "sec_feature_eligible_event_days_90s": 0,
        "sec_days_since_latest_8k": None,
        "sec_days_since_latest_10q": None,
        "sec_days_since_latest_10k": None,
        "sec_days_since_latest_core_periodic": None,
        "sec_days_since_latest_current_event": None,
        "sec_days_since_latest_ownership": None,
        "sec_days_since_latest_equity_financing": None,
        "sec_days_since_latest_debt_financing": None,
        "sec_days_since_latest_structured_note": None,
        "sec_days_since_latest_registration_or_prospectus_other": None,
        "sec_days_since_latest_amendment": None,
        "sec_recent_equity_financing_flag": False,
        "sec_recent_structured_note_flag": False,
        "sec_recent_registration_or_prospectus_other_flag": False,
        "sec_recent_form4_count": 0,
        "sec_amendment_count_30s": 0,
        "sec_needs_review_filing_flag": False,
        "sec_unknown_classification_count_30s": 0,
        "sec_feature_excluded_count_30s": 0,
        "sec_max_raw_filings_single_day_30s_audit": 0,
    }
    for category in SEC_FEATURE_CATEGORIES:
        for window in SEC_FEATURE_WINDOWS:
            base[f"sec_{category}_event_days_{window}s"] = 0
            base[f"sec_{category}_filing_count_{window}s_audit"] = 0
        base[f"sec_{category}_present_30s"] = False
    base["sec_max_feature_eligible_filings_single_day_30s"] = 0
    return base


def _sec_filing_features(catalysts: pd.DataFrame, trading_date: date) -> dict[str, Any]:
    categories = SEC_FEATURE_CATEGORIES
    windows = SEC_FEATURE_WINDOWS
    base = _sec_feature_base()
    if catalysts is None or catalysts.empty or "event_type" not in catalysts.columns:
        return base

    events = catalysts[catalysts["event_type"].astype(str).eq("sec_filing")].copy()
    if events.empty:
        return base

    if "_available_at_parsed" in events.columns:
        available = pd.to_datetime(events["_available_at_parsed"], utc=True, errors="coerce")
    else:
        availability = events["available_at"].where(
            events["available_at"].notna() & events["available_at"].astype(str).ne(""),
            events.get("created_at"),
        )
        available = pd.to_datetime(availability, utc=True, errors="coerce")
    fallback_dates = (
        pd.to_datetime(events["_event_date_parsed"], errors="coerce")
        if "_event_date_parsed" in events.columns
        else pd.to_datetime(events["event_date"], errors="coerce")
    )
    available_dates = pd.Series(available.dt.date, index=events.index)
    fallback_date_values = pd.Series(fallback_dates.dt.date, index=events.index)
    events["available_date"] = available_dates.where(available.notna(), fallback_date_values)
    events = events.dropna(subset=["available_date"])
    if events.empty:
        return base

    if "_sec_form" in events.columns:
        events["sec_form"] = events["_sec_form"].astype(str)
    else:
        def raw_form(value: Any) -> Any:
            payload = _json_loads(value) or {}
            return payload.get("form") if isinstance(payload, dict) else None

        events["sec_form"] = events["raw_payload_json"].map(raw_form)
    title_forms = events["title"].astype(str).str.replace("SEC ", "", regex=False).str.split(" filing").str[0]
    form_source = events["sec_form"].where(events["sec_form"].astype(str).str.strip().ne(""), title_forms)
    events["form_family"] = form_source.map(_form_family)
    events["sec_classification"] = events.get("sec_classification", pd.Series("unknown", index=events.index)).fillna("unknown")
    events["sec_feature_eligible"] = (
        pd.to_numeric(events.get("sec_feature_eligible", pd.Series(0, index=events.index)), errors="coerce")
        .fillna(0)
        .astype(int)
        .astype(bool)
    )

    def days_since(forms: tuple[str, ...]) -> int | None:
        matched = events[events["form_family"].isin(forms)]
        if matched.empty:
            return None
        latest = max(matched["available_date"])
        return int((trading_date - latest).days)

    output = dict(base)
    output["sec_metadata_available"] = True

    windows_by_session = {window: _sessions_ago(trading_date, window) for window in windows}
    for window, start_date in windows_by_session.items():
        window_events = events[(events["available_date"] >= start_date) & (events["available_date"] <= trading_date)]
        eligible = window_events[window_events["sec_feature_eligible"]]
        output[f"sec_raw_filing_count_{window}s_audit"] = int(len(window_events))
        output[f"sec_feature_eligible_filing_count_{window}s"] = int(len(eligible))
        output[f"sec_feature_eligible_event_days_{window}s"] = int(eligible["available_date"].nunique())
        for category in categories:
            category_rows = window_events[window_events["sec_classification"].eq(category)]
            output[f"sec_{category}_filing_count_{window}s_audit"] = int(len(category_rows))
            if category == "unknown":
                # Unknown filings are intentionally not feature eligible.
                output[f"sec_{category}_event_days_{window}s"] = 0
            else:
                eligible_category_rows = category_rows[category_rows["sec_feature_eligible"]]
                output[f"sec_{category}_event_days_{window}s"] = int(eligible_category_rows["available_date"].nunique())
            if window == 30:
                output[f"sec_{category}_present_30s"] = bool(output[f"sec_{category}_event_days_{window}s"] > 0)

    window_30 = events[(events["available_date"] >= windows_by_session[30]) & (events["available_date"] <= trading_date)]
    eligible_30 = window_30[window_30["sec_feature_eligible"]]
    equity_financing = window_30["sec_classification"].eq("equity_financing")
    structured_note = window_30["sec_classification"].eq("structured_note")
    registration_other = window_30["sec_classification"].eq("registration_or_prospectus_other")
    form4 = window_30["sec_classification"].eq("ownership")
    amendments = window_30["sec_classification"].eq("amendment")
    needs_review = (
        window_30["title"].astype(str).str.contains("Needs Review", case=False, na=False)
        if "title" in window_30.columns
        else pd.Series(dtype=bool)
    )
    output["sec_days_since_latest_8k"] = days_since(("8-K", "8-K-A"))
    output["sec_days_since_latest_10q"] = days_since(("10-Q", "10-Q-A"))
    output["sec_days_since_latest_10k"] = days_since(("10-K", "10-K-A"))
    for category in categories:
        if category == "unknown":
            continue
        matched = events[events["sec_classification"].eq(category) & events["sec_feature_eligible"]]
        if not matched.empty:
            output[f"sec_days_since_latest_{category}"] = int((trading_date - max(matched["available_date"])).days)
    output["sec_recent_equity_financing_flag"] = bool(equity_financing.any()) if len(equity_financing) else False
    output["sec_recent_structured_note_flag"] = bool(structured_note.any()) if len(structured_note) else False
    output["sec_recent_registration_or_prospectus_other_flag"] = (
        bool(registration_other.any()) if len(registration_other) else False
    )
    output["sec_recent_form4_count"] = int(form4.sum()) if len(form4) else 0
    output["sec_amendment_count_30s"] = int(amendments.sum()) if len(amendments) else 0
    output["sec_needs_review_filing_flag"] = bool(needs_review.any()) if len(needs_review) else False
    output["sec_unknown_classification_count_30s"] = int(window_30["sec_classification"].eq("unknown").sum()) if len(window_30) else 0
    output["sec_feature_excluded_count_30s"] = int((~window_30["sec_feature_eligible"]).sum()) if len(window_30) else 0
    if not window_30.empty:
        output["sec_max_raw_filings_single_day_30s_audit"] = int(window_30.groupby("available_date").size().max())
    if not eligible_30.empty:
        output["sec_max_feature_eligible_filings_single_day_30s"] = int(eligible_30.groupby("available_date").size().max())
    return output


def _sec_metadata_signature(db_path: str | Path, ticker: str) -> tuple[int, str, int, int, str]:
    db_key = str(Path(db_path).resolve())
    ticker_key = ticker.upper().strip()
    with storage.connect(db_path) as conn:
        sec_signature = conn.execute(
            """
            SELECT COUNT(*) AS row_count,
                   MAX(updated_at) AS max_updated_at,
                   MAX(id) AS max_id
            FROM catalysts
            WHERE ticker = ?
              AND event_type = 'sec_filing'
              AND source = 'SEC EDGAR'
            """,
            (ticker_key,),
        ).fetchone()
    sec_row_count = int(sec_signature["row_count"] or 0)
    _ensure_sec_classifications_current(
        db_key,
        ticker_key,
        sec_row_count,
        str(sec_signature["max_updated_at"] or ""),
        int(sec_signature["max_id"] or 0),
    )
    with storage.connect(db_path) as conn:
        class_signature = conn.execute(
            """
            SELECT COUNT(*) AS row_count, MAX(updated_at) AS max_updated_at
            FROM sec_filing_classifications
            WHERE ticker = ?
            """,
            (ticker_key,),
        ).fetchone()
    return (
        sec_row_count,
        str(sec_signature["max_updated_at"] or ""),
        int(sec_signature["max_id"] or 0),
        int(class_signature["row_count"] or 0),
        str(class_signature["max_updated_at"] or ""),
    )


def _compact_sec_metadata_frame(db_path: str | Path, ticker: str) -> pd.DataFrame:
    storage.init_db(db_path)
    db_key = str(Path(db_path).resolve())
    ticker_key = ticker.upper().strip()
    signature = _sec_metadata_signature(db_path, ticker_key)
    return _cached_compact_sec_metadata_frame(db_key, ticker_key, SEC_FEATURE_POLICY_VERSION, *signature).copy()


@lru_cache(maxsize=128)
def _cached_compact_sec_metadata_frame(
    db_key: str,
    ticker: str,
    policy_version: str,
    sec_row_count: int,
    sec_max_updated_at: str,
    sec_max_id: int,
    classification_count: int,
    classification_updated_at: str,
) -> pd.DataFrame:
    del policy_version, sec_row_count, sec_max_updated_at, sec_max_id, classification_count, classification_updated_at
    with storage.connect(db_key) as conn:
        frame = pd.read_sql_query(
            """
            SELECT
                c.id,
                c.ticker,
                c.event_date,
                c.title,
                c.available_at,
                c.created_at,
                s.form AS sec_form,
                s.classification AS sec_classification,
                s.feature_eligible AS sec_feature_eligible
            FROM catalysts c
            LEFT JOIN sec_filing_classifications s ON s.catalyst_id = c.id
            WHERE c.ticker = ?
              AND c.event_type = 'sec_filing'
              AND c.source = 'SEC EDGAR'
            ORDER BY c.available_at, c.event_date, c.id
            """,
            conn,
            params=(ticker,),
        )
    if frame.empty:
        return frame
    availability = frame["available_at"].where(
        frame["available_at"].notna() & frame["available_at"].astype(str).ne(""),
        frame["created_at"],
    )
    parsed_available = pd.to_datetime(availability, utc=True, errors="coerce")
    fallback_dates = pd.to_datetime(frame["event_date"], errors="coerce")
    available_dates = pd.Series(parsed_available.dt.date, index=frame.index)
    fallback_date_values = pd.Series(fallback_dates.dt.date, index=frame.index)
    frame["available_date"] = available_dates.where(parsed_available.notna(), fallback_date_values)
    frame["form_family"] = frame["sec_form"].map(_form_family)
    frame["sec_classification"] = frame["sec_classification"].fillna("unknown")
    frame["sec_feature_eligible"] = (
        pd.to_numeric(frame["sec_feature_eligible"], errors="coerce").fillna(0).astype(int).astype(bool)
    )
    frame = frame.dropna(subset=["available_date"])
    return frame[
        [
            "id",
            "ticker",
            "available_date",
            "title",
            "form_family",
            "sec_classification",
            "sec_feature_eligible",
        ]
    ].copy()


def _latest_days_since(day_list: list[date], trading_date: date) -> int | None:
    pos = bisect_right(day_list, trading_date)
    if pos <= 0:
        return None
    return int((trading_date - day_list[pos - 1]).days)


def _prefix_counts(counts: dict[date, int]) -> tuple[list[date], list[int]]:
    days = sorted(counts)
    prefix = [0]
    total = 0
    for day in days:
        total += int(counts.get(day, 0) or 0)
        prefix.append(total)
    return days, prefix


def _prefix_sum(days: list[date], prefix: list[int], start_date: date, end_date: date) -> int:
    left = bisect_left(days, start_date)
    right = bisect_right(days, end_date)
    return int(prefix[right] - prefix[left])


def _range_day_count(days: list[date], start_date: date, end_date: date) -> int:
    left = bisect_left(days, start_date)
    right = bisect_right(days, end_date)
    return max(0, int(right - left))


def _range_max(counts: dict[date, int], days: list[date], start_date: date, end_date: date) -> int:
    left = bisect_left(days, start_date)
    right = bisect_right(days, end_date)
    if right <= left:
        return 0
    return max(int(counts.get(day, 0) or 0) for day in days[left:right])


def precompute_sec_features_for_dates(
    db_path: str | Path,
    ticker: str,
    snapshot_dates: list[date],
) -> dict[date, dict[str, Any]]:
    dates = sorted(set(snapshot_dates))
    if not dates:
        return {}
    events = _compact_sec_metadata_frame(db_path, ticker)
    if events.empty:
        return {trading_date: _sec_feature_base() for trading_date in dates}

    events = events[events["available_date"].notna()].copy()
    if events.empty:
        return {trading_date: _sec_feature_base() for trading_date in dates}

    daily_total: dict[date, int] = {}
    daily_eligible: dict[date, int] = {}
    daily_excluded: dict[date, int] = {}
    daily_needs_review: dict[date, int] = {}
    daily_category: dict[str, dict[date, int]] = {category: {} for category in SEC_FEATURE_CATEGORIES}
    eligible_category: dict[str, dict[date, int]] = {category: {} for category in SEC_FEATURE_CATEGORIES}
    form_day_counts: dict[str, dict[date, int]] = {}

    for row in events.itertuples(index=False):
        available_date = row.available_date
        category = str(row.sec_classification or "unknown")
        if category not in daily_category:
            category = "unknown"
        eligible = bool(row.sec_feature_eligible)
        title = str(row.title or "")
        form_family = str(row.form_family or "")

        daily_total[available_date] = daily_total.get(available_date, 0) + 1
        daily_category[category][available_date] = daily_category[category].get(available_date, 0) + 1
        form_counts = form_day_counts.setdefault(form_family, {})
        form_counts[available_date] = form_counts.get(available_date, 0) + 1
        if eligible:
            daily_eligible[available_date] = daily_eligible.get(available_date, 0) + 1
            eligible_category[category][available_date] = eligible_category[category].get(available_date, 0) + 1
        else:
            daily_excluded[available_date] = daily_excluded.get(available_date, 0) + 1
        if "needs review" in title.lower():
            daily_needs_review[available_date] = daily_needs_review.get(available_date, 0) + 1

    total_days, total_prefix = _prefix_counts(daily_total)
    eligible_days, eligible_prefix = _prefix_counts(daily_eligible)
    excluded_days, excluded_prefix = _prefix_counts(daily_excluded)
    needs_review_days, needs_review_prefix = _prefix_counts(daily_needs_review)
    category_prefixes = {category: (*_prefix_counts(counts), counts) for category, counts in daily_category.items()}
    eligible_category_prefixes = {
        category: (*_prefix_counts(counts), counts) for category, counts in eligible_category.items()
    }
    category_days = {
        category: sorted(counts)
        for category, counts in eligible_category.items()
        if category != "unknown"
    }

    def form_days(forms: tuple[str, ...]) -> list[date]:
        days: set[date] = set()
        for form in forms:
            days.update(form_day_counts.get(form, {}))
        return sorted(days)

    form_8k_days = form_days(("8-K", "8-K-A"))
    form_10q_days = form_days(("10-Q", "10-Q-A"))
    form_10k_days = form_days(("10-K", "10-K-A"))

    outputs: dict[date, dict[str, Any]] = {}
    first_available = min(daily_total) if daily_total else None
    for trading_date in dates:
        output = _sec_feature_base()
        if first_available is None or first_available > trading_date:
            outputs[trading_date] = output
            continue
        output["sec_metadata_available"] = True
        windows_by_session = {window: _sessions_ago(trading_date, window) for window in SEC_FEATURE_WINDOWS}
        for window, start_date in windows_by_session.items():
            output[f"sec_raw_filing_count_{window}s_audit"] = _prefix_sum(
                total_days,
                total_prefix,
                start_date,
                trading_date,
            )
            output[f"sec_feature_eligible_filing_count_{window}s"] = _prefix_sum(
                eligible_days,
                eligible_prefix,
                start_date,
                trading_date,
            )
            output[f"sec_feature_eligible_event_days_{window}s"] = _range_day_count(
                eligible_days,
                start_date,
                trading_date,
            )
            for category in SEC_FEATURE_CATEGORIES:
                raw_days, raw_prefix, _raw_counts = category_prefixes[category]
                output[f"sec_{category}_filing_count_{window}s_audit"] = _prefix_sum(
                    raw_days,
                    raw_prefix,
                    start_date,
                    trading_date,
                )
                if category == "unknown":
                    output[f"sec_{category}_event_days_{window}s"] = 0
                else:
                    eligible_category_days, _eligible_prefix, _eligible_counts = eligible_category_prefixes[category]
                    output[f"sec_{category}_event_days_{window}s"] = _range_day_count(
                        eligible_category_days,
                        start_date,
                        trading_date,
                    )
                if window == 30:
                    output[f"sec_{category}_present_30s"] = bool(output[f"sec_{category}_event_days_{window}s"] > 0)

        window_30_start = windows_by_session[30]
        output["sec_days_since_latest_8k"] = _latest_days_since(form_8k_days, trading_date)
        output["sec_days_since_latest_10q"] = _latest_days_since(form_10q_days, trading_date)
        output["sec_days_since_latest_10k"] = _latest_days_since(form_10k_days, trading_date)
        for category, day_list in category_days.items():
            output[f"sec_days_since_latest_{category}"] = _latest_days_since(day_list, trading_date)
        output["sec_recent_equity_financing_flag"] = bool(output.get("sec_equity_financing_event_days_30s", 0) > 0)
        output["sec_recent_structured_note_flag"] = bool(output.get("sec_structured_note_event_days_30s", 0) > 0)
        output["sec_recent_registration_or_prospectus_other_flag"] = bool(
            output.get("sec_registration_or_prospectus_other_event_days_30s", 0) > 0
        )
        ownership_days, ownership_prefix, _ownership_counts = category_prefixes["ownership"]
        amendment_days, amendment_prefix, _amendment_counts = category_prefixes["amendment"]
        unknown_days, unknown_prefix, _unknown_counts = category_prefixes["unknown"]
        output["sec_recent_form4_count"] = _prefix_sum(ownership_days, ownership_prefix, window_30_start, trading_date)
        output["sec_amendment_count_30s"] = _prefix_sum(amendment_days, amendment_prefix, window_30_start, trading_date)
        output["sec_needs_review_filing_flag"] = bool(
            _prefix_sum(needs_review_days, needs_review_prefix, window_30_start, trading_date) > 0
        )
        output["sec_unknown_classification_count_30s"] = _prefix_sum(
            unknown_days,
            unknown_prefix,
            window_30_start,
            trading_date,
        )
        output["sec_feature_excluded_count_30s"] = _prefix_sum(
            excluded_days,
            excluded_prefix,
            window_30_start,
            trading_date,
        )
        output["sec_max_raw_filings_single_day_30s_audit"] = _range_max(
            daily_total,
            total_days,
            window_30_start,
            trading_date,
        )
        output["sec_max_feature_eligible_filings_single_day_30s"] = _range_max(
            daily_eligible,
            eligible_days,
            window_30_start,
            trading_date,
        )
        outputs[trading_date] = output
    return outputs


def _empty_llm_supported_features() -> dict[str, Any]:
    return {
        "published_llm_supported_catalyst": False,
        "published_llm_supported_count": 0,
        "llm_max_confidence": 0.0,
        "llm_max_risk_severity": 0,
        "llm_relevant_count": 0,
        "llm_sufficient_or_limited_count": 0,
    }


def _llm_supported_features(db_path: str | Path, catalysts: pd.DataFrame) -> dict[str, Any]:
    if catalysts is None or catalysts.empty:
        return _empty_llm_supported_features()

    llm_rows: list[dict[str, Any]] = []
    for _, row in catalysts.iterrows():
        payload = _json_loads(row.get("raw_payload_json")) or {}
        if not payload.get("llm_supported") and str(row.get("source") or "") != "llm_supported":
            continue
        ref = _latest_publication_ref(payload)
        extraction_id = ref.get("extraction_id")
        extraction = _extraction_row(db_path, int(extraction_id)) if extraction_id is not None else None
        llm_rows.append(
            {
                "confidence": float(row.get("confidence", 0) or 0),
                "risk_severity": int((extraction or {}).get("risk_severity", 0) or 0),
                "document_relevance": (extraction or {}).get("document_relevance", "unknown"),
                "evidence_sufficiency": (extraction or {}).get("evidence_sufficiency", "unknown"),
                "provider": (extraction or {}).get("provider"),
                "model_name": (extraction or {}).get("model_name"),
            }
        )

    if not llm_rows:
        return _empty_llm_supported_features()
    return {
        "published_llm_supported_catalyst": True,
        "published_llm_supported_count": len(llm_rows),
        "llm_max_confidence": max(item["confidence"] for item in llm_rows),
        "llm_max_risk_severity": max(item["risk_severity"] for item in llm_rows),
        "llm_relevant_count": sum(1 for item in llm_rows if item["document_relevance"] == "relevant"),
        "llm_sufficient_or_limited_count": sum(
            1 for item in llm_rows if item["evidence_sufficiency"] in {"sufficient", "limited"}
        ),
    }


def build_feature_snapshot(
    db_path: str | Path,
    ticker: str,
    trading_date: date,
    histories: dict[str, pd.DataFrame],
    feature_version: str = FEATURE_VERSION,
    sec_features_override: dict[str, Any] | None = None,
    earnings_features_override: dict[str, Any] | None = None,
    regime_override: dict[str, Any] | None = None,
    catalyst_features_override: dict[str, Any] | None = None,
    feature_set_override: dict[str, Any] | None = None,
) -> FeatureSnapshot | None:
    ticker = ticker.upper().strip()
    if feature_set_override is None:
        ticker_df = _slice_through(histories.get(ticker, pd.DataFrame()), trading_date)
        spy_df = _slice_through(histories.get("SPY", pd.DataFrame()), trading_date)
        if ticker_df.empty:
            return None
        feature_set = build_feature_set(ticker, ticker_df, spy_df)
    else:
        feature_set = feature_set_override
        if not feature_set.get("has_data", False):
            return None
        ticker_df = pd.DataFrame()
    if feature_set is None:
        return None

    as_of_timestamp = as_of_after_close(trading_date)
    regime = regime_override or classify_market_regime(_history_map_as_of(histories, trading_date))
    if catalyst_features_override is None:
        catalysts = active_catalysts_as_of(
            db_path,
            ticker,
            trading_date,
            as_of_timestamp,
            include_all=False,
            exclude_sec_edgar=True,
        )
        scoring_catalysts = _dataset_scoring_catalysts(catalysts)
        catalyst_features = get_catalyst_features(ticker, scoring_catalysts, as_of_date=trading_date)
        catalyst_counts = _recent_catalyst_counts(scoring_catalysts, trading_date)
        catalyst_revision_warnings = catalysts.attrs.get("revision_history_warnings", [])
        llm_features = _llm_supported_features(db_path, scoring_catalysts)
        active_catalyst_count = int(len(scoring_catalysts))
        available_catalyst_ids = [
            int(value) for value in scoring_catalysts.get("id", pd.Series(dtype=int)).dropna().tolist()
        ]
    else:
        catalyst_features = catalyst_features_override.get("catalyst_features", {})
        catalyst_counts = catalyst_features_override.get("catalyst_counts", {})
        catalyst_revision_warnings = catalyst_features_override.get("revision_history_warnings", [])
        llm_features = catalyst_features_override.get("llm_features", _empty_llm_supported_features())
        active_catalyst_count = int(catalyst_features_override.get("active_catalyst_count", 0) or 0)
        available_catalyst_ids = [
            int(value) for value in catalyst_features_override.get("available_catalyst_ids", []) if value is not None
        ]
    if sec_features_override is None:
        sec_catalysts = _sec_metadata_rows_as_of(db_path, ticker, as_of_timestamp, trading_date)
        sec_features = _sec_filing_features(sec_catalysts, trading_date)
    else:
        sec_features = sec_features_override
    earnings_features = earnings_features_override if earnings_features_override is not None else earnings_feature_base()

    market_regime = {
        "market_regime": regime.get("regime", "Neutral"),
        "market_regime_confidence": regime.get("confidence", "unknown"),
        "regime_vix": regime.get("vix"),
        "regime_vix_elevated": bool(regime.get("vix_elevated", False)),
        "regime_qqq_spy_rs_20": regime.get("qqq_spy_rs_20"),
        "regime_iwm_spy_rs_20": regime.get("iwm_spy_rs_20"),
    }
    technical = {
        key: feature_set.get(key)
        for key in [
            "last_price",
            "daily_return",
            "ret_5d",
            "ret_20d",
            "ret_60d",
            "ret_120d",
            "volatility_20d",
            "ma_20",
            "ma_50",
            "ma_200",
            "distance_20d_ma",
            "distance_50d_ma",
            "above_50d_ma",
            "above_200d_ma",
        ]
    }
    relative_strength = {
        "relative_strength_20d": feature_set.get("relative_strength_20d"),
        "relative_strength_60d": feature_set.get("relative_strength_60d"),
        "relative_strength_score_raw": feature_set.get("relative_strength_score_raw"),
    }
    volume_liquidity = {
        key: feature_set.get(key)
        for key in [
            "current_volume",
            "avg_volume_20d",
            "volume_ratio_20d",
            "avg_dollar_volume_20d",
            "volume_anomaly",
            "avg_dollar_volume_ok",
            "price_ok",
            "liquidity_score_raw",
            "liquidity_label",
        ]
    }
    catalyst = {
        "catalyst_score": catalyst_features.get("catalyst_score", 0.0),
        "catalyst_penalty": catalyst_features.get("catalyst_penalty", 0.0),
        "catalyst_net": round(
            float(catalyst_features.get("catalyst_score", 0) or 0)
            + float(catalyst_features.get("catalyst_penalty", 0) or 0),
            2,
        ),
        "active_catalyst_count": active_catalyst_count,
        "available_catalyst_ids": available_catalyst_ids,
        **catalyst_counts,
        **sec_features,
    }
    data_quality = {
        "has_data": bool(feature_set.get("has_data", False)),
        "data_quality": feature_set.get("data_quality", "missing"),
        "bars_available": int(feature_set.get("bars", len(ticker_df)) or 0),
        "insufficient_history_200d": bool(feature_set.get("ma_200") is None),
        "missing_volume": bool(feature_set.get("current_volume") is None),
        "failed_spy_comparison": bool(feature_set.get("relative_strength_20d") is None),
        "catalyst_revision_history_unavailable": bool(catalyst_revision_warnings),
        "regime_warnings": regime.get("warnings", []),
        "catalyst_warnings": [
            *catalyst_features.get("catalyst_warnings", []),
            *catalyst_revision_warnings,
        ],
    }
    features = {
        **market_regime,
        **technical,
        **relative_strength,
        **volume_liquidity,
        **catalyst,
        **earnings_features,
        **llm_features,
        **data_quality,
    }

    return FeatureSnapshot(
        ticker=ticker,
        trading_date=trading_date,
        as_of_timestamp=as_of_timestamp,
        feature_version=feature_version,
        market_regime=market_regime,
        technical=technical,
        relative_strength=relative_strength,
        volume_liquidity=volume_liquidity,
        catalyst=catalyst,
        llm_supported=llm_features,
        data_quality=data_quality,
        features=features,
    )


def calculate_outcome_labels(
    snapshot: FeatureSnapshot,
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
) -> list[OutcomeLabel]:
    if ticker_df is None or ticker_df.empty:
        return []
    ticker_cache = _label_price_cache(ticker_df)
    ordered = ticker_cache["ordered"]
    dates = ticker_cache["dates"]
    try:
        snapshot_pos = dates.index(snapshot.trading_date)
    except ValueError:
        return []
    entry_pos = snapshot_pos + 1
    if entry_pos >= len(ordered):
        return []

    labels: list[OutcomeLabel] = []
    entry_date = dates[entry_pos]
    entry_price = float(ordered["close"].iloc[entry_pos])
    spy_close_by_date = _label_price_cache(spy_df)["close_by_date"]

    for horizon in horizons:
        exit_pos = entry_pos + int(horizon)
        if exit_pos >= len(ordered):
            continue
        exit_date = dates[exit_pos]
        exit_price = float(ordered["close"].iloc[exit_pos])
        forward_return = exit_price / entry_price - 1
        spy_forward_return = None
        if entry_date in spy_close_by_date and exit_date in spy_close_by_date and spy_close_by_date[entry_date] != 0:
            spy_forward_return = spy_close_by_date[exit_date] / spy_close_by_date[entry_date] - 1
        labels.append(
            OutcomeLabel(
                snapshot_id=snapshot.snapshot_id,
                ticker=snapshot.ticker,
                entry_date=entry_date,
                horizon=f"{int(horizon)}_session",
                entry_price=entry_price,
                exit_date=exit_date,
                exit_price=exit_price,
                forward_return=float(forward_return),
                spy_forward_return=spy_forward_return,
                excess_return=None if spy_forward_return is None else float(forward_return - spy_forward_return),
                label_available_at=as_of_after_close(exit_date),
            )
        )
    return labels


def _label_price_cache(df: pd.DataFrame | None) -> dict[str, Any]:
    if df is None or df.empty:
        return {"ordered": pd.DataFrame(), "dates": [], "close_by_date": {}}
    first_index = str(df.index[0]) if len(df.index) else ""
    last_index = str(df.index[-1]) if len(df.index) else ""
    first_close = str(df["close"].iloc[0]) if "close" in df.columns and len(df) else ""
    last_close = str(df["close"].iloc[-1]) if "close" in df.columns and len(df) else ""
    cache_key = (id(df), len(df), first_index, last_index, first_close, last_close)
    cache = _LABEL_PRICE_CACHE.get(cache_key)
    if isinstance(cache, dict):
        return cache
    ordered = _sorted_ohlcv(df)
    dates = [pd.Timestamp(idx).date() for idx in ordered.index]
    close_by_date = (
        {pd.Timestamp(idx).date(): float(value) for idx, value in ordered["close"].dropna().items()}
        if not ordered.empty and "close" in ordered.columns
        else {}
    )
    cache = {"ordered": ordered, "dates": dates, "close_by_date": close_by_date}
    _LABEL_PRICE_CACHE[cache_key] = cache
    return cache


def clear_dataset_build_caches() -> None:
    """Release dataset-build caches that can otherwise retain per-ticker frames."""
    _LABEL_PRICE_CACHE.clear()
    _ensure_sec_classifications_current.cache_clear()
    _cached_current_catalyst_rows.cache_clear()
    _cached_sec_metadata_rows.cache_clear()
    _cached_compact_sec_metadata_frame.cache_clear()


def _flatten_snapshot_rows(
    snapshots: list[FeatureSnapshot],
    labels_by_key: dict[tuple[str, date], list[OutcomeLabel]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for snapshot in snapshots:
        row = {
            "ticker": snapshot.ticker,
            "trading_date": snapshot.trading_date.isoformat(),
            "as_of_timestamp": snapshot.as_of_timestamp.isoformat(timespec="seconds"),
            **snapshot.features,
        }
        for label in labels_by_key.get((snapshot.ticker, snapshot.trading_date), []):
            prefix = f"label_{label.horizon}"
            row[f"{prefix}_entry_date"] = label.entry_date.isoformat()
            row[f"{prefix}_exit_date"] = label.exit_date.isoformat()
            row[f"{prefix}_forward_return"] = label.forward_return
            row[f"{prefix}_spy_forward_return"] = label.spy_forward_return
            row[f"{prefix}_excess_return"] = label.excess_return
            row[f"{prefix}_available_at"] = label.label_available_at.isoformat(timespec="seconds")
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["ticker", "trading_date"]).reset_index(drop=True) if rows else pd.DataFrame()


def feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    return role_sets_from_frame(frame).model_features


def dataset_hash(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return hashlib.sha256(b"empty").hexdigest()
    stable = frame.drop(columns=["snapshot_id", "dataset_id"], errors="ignore").copy()
    stable = stable.sort_index(axis=1).sort_values(
        [column for column in ["ticker", "trading_date"] if column in frame.columns]
    )
    payload = stable.to_csv(index=False, na_rep="").encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def label_definitions(horizons: tuple[int, ...] = DEFAULT_HORIZONS) -> dict[str, Any]:
    return {
        "horizons": [f"{int(horizon)}_session" for horizon in horizons],
        "timing": (
            "For snapshot date T, features are available after T close. Entry uses the next cached "
            "trading session close. Exit uses the close N sessions after entry. The signal-date-to-entry "
            "return is not included in the label."
        ),
        "benchmark": "SPY forward return over the same entry and exit dates when available.",
        "sec_feature_policy": SEC_FEATURE_POLICY,
        "earnings_feature_policy": {
            "version": "earnings_feature_policy_v1",
            "availability": "Events are included only when available_at is no later than the post-close snapshot timestamp.",
            "timing": (
                "Before-market events may be present on that session. After-market events are available after close. "
                "Unknown timing uses conservative after-close availability from the provider layer."
            ),
            "scoring": "Earnings features are dataset-only and do not affect scanner scoring.",
        },
    }


def build_point_in_time_dataset(
    db_path: str | Path,
    tickers: list[str],
    start_date: date,
    end_date: date,
    output_dir: str | Path = "data/processed",
    horizons: tuple[int, ...] = DEFAULT_HORIZONS,
    feature_version: str = FEATURE_VERSION,
    version: str | None = None,
) -> DatasetBuildResult:
    storage.init_db(db_path)
    clean_tickers = sorted({ticker.upper().strip() for ticker in tickers if ticker and ticker.strip()})
    warnings: list[str] = []
    if not clean_tickers:
        warnings.append("No tickers selected.")

    required_history_tickers = sorted(set(clean_tickers) | {"SPY", "QQQ", "IWM", "^VIX"})
    histories = {ticker: _sorted_ohlcv(storage.load_ohlcv(db_path, ticker)) for ticker in required_history_tickers}
    all_snapshot_dates: list[date] = []
    ticker_dates_map: dict[str, list[date]] = {}
    for ticker in clean_tickers:
        dates = _trading_dates(histories.get(ticker, pd.DataFrame()), start_date, end_date)
        ticker_dates_map[ticker] = dates
        all_snapshot_dates.extend(dates)
    regime_feature_map = precompute_market_regimes_for_dates(histories, all_snapshot_dates)
    snapshots: list[FeatureSnapshot] = []
    labels_by_key: dict[tuple[str, date], list[OutcomeLabel]] = {}

    for ticker in clean_tickers:
        ticker_df = histories.get(ticker, pd.DataFrame())
        if ticker_df.empty:
            warnings.append(f"{ticker}: no cached OHLCV data; skipped.")
            continue
        ticker_dates = ticker_dates_map.get(ticker, [])
        if not ticker_dates:
            warnings.append(f"{ticker}: no cached rows inside requested date range.")
            continue
        sec_feature_map = precompute_sec_features_for_dates(db_path, ticker, ticker_dates)
        earnings_feature_map = precompute_earnings_features_for_dates(db_path, ticker, ticker_dates)
        catalyst_override_map = precompute_catalyst_overrides_for_dates(db_path, ticker, ticker_dates)
        for trading_date in ticker_dates:
            snapshot = build_feature_snapshot(
                db_path,
                ticker,
                trading_date,
                histories,
                feature_version,
                sec_features_override=sec_feature_map.get(trading_date),
                earnings_features_override=earnings_feature_map.get(trading_date),
                regime_override=regime_feature_map.get(trading_date),
                catalyst_features_override=catalyst_override_map.get(trading_date),
            )
            if snapshot is None:
                warnings.append(f"{ticker} {trading_date}: snapshot unavailable.")
                continue
            labels = calculate_outcome_labels(snapshot, ticker_df, histories.get("SPY", pd.DataFrame()), horizons)
            if not labels:
                warnings.append(f"{ticker} {trading_date}: no forward labels available.")
            labels_by_key[(ticker, trading_date)] = labels
            snapshots.append(snapshot)

    export_frame = _flatten_snapshot_rows(snapshots, labels_by_key)
    role_sets = role_sets_from_frame(export_frame)
    feature_columns = role_sets.model_features
    hash_value = dataset_hash(export_frame)
    build_timestamp = datetime.now(UTC)
    build = DatasetBuild(
        version=version or feature_version,
        build_timestamp=build_timestamp,
        requested_start_date=start_date,
        requested_end_date=end_date,
        ticker_universe=clean_tickers,
        feature_columns=feature_columns,
        label_definitions=label_definitions(horizons),
        row_count=int(len(export_frame)),
        data_hash=hash_value,
        audit_columns=role_sets.audit_columns,
        label_columns=role_sets.label_columns,
        identifier_columns=role_sets.identifier_columns,
        metadata_columns=role_sets.metadata_columns,
        feature_manifest=role_sets.manifest,
        warnings=warnings,
    )
    dataset_id = insert_dataset_build(db_path, build)

    persisted_labels: list[OutcomeLabel] = []
    snapshot_ids = insert_feature_snapshots(db_path, dataset_id, snapshots)
    for snapshot in snapshots:
        snapshot_id = snapshot_ids.get((snapshot.ticker.upper(), snapshot.trading_date))
        if snapshot_id is None:
            continue
        snapshot.snapshot_id = snapshot_id
        snapshot.dataset_id = dataset_id
        for label in labels_by_key.get((snapshot.ticker, snapshot.trading_date), []):
            label.snapshot_id = snapshot_id
            persisted_labels.append(label)
    insert_outcome_labels(db_path, persisted_labels)

    export_path = None
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    if not export_frame.empty:
        export_frame = export_frame.copy()
        export_frame.insert(0, "dataset_id", dataset_id)
        export_path = str(output / f"dataset_{dataset_id}_{build.version}_{hash_value[:8]}.csv")
        export_frame.to_csv(export_path, index=False)
        update_dataset_export_path(db_path, dataset_id, export_path)
        build.export_path = export_path
    build.dataset_id = dataset_id
    return DatasetBuildResult(
        dataset_id=dataset_id,
        dataset_frame=export_frame,
        build=build,
        warnings=warnings,
        export_path=export_path,
    )
