from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from src.data import storage
from src.utils.trading_calendar import (
    DAILY_BAR_READY_TIME_ET,
    NEW_YORK_TZ,
    is_trading_day,
    latest_expected_trading_day,
)


DEFAULT_OPTIONS_UNIVERSE = ("AAPL", "AMD", "AMZN", "COIN", "META", "MSFT", "NVDA", "TSLA")
DEFAULT_PROVIDER = "yfinance"


class OptionsSnapshotError(RuntimeError):
    pass


class OptionsProvider(Protocol):
    name: str

    def collect_ticker(
        self, ticker: str, max_expirations: int, snapshot_date: date
    ) -> tuple[float | None, list[dict[str, Any]], list[str]]:
        """Return underlying price, normalized option rows, and warnings."""


@dataclass(frozen=True)
class YFinanceOptionsProvider:
    name: str = DEFAULT_PROVIDER

    def collect_ticker(
        self, ticker: str, max_expirations: int, snapshot_date: date
    ) -> tuple[float | None, list[dict[str, Any]], list[str]]:
        try:
            import yfinance as yf
        except ImportError as exc:
            raise OptionsSnapshotError("yfinance is not installed.") from exc

        instrument = yf.Ticker(ticker)
        expirations = sorted(str(value) for value in (instrument.options or ()))[:max_expirations]
        if not expirations:
            raise OptionsSnapshotError("provider returned no valid expirations")
        underlying_price = _latest_underlying_price(instrument, snapshot_date)
        rows: list[dict[str, Any]] = []
        warnings: list[str] = []
        if underlying_price is None:
            warnings.append("underlying_price_unavailable")
        for expiration in expirations:
            chain = instrument.option_chain(expiration)
            rows.extend(_normalize_chain_frame(chain.calls, ticker, expiration, "call"))
            rows.extend(_normalize_chain_frame(chain.puts, ticker, expiration, "put"))
        if not rows:
            raise OptionsSnapshotError("provider returned no option contracts")
        return underlying_price, rows, warnings


def _latest_underlying_price(instrument: Any, snapshot_date: date) -> float | None:
    history = instrument.history(period="5d", interval="1d", auto_adjust=False)
    if history is None or history.empty:
        return None
    close_column = next((column for column in history.columns if str(column).lower() == "close"), None)
    if close_column is None:
        return None
    history_dates = pd.to_datetime(history.index, utc=True, errors="coerce").date
    bounded = history.loc[history_dates <= snapshot_date]
    values = pd.to_numeric(bounded[close_column], errors="coerce").dropna()
    return None if values.empty else float(values.iloc[-1])


def _nullable_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _normalize_timestamp(value: Any) -> str | None:
    if value is None or pd.isna(value):
        return None
    timestamp = pd.to_datetime(value, utc=True, errors="coerce")
    return None if pd.isna(timestamp) else timestamp.isoformat()


def _normalize_chain_frame(frame: pd.DataFrame, ticker: str, expiration: str, option_type: str) -> list[dict[str, Any]]:
    if frame is None or frame.empty:
        return []
    rows: list[dict[str, Any]] = []
    for raw in frame.to_dict(orient="records"):
        contract_symbol = str(raw.get("contractSymbol") or "").strip()
        if not contract_symbol:
            continue
        flags: list[str] = []
        for source, flag in (
            ("volume", "missing_volume"),
            ("openInterest", "missing_open_interest"),
            ("impliedVolatility", "missing_implied_volatility"),
        ):
            if _nullable_number(raw.get(source)) is None:
                flags.append(flag)
        bid = _nullable_number(raw.get("bid"))
        ask = _nullable_number(raw.get("ask"))
        if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
            flags.append("unusable_quote")
        rows.append(
            {
                "ticker": ticker,
                "expiration": expiration,
                "option_type": option_type,
                "strike": _nullable_number(raw.get("strike")),
                "bid": bid,
                "ask": ask,
                "last_price": _nullable_number(raw.get("lastPrice")),
                "volume": _nullable_number(raw.get("volume")),
                "open_interest": _nullable_number(raw.get("openInterest")),
                "implied_volatility": _nullable_number(raw.get("impliedVolatility")),
                "in_the_money": None if raw.get("inTheMoney") is None else int(bool(raw.get("inTheMoney"))),
                "contract_symbol": contract_symbol,
                "provider_timestamp": _normalize_timestamp(raw.get("lastTradeDate")),
                "data_quality_flags": flags,
            }
        )
    return rows


def _universe(tickers: list[str] | tuple[str, ...] | None) -> list[str]:
    values = DEFAULT_OPTIONS_UNIVERSE if tickers is None else tickers
    normalized = sorted({str(value).strip().upper() for value in values if str(value).strip()})
    if not normalized:
        raise OptionsSnapshotError("Options snapshot universe is empty.")
    return normalized


def _universe_hash(tickers: list[str]) -> str:
    return hashlib.sha256(json.dumps(tickers, separators=(",", ":")).encode("utf-8")).hexdigest()


def _backup_database(db_path: str | Path) -> Path:
    source = Path(db_path)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S_%f")
    backup = source.with_name(f"{source.stem}_backup_phase3b1_options_{stamp}{source.suffix}")
    with sqlite3.connect(source) as src, sqlite3.connect(backup) as dst:
        src.backup(dst)
    return backup


def _collection_window_ready(reference_time: datetime | date | None) -> bool:
    if reference_time is None:
        reference_et = datetime.now(UTC).astimezone(NEW_YORK_TZ)
    elif isinstance(reference_time, datetime):
        normalized = reference_time if reference_time.tzinfo is not None else reference_time.replace(tzinfo=UTC)
        reference_et = normalized.astimezone(NEW_YORK_TZ)
    else:
        return True
    return not (
        is_trading_day(reference_et.date())
        and reference_et.time() < DAILY_BAR_READY_TIME_ET
    )


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone() is not None


def collect_options_snapshots(
    db_path: str | Path,
    *,
    apply: bool = False,
    tickers: list[str] | tuple[str, ...] | None = None,
    max_expirations: int = 2,
    provider: OptionsProvider | None = None,
    reference_time: datetime | date | None = None,
    create_backup: bool = True,
) -> dict[str, Any]:
    if max_expirations < 1:
        raise OptionsSnapshotError("max_expirations must be at least 1")
    universe = _universe(tickers)
    snapshot_date = latest_expected_trading_day(reference_time)
    plan: dict[str, Any] = {
        "status": "planned" if not apply else "pending",
        "mode": "apply" if apply else "dry_run",
        "snapshot_date": snapshot_date.isoformat(),
        "provider": DEFAULT_PROVIDER if provider is None else provider.name,
        "universe": universe,
        "universe_hash": _universe_hash(universe),
        "ticker_count": len(universe),
        "max_expirations": max_expirations,
        "expiration_policy": "nearest valid expirations in ascending date order",
        "network_calls_made": False,
        "database_mutated": False,
        "database_backup": None,
        "expected_tables": ["options_snapshot_runs", "options_snapshots"],
        "successful_tickers": [],
        "failed_tickers": [],
        "contract_count": 0,
        "run_id": None,
    }
    if not apply:
        return plan

    if not _collection_window_ready(reference_time):
        raise OptionsSnapshotError(
            "The current U.S. session is not yet complete; refusing to persist an intraday chain as a daily snapshot."
        )

    db_path = Path(db_path)
    if not db_path.exists():
        raise OptionsSnapshotError(f"Database does not exist: {db_path}")
    backup = _backup_database(db_path) if create_backup else None
    plan["database_backup"] = None if backup is None else str(backup)
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        duplicate = conn.execute(
            "SELECT run_id FROM options_snapshot_runs WHERE snapshot_date=? AND provider=?",
            (snapshot_date.isoformat(), plan["provider"]),
        ).fetchone()
    if duplicate is not None:
        raise OptionsSnapshotError(
            f"Options snapshot run already exists for {snapshot_date.isoformat()} and {plan['provider']}: "
            f"run_id={duplicate['run_id']}"
        )

    active_provider = provider or YFinanceOptionsProvider()
    collected: list[dict[str, Any]] = []
    warnings: list[str] = []
    plan["network_calls_made"] = provider is None
    for ticker in universe:
        try:
            underlying_price, rows, ticker_warnings = active_provider.collect_ticker(
                ticker, max_expirations, snapshot_date
            )
            for row in rows:
                row["underlying_price"] = underlying_price
            collected.extend(rows)
            plan["successful_tickers"].append(
                {
                    "ticker": ticker,
                    "contract_count": len(rows),
                    "expiration_count": len({row["expiration"] for row in rows}),
                    "warnings": ticker_warnings,
                }
            )
            warnings.extend(f"{ticker}:{warning}" for warning in ticker_warnings)
        except Exception as exc:
            plan["failed_tickers"].append({"ticker": ticker, "error": str(exc)})
            warnings.append(f"{ticker}:provider_failure:{exc}")
    if not collected:
        raise OptionsSnapshotError("No options contracts were collected; no run was persisted.")

    now = datetime.now(UTC).isoformat(timespec="seconds")
    status = "completed" if not plan["failed_tickers"] else "partial"
    with storage.connect(db_path) as conn:
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            cursor = conn.execute(
                """
                INSERT INTO options_snapshot_runs (
                    snapshot_date, as_of_timestamp, provider, universe_hash, status,
                    ticker_count, warnings, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    snapshot_date.isoformat(),
                    now,
                    plan["provider"],
                    plan["universe_hash"],
                    status,
                    len(plan["successful_tickers"]),
                    json.dumps(warnings, sort_keys=True),
                    now,
                ),
            )
            run_id = int(cursor.lastrowid)
            conn.executemany(
                """
                INSERT INTO options_snapshots (
                    run_id, ticker, expiration, underlying_price, option_type, strike,
                    bid, ask, last_price, volume, open_interest, implied_volatility,
                    in_the_money, contract_symbol, provider_timestamp, created_at,
                    data_quality_flags
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        run_id,
                        row["ticker"],
                        row["expiration"],
                        row["underlying_price"],
                        row["option_type"],
                        row["strike"],
                        row["bid"],
                        row["ask"],
                        row["last_price"],
                        row["volume"],
                        row["open_interest"],
                        row["implied_volatility"],
                        row["in_the_money"],
                        row["contract_symbol"],
                        row["provider_timestamp"],
                        now,
                        json.dumps(row["data_quality_flags"], sort_keys=True),
                    )
                    for row in collected
                ],
            )
        except Exception:
            conn.rollback()
            raise
    plan.update(
        {
            "status": status,
            "database_mutated": True,
            "contract_count": len(collected),
            "run_id": run_id,
            "warnings": warnings,
        }
    )
    return plan


def _safe_ratio(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def _atm_iv(frame: pd.DataFrame, underlying: float | None, option_type: str) -> float | None:
    if underlying is None:
        return None
    eligible = frame[frame["option_type"].eq(option_type)].dropna(subset=["strike", "implied_volatility"])
    if eligible.empty:
        return None
    ordered = eligible.assign(distance=(eligible["strike"] - underlying).abs()).sort_values(
        ["distance", "strike", "contract_symbol"]
    )
    return float(ordered.iloc[0]["implied_volatility"])


def _concentration(frame: pd.DataFrame, option_type: str, underlying: float | None) -> tuple[float | None, float | None]:
    eligible = frame[frame["option_type"].eq(option_type)].dropna(subset=["strike", "open_interest"])
    if eligible.empty:
        return None, None
    row = eligible.sort_values(["open_interest", "strike", "contract_symbol"], ascending=[False, True, True]).iloc[0]
    strike = float(row["strike"])
    distance = None if underlying is None or underlying == 0 else (strike - underlying) / underlying
    return strike, distance


def summarize_options_frame(frame: pd.DataFrame) -> dict[str, Any]:
    if frame.empty:
        return {}
    frame = frame.copy()
    numeric = ["underlying_price", "strike", "bid", "ask", "volume", "open_interest", "implied_volatility"]
    for column in numeric:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    underlying_values = frame["underlying_price"].dropna()
    underlying = None if underlying_values.empty else float(underlying_values.iloc[0])
    calls = frame[frame["option_type"].eq("call")]
    puts = frame[frame["option_type"].eq("put")]
    def total(frame: pd.DataFrame, column: str) -> float:
        value = frame[column].sum(min_count=1)
        return 0.0 if pd.isna(value) else float(value)

    call_oi = total(calls, "open_interest")
    put_oi = total(puts, "open_interest")
    call_volume = total(calls, "volume")
    put_volume = total(puts, "volume")
    expirations = sorted(frame["expiration"].dropna().astype(str).unique().tolist())
    front = frame[frame["expiration"].eq(expirations[0])] if expirations else frame.iloc[0:0]
    next_expiry = frame[frame["expiration"].eq(expirations[1])] if len(expirations) > 1 else frame.iloc[0:0]
    front_call_iv = _atm_iv(front, underlying, "call")
    front_put_iv = _atm_iv(front, underlying, "put")
    next_call_iv = _atm_iv(next_expiry, underlying, "call")
    next_put_iv = _atm_iv(next_expiry, underlying, "put")
    front_ivs = [value for value in (front_call_iv, front_put_iv) if value is not None]
    next_ivs = [value for value in (next_call_iv, next_put_iv) if value is not None]
    front_mean = None if not front_ivs else sum(front_ivs) / len(front_ivs)
    next_mean = None if not next_ivs else sum(next_ivs) / len(next_ivs)
    call_strike, call_distance = _concentration(frame, "call", underlying)
    put_strike, put_distance = _concentration(frame, "put", underlying)
    quote_frame = frame[(frame["bid"] > 0) & (frame["ask"] > 0) & (frame["ask"] >= frame["bid"])].copy()
    if quote_frame.empty:
        median_spread = None
    else:
        midpoint = (quote_frame["ask"] + quote_frame["bid"]) / 2
        median_spread = float(((quote_frame["ask"] - quote_frame["bid"]) / midpoint).median())
    count = len(frame)
    return {
        "ticker": str(frame.iloc[0]["ticker"]),
        "snapshot_date": str(frame.iloc[0]["snapshot_date"]),
        "underlying_price": underlying,
        "expiration_count": len(expirations),
        "total_call_open_interest": call_oi,
        "total_put_open_interest": put_oi,
        "put_call_open_interest_ratio": _safe_ratio(put_oi, call_oi),
        "total_call_volume": call_volume,
        "total_put_volume": put_volume,
        "put_call_volume_ratio": _safe_ratio(put_volume, call_volume),
        "nearest_expiry_atm_call_iv": front_call_iv,
        "nearest_expiry_atm_put_iv": front_put_iv,
        "atm_put_minus_call_iv": None if front_call_iv is None or front_put_iv is None else front_put_iv - front_call_iv,
        "front_minus_next_expiry_atm_iv": None if front_mean is None or next_mean is None else front_mean - next_mean,
        "highest_call_oi_strike": call_strike,
        "highest_put_oi_strike": put_strike,
        "highest_call_oi_strike_distance": call_distance,
        "highest_put_oi_strike_distance": put_distance,
        "valid_contract_count": count,
        "missing_volume_pct": float(frame["volume"].isna().mean()),
        "missing_open_interest_pct": float(frame["open_interest"].isna().mean()),
        "missing_implied_volatility_pct": float(frame["implied_volatility"].isna().mean()),
        "median_relative_bid_ask_spread": median_spread,
    }


def latest_options_summaries(db_path: str | Path, run_id: int | None = None) -> pd.DataFrame:
    with storage.connect(db_path) as conn:
        if not _table_exists(conn, "options_snapshot_runs"):
            return pd.DataFrame()
        selected = run_id
        if selected is None:
            row = conn.execute("SELECT run_id FROM options_snapshot_runs ORDER BY snapshot_date DESC, run_id DESC LIMIT 1").fetchone()
            if row is None:
                return pd.DataFrame()
            selected = int(row["run_id"])
        frame = pd.read_sql_query(
            """
            SELECT s.*, r.snapshot_date
            FROM options_snapshots s
            JOIN options_snapshot_runs r ON r.run_id=s.run_id
            WHERE s.run_id=?
            ORDER BY s.ticker, s.expiration, s.option_type, s.strike, s.contract_symbol
            """,
            conn,
            params=(selected,),
        )
    if frame.empty:
        return pd.DataFrame()
    return pd.DataFrame([summarize_options_frame(group) for _, group in frame.groupby("ticker", sort=True)])


def options_status_report(db_path: str | Path) -> dict[str, Any]:
    with storage.connect(db_path) as conn:
        if not _table_exists(conn, "options_snapshot_runs"):
            return {
                "status": "passed",
                "run_count": 0,
                "snapshot_date_count": 0,
                "sample_status": "collection_only",
                "violations": [],
                "latest_summaries": [],
            }
        runs = [dict(row) for row in conn.execute("SELECT * FROM options_snapshot_runs ORDER BY snapshot_date, run_id")]
        duplicate_runs = [
            dict(row)
            for row in conn.execute(
                "SELECT snapshot_date, provider, COUNT(*) AS count FROM options_snapshot_runs "
                "GROUP BY snapshot_date, provider HAVING COUNT(*)>1"
            )
        ]
        duplicate_contracts = int(
            conn.execute(
                """
                SELECT COUNT(*) FROM (
                    SELECT run_id, ticker, expiration, option_type, contract_symbol, COUNT(*)
                    FROM options_snapshots
                    GROUP BY run_id, ticker, expiration, option_type, contract_symbol
                    HAVING COUNT(*)>1
                )
                """
            ).fetchone()[0]
        )
        total_contracts = int(conn.execute("SELECT COUNT(*) FROM options_snapshots").fetchone()[0])
        ticker_coverage = [
            dict(row)
            for row in conn.execute(
                "SELECT ticker, COUNT(DISTINCT run_id) AS run_count, COUNT(*) AS contract_count "
                "FROM options_snapshots GROUP BY ticker ORDER BY ticker"
            )
        ]
        missing = dict(
            conn.execute(
                """
                SELECT
                    SUM(CASE WHEN volume IS NULL THEN 1 ELSE 0 END) AS missing_volume,
                    SUM(CASE WHEN open_interest IS NULL THEN 1 ELSE 0 END) AS missing_open_interest,
                    SUM(CASE WHEN implied_volatility IS NULL THEN 1 ELSE 0 END) AS missing_iv
                FROM options_snapshots
                """
            ).fetchone()
        )
    dates = sorted({str(row["snapshot_date"]) for row in runs})
    date_count = len(dates)
    sample_status = "collection_only" if date_count < 20 else "preliminary_research" if date_count < 60 else "eligible_for_feature_evaluation"
    violations = []
    if duplicate_runs:
        violations.append("duplicate snapshot-date/provider runs detected")
    if duplicate_contracts:
        violations.append("duplicate option contracts detected")
    summaries = latest_options_summaries(db_path).to_dict(orient="records") if runs else []
    return {
        "status": "passed" if not violations else "failed",
        "run_count": len(runs),
        "snapshot_date_count": date_count,
        "snapshot_date_range": None if not dates else [dates[0], dates[-1]],
        "sample_status": sample_status,
        "total_contract_count": total_contracts,
        "ticker_coverage": ticker_coverage,
        "missingness": {
            "volume_pct": 0.0 if total_contracts == 0 else int(missing.get("missing_volume") or 0) / total_contracts,
            "open_interest_pct": 0.0 if total_contracts == 0 else int(missing.get("missing_open_interest") or 0) / total_contracts,
            "implied_volatility_pct": 0.0 if total_contracts == 0 else int(missing.get("missing_iv") or 0) / total_contracts,
        },
        "latest_run": None if not runs else runs[-1],
        "latest_summaries": summaries,
        "duplicate_run_count": len(duplicate_runs),
        "duplicate_contract_count": duplicate_contracts,
        "violations": violations,
    }
