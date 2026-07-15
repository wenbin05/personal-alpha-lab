from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

import pandas as pd
import pytest

from src.data import storage
from src.options_research.snapshots import (
    OptionsSnapshotError,
    _latest_underlying_price,
    collect_options_snapshots,
    latest_options_summaries,
    options_status_report,
    summarize_options_frame,
)


REFERENCE_TIME = datetime(2026, 7, 14, 23, 0, tzinfo=UTC)


def _contract(ticker: str, expiration: str, option_type: str, strike: float, suffix: str) -> dict:
    return {
        "ticker": ticker,
        "expiration": expiration,
        "option_type": option_type,
        "strike": strike,
        "bid": 1.0,
        "ask": 1.2,
        "last_price": 1.1,
        "volume": 10.0,
        "open_interest": 100.0,
        "implied_volatility": 0.3,
        "in_the_money": 0,
        "contract_symbol": f"{ticker}{suffix}",
        "provider_timestamp": None,
        "data_quality_flags": [],
    }


class FakeProvider:
    name = "yfinance"

    def __init__(self, failing: set[str] | None = None) -> None:
        self.failing = failing or set()
        self.calls: list[tuple[str, int]] = []

    def collect_ticker(self, ticker: str, max_expirations: int, snapshot_date):
        self.calls.append((ticker, max_expirations))
        if ticker in self.failing:
            raise OptionsSnapshotError("synthetic ticker failure")
        return (
            100.0,
            [
                _contract(ticker, "2026-07-17", "call", 100.0, "C100"),
                _contract(ticker, "2026-07-17", "put", 100.0, "P100"),
                _contract(ticker, "2026-07-24", "call", 105.0, "C105"),
                _contract(ticker, "2026-07-24", "put", 95.0, "P095"),
            ],
            [],
        )


def test_dry_run_has_no_network_or_database_mutation(tmp_path) -> None:
    db_path = tmp_path / "missing.db"
    provider = FakeProvider()

    report = collect_options_snapshots(
        db_path,
        apply=False,
        tickers=["AAPL"],
        provider=provider,
        reference_time=REFERENCE_TIME,
    )

    assert report["mode"] == "dry_run"
    assert report["network_calls_made"] is False
    assert report["database_mutated"] is False
    assert provider.calls == []
    assert not db_path.exists()


def test_apply_before_completed_session_is_blocked_without_provider_call(tmp_path) -> None:
    db_path = tmp_path / "alpha.db"
    storage.init_db(db_path)
    provider = FakeProvider()

    with pytest.raises(OptionsSnapshotError, match="not yet complete"):
        collect_options_snapshots(
            db_path,
            apply=True,
            tickers=["AAPL"],
            provider=provider,
            reference_time=datetime(2026, 7, 14, 15, 0, tzinfo=UTC),
            create_backup=False,
        )

    assert provider.calls == []
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM options_snapshot_runs").fetchone()[0] == 0


def test_apply_is_immutable_duplicate_safe_and_partial_failures_survive(tmp_path) -> None:
    db_path = tmp_path / "alpha.db"
    storage.init_db(db_path)
    provider = FakeProvider(failing={"AMD"})

    report = collect_options_snapshots(
        db_path,
        apply=True,
        tickers=["AAPL", "AMD"],
        provider=provider,
        reference_time=REFERENCE_TIME,
        create_backup=False,
    )

    assert report["status"] == "partial"
    assert report["contract_count"] == 4
    assert [row["ticker"] for row in report["failed_tickers"]] == ["AMD"]
    with storage.connect(db_path) as conn:
        assert conn.execute("SELECT COUNT(*) FROM options_snapshot_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM options_snapshots").fetchone()[0] == 4
        with pytest.raises(sqlite3.IntegrityError, match="immutable"):
            conn.execute("UPDATE options_snapshots SET bid=99")
    with pytest.raises(OptionsSnapshotError, match="already exists"):
        collect_options_snapshots(
            db_path,
            apply=True,
            tickers=["AAPL"],
            provider=FakeProvider(),
            reference_time=REFERENCE_TIME,
            create_backup=False,
        )


def test_summary_calculations_expiration_order_and_missing_quotes() -> None:
    frame = pd.DataFrame(
        [
            {
                **_contract("AAPL", "2026-07-24", "call", 105.0, "C105"),
                "snapshot_date": "2026-07-14",
                "underlying_price": 100.0,
                "open_interest": 500.0,
                "implied_volatility": 0.25,
            },
            {
                **_contract("AAPL", "2026-07-17", "call", 100.0, "C100"),
                "snapshot_date": "2026-07-14",
                "underlying_price": 100.0,
                "open_interest": 200.0,
                "volume": 20.0,
                "implied_volatility": 0.30,
                "bid": 0.0,
                "ask": 0.0,
            },
            {
                **_contract("AAPL", "2026-07-17", "put", 100.0, "P100"),
                "snapshot_date": "2026-07-14",
                "underlying_price": 100.0,
                "open_interest": 300.0,
                "volume": 10.0,
                "implied_volatility": 0.35,
            },
            {
                **_contract("AAPL", "2026-07-24", "put", 95.0, "P095"),
                "snapshot_date": "2026-07-14",
                "underlying_price": 100.0,
                "open_interest": None,
                "volume": None,
                "implied_volatility": None,
            },
        ]
    )

    result = summarize_options_frame(frame)

    assert result["total_call_open_interest"] == 700.0
    assert result["total_put_open_interest"] == 300.0
    assert result["put_call_volume_ratio"] == pytest.approx(1 / 3)
    assert result["nearest_expiry_atm_call_iv"] == 0.30
    assert result["nearest_expiry_atm_put_iv"] == 0.35
    assert result["atm_put_minus_call_iv"] == pytest.approx(0.05)
    assert result["highest_call_oi_strike"] == 105.0
    assert result["missing_open_interest_pct"] == 0.25
    assert result["missing_implied_volatility_pct"] == 0.25
    assert result["median_relative_bid_ask_spread"] == pytest.approx(0.2 / 1.1)


def test_underlying_price_excludes_later_incomplete_bar() -> None:
    class Instrument:
        @staticmethod
        def history(**_kwargs):
            return pd.DataFrame(
                {"Close": [100.0, 999.0]},
                index=pd.to_datetime(["2026-07-14", "2026-07-15"], utc=True),
            )

    assert _latest_underlying_price(Instrument(), pd.Timestamp("2026-07-14").date()) == 100.0


def test_status_and_latest_summary_empty_and_populated(tmp_path) -> None:
    db_path = tmp_path / "alpha.db"
    storage.init_db(db_path)
    assert options_status_report(db_path)["sample_status"] == "collection_only"

    collect_options_snapshots(
        db_path,
        apply=True,
        tickers=["AAPL"],
        provider=FakeProvider(),
        reference_time=REFERENCE_TIME,
        create_backup=False,
    )

    status = options_status_report(db_path)
    summaries = latest_options_summaries(db_path)
    assert status["status"] == "passed"
    assert status["snapshot_date_count"] == 1
    assert status["total_contract_count"] == 4
    assert status["sample_status"] == "collection_only"
    assert summaries.iloc[0]["ticker"] == "AAPL"


def test_options_collection_does_not_touch_scanner_shadow_or_datasets(tmp_path) -> None:
    db_path = tmp_path / "alpha.db"
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        before = {
            "scanner": conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0],
            "datasets": conn.execute("SELECT COUNT(*) FROM dataset_builds").fetchone()[0],
            "shadow_runs": conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='shadow_prediction_runs'").fetchone()[0],
        }
    collect_options_snapshots(
        db_path,
        apply=True,
        tickers=["AAPL"],
        provider=FakeProvider(),
        reference_time=REFERENCE_TIME,
        create_backup=False,
    )
    with storage.connect(db_path) as conn:
        after = {
            "scanner": conn.execute("SELECT COUNT(*) FROM scan_results").fetchone()[0],
            "datasets": conn.execute("SELECT COUNT(*) FROM dataset_builds").fetchone()[0],
            "shadow_runs": conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table' AND name='shadow_prediction_runs'").fetchone()[0],
        }
    assert after == before
