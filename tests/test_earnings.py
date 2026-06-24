from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from src.data import storage
from src.datasets.builder import build_point_in_time_dataset, build_feature_snapshot
from src.earnings.backfill import backfill_earnings_events
from src.earnings.csv_import import parse_earnings_import_frame
from src.earnings.features import precompute_earnings_features_for_dates
from src.earnings.models import EarningsEvent
from src.earnings.repository import (
    classify_earnings_coverage,
    earnings_coverage_summary,
    earnings_coverage_report,
    expected_quarterly_events,
    insert_earnings_event,
    list_earnings_by_ticker,
)
from src.earnings.yfinance_provider import EarningsProviderResult, YFinanceHistoricalEarningsProvider
from src.scoring.score_engine import score_ticker_from_features


def _event(
    ticker: str = "AAPL",
    available_at: datetime = datetime(2024, 1, 12, 21, 5, tzinfo=UTC),
    provider_event_id: str = "test:AAPL:2024Q1",
    eps_actual: float | None = 2.0,
    eps_estimate: float | None = 1.8,
    eps_surprise_percent: float | None = 11.1,
) -> EarningsEvent:
    return EarningsEvent(
        ticker=ticker,
        fiscal_period_end=date(2023, 12, 31),
        announced_at=available_at,
        available_at=available_at,
        timing="after_market",
        eps_estimate=eps_estimate,
        eps_actual=eps_actual,
        eps_surprise=None if eps_actual is None or eps_estimate is None else eps_actual - eps_estimate,
        eps_surprise_percent=eps_surprise_percent,
        provider="unit_test",
        provider_event_id=provider_event_id,
        fetched_at=datetime(2024, 1, 13, tzinfo=UTC),
        raw_payload_json=f'{{"provider_event_id": "{provider_event_id}", "epsActual": {eps_actual}}}',
        data_quality_status="partial",
        warnings=["Revenue fields unavailable from provider."],
    )


def _price_frame(start: str = "2024-01-02", periods: int = 45, base: float = 100.0) -> pd.DataFrame:
    dates = pd.bdate_range(start, periods=periods)
    close = [base + idx for idx in range(periods)]
    return pd.DataFrame(
        {
            "date": dates,
            "open": [value - 0.25 for value in close],
            "high": [value + 0.5 for value in close],
            "low": [value - 0.5 for value in close],
            "close": close,
            "adj_close": close,
            "volume": [1_000_000] * periods,
        }
    )


def _seed_prices(db_path) -> None:
    storage.upsert_ohlcv(db_path, "AAPL", _price_frame())
    storage.upsert_ohlcv(db_path, "SPY", _price_frame(base=400))
    storage.upsert_ohlcv(db_path, "QQQ", _price_frame(base=300))
    storage.upsert_ohlcv(db_path, "IWM", _price_frame(base=200))
    storage.upsert_ohlcv(db_path, "^VIX", _price_frame(base=15))


def test_earnings_table_insert_list_and_coverage(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    event_id, status = insert_earnings_event(db_path, _event())

    frame = list_earnings_by_ticker(db_path, "AAPL")
    coverage = earnings_coverage_summary(db_path, ["AAPL"])

    assert event_id > 0
    assert status == "inserted"
    assert len(frame) == 1
    assert int(coverage.iloc[0]["events"]) == 1
    assert int(coverage.iloc[0]["missing_revenue_actual"]) == 1


def test_earnings_dedupes_and_records_revision(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    first_id, first_status = insert_earnings_event(db_path, _event(eps_actual=2.0))
    second_id, second_status = insert_earnings_event(db_path, _event(eps_actual=2.1))
    third_id, third_status = insert_earnings_event(db_path, _event(eps_actual=2.1))

    with storage.connect(db_path) as conn:
        revisions = conn.execute("SELECT action FROM earnings_event_revisions ORDER BY revision_id").fetchall()
        active = conn.execute(
            "SELECT eps_actual FROM earnings_events WHERE earnings_event_id = ?",
            (first_id,),
        ).fetchone()

    assert first_id == second_id == third_id
    assert first_status == "inserted"
    assert second_status == "updated"
    assert third_status == "duplicate"
    assert active["eps_actual"] == 2.0
    assert [row["action"] for row in revisions] == ["create", "update"]


def test_yfinance_provider_empty_and_error_handling(monkeypatch, tmp_path) -> None:
    class EmptyTicker:
        def get_earnings_dates(self, limit=12):
            raise ImportError("missing lxml")

        def get_earnings_history(self):
            return pd.DataFrame()

    provider = YFinanceHistoricalEarningsProvider(db_path=tmp_path / "alpha_lab.db", min_interval_seconds=0)
    calls = {"count": 0}

    def get_ticker(_ticker):
        calls["count"] += 1
        return EmptyTicker()

    monkeypatch.setattr(provider, "_get_ticker", get_ticker)

    result = provider.fetch_earnings_events("AAPL", date(2024, 1, 1), date(2024, 12, 31), use_cache=False)
    cached = provider.fetch_earnings_events("AAPL", date(2024, 1, 1), date(2024, 12, 31), use_cache=True)

    assert result.events == []
    assert result.warnings
    assert "No historical earnings events" in result.warnings[-1]
    assert cached.events == []
    assert cached.metadata["source"] == "cache"
    assert calls["count"] == 1


def test_earnings_csv_validation_and_missing_values(tmp_path) -> None:
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAPL",
                "fiscal_period_end": "2023-12-31",
                "eps_estimate": "1.8",
                "eps_actual": "2.0",
            },
            {"ticker": "", "available_at": "2024-01-01"},
        ]
    )

    parsed = parse_earnings_import_frame(frame)

    assert len(parsed.events) == 1
    assert parsed.events[0].available_at > datetime(2023, 12, 31, tzinfo=UTC)
    assert parsed.events[0].revenue_actual is None
    assert parsed.errors


def test_earnings_point_in_time_features_do_not_leak_before_available_at(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    insert_earnings_event(db_path, _event(available_at=datetime(2024, 1, 12, 21, 5, tzinfo=UTC)))
    dates = [date(2024, 1, 11), date(2024, 1, 12), date(2024, 1, 16)]

    features = precompute_earnings_features_for_dates(db_path, "AAPL", dates)

    assert features[date(2024, 1, 11)]["earnings_data_available"] is False
    assert features[date(2024, 1, 12)]["earnings_event_present_1s"] is True
    assert features[date(2024, 1, 16)]["latest_eps_surprise_direction"] == 1
    assert features[date(2024, 1, 16)]["sessions_since_latest_earnings"] == 1


def test_dataset_build_includes_earnings_features_and_excludes_audit_from_model_features(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    insert_earnings_event(db_path, _event(available_at=datetime(2024, 1, 12, 21, 5, tzinfo=UTC)))

    result = build_point_in_time_dataset(db_path, ["AAPL"], date(2024, 1, 10), date(2024, 1, 18), output_dir=tmp_path)

    assert "earnings_event_present_20s" in result.dataset_frame.columns
    assert "earnings_event_count_20s" in result.build.audit_columns
    assert "earnings_event_count_20s" not in result.build.feature_columns
    assert "earnings_event_present_20s" in result.build.feature_columns


def test_build_feature_snapshot_accepts_earnings_override(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    _seed_prices(db_path)
    histories = {ticker: storage.load_ohlcv(db_path, ticker) for ticker in ["AAPL", "SPY", "QQQ", "IWM", "^VIX"]}

    snapshot = build_feature_snapshot(
        db_path,
        "AAPL",
        date(2024, 1, 12),
        histories,
        earnings_features_override={"earnings_data_available": True, "earnings_event_present_1s": True},
    )

    assert snapshot is not None
    assert snapshot.features["earnings_data_available"] is True
    assert snapshot.features["earnings_event_present_1s"] is True


def test_backfill_isolated_provider_failure(tmp_path) -> None:
    class FakeProvider:
        def fetch_earnings_events(self, ticker, start_date, end_date, use_cache=True):
            if ticker == "BAD":
                raise OSError("provider offline")
            return EarningsProviderResult(events=[_event(ticker=ticker, provider_event_id=f"test:{ticker}")])

    result = backfill_earnings_events(
        tmp_path / "alpha_lab.db",
        ["AAPL", "BAD"],
        date(2024, 1, 1),
        date(2024, 12, 31),
        provider=FakeProvider(),
    )

    assert result.inserted == 1
    assert result.failed_tickers == 1
    assert any("BAD" in warning for warning in result.warnings)


def test_duplicate_provider_event_timestamps_are_stable_on_cached_rerun(monkeypatch, tmp_path) -> None:
    class DuplicateTicker:
        def get_earnings_dates(self, limit=12):
            idx = pd.to_datetime(["2024-02-01 16:00:00-05:00", "2024-02-01 16:00:00-05:00"])
            return pd.DataFrame(
                {
                    "EPS Estimate": [0.6, 0.75],
                    "Reported EPS": [0.61, 0.74],
                    "Surprise(%)": [1.0, -1.33],
                },
                index=idx,
            )

        def get_earnings_history(self):
            return pd.DataFrame()

    db_path = tmp_path / "alpha_lab.db"
    provider = YFinanceHistoricalEarningsProvider(db_path=db_path, min_interval_seconds=0)
    monkeypatch.setattr(provider, "_get_ticker", lambda ticker: DuplicateTicker())

    first = backfill_earnings_events(db_path, ["DUP"], date(2024, 1, 1), date(2024, 12, 31), provider=provider)
    second = backfill_earnings_events(db_path, ["DUP"], date(2024, 1, 1), date(2024, 12, 31), provider=provider)
    stored = list_earnings_by_ticker(db_path, "DUP")

    assert first.inserted == 2
    assert first.updated == 0
    assert second.inserted == 0
    assert second.updated == 0
    assert second.duplicates == 2
    assert len(stored) == 2
    assert sorted(stored["provider_event_id"].tolist()) == [
        "earnings_dates:2024-02-01 16:00:00-05:00",
        "earnings_dates:2024-02-01 16:00:00-05:00:duplicate_1",
    ]


def test_scanner_score_is_unchanged_by_earnings_events(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    base_features = {
        "has_data": True,
        "data_quality": "ok",
        "last_price": 50.0,
        "ret_20d": 0.12,
        "ret_60d": 0.25,
        "above_50d_ma": True,
        "above_200d_ma": True,
        "relative_strength_20d": 0.05,
        "relative_strength_60d": 0.08,
        "volume_ratio_20d": 1.8,
        "avg_dollar_volume_20d": 50_000_000,
        "avg_dollar_volume_ok": True,
        "liquidity_score_raw": 1.0,
        "liquidity_label": "Acceptable",
        "distance_20d_ma": 0.04,
        "volatility_20d": 0.35,
    }
    before = score_ticker_from_features("AAPL", base_features, {"regime": "Risk-On"})
    insert_earnings_event(db_path, _event())
    after = score_ticker_from_features("AAPL", base_features, {"regime": "Risk-On"})

    assert after["score"] == before["score"]
    assert after["breakdown"] == before["breakdown"]


def test_earnings_coverage_classification_and_listing_adjustment(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    storage.upsert_ohlcv(db_path, "NEW", _price_frame(start="2024-01-02", periods=20))
    for idx, available_at in enumerate(
        [
            datetime(2024, 2, 1, 21, 0, tzinfo=UTC),
            datetime(2024, 5, 1, 21, 0, tzinfo=UTC),
            datetime(2024, 8, 1, 21, 0, tzinfo=UTC),
            datetime(2024, 11, 1, 21, 0, tzinfo=UTC),
        ],
        start=1,
    ):
        insert_earnings_event(db_path, _event("NEW", available_at=available_at, provider_event_id=f"earnings_dates:NEW:{idx}"))

    report = earnings_coverage_report(db_path, ["NEW", "EMPTY"], date(2023, 1, 1), date(2024, 12, 31))

    new_row = report[report["ticker"].eq("NEW")].iloc[0]
    empty_row = report[report["ticker"].eq("EMPTY")].iloc[0]
    assert expected_quarterly_events(date(2024, 1, 2), date(2024, 12, 31)) == 4
    assert classify_earnings_coverage(4, 4) == "complete"
    assert classify_earnings_coverage(2, 4) == "partial"
    assert classify_earnings_coverage(1, 4) == "sparse"
    assert classify_earnings_coverage(0, 4) == "unavailable"
    assert new_row["coverage_classification"] == "complete"
    assert new_row["provider_path"] == "earnings_dates"
    assert empty_row["coverage_classification"] == "unavailable"
