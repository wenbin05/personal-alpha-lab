from __future__ import annotations

from datetime import UTC, date, datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data import market_data, storage
from src.data.market_data import MarketDataError


def sample_ohlcv(rows: int) -> pd.DataFrame:
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=rows)
    close = 100 + np.arange(rows, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(rows, 1_000_000),
        },
        index=dates,
    )


def test_normalize_ohlcv_accepts_lowercase_datetime_index() -> None:
    normalized = storage.normalize_ohlcv(sample_ohlcv(5))

    assert not normalized.empty
    assert list(normalized.columns) == storage.OHLCV_COLUMNS
    assert normalized["date"].notna().all()


def test_yfinance_daily_end_is_advanced_for_exclusive_provider_boundary(monkeypatch) -> None:
    calls: list[tuple[str, dict]] = []

    class FakeTicker:
        def __init__(self, ticker: str):
            self.ticker = ticker

        def history(self, **kwargs):
            calls.append((self.ticker, kwargs))
            frame = sample_ohlcv(1)
            frame.index = pd.to_datetime(["2026-07-13"])
            return frame

    monkeypatch.setitem(__import__("sys").modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))
    result = market_data.YFinanceProvider().download_history(
        "AAPL", start=date(2026, 7, 13), end=date(2026, 7, 13)
    )

    assert calls == [
        (
            "AAPL",
            {
                "interval": "1d",
                "auto_adjust": False,
                "start": "2026-07-13",
                "end": "2026-07-14",
            },
        )
    ]
    assert result["date"].tolist() == ["2026-07-13"]


def test_yfinance_exclusive_end_handles_friday_holiday_boundary_and_vix(monkeypatch) -> None:
    calls: list[tuple[str, str]] = []

    class FakeTicker:
        def __init__(self, ticker: str):
            self.ticker = ticker

        def history(self, **kwargs):
            calls.append((self.ticker, kwargs["end"]))
            frame = sample_ohlcv(1)
            frame.index = pd.to_datetime([kwargs["start"]])
            return frame

    monkeypatch.setitem(__import__("sys").modules, "yfinance", SimpleNamespace(Ticker=FakeTicker))
    provider = market_data.YFinanceProvider()
    provider.download_history("AAPL", start="2026-07-10", end="2026-07-10")
    provider.download_history("AAPL", start="2025-01-08", end="2025-01-08")
    provider.download_history("^VIX", start="2026-07-13", end="2026-07-13")

    assert calls == [
        ("AAPL", "2026-07-11"),
        ("AAPL", "2025-01-09"),
        ("^VIX", "2026-07-14"),
    ]


def test_get_history_refreshes_when_cache_is_too_short_for_period(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "alpha_lab.db"
    storage.init_db(db_path)
    storage.upsert_ohlcv(db_path, "AAA", sample_ohlcv(500))

    calls: list[str] = []

    class FakeProvider:
        def download_history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
            calls.append(period)
            return sample_ohlcv(1300)

    monkeypatch.setattr(market_data, "get_provider", lambda name: FakeProvider())

    history = market_data.get_history("AAA", db_path, period="5y", refresh=False)

    assert calls == ["5y"]
    assert len(history) >= 1200


def test_get_history_returns_empty_metadata_on_download_failure_without_cache(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "alpha_lab.db"

    class FailingProvider:
        def download_history(self, ticker: str, period: str = "2y") -> pd.DataFrame:
            raise MarketDataError("offline")

    monkeypatch.setattr(market_data, "get_provider", lambda name: FailingProvider())

    result = market_data.get_history_with_metadata("BAD", db_path, period="2y", refresh=True)

    assert result.data.empty
    assert result.metadata["source"] == "download_error_no_cache"
    assert "offline" in result.metadata["download_error"]
    assert result.metadata["latest_expected_trading_day"]
    assert result.metadata["calendar_source"]


def test_cache_freshness_uses_trading_calendar_cutoff() -> None:
    df = sample_ohlcv(3)
    df.index = pd.to_datetime(["2024-01-03", "2024-01-04", "2024-01-05"])

    assert market_data.cache_is_fresh(df, reference_datetime=datetime(2024, 1, 8, 16, 0, tzinfo=UTC))


def test_trade_journal_persists_optional_fields(tmp_path) -> None:
    db_path = tmp_path / "journal.db"

    storage.add_trade(
        db_path,
        {
            "ticker": "AAPL",
            "direction": "long",
            "entry_date": "2026-01-02",
            "entry_price": 100.0,
            "thesis": "Test thesis",
        },
    )
    trades = storage.load_trades(db_path)

    assert len(trades) == 1
    assert trades.iloc[0]["ticker"] == "AAPL"
    assert trades.iloc[0]["thesis"] == "Test thesis"
