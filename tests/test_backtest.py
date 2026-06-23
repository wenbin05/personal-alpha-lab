from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.backtest.metrics import performance_metrics
from src.backtest.simple_backtester import (
    _signal_backtest,
    backtest_moving_average_strategy,
    backtest_top_score_strategy,
)


def sample_trend(rows: int = 320) -> pd.DataFrame:
    dates = pd.bdate_range("2023-01-01", periods=rows)
    close = 100 + np.arange(rows) * 0.25
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.01,
            "low": close * 0.99,
            "close": close,
            "adj_close": close,
            "volume": np.full(rows, 2_000_000),
        },
        index=dates,
    )


def test_backtest_metrics_calculate_correctly() -> None:
    equity = pd.Series([100_000, 101_000, 99_000, 105_000], index=pd.bdate_range("2024-01-01", periods=4))
    trades = pd.DataFrame({"pnl_pct": [0.05, -0.02], "turnover": [2, 2]})
    metrics = performance_metrics(equity, trades)

    assert metrics["total_return"] == pytest.approx(0.05)
    assert metrics["max_drawdown"] < 0
    assert metrics["number_of_trades"] == 2
    assert metrics["profit_factor"] > 0


def test_moving_average_backtest_runs() -> None:
    df = sample_trend()
    result = backtest_moving_average_strategy(df, df)

    assert not result["equity"].empty
    assert "total_return" in result["metrics"]


def test_signal_backtest_enters_next_day_close_without_signal_day_return() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame(
        {
            "open": [100.0, 200.0, 400.0, 800.0],
            "high": [100.0, 200.0, 400.0, 800.0],
            "low": [100.0, 200.0, 400.0, 800.0],
            "close": [100.0, 200.0, 400.0, 800.0],
            "adj_close": [100.0, 200.0, 400.0, 800.0],
            "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
        },
        index=dates,
    )
    signal = pd.Series([True, False, False, False], index=dates)

    result = _signal_backtest(df, signal, initial_capital=100_000, slippage=0, commission=0)
    trades = result["trades"]

    assert result["equity"].loc[dates[1]] == pytest.approx(100_000)
    assert result["equity"].loc[dates[2]] == pytest.approx(200_000)
    assert trades.iloc[0]["entry_date"] == dates[1]
    assert trades.iloc[0]["entry_price"] == 200.0
    assert trades.iloc[0]["exit_date"] == dates[2]
    assert trades.iloc[0]["exit_price"] == 400.0


def test_signal_backtest_applies_slippage_and_benchmark() -> None:
    dates = pd.bdate_range("2024-01-01", periods=4)
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 110.0, 120.0],
            "high": [100.0, 100.0, 110.0, 120.0],
            "low": [100.0, 100.0, 110.0, 120.0],
            "close": [100.0, 100.0, 110.0, 120.0],
            "adj_close": [100.0, 100.0, 110.0, 120.0],
            "volume": [1_000_000, 1_000_000, 1_000_000, 1_000_000],
        },
        index=dates,
    )
    signal = pd.Series([True, False, False, False], index=dates)

    no_slip = _signal_backtest(df, signal, initial_capital=100_000, slippage=0, commission=0, benchmark_df=df)
    slipped = _signal_backtest(df, signal, initial_capital=100_000, slippage=0.01, commission=0, benchmark_df=df)

    assert slipped["equity"].iloc[-1] < no_slip["equity"].iloc[-1]
    assert not slipped["benchmark"].empty
    assert slipped["benchmark"].iloc[-1] == pytest.approx(120_000)


def test_top_score_backtest_records_rebalance_period_pnl() -> None:
    spy = sample_trend(260)
    histories = {
        "SPY": spy,
        "QQQ": sample_trend(260),
        "IWM": sample_trend(260),
        "AAA": sample_trend(260),
        "BBB": sample_trend(260),
    }

    result = backtest_top_score_strategy(
        histories,
        spy,
        top_n=1,
        rebalance_every=10,
        initial_capital=100_000,
        slippage=0,
        commission=0,
    )

    assert not result["equity"].empty
    assert not result["benchmark"].empty
    assert not result["trades"].empty
    assert "holdings" in result["trades"].columns
    assert result["trades"]["pnl_pct"].abs().sum() > 0


def test_top_score_backtest_reports_rebalance_period_approximation() -> None:
    spy = sample_trend(260)
    histories = {
        "SPY": spy,
        "QQQ": sample_trend(260),
        "IWM": sample_trend(260),
        "AAA": sample_trend(260),
    }

    result = backtest_top_score_strategy(histories, spy, top_n=1, rebalance_every=10)
    reporting = result["reporting"]

    assert reporting["mode"] == "rebalance_period_approximation"
    assert "Rebalance-period win rate" == reporting["win_rate_label"]
    assert "Rebalance periods" == reporting["trade_count_label"]
    assert "not precise order-level fills" in reporting["note"]
