from __future__ import annotations

from typing import Any

import pandas as pd

from src.backtest.metrics import benchmark_buy_hold, performance_metrics
from src.features.momentum import add_technical_columns
from src.features.regime import classify_market_regime
from src.features.volume import compute_volume_features
from src.scoring.score_engine import score_ticker
from src.utils.trading_calendar import is_trading_day


PORTFOLIO_REBALANCE_REPORTING = {
    "mode": "rebalance_period_approximation",
    "metrics_context": "rebalance_period",
    "win_rate_label": "Rebalance-period win rate",
    "average_win_label": "Avg rebalance-period win",
    "average_loss_label": "Avg rebalance-period loss",
    "trade_count_label": "Rebalance periods",
    "trades_table_label": "Rebalance periods / portfolio legs",
    "note": (
        "Scanner score portfolio results are rebalance-period approximations. "
        "Rows summarize portfolio holding periods and selected legs, not precise order-level fills."
    ),
}


def _empty_backtest_result(reporting: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "equity": pd.Series(dtype=float),
        "trades": pd.DataFrame(),
        "metrics": performance_metrics(pd.Series(dtype=float)),
        "reporting": reporting or {},
    }


def _calendar_filtered_index(index: pd.Index) -> pd.DatetimeIndex:
    ordered = pd.DatetimeIndex(index).sort_values().unique()
    return pd.DatetimeIndex([timestamp for timestamp in ordered if is_trading_day(pd.Timestamp(timestamp).date())])


def _signal_backtest(
    df: pd.DataFrame,
    signal: pd.Series,
    initial_capital: float = 100_000,
    slippage: float = 0.001,
    commission: float = 0.0,
    benchmark_df: pd.DataFrame | None = None,
) -> dict[str, Any]:
    if df is None or df.empty or "close" not in df.columns:
        return _empty_backtest_result()

    close = df["close"].dropna()
    signal = signal.reindex(close.index).fillna(False).astype(int)
    # Signals are assumed known only after the signal-date close. The target
    # position is entered/exited at the next trading day's close; returns begin
    # on the following close-to-close interval.
    holdings_after_close = signal.shift(1).fillna(0)
    return_position = holdings_after_close.shift(1).fillna(0)
    daily_returns = close.pct_change().fillna(0)
    position_change = holdings_after_close.diff().fillna(holdings_after_close)
    turnover = position_change.abs()
    cost = turnover * slippage
    if commission:
        cost += turnover * (commission / initial_capital)
    strategy_returns = return_position * daily_returns - cost
    equity = (1 + strategy_returns).cumprod() * initial_capital

    trade_rows = []
    current_entry_date = None
    current_entry_price = None
    for date, change in position_change.items():
        if change == 0:
            continue
        pos = holdings_after_close.loc[date]
        if change > 0 and pos == 1 and current_entry_date is None:
            current_entry_date = date
            current_entry_price = close.loc[date] * (1 + slippage)
        elif change < 0 and pos == 0 and current_entry_date is not None:
            exit_price = close.loc[date] * (1 - slippage)
            pnl_pct = exit_price / current_entry_price - 1
            trade_rows.append(
                {
                    "entry_date": current_entry_date,
                    "exit_date": date,
                    "entry_price": current_entry_price,
                    "exit_price": exit_price,
                    "pnl_pct": pnl_pct,
                    "turnover": 2.0,
                }
            )
            current_entry_date = None
            current_entry_price = None

    trades = pd.DataFrame(trade_rows)
    metrics = performance_metrics(equity, trades)
    benchmark = benchmark_buy_hold(benchmark_df, initial_capital) if benchmark_df is not None else pd.Series(dtype=float)
    return {"equity": equity, "trades": trades, "metrics": metrics, "benchmark": benchmark}


def backtest_moving_average_strategy(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    initial_capital: float = 100_000,
    slippage: float = 0.001,
    commission: float = 0.0,
) -> dict[str, Any]:
    data = add_technical_columns(df)
    signal = (data["close"] > data["ma_50"]) & (data["ma_50"] > data["ma_200"])
    return _signal_backtest(data, signal, initial_capital, slippage, commission, benchmark_df)


def backtest_momentum_breakout_strategy(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    volume_threshold: float = 1.2,
    initial_capital: float = 100_000,
    slippage: float = 0.001,
    commission: float = 0.0,
) -> dict[str, Any]:
    data = add_technical_columns(df)
    if "volume" not in data.columns:
        rolling_ratio = pd.Series(False, index=data.index)
    else:
        rolling_ratio = data["volume"] / data["volume"].shift(1).rolling(20).mean()
    signal = (data["close"] > data["ma_50"]) & (data["ret_20d"] > 0) & (rolling_ratio > volume_threshold)
    return _signal_backtest(data, signal, initial_capital, slippage, commission, benchmark_df)


def backtest_mean_reversion_strategy(
    df: pd.DataFrame,
    benchmark_df: pd.DataFrame | None = None,
    pullback_pct: float = 0.04,
    initial_capital: float = 100_000,
    slippage: float = 0.001,
    commission: float = 0.0,
) -> dict[str, Any]:
    data = add_technical_columns(df)
    signal = (data["distance_20d_ma"] < -abs(pullback_pct)) & (data["close"] > data["ma_200"])
    return _signal_backtest(data, signal, initial_capital, slippage, commission, benchmark_df)


def backtest_top_score_strategy(
    histories: dict[str, pd.DataFrame],
    spy_df: pd.DataFrame,
    top_n: int = 5,
    rebalance_every: int = 5,
    initial_capital: float = 100_000,
    slippage: float = 0.001,
    commission: float = 0.0,
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
) -> dict[str, Any]:
    if spy_df is None or spy_df.empty:
        return _empty_backtest_result(PORTFOLIO_REBALANCE_REPORTING)

    tickers = [ticker for ticker, df in histories.items() if ticker != "SPY" and df is not None and not df.empty]
    common_dates = _calendar_filtered_index(spy_df.index)
    if len(common_dates) < 220 or not tickers:
        return _empty_backtest_result(PORTFOLIO_REBALANCE_REPORTING)

    weights = pd.DataFrame(0.0, index=common_dates, columns=tickers)
    rebalance_dates = list(common_dates[200::rebalance_every])
    for signal_date in rebalance_dates:
        available_histories = {
            "SPY": spy_df.loc[:signal_date],
            "QQQ": histories.get("QQQ", pd.DataFrame()).loc[:signal_date] if "QQQ" in histories else pd.DataFrame(),
            "IWM": histories.get("IWM", pd.DataFrame()).loc[:signal_date] if "IWM" in histories else pd.DataFrame(),
        }
        regime = classify_market_regime(available_histories)
        scored = []
        for ticker in tickers:
            hist = histories[ticker].loc[:signal_date]
            if len(hist) < 200:
                continue
            result = score_ticker(ticker, hist, spy_df.loc[:signal_date], regime, None, min_price, min_avg_dollar_volume)
            scored.append((ticker, result["score"]))
        selected = [ticker for ticker, _ in sorted(scored, key=lambda item: item[1], reverse=True)[:top_n]]
        if not selected:
            continue
        start_idx = common_dates.get_loc(signal_date) + 1
        end_idx = min(start_idx + rebalance_every, len(common_dates))
        if start_idx >= len(common_dates):
            continue
        weights.loc[common_dates[start_idx:end_idx], selected] = 1 / len(selected)

    returns = pd.DataFrame(index=common_dates)
    for ticker in tickers:
        returns[ticker] = histories[ticker]["close"].reindex(common_dates).pct_change().fillna(0)
    portfolio_returns = (weights.shift(1).fillna(0) * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.abs().sum(axis=1))
    strategy_returns = portfolio_returns - turnover * slippage
    if commission:
        strategy_returns -= turnover * (commission / initial_capital)
    equity = (1 + strategy_returns).cumprod() * initial_capital
    rebalance_dates = list(turnover[turnover > 0].index)
    trade_rows = []
    for idx, start_date in enumerate(rebalance_dates):
        end_date = rebalance_dates[idx + 1] if idx + 1 < len(rebalance_dates) else common_dates[-1]
        if start_date not in equity.index or end_date not in equity.index or end_date <= start_date:
            continue
        start_equity = equity.loc[start_date]
        end_equity = equity.loc[end_date]
        if start_equity == 0:
            continue
        selected = weights.loc[start_date]
        selected = selected[selected > 0].index.tolist()
        trade_rows.append(
            {
                "date": start_date,
                "end_date": end_date,
                "turnover": float(turnover.loc[start_date]),
                "pnl_pct": float(end_equity / start_equity - 1),
                "holdings": ", ".join(selected),
            }
        )
    trades = pd.DataFrame(trade_rows)
    metrics = performance_metrics(equity, trades)
    benchmark = benchmark_buy_hold(spy_df.reindex(common_dates), initial_capital)
    return {
        "equity": equity,
        "trades": trades,
        "metrics": metrics,
        "benchmark": benchmark,
        "weights": weights,
        "reporting": PORTFOLIO_REBALANCE_REPORTING,
    }
