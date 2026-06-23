from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    running_max = equity.cummax()
    drawdown = equity / running_max - 1
    return float(drawdown.min())


def performance_metrics(equity: pd.Series, trades: pd.DataFrame | None = None) -> dict[str, Any]:
    equity = equity.dropna()
    if equity.empty:
        return {
            "total_return": 0.0,
            "annualized_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe_ratio": 0.0,
            "win_rate": 0.0,
            "average_win": 0.0,
            "average_loss": 0.0,
            "profit_factor": 0.0,
            "number_of_trades": 0,
            "turnover_estimate": 0.0,
        }

    daily_returns = equity.pct_change().dropna()
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = max((equity.index[-1] - equity.index[0]).days / 365.25, 1 / 365.25)
    annualized = (1 + total_return) ** (1 / years) - 1
    sharpe = 0.0
    if not daily_returns.empty and daily_returns.std() > 0:
        sharpe = float(daily_returns.mean() / daily_returns.std() * np.sqrt(252))

    pnl = pd.Series(dtype=float)
    turnover = 0.0
    if trades is not None and not trades.empty:
        if "pnl_pct" in trades.columns:
            pnl = pd.to_numeric(trades["pnl_pct"], errors="coerce").dropna()
        if "turnover" in trades.columns:
            turnover = float(pd.to_numeric(trades["turnover"], errors="coerce").fillna(0).sum())

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    gross_win = wins.sum()
    gross_loss = abs(losses.sum())

    return {
        "total_return": float(total_return),
        "annualized_return": float(annualized),
        "max_drawdown": max_drawdown(equity),
        "sharpe_ratio": sharpe,
        "win_rate": float(len(wins) / len(pnl)) if len(pnl) else 0.0,
        "average_win": float(wins.mean()) if len(wins) else 0.0,
        "average_loss": float(losses.mean()) if len(losses) else 0.0,
        "profit_factor": float(gross_win / gross_loss) if gross_loss > 0 else (float("inf") if gross_win > 0 else 0.0),
        "number_of_trades": int(len(pnl)),
        "turnover_estimate": turnover,
    }


def benchmark_buy_hold(df: pd.DataFrame, initial_capital: float = 100_000) -> pd.Series:
    if df is None or df.empty or "close" not in df.columns:
        return pd.Series(dtype=float)
    close = df["close"].dropna()
    return close / close.iloc[0] * initial_capital

