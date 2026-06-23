from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from src.features.momentum import add_technical_columns


def price_volume_chart(df: pd.DataFrame, title: str = "Price") -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        row_heights=[0.72, 0.28],
    )
    if df is None or df.empty:
        fig.update_layout(title=f"{title}: no data")
        return fig

    data = add_technical_columns(df)
    fig.add_trace(go.Scatter(x=data.index, y=data["close"], name="Close", line=dict(color="#2563eb")), row=1, col=1)
    for col, color in [("ma_20", "#0f766e"), ("ma_50", "#f59e0b"), ("ma_200", "#7c3aed")]:
        if col in data:
            fig.add_trace(go.Scatter(x=data.index, y=data[col], name=col.upper(), line=dict(color=color, width=1.5)), row=1, col=1)
    if "volume" in data:
        fig.add_trace(go.Bar(x=data.index, y=data["volume"], name="Volume", marker_color="#94a3b8"), row=2, col=1)
    fig.update_layout(
        title=title,
        height=520,
        margin=dict(l=20, r=20, t=48, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hovermode="x unified",
    )
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)
    return fig


def equity_curve_chart(equity: pd.Series, benchmark: pd.Series | None = None, title: str = "Equity Curve") -> go.Figure:
    fig = go.Figure()
    if equity is not None and not equity.empty:
        fig.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy", line=dict(color="#16a34a", width=2)))
    if benchmark is not None and not benchmark.empty:
        aligned = benchmark.reindex(equity.index).ffill() if equity is not None and not equity.empty else benchmark
        fig.add_trace(go.Scatter(x=aligned.index, y=aligned, name="SPY Benchmark", line=dict(color="#64748b", width=2)))
    fig.update_layout(title=title, height=420, margin=dict(l=20, r=20, t=48, b=20), hovermode="x unified")
    return fig


def score_breakdown_chart(breakdown: dict[str, float]) -> go.Figure:
    fig = go.Figure(go.Bar(x=list(breakdown.keys()), y=list(breakdown.values()), marker_color="#2563eb"))
    fig.update_layout(title="Score Breakdown", height=320, margin=dict(l=20, r=20, t=48, b=80))
    return fig

