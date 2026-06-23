from __future__ import annotations

from pathlib import Path

import pandas as pd


DEFAULT_UNIVERSE = [
    ("SPY", "SPDR S&P 500 ETF Trust", "Benchmark"),
    ("QQQ", "Invesco QQQ Trust", "Benchmark"),
    ("IWM", "iShares Russell 2000 ETF", "Benchmark"),
    ("AAPL", "Apple Inc.", "Technology"),
    ("MSFT", "Microsoft Corporation", "Technology"),
    ("NVDA", "NVIDIA Corporation", "Technology"),
]


def load_universe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(DEFAULT_UNIVERSE, columns=["ticker", "name", "sector"])

    df = pd.read_csv(path)
    if "ticker" not in df.columns:
        raise ValueError("Universe file must contain a 'ticker' column.")
    if "name" not in df.columns:
        df["name"] = ""
    if "sector" not in df.columns:
        df["sector"] = "Unknown"

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].ne("")]
    return df.drop_duplicates("ticker").reset_index(drop=True)


def ticker_name_map(universe: pd.DataFrame) -> dict[str, str]:
    if universe.empty or "ticker" not in universe:
        return {}
    return dict(zip(universe["ticker"], universe.get("name", ""), strict=False))

