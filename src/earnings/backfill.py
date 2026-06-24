from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from src.earnings.repository import bulk_insert_earnings_events
from src.earnings.yfinance_provider import EarningsProviderResult, YFinanceHistoricalEarningsProvider


@dataclass
class EarningsBackfillResult:
    tickers: list[str]
    inserted: int = 0
    updated: int = 0
    duplicates: int = 0
    failed_tickers: int = 0
    warnings: list[str] = field(default_factory=list)
    per_ticker: list[dict[str, Any]] = field(default_factory=list)
    elapsed_seconds: float = 0.0


def backfill_earnings_events(
    db_path: str | Path,
    tickers: list[str],
    start_date: date,
    end_date: date,
    provider: YFinanceHistoricalEarningsProvider | None = None,
    use_cache: bool = True,
) -> EarningsBackfillResult:
    start = time.perf_counter()
    clean_tickers = sorted({ticker.upper().strip() for ticker in tickers if ticker and ticker.strip()})
    provider = provider or YFinanceHistoricalEarningsProvider(db_path=db_path)
    result = EarningsBackfillResult(tickers=clean_tickers)
    for ticker in clean_tickers:
        ticker_start = time.perf_counter()
        try:
            provider_result: EarningsProviderResult = provider.fetch_earnings_events(
                ticker,
                start_date,
                end_date,
                use_cache=use_cache,
            )
            counts = bulk_insert_earnings_events(db_path, provider_result.events)
            result.inserted += int(counts.get("inserted", 0))
            result.updated += int(counts.get("updated", 0))
            result.duplicates += int(counts.get("duplicate", 0))
            result.warnings.extend(provider_result.warnings)
            result.per_ticker.append(
                {
                    "ticker": ticker,
                    "events_returned": len(provider_result.events),
                    "inserted": int(counts.get("inserted", 0)),
                    "updated": int(counts.get("updated", 0)),
                    "duplicates": int(counts.get("duplicate", 0)),
                    "provider_source": provider_result.metadata.get("source"),
                    "record_source": provider_result.metadata.get("record_source"),
                    "warnings": "; ".join(provider_result.warnings[:3]),
                    "elapsed_seconds": round(time.perf_counter() - ticker_start, 3),
                }
            )
        except Exception as exc:
            result.failed_tickers += 1
            message = f"{ticker}: earnings backfill failed: {exc}"
            result.warnings.append(message)
            result.per_ticker.append(
                {
                    "ticker": ticker,
                    "events_returned": 0,
                    "inserted": 0,
                    "updated": 0,
                    "duplicates": 0,
                    "provider_source": "error",
                    "record_source": None,
                    "warnings": message,
                    "elapsed_seconds": round(time.perf_counter() - ticker_start, 3),
                }
            )
    result.elapsed_seconds = round(time.perf_counter() - start, 3)
    return result

