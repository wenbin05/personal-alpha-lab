from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.catalysts.repository import catalyst_display_frame, list_catalysts_by_ticker
from src.catalysts.proposals import (
    link_display_frame,
    list_links_by_ticker,
    list_proposals_by_ticker,
    proposal_display_frame,
    proposal_score_contribution,
)
from src.catalysts.publications import list_publications_by_ticker, publication_display_frame
from src.data import market_data
from src.documents.repository import document_display_frame, list_documents_by_ticker
from src.features.regime import classify_market_regime
from src.scoring.score_engine import score_ticker
from src.utils.trading_calendar import calendar_source, latest_expected_trading_day


DISPLAY_OHLCV_COLUMNS = {
    "open": "Open",
    "high": "High",
    "low": "Low",
    "close": "Close",
    "adj_close": "Adj Close",
    "volume": "Volume",
}


FEATURE_LABELS = {
    "ret_5d": "5D return",
    "ret_20d": "20D return",
    "ret_60d": "60D return",
    "ret_120d": "120D return",
    "volatility_20d": "20D volatility",
    "ma_50": "50D moving average",
    "ma_200": "200D moving average",
    "distance_20d_ma": "Distance from 20D MA",
    "distance_50d_ma": "Distance from 50D MA",
    "above_50d_ma": "Above 50D MA",
    "above_200d_ma": "Above 200D MA",
    "current_volume": "Current volume",
    "avg_volume_20d": "Prior 20D average volume",
    "volume_ratio_20d": "Volume ratio",
    "avg_dollar_volume_20d": "Average daily dollar volume",
    "relative_strength_20d": "20D relative strength vs SPY",
    "relative_strength_60d": "60D relative strength vs SPY",
}


SCORE_LABELS = {
    "market_regime": "Market regime compatibility",
    "momentum_trend": "Momentum/trend score",
    "relative_strength": "Relative strength score",
    "volume_anomaly": "Volume score",
    "liquidity_quality": "Liquidity score",
    "catalyst": "Catalyst placeholder score",
    "options": "Options placeholder score",
    "risk_reward": "Risk/reward score",
}


def _date_or_none(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    return pd.to_datetime(value).date()


def _cache_age_days(latest_date: date | None, today: date | None = None) -> int | None:
    if latest_date is None:
        return None
    today = today or datetime.now(UTC).date()
    return max(0, (today - latest_date).days)


def ohlcv_metadata(
    df: pd.DataFrame,
    requested_period: str,
    fetch_metadata: dict[str, Any] | None = None,
    today: date | None = None,
) -> dict[str, Any]:
    fetch_metadata = fetch_metadata or {}
    if df is None or df.empty:
        expected_latest = latest_expected_trading_day(today)
        return {
            "requested_period": requested_period,
            "actual_start_date": None,
            "actual_end_date": None,
            "rows": 0,
            "latest_available_trading_date": None,
            "latest_expected_trading_day": expected_latest.isoformat(),
            "cache_age_days": None,
            "calendar_source": fetch_metadata.get("calendar_source", calendar_source()),
            "data_source": fetch_metadata.get("source", "none"),
            "cache_satisfies_requested_period": fetch_metadata.get("cache_satisfies_requested_period_before", False),
            "minimum_bars_for_requested_period": fetch_metadata.get(
                "minimum_bars_for_requested_period",
                market_data.minimum_bars_for_period(requested_period),
            ),
        }

    ordered = df.sort_index()
    start = _date_or_none(ordered.index.min())
    end = _date_or_none(ordered.index.max())
    expected_latest = latest_expected_trading_day(today)
    return {
        "requested_period": requested_period,
        "actual_start_date": start.isoformat() if start else None,
        "actual_end_date": end.isoformat() if end else None,
        "rows": int(len(ordered)),
        "latest_available_trading_date": end.isoformat() if end else None,
        "latest_expected_trading_day": expected_latest.isoformat(),
        "cache_age_days": _cache_age_days(end, today),
        "calendar_source": fetch_metadata.get("calendar_source", calendar_source()),
        "data_source": fetch_metadata.get("source", "unknown"),
        "cached_rows_before": fetch_metadata.get("cached_rows_before"),
        "cached_latest_date_before": fetch_metadata.get("cached_latest_date_before"),
        "cache_fresh_before": fetch_metadata.get("cache_fresh_before"),
        "cache_satisfies_requested_period": fetch_metadata.get("cache_satisfies_requested_period_before"),
        "minimum_bars_for_requested_period": fetch_metadata.get(
            "minimum_bars_for_requested_period",
            market_data.minimum_bars_for_period(requested_period),
        ),
        "download_error": fetch_metadata.get("download_error"),
    }


def latest_ohlcv_rows(df: pd.DataFrame, rows: int = 5) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=["Date", *DISPLAY_OHLCV_COLUMNS.values()])
    latest = df.sort_index().tail(rows).copy()
    latest = latest.rename(columns=DISPLAY_OHLCV_COLUMNS)
    latest = latest.reset_index().rename(columns={"date": "Date", "index": "Date"})
    return latest


def missing_value_counts(df: pd.DataFrame) -> dict[str, int]:
    counts: dict[str, int] = {}
    for column, label in DISPLAY_OHLCV_COLUMNS.items():
        if df is None or df.empty or column not in df.columns:
            counts[label] = int(len(df)) if df is not None and not df.empty else 0
        else:
            counts[label] = int(df[column].isna().sum())
    return counts


def feature_table(features: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"feature": label, "value": features.get(key)} for key, label in FEATURE_LABELS.items()]
    )


def score_breakdown_table(breakdown: dict[str, Any]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"component": label, "score": breakdown.get(key, 0)} for key, label in SCORE_LABELS.items()]
    )


def validation_warnings(
    df: pd.DataFrame,
    spy_df: pd.DataFrame,
    features: dict[str, Any],
    metadata: dict[str, Any],
    requested_period: str,
    catalyst_features: dict[str, Any] | None = None,
    today: date | None = None,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    today = today or datetime.now(UTC).date()

    if not features.get("has_data"):
        warnings.append(
            {
                "name": "missing_data",
                "severity": "high",
                "message": "No usable OHLCV data is available for this ticker.",
            }
        )

    if int(metadata.get("rows") or 0) < 200 or features.get("ma_200") is None:
        warnings.append(
            {
                "name": "insufficient_history_for_200d_ma",
                "severity": "medium",
                "message": "Fewer than 200 rows are available, so the 200D moving average may be missing.",
            }
        )

    latest = _date_or_none(metadata.get("latest_available_trading_date"))
    expected_latest = latest_expected_trading_day(today)
    if latest is not None and latest < expected_latest:
        warnings.append(
            {
                "name": "stale_latest_date",
                "severity": "medium",
                "message": (
                    f"Latest bar is {latest.isoformat()}, older than the expected U.S. trading-day bar "
                    f"{expected_latest.isoformat()}. This may reflect delayed free data, a holiday/weekend "
                    "timing edge, a failed download, or stale cache."
                ),
            }
        )

    if df is not None and not df.empty:
        missing_required = [label for column, label in DISPLAY_OHLCV_COLUMNS.items() if column not in df.columns]
        if missing_required:
            warnings.append(
                {
                    "name": "missing_required_ohlcv_columns",
                    "severity": "high",
                    "message": "Missing required OHLCV columns: " + ", ".join(missing_required) + ".",
                }
            )

        if "volume" not in df.columns or df["volume"].isna().any():
            warnings.append(
                {
                    "name": "missing_volume",
                    "severity": "medium",
                    "message": "Volume data is missing for one or more rows.",
                }
            )
        elif df["volume"].tail(20).eq(0).any():
            warnings.append(
                {
                    "name": "zero_volume",
                    "severity": "medium",
                    "message": "One or more recent rows have zero volume.",
                }
            )

        price_cols = [col for col in ["open", "high", "low", "close", "adj_close"] if col in df.columns]
        if price_cols:
            prices = df[price_cols]
            high_low_bad = "high" in df.columns and "low" in df.columns and (df["high"] < df["low"]).any()
            non_positive = prices.le(0).any().any()
            very_large = prices.gt(100_000).any().any()
            if bool(high_low_bad or non_positive or very_large):
                warnings.append(
                    {
                        "name": "suspicious_price_values",
                        "severity": "high",
                        "message": "Price data contains non-positive values, high < low, or very large values.",
                    }
                )

    if (
        spy_df is None
        or spy_df.empty
        or features.get("relative_strength_20d") is None
        or features.get("relative_strength_60d") is None
    ):
        warnings.append(
            {
                "name": "failed_spy_comparison",
                "severity": "medium",
                "message": "Relative strength vs SPY could not be calculated.",
            }
        )

    minimum_bars = market_data.minimum_bars_for_period(requested_period)
    if int(metadata.get("cached_rows_before") or 0) > 0 and int(metadata.get("cached_rows_before") or 0) < minimum_bars:
        warnings.append(
            {
                "name": "cache_period_shorter_than_requested_period",
                "severity": "low",
                "message": "Cached rows before fetch were fewer than expected for the requested period.",
            }
        )

    catalyst_features = catalyst_features or {}
    if catalyst_features and not catalyst_features.get("has_catalyst"):
        warnings.append(
            {
                "name": "no_catalyst_data",
                "severity": "low",
                "message": "No catalyst events are stored for this ticker; catalyst contribution is neutral.",
            }
        )
    for message in catalyst_features.get("catalyst_warnings", []) or []:
        warnings.append(
            {
                "name": "catalyst_warning",
                "severity": "low",
                "message": message,
            }
        )

    return warnings


def document_availability_warnings(
    documents: pd.DataFrame | None,
    catalyst_events: pd.DataFrame | None = None,
) -> list[dict[str, str]]:
    warnings: list[dict[str, str]] = []
    documents = documents if documents is not None else pd.DataFrame()
    catalyst_events = catalyst_events if catalyst_events is not None else pd.DataFrame()

    if documents.empty:
        warnings.append(
            {
                "name": "no_source_documents",
                "severity": "low",
                "message": "No source documents are stored for this ticker yet; future extraction has no local text context.",
            }
        )
    else:
        failed = documents[documents["parsing_status"].eq("failed")] if "parsing_status" in documents.columns else pd.DataFrame()
        partial = documents[documents["parsing_status"].eq("partial")] if "parsing_status" in documents.columns else pd.DataFrame()
        if not failed.empty:
            warnings.append(
                {
                    "name": "failed_source_document_parse",
                    "severity": "low",
                    "message": f"{len(failed)} source document(s) have failed parsing/fetch status.",
                }
            )
        if not partial.empty:
            warnings.append(
                {
                    "name": "partial_source_document_parse",
                    "severity": "low",
                    "message": f"{len(partial)} source document(s) have partial parsing/fetch status.",
                }
            )

    if not catalyst_events.empty:
        sec_events = catalyst_events[catalyst_events["event_type"].eq("sec_filing")] if "event_type" in catalyst_events.columns else pd.DataFrame()
        if not sec_events.empty:
            linked_ids = set()
            if not documents.empty and "catalyst_id" in documents.columns:
                linked_ids = {int(value) for value in documents["catalyst_id"].dropna().tolist()}
            missing_sec_text = [
                int(row["id"])
                for _, row in sec_events.iterrows()
                if pd.notna(row.get("id")) and int(row["id"]) not in linked_ids
            ]
            if missing_sec_text:
                warnings.append(
                    {
                        "name": "sec_catalyst_without_source_text",
                        "severity": "low",
                        "message": (
                            f"{len(missing_sec_text)} SEC catalyst event(s) do not have linked source text. "
                            "This is an auditability warning, not an alpha penalty."
                        ),
                    }
                )

    return warnings


def build_validation_report_from_data(
    ticker: str,
    ticker_df: pd.DataFrame,
    spy_df: pd.DataFrame,
    regime: dict[str, Any],
    fetch_metadata: dict[str, Any] | None = None,
    requested_period: str = "2y",
    catalyst_events: pd.DataFrame | None = None,
    documents: pd.DataFrame | None = None,
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
    today: date | None = None,
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    score_result = score_ticker(
        ticker,
        ticker_df,
        spy_df,
        regime,
        catalyst_events,
        min_price=min_price,
        min_avg_dollar_volume=min_avg_dollar_volume,
    )
    metadata = ohlcv_metadata(ticker_df, requested_period, fetch_metadata, today)
    features = score_result["features"]
    catalyst_features = score_result.get("catalyst_features", {})
    documents = documents if documents is not None else pd.DataFrame()
    document_warnings = document_availability_warnings(documents, catalyst_events)

    return {
        "ticker": ticker,
        "metadata": metadata,
        "latest_rows": latest_ohlcv_rows(ticker_df),
        "missing_value_counts": missing_value_counts(ticker_df),
        "features": features,
        "feature_table": feature_table(features),
        "score_result": score_result,
        "score_breakdown_table": score_breakdown_table(score_result.get("breakdown", {})),
        "penalties": score_result.get("penalties", []),
        "warnings": [
            *validation_warnings(ticker_df, spy_df, features, metadata, requested_period, catalyst_features, today),
            *document_warnings,
        ],
        "regime": regime,
        "catalyst_features": catalyst_features,
        "catalyst_table": catalyst_display_frame(catalyst_events if catalyst_events is not None else pd.DataFrame()),
        "document_count": int(len(documents)),
        "document_table": document_display_frame(documents),
        "document_warnings": document_warnings,
    }


def load_validation_report(
    ticker: str,
    db_path: str | Path,
    provider_name: str,
    requested_period: str = "2y",
    refresh: bool = False,
    min_price: float = 5.0,
    min_avg_dollar_volume: float = 10_000_000,
) -> dict[str, Any]:
    ticker = ticker.upper().strip()
    ticker_fetch = market_data.get_history_with_metadata(ticker, db_path, provider_name, requested_period, refresh)
    spy_fetch = (
        ticker_fetch
        if ticker == "SPY"
        else market_data.get_history_with_metadata("SPY", db_path, provider_name, requested_period, refresh)
    )
    histories = {
        "SPY": spy_fetch.data,
        "QQQ": market_data.get_history("QQQ", db_path, provider_name, requested_period, refresh),
        "IWM": market_data.get_history("IWM", db_path, provider_name, requested_period, refresh),
        "^VIX": market_data.get_history("^VIX", db_path, provider_name, requested_period, refresh),
    }
    regime = classify_market_regime(histories)
    catalyst_events = list_catalysts_by_ticker(db_path, ticker, limit=100)
    documents = list_documents_by_ticker(db_path, ticker, limit=100)
    proposals = list_proposals_by_ticker(db_path, ticker, limit=100)
    extraction_links = list_links_by_ticker(db_path, ticker, limit=100)
    publications = list_publications_by_ticker(db_path, ticker, limit=100)
    report = build_validation_report_from_data(
        ticker,
        ticker_fetch.data,
        spy_fetch.data,
        regime,
        ticker_fetch.metadata,
        requested_period,
        catalyst_events,
        documents,
        min_price,
        min_avg_dollar_volume,
    )
    report.update(
        {
            "proposal_count": int(len(proposals)),
            "proposal_table": proposal_display_frame(proposals),
            "extraction_link_count": int(len(extraction_links)),
            "extraction_link_table": link_display_frame(extraction_links),
            "publication_count": int(len(publications)),
            "publication_table": publication_display_frame(publications),
            "llm_proposal_score_contribution": proposal_score_contribution(),
            "proposal_scoring_note": (
                "Only active catalyst records feed catalyst scoring. Pending extractions, proposal-only rows, "
                "audit links, and reverted publication rows contribute 0. Duplicate active publication per proposal is blocked."
            ),
        }
    )
    return report
