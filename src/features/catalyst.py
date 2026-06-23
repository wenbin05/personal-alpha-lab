from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from typing import Any

import pandas as pd


RISKY_REVIEW_FORMS = ("8-K", "S-1", "S-3", "424B", "Form 4", "FORM 4", "SEC 4")


def _as_date(value: Any) -> date | None:
    try:
        if value is None or pd.isna(value):
            return None
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _legacy_manual_features(ticker: str, manual_catalysts: pd.DataFrame | None) -> dict[str, Any]:
    ticker = ticker.upper()
    if manual_catalysts is None or manual_catalysts.empty:
        return _empty_catalyst_features()

    row = manual_catalysts[manual_catalysts["ticker"].str.upper() == ticker]
    if row.empty:
        return _empty_catalyst_features()

    record = row.iloc[0]
    score = float(record.get("catalyst_score", 0) or 0)
    note = str(record.get("note", "") or "")
    return {
        "catalyst_score": max(0.0, min(10.0, score)),
        "catalyst_penalty": 0.0,
        "catalyst_note": note,
        "has_manual_catalyst": True,
        "has_catalyst": True,
        "catalyst_events_count": 1,
        "recent_catalysts": [
            {
                "event_date": str(record.get("updated_at", ""))[:10],
                "event_type": "manual_note",
                "title": "Legacy manual catalyst note",
                "summary": note,
                "sentiment_label": "unknown",
                "catalyst_strength": score,
                "confidence": 1.0,
                "source": "manual_legacy",
                "is_manual": True,
            }
        ],
        "catalyst_warnings": [],
        "needs_review": [],
    }


def _empty_catalyst_features() -> dict[str, Any]:
    return {
        "catalyst_score": 0.0,
        "catalyst_penalty": 0.0,
        "catalyst_note": "",
        "has_manual_catalyst": False,
        "has_catalyst": False,
        "catalyst_events_count": 0,
        "recent_catalysts": [],
        "catalyst_warnings": ["No catalyst events found; catalyst contribution is neutral."],
        "needs_review": [],
    }


def _event_needs_review(event: pd.Series) -> bool:
    title = str(event.get("title", ""))
    event_type = str(event.get("event_type", ""))
    if event_type != "sec_filing":
        return False
    return any(form in title for form in RISKY_REVIEW_FORMS)


def get_catalyst_features(
    ticker: str,
    catalyst_events: pd.DataFrame | None = None,
    as_of_date: date | None = None,
    lookback_days: int = 45,
    future_days: int = 45,
) -> dict[str, Any]:
    """Calculate conservative, auditable catalyst contribution from event rows."""
    ticker = ticker.upper()
    if catalyst_events is None or catalyst_events.empty:
        return _empty_catalyst_features()

    if "event_type" not in catalyst_events.columns:
        return _legacy_manual_features(ticker, catalyst_events)

    as_of = as_of_date or datetime.now(UTC).date()
    start = as_of - timedelta(days=lookback_days)
    end = as_of + timedelta(days=future_days)

    events = catalyst_events.copy()
    events = events[events["ticker"].astype(str).str.upper() == ticker]
    if events.empty:
        return _empty_catalyst_features()

    if "_event_date_parsed" in events.columns:
        parsed_event_dates = pd.to_datetime(events["_event_date_parsed"], errors="coerce")
        parsed_dates = pd.Series(parsed_event_dates.dt.date, index=events.index)
        missing_dates = parsed_event_dates.isna()
        if bool(missing_dates.any()):
            parsed_dates.loc[missing_dates] = events.loc[missing_dates, "event_date"].apply(_as_date)
        events["event_date_parsed"] = parsed_dates
    else:
        events["event_date_parsed"] = events["event_date"].apply(_as_date)
    events = events.dropna(subset=["event_date_parsed"])
    events = events[(events["event_date_parsed"] >= start) & (events["event_date_parsed"] <= end)]
    if events.empty:
        features = _empty_catalyst_features()
        features["catalyst_warnings"] = ["Catalyst events exist, but none are recent enough for scoring."]
        return features

    positive_score = 0.0
    negative_penalty = 0.0
    reasons: list[str] = []
    needs_review: list[str] = []
    event_payloads: list[dict[str, Any]] = []

    for _, event in events.sort_values("event_date_parsed", ascending=False).iterrows():
        strength = max(0.0, min(10.0, float(event.get("catalyst_strength", 0) or 0)))
        confidence = max(0.0, min(1.0, float(event.get("confidence", 0) or 0)))
        sentiment = str(event.get("sentiment_label", "unknown") or "unknown").lower()
        weighted = strength * confidence
        title = str(event.get("title", "") or "")
        summary = str(event.get("summary", "") or "")

        if sentiment == "positive":
            positive_score += weighted
        elif sentiment == "negative":
            negative_penalty -= max(5.0 if weighted > 0 else 0.0, weighted * 1.5)

        if _event_needs_review(event):
            needs_review.append(title)

        event_payloads.append(
            {
                "id": event.get("id"),
                "event_date": event.get("event_date_parsed").isoformat(),
                "event_type": event.get("event_type"),
                "title": title,
                "summary": summary,
                "source": event.get("source"),
                "source_url": event.get("source_url"),
                "sentiment_label": sentiment,
                "catalyst_strength": strength,
                "confidence": confidence,
                "is_manual": bool(event.get("is_manual", False)),
            }
        )

    catalyst_score = max(0.0, min(10.0, positive_score))
    catalyst_penalty = max(-15.0, negative_penalty)
    if catalyst_score > 0:
        reasons.append(f"Recent positive catalysts add {catalyst_score:.1f}/10.")
    if catalyst_penalty < 0:
        reasons.append(f"Recent negative catalysts apply a {catalyst_penalty:.1f} penalty.")
    if needs_review:
        reasons.append("Recent SEC filing metadata needs manual review.")

    manual_events = [event for event in event_payloads if event["is_manual"]]
    note = manual_events[0]["summary"] if manual_events else (event_payloads[0]["summary"] if event_payloads else "")
    warnings = []
    if needs_review:
        warnings.append("One or more SEC filings are flagged Needs Review; no LLM or NLP interpretation has been applied.")

    return {
        "catalyst_score": round(catalyst_score, 2),
        "catalyst_penalty": round(catalyst_penalty, 2),
        "catalyst_note": note,
        "has_manual_catalyst": bool(manual_events),
        "has_catalyst": True,
        "catalyst_events_count": int(len(events)),
        "recent_catalysts": event_payloads[:8],
        "catalyst_warnings": warnings,
        "needs_review": needs_review,
        "catalyst_reasons": reasons,
    }
