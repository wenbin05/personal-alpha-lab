from __future__ import annotations


def get_news_placeholder(ticker: str) -> dict:
    return {
        "ticker": ticker.upper(),
        "headline_count": 0,
        "summary": "No news provider configured in v1. Add a future news adapter here.",
        "provider": "placeholder",
    }

