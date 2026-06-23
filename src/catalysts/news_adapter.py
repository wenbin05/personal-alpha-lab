from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from src.catalysts.models import CatalystEvent


@dataclass
class NewsResult:
    events: list[CatalystEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class PlaceholderNewsProvider:
    name = "placeholder_news"

    def get_recent_news(self, ticker: str, start_date: date, end_date: date) -> NewsResult:
        return NewsResult(
            events=[],
            warnings=[
                (
                    f"No news provider is configured for {ticker.upper()}. "
                    "Manual catalyst notes and future CSV/API adapters can supply news events."
                )
            ],
        )
