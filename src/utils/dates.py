from __future__ import annotations

from datetime import date, datetime, time
from zoneinfo import ZoneInfo


MARKET_TZ = ZoneInfo("America/New_York")
USER_TZ = ZoneInfo("Asia/Singapore")


def now_in_user_tz() -> datetime:
    return datetime.now(USER_TZ)


def now_in_market_tz() -> datetime:
    return datetime.now(MARKET_TZ)


def today_user_tz() -> date:
    return now_in_user_tz().date()


def market_session_label() -> str:
    """Return a simple US market-hours label from the current Singapore session."""
    now = now_in_market_tz()
    open_time = datetime.combine(now.date(), time(9, 30), MARKET_TZ)
    close_time = datetime.combine(now.date(), time(16, 0), MARKET_TZ)
    if now.weekday() >= 5:
        return "U.S. market closed (weekend)"
    if now < open_time:
        return "Pre-market / before U.S. open"
    if now <= close_time:
        return "U.S. market hours"
    return "After U.S. close"

