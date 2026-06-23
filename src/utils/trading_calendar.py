from __future__ import annotations

import calendar
from datetime import UTC, date, datetime, time, timedelta
from functools import lru_cache
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd


NEW_YORK_TZ = ZoneInfo("America/New_York")
DAILY_BAR_READY_TIME_ET = time(18, 0)


EXCEPTIONAL_FULL_DAY_CLOSURES: dict[int, frozenset[date]] = {
    2025: frozenset(
        {
            # National Day of Mourning for President Jimmy Carter.
            date(2025, 1, 9),
        }
    )
}


def _load_real_calendar() -> tuple[str, Any | None]:
    try:
        import pandas_market_calendars as mcal

        return "pandas_market_calendars:NYSE", mcal.get_calendar("NYSE")
    except Exception:
        pass

    try:
        import exchange_calendars as xcals

        return "exchange_calendars:XNYS", xcals.get_calendar("XNYS")
    except Exception:
        return "fallback_nyse_holiday_rules", None


_CALENDAR_SOURCE, _REAL_CALENDAR = _load_real_calendar()


def calendar_source() -> str:
    return _CALENDAR_SOURCE


def _as_date(value: date | datetime | str | pd.Timestamp) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.to_datetime(value).date()


def _observed_fixed_holiday(year: int, month: int, day: int) -> date:
    holiday = date(year, month, day)
    if holiday.weekday() == calendar.SATURDAY:
        return holiday - timedelta(days=1)
    if holiday.weekday() == calendar.SUNDAY:
        return holiday + timedelta(days=1)
    return holiday


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    current = date(year, month, 1)
    days_until = (weekday - current.weekday()) % 7
    return current + timedelta(days=days_until + 7 * (n - 1))


def _last_weekday(year: int, month: int, weekday: int) -> date:
    current = date(year, month, calendar.monthrange(year, month)[1])
    days_back = (current.weekday() - weekday) % 7
    return current - timedelta(days=days_back)


def _easter_date(year: int) -> date:
    """Gregorian Easter date, used for the NYSE Good Friday closure."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    ll = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * ll) // 451
    month = (h + ll - 7 * m + 114) // 31
    day = ((h + ll - 7 * m + 114) % 31) + 1
    return date(year, month, day)


@lru_cache(maxsize=64)
def _fallback_holidays(year: int) -> frozenset[date]:
    holidays = {
        _observed_fixed_holiday(year, 1, 1),
        _nth_weekday(year, 1, calendar.MONDAY, 3),
        _nth_weekday(year, 2, calendar.MONDAY, 3),
        _easter_date(year) - timedelta(days=2),
        _last_weekday(year, 5, calendar.MONDAY),
        _observed_fixed_holiday(year, 7, 4),
        _nth_weekday(year, 9, calendar.MONDAY, 1),
        _nth_weekday(year, 11, calendar.THURSDAY, 4),
        _observed_fixed_holiday(year, 12, 25),
    }

    if year >= 2022:
        holidays.add(_observed_fixed_holiday(year, 6, 19))

    next_new_year_observed = _observed_fixed_holiday(year + 1, 1, 1)
    if next_new_year_observed.year == year:
        holidays.add(next_new_year_observed)

    holidays.update(EXCEPTIONAL_FULL_DAY_CLOSURES.get(year, frozenset()))

    return frozenset(holidays)


def _real_calendar_days(start: date, end: date) -> list[date] | None:
    if _REAL_CALENDAR is None:
        return None

    if _CALENDAR_SOURCE.startswith("pandas_market_calendars"):
        schedule = _REAL_CALENDAR.schedule(start_date=start.isoformat(), end_date=end.isoformat())
        return [pd.Timestamp(idx).date() for idx in schedule.index]

    if _CALENDAR_SOURCE.startswith("exchange_calendars"):
        sessions = _REAL_CALENDAR.sessions_in_range(pd.Timestamp(start), pd.Timestamp(end))
        return [pd.Timestamp(session).date() for session in sessions]

    return None


def is_trading_day(day: date | datetime | str | pd.Timestamp) -> bool:
    check_date = _as_date(day)
    real_days = _real_calendar_days(check_date, check_date)
    if real_days is not None:
        return check_date in real_days

    return check_date.weekday() < 5 and check_date not in _fallback_holidays(check_date.year)


def previous_trading_day(day: date | datetime | str | pd.Timestamp) -> date:
    current = _as_date(day) - timedelta(days=1)
    while not is_trading_day(current):
        current -= timedelta(days=1)
    return current


def next_trading_day(day: date | datetime | str | pd.Timestamp) -> date:
    current = _as_date(day) + timedelta(days=1)
    while not is_trading_day(current):
        current += timedelta(days=1)
    return current


def trading_days_between(
    start: date | datetime | str | pd.Timestamp,
    end: date | datetime | str | pd.Timestamp,
) -> list[date]:
    start_date = _as_date(start)
    end_date = _as_date(end)
    if end_date < start_date:
        return []

    real_days = _real_calendar_days(start_date, end_date)
    if real_days is not None:
        return real_days

    days: list[date] = []
    current = start_date
    while current <= end_date:
        if is_trading_day(current):
            days.append(current)
        current += timedelta(days=1)
    return days


def _reference_as_new_york_datetime(reference_datetime: datetime | date | None) -> datetime:
    if reference_datetime is None:
        return datetime.now(UTC).astimezone(NEW_YORK_TZ)

    if isinstance(reference_datetime, datetime):
        if reference_datetime.tzinfo is None:
            reference_datetime = reference_datetime.replace(tzinfo=UTC)
        return reference_datetime.astimezone(NEW_YORK_TZ)

    return datetime.combine(reference_datetime, time(23, 59), NEW_YORK_TZ)


def latest_expected_trading_day(reference_datetime: datetime | date | None = None) -> date:
    """Return the latest daily U.S. equities bar that should reasonably exist.

    Daily OHLCV vendors often publish the current session's full bar after the
    close, so a trading day is only considered expected after 18:00 ET.
    """
    reference_et = _reference_as_new_york_datetime(reference_datetime)
    reference_date = reference_et.date()

    if is_trading_day(reference_date):
        if reference_et.time() >= DAILY_BAR_READY_TIME_ET:
            return reference_date
        return previous_trading_day(reference_date)

    return previous_trading_day(reference_date)
