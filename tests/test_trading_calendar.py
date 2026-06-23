from __future__ import annotations

from datetime import UTC, date, datetime

from src.utils.trading_calendar import (
    calendar_source,
    is_trading_day,
    latest_expected_trading_day,
    next_trading_day,
    previous_trading_day,
    trading_days_between,
)


def test_previous_trading_day_around_weekend() -> None:
    assert previous_trading_day(date(2024, 1, 7)) == date(2024, 1, 5)


def test_next_trading_day_around_weekend() -> None:
    assert next_trading_day(date(2024, 1, 6)) == date(2024, 1, 8)


def test_known_us_market_holiday() -> None:
    christmas = date(2024, 12, 25)

    assert not is_trading_day(christmas)
    assert previous_trading_day(christmas) == date(2024, 12, 24)
    assert next_trading_day(christmas) == date(2024, 12, 26)


def test_exceptional_full_day_closure_january_9_2025() -> None:
    assert is_trading_day(date(2025, 1, 8))
    assert not is_trading_day(date(2025, 1, 9))
    assert is_trading_day(date(2025, 1, 10))
    assert previous_trading_day(date(2025, 1, 10)) == date(2025, 1, 8)
    assert next_trading_day(date(2025, 1, 8)) == date(2025, 1, 10)


def test_trading_days_between_is_inclusive_and_skips_holidays() -> None:
    days = trading_days_between(date(2024, 12, 24), date(2024, 12, 26))

    assert days == [date(2024, 12, 24), date(2024, 12, 26)]


def test_latest_expected_trading_day_respects_et_daily_bar_cutoff() -> None:
    before_bar_ready = datetime(2024, 1, 8, 16, 0, tzinfo=UTC)
    after_bar_ready = datetime(2024, 1, 9, 0, 30, tzinfo=UTC)

    assert latest_expected_trading_day(before_bar_ready) == date(2024, 1, 5)
    assert latest_expected_trading_day(after_bar_ready) == date(2024, 1, 8)


def test_calendar_source_is_reportable() -> None:
    assert calendar_source()
