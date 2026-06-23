from __future__ import annotations

from datetime import date
from typing import Any

import pandas as pd


def _as_date(value: Any) -> date:
    return pd.to_datetime(value).date()


def chronological_split_dates(
    trading_dates: list[date] | pd.Series,
    train_end: date,
    validation_end: date,
    test_end: date | None = None,
    gap_sessions: int = 0,
) -> dict[str, list[date]]:
    """Create deterministic non-overlapping chronological splits with optional session gaps."""
    dates = sorted({_as_date(value) for value in list(trading_dates)})
    gap_sessions = max(0, int(gap_sessions))
    train = [value for value in dates if value <= train_end]

    after_train = [value for value in dates if value > train_end]
    validation_pool = after_train[gap_sessions:]
    validation = [value for value in validation_pool if value <= validation_end]

    after_validation = [value for value in dates if value > validation_end]
    test_pool = after_validation[gap_sessions:]
    test = [value for value in test_pool if test_end is None or value <= test_end]
    return {"train": train, "validation": validation, "test": test}


def assign_chronological_splits(
    frame: pd.DataFrame,
    train_end: date,
    validation_end: date,
    test_end: date | None = None,
    gap_sessions: int = 0,
    date_column: str = "trading_date",
) -> pd.DataFrame:
    if frame is None or frame.empty or date_column not in frame.columns:
        result = pd.DataFrame() if frame is None else frame.copy()
        result["split"] = pd.Series(dtype="string")
        return result

    result = frame.copy()
    split_dates = chronological_split_dates(
        result[date_column],
        train_end=train_end,
        validation_end=validation_end,
        test_end=test_end,
        gap_sessions=gap_sessions,
    )
    mapping = {value: split for split, values in split_dates.items() for value in values}
    result["split"] = result[date_column].map(lambda value: mapping.get(_as_date(value), "gap"))
    return result

