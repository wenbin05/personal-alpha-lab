from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class ModelSplit:
    fold_name: str
    split_name: str
    train_dates: list[date]
    eval_dates: list[date]
    purge_sessions: int
    embargo_sessions: int

    @property
    def train_start(self) -> date | None:
        return min(self.train_dates) if self.train_dates else None

    @property
    def train_end(self) -> date | None:
        return max(self.train_dates) if self.train_dates else None

    @property
    def eval_start(self) -> date | None:
        return min(self.eval_dates) if self.eval_dates else None

    @property
    def eval_end(self) -> date | None:
        return max(self.eval_dates) if self.eval_dates else None


def horizon_sessions_from_label(label_column: str) -> int:
    if "1_session" in label_column:
        return 1
    if "5_session" in label_column:
        return 5
    if "20_session" in label_column:
        return 20
    raise ValueError(f"Cannot infer horizon sessions from {label_column!r}")


def horizon_from_label(label_column: str) -> str:
    for horizon in ("1_session", "5_session", "20_session"):
        if horizon in label_column:
            return horizon
    raise ValueError(f"Cannot infer horizon from {label_column!r}")


def make_walk_forward_splits(
    metadata: pd.DataFrame,
    label_values: pd.Series,
    horizon_sessions: int,
    n_folds: int = 3,
    final_test_fraction: float = 0.20,
    purge_sessions: int | None = None,
    embargo_sessions: int | None = None,
) -> list[ModelSplit]:
    """Create expanding-window validation folds plus one untouched final test.

    The purge/embargo gap is enforced before each validation/test window so
    training rows cannot overlap with the target horizon being evaluated.
    """
    if "trading_date" not in metadata.columns:
        raise ValueError("metadata must contain trading_date")
    valid_dates = pd.to_datetime(metadata.loc[label_values.notna(), "trading_date"]).dt.date
    dates = sorted(set(valid_dates.tolist()))
    if len(dates) < 20:
        raise ValueError("Not enough labeled dates for chronological modeling splits.")

    purge_sessions = horizon_sessions if purge_sessions is None else max(0, int(purge_sessions))
    embargo_sessions = horizon_sessions if embargo_sessions is None else max(0, int(embargo_sessions))
    gap_sessions = max(purge_sessions, embargo_sessions)
    n_folds = max(1, int(n_folds))
    final_test_count = max(1, int(round(len(dates) * float(final_test_fraction))))
    final_test_count = min(final_test_count, max(1, len(dates) // 3))
    test_start_idx = len(dates) - final_test_count
    development_dates = dates[:test_start_idx]
    test_dates = dates[test_start_idx:]

    folds: list[ModelSplit] = []
    fold_size = max(1, len(development_dates) // (n_folds + 1))
    for fold_idx in range(n_folds):
        val_start_idx = fold_size * (fold_idx + 1)
        val_end_idx = len(development_dates) if fold_idx == n_folds - 1 else min(len(development_dates), val_start_idx + fold_size)
        train_end_idx = max(0, val_start_idx - gap_sessions)
        train_dates = development_dates[:train_end_idx]
        eval_dates = development_dates[val_start_idx:val_end_idx]
        if not train_dates or not eval_dates:
            continue
        folds.append(
            ModelSplit(
                fold_name=f"fold_{fold_idx + 1}",
                split_name="validation",
                train_dates=train_dates,
                eval_dates=eval_dates,
                purge_sessions=purge_sessions,
                embargo_sessions=embargo_sessions,
            )
        )

    final_train_end = max(0, test_start_idx - gap_sessions)
    final_train_dates = dates[:final_train_end]
    if not final_train_dates or not test_dates:
        raise ValueError("Not enough labeled dates for final test split after purge/embargo.")
    folds.append(
        ModelSplit(
            fold_name="final_test",
            split_name="test",
            train_dates=final_train_dates,
            eval_dates=test_dates,
            purge_sessions=purge_sessions,
            embargo_sessions=embargo_sessions,
        )
    )
    return folds


def split_config_dict(
    label_column: str,
    n_folds: int,
    final_test_fraction: float,
    purge_sessions: int,
    embargo_sessions: int,
) -> dict[str, Any]:
    return {
        "split_type": "expanding_walk_forward",
        "label_column": label_column,
        "horizon_sessions": horizon_sessions_from_label(label_column),
        "n_folds": int(n_folds),
        "final_test_fraction": float(final_test_fraction),
        "purge_sessions": int(purge_sessions),
        "embargo_sessions": int(embargo_sessions),
        "final_test_policy": "untouched_until_final_evaluation",
    }

