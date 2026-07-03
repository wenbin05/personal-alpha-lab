from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


MISSING_CATEGORY = "__MISSING__"


@dataclass
class MatrixPreprocessor:
    numeric_columns: list[str]
    categorical_columns: list[str]
    medians: dict[str, float]
    categories: dict[str, list[str]]
    output_columns: list[str]

    @classmethod
    def fit(cls, frame: pd.DataFrame) -> "MatrixPreprocessor":
        numeric_columns: list[str] = []
        categorical_columns: list[str] = []
        for column in frame.columns:
            series = frame[column]
            if pd.api.types.is_bool_dtype(series) or pd.api.types.is_numeric_dtype(series):
                numeric_columns.append(column)
                continue
            converted = pd.to_numeric(series, errors="coerce")
            non_missing = series.notna().sum()
            if non_missing > 0 and converted.notna().sum() == non_missing:
                numeric_columns.append(column)
            else:
                categorical_columns.append(column)

        medians: dict[str, float] = {}
        for column in numeric_columns:
            numeric = pd.to_numeric(frame[column], errors="coerce")
            median = numeric.median()
            medians[column] = 0.0 if pd.isna(median) else float(median)

        categories: dict[str, list[str]] = {}
        for column in categorical_columns:
            values = frame[column].astype("string").fillna(MISSING_CATEGORY)
            categories[column] = sorted({str(value) for value in values.tolist()})

        output_columns = list(numeric_columns)
        for column in categorical_columns:
            output_columns.extend(f"{column}={category}" for category in categories[column])

        return cls(
            numeric_columns=numeric_columns,
            categorical_columns=categorical_columns,
            medians=medians,
            categories=categories,
            output_columns=output_columns,
        )

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        parts: list[pd.DataFrame] = []
        if self.numeric_columns:
            numeric_data = {
                column: (
                    pd.to_numeric(frame[column], errors="coerce")
                    if column in frame.columns
                    else pd.Series(np.nan, index=frame.index)
                )
                .fillna(self.medians.get(column, 0.0))
                .astype(float)
                for column in self.numeric_columns
            }
            parts.append(pd.DataFrame(numeric_data, index=frame.index))

        for column in self.categorical_columns:
            raw = frame[column] if column in frame.columns else pd.Series(MISSING_CATEGORY, index=frame.index)
            values = raw.astype("string").fillna(MISSING_CATEGORY).map(str)
            category_data = {
                f"{column}={category}": (values == category).astype(float)
                for category in self.categories.get(column, [])
            }
            if category_data:
                parts.append(pd.DataFrame(category_data, index=frame.index))

        if not parts:
            return pd.DataFrame(index=frame.index, columns=self.output_columns).fillna(0.0)
        matrix = pd.concat(parts, axis=1)
        return matrix.reindex(columns=self.output_columns, fill_value=0.0).astype(float)


def fit_transform_matrices(train: pd.DataFrame, evaluation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, MatrixPreprocessor]:
    transformer = MatrixPreprocessor.fit(train)
    return transformer.transform(train), transformer.transform(evaluation), transformer
