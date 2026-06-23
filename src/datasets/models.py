from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any


@dataclass
class FeatureSnapshot:
    ticker: str
    trading_date: date
    as_of_timestamp: datetime
    feature_version: str
    market_regime: dict[str, Any]
    technical: dict[str, Any]
    relative_strength: dict[str, Any]
    volume_liquidity: dict[str, Any]
    catalyst: dict[str, Any]
    llm_supported: dict[str, Any]
    data_quality: dict[str, Any]
    features: dict[str, Any]
    snapshot_id: int | None = None
    dataset_id: int | None = None
    created_at: datetime | None = None


@dataclass
class OutcomeLabel:
    snapshot_id: int | None
    ticker: str
    entry_date: date
    horizon: str
    entry_price: float
    exit_date: date
    exit_price: float
    forward_return: float
    spy_forward_return: float | None
    excess_return: float | None
    label_available_at: datetime


@dataclass
class DatasetBuild:
    version: str
    build_timestamp: datetime
    requested_start_date: date
    requested_end_date: date
    ticker_universe: list[str]
    feature_columns: list[str]
    label_definitions: dict[str, Any]
    row_count: int
    data_hash: str
    audit_columns: list[str] = field(default_factory=list)
    label_columns: list[str] = field(default_factory=list)
    identifier_columns: list[str] = field(default_factory=list)
    metadata_columns: list[str] = field(default_factory=list)
    feature_manifest: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    export_path: str | None = None
    dataset_id: int | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
