from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class ScannerSettings(BaseModel):
    max_tickers: int = 35
    min_price: float = 5.0
    min_avg_dollar_volume: float = 10_000_000
    lookback_period: str = "2y"


class RiskSettings(BaseModel):
    default_portfolio_size: float = 100_000
    default_risk_per_trade_pct: float = 0.75
    max_position_pct: float = 10
    max_open_positions: int = 8
    max_sector_exposure_pct: float = 35


class BacktestSettings(BaseModel):
    initial_capital: float = 100_000
    default_slippage_bps: float = 10
    commission_per_trade: float = 0


class LLMSettings(BaseModel):
    provider: str = ""
    model: str = ""
    max_input_chars: int = 12_000
    timeout_seconds: int = 60


class Settings(BaseModel):
    database_path: str = "data/alpha_lab.db"
    market_data_provider: str = "yfinance"
    universe_path: str = "config/universe.csv"
    default_history_period: str = "2y"
    default_benchmark: str = "SPY"
    market_timezone: str = "America/New_York"
    user_timezone: str = "Asia/Singapore"
    scanner: ScannerSettings = Field(default_factory=ScannerSettings)
    risk: RiskSettings = Field(default_factory=RiskSettings)
    backtest: BacktestSettings = Field(default_factory=BacktestSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)

    @property
    def database_file(self) -> Path:
        return resolve_project_path(self.database_path)

    @property
    def universe_file(self) -> Path:
        return resolve_project_path(self.universe_path)


def resolve_project_path(path: str | Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def load_settings(path: str | Path | None = None) -> Settings:
    """Load YAML settings and allow selected values to be overridden by .env."""
    load_dotenv(PROJECT_ROOT / ".env")
    settings_path = resolve_project_path(path or "config/settings.yaml")
    raw: dict[str, Any] = {}
    if settings_path.exists():
        with settings_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

    if os.getenv("DATABASE_PATH"):
        raw["database_path"] = os.environ["DATABASE_PATH"]
    if os.getenv("MARKET_DATA_PROVIDER"):
        raw["market_data_provider"] = os.environ["MARKET_DATA_PROVIDER"]

    llm_raw = dict(raw.get("llm") or {})
    if os.getenv("LLM_PROVIDER") is not None:
        llm_raw["provider"] = os.environ["LLM_PROVIDER"]
    if os.getenv("LLM_MODEL") is not None:
        llm_raw["model"] = os.environ["LLM_MODEL"]
    if os.getenv("LLM_MAX_INPUT_CHARS") is not None:
        llm_raw["max_input_chars"] = os.environ["LLM_MAX_INPUT_CHARS"]
    if os.getenv("LLM_TIMEOUT_SECONDS") is not None:
        llm_raw["timeout_seconds"] = os.environ["LLM_TIMEOUT_SECONDS"]
    if llm_raw:
        raw["llm"] = llm_raw

    return Settings(**raw)
