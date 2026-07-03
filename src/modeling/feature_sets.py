from __future__ import annotations

from dataclasses import dataclass


FEATURE_SET_NAMES = [
    "technical_only",
    "technical_plus_sec",
    "technical_plus_earnings",
    "all_model_features",
    "technical_core",
    "technical_pruned",
    "event_features_only",
    "technical_core_plus_events",
    "low_missing_low_correlation",
]

EARNINGS_COLUMNS = {
    "earnings_data_available",
    "earnings_event_present_1s",
    "earnings_event_present_5s",
    "earnings_event_present_20s",
    "earnings_timing_known",
    "latest_eps_surprise_direction",
    "latest_eps_surprise_percent",
    "latest_revenue_surprise_percent",
    "sessions_since_latest_earnings",
}

CATALYST_OR_LLM_COLUMNS = {
    "active_catalyst_count",
    "catalyst_net",
    "catalyst_penalty",
    "catalyst_revision_history_unavailable",
    "catalyst_score",
    "negative_catalyst_count_45d",
    "positive_catalyst_count_45d",
    "published_llm_supported_catalyst",
    "published_llm_supported_count",
    "recent_catalyst_count_45d",
    "llm_max_confidence",
    "llm_max_risk_severity",
    "llm_relevant_count",
    "llm_sufficient_or_limited_count",
}

TECHNICAL_CORE_CANDIDATES = [
    "ret_5d",
    "ret_20d",
    "ret_60d",
    "ret_120d",
    "daily_return",
    "volatility_20d",
    "distance_20d_ma",
    "distance_50d_ma",
    "above_50d_ma",
    "above_200d_ma",
    "relative_strength_20d",
    "relative_strength_60d",
    "relative_strength_score_raw",
    "volume_ratio_20d",
    "volume_anomaly",
    "avg_dollar_volume_20d",
    "avg_dollar_volume_ok",
    "liquidity_score_raw",
    "price_ok",
    "market_regime",
    "market_regime_confidence",
    "regime_qqq_spy_rs_20",
    "regime_iwm_spy_rs_20",
    "regime_vix",
    "regime_vix_elevated",
    "bars_available",
    "has_data",
    "insufficient_history_200d",
    "failed_spy_comparison",
    "missing_volume",
]

TECHNICAL_PRUNING_EXCLUDE = {
    "last_price",
    "current_volume",
    "avg_volume_20d",
    "ma_20",
    "ma_50",
    "ma_200",
    "liquidity_label",
    "data_quality",
}


@dataclass(frozen=True)
class FeatureSetDefinition:
    name: str
    columns: list[str]
    description: str


def _is_sec_column(column: str) -> bool:
    return column.startswith("sec_")


def is_generic_sec_volume_proxy(column: str) -> bool:
    return column.startswith("sec_feature_eligible_")


def _is_earnings_column(column: str) -> bool:
    return column in EARNINGS_COLUMNS or column.startswith("earnings_")


def _is_catalyst_or_llm_column(column: str) -> bool:
    return column in CATALYST_OR_LLM_COLUMNS or column.startswith("llm_") or column.startswith("published_llm_")


def feature_group(column: str) -> str:
    lowered = column.lower()
    if _is_sec_column(lowered):
        return "sec"
    if _is_earnings_column(lowered):
        return "earnings"
    if _is_catalyst_or_llm_column(lowered) or "catalyst" in lowered:
        return "catalyst_llm"
    if lowered.startswith("regime_") or lowered == "market_regime" or lowered == "market_regime_confidence":
        return "regime"
    if "quality" in lowered or "missing" in lowered or "stale" in lowered or "warning" in lowered:
        return "data_quality"
    return "technical"


def technical_feature_columns(feature_columns: list[str]) -> list[str]:
    return [
        column
        for column in feature_columns
        if not _is_sec_column(column)
        and not _is_earnings_column(column)
        and not _is_catalyst_or_llm_column(column)
    ]


def technical_core_columns(feature_columns: list[str]) -> list[str]:
    available = set(feature_columns)
    return [column for column in TECHNICAL_CORE_CANDIDATES if column in available]


def event_feature_columns(feature_columns: list[str]) -> list[str]:
    """Return curated SEC/earnings event features without catalyst/LLM signals."""
    return [
        column
        for column in feature_columns
        if (_is_sec_column(column) and not is_generic_sec_volume_proxy(column))
        or _is_earnings_column(column)
    ]


def static_pruned_feature_columns(feature_columns: list[str]) -> list[str]:
    """Deterministic low-risk pruning usable before a data-dependent audit exists."""
    excluded = set(TECHNICAL_PRUNING_EXCLUDE)
    return [
        column
        for column in feature_columns
        if column not in excluded
        and not is_generic_sec_volume_proxy(column)
        and not column.startswith("llm_")
        and not column.startswith("published_llm_")
        and column not in {"active_catalyst_count", "recent_catalyst_count_45d", "positive_catalyst_count_45d", "negative_catalyst_count_45d"}
    ]


def select_feature_columns(feature_columns: list[str], feature_set_name: str) -> list[str]:
    """Return a deterministic model-feature subset without audit/label columns.

    The input must already come from the dataset's `feature_columns_json`; this
    function only performs ablation selection within the approved model feature
    contract.
    """
    feature_set_name = feature_set_name.strip().lower()
    base = technical_feature_columns(feature_columns)
    if feature_set_name == "technical_only":
        return base
    if feature_set_name == "technical_plus_sec":
        sec = [column for column in feature_columns if _is_sec_column(column)]
        return [column for column in feature_columns if column in set(base).union(sec)]
    if feature_set_name == "technical_plus_earnings":
        earnings = [column for column in feature_columns if _is_earnings_column(column)]
        return [column for column in feature_columns if column in set(base).union(earnings)]
    if feature_set_name == "all_model_features":
        return list(feature_columns)
    if feature_set_name == "technical_core":
        return technical_core_columns(feature_columns)
    if feature_set_name == "technical_pruned":
        return [
            column
            for column in technical_feature_columns(feature_columns)
            if column not in TECHNICAL_PRUNING_EXCLUDE
        ]
    if feature_set_name == "event_features_only":
        return event_feature_columns(feature_columns)
    if feature_set_name == "technical_core_plus_events":
        selected = set(technical_core_columns(feature_columns)).union(event_feature_columns(feature_columns))
        return [column for column in feature_columns if column in selected]
    if feature_set_name == "low_missing_low_correlation":
        return static_pruned_feature_columns(feature_columns)
    raise ValueError(f"Unknown feature set: {feature_set_name}")


def feature_set_definitions(feature_columns: list[str]) -> list[FeatureSetDefinition]:
    descriptions = {
        "technical_only": "Price, trend, volume/liquidity, relative strength, market-regime, and data-quality controls.",
        "technical_plus_sec": "Technical feature set plus curated point-in-time SEC filing features.",
        "technical_plus_earnings": "Technical feature set plus point-in-time historical earnings features.",
        "all_model_features": "All dataset-approved model features, including active catalyst and published LLM-supported fields.",
        "technical_core": "Compact technical/regime/liquidity core that removes raw price and moving-average level fields.",
        "technical_pruned": "Technical-only fields after deterministic removal of raw levels and categorical quality labels.",
        "event_features_only": "Curated SEC and earnings event features only; no catalyst/LLM or technical fields.",
        "technical_core_plus_events": "Compact technical core plus curated SEC and earnings event features.",
        "low_missing_low_correlation": "Static fallback pruning that excludes obvious workflow/sparse catalyst fields; Phase 2D-4 audit artifacts store the data-dependent version.",
    }
    return [
        FeatureSetDefinition(name=name, columns=select_feature_columns(feature_columns, name), description=descriptions[name])
        for name in FEATURE_SET_NAMES
    ]
