from __future__ import annotations

import json
import math
import re
from bisect import bisect_left
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.annotations.repository import list_annotations
from src.annotations.source_quality import SOURCE_QUALITY_CATEGORIES, enrich_quality_frame, quality_distribution
from src.data import storage
from src.datasets.training_loader import TrainingDataset, load_training_dataset
from src.modeling.feature_sets import technical_core_columns
from src.modeling.splits import make_walk_forward_splits
from src.modeling.targets import RAW_TARGET_5_SESSION, get_target_definition, precompute_target_series, target_metadata
from src.utils.trading_calendar import trading_days_between


ANNOTATION_FEATURE_VERSION = "research_annotation_features_v2"
ANNOTATION_FEATURE_COLUMNS = [
    "recent_positive_annotation_count_20s",
    "recent_negative_annotation_count_20s",
    "days_since_latest_positive_annotation",
    "days_since_latest_negative_annotation",
    "max_recent_annotation_strength",
    "weighted_recent_annotation_sentiment",
    "annotation_coverage_available",
]
HIGH_SIGNAL_ANNOTATION_FEATURE_COLUMNS = [
    "high_signal_annotation_count_20s",
    "high_signal_positive_count_20s",
    "high_signal_negative_mixed_count_20s",
    "high_signal_max_recent_strength",
    "high_signal_weighted_recent_sentiment",
    "high_signal_coverage_available",
]
NON_SEC_ANNOTATION_FEATURE_COLUMNS = [
    "non_sec_event_count_20s",
    "non_sec_positive_count_20s",
    "non_sec_negative_mixed_count_20s",
    "non_sec_max_recent_strength",
    "non_sec_weighted_recent_sentiment",
]
NEGATIVE_MIXED_ANNOTATION_FEATURE_COLUMNS = [
    "negative_mixed_event_count_20s",
    "days_since_latest_negative_mixed_annotation",
    "negative_mixed_max_recent_strength",
    "negative_mixed_weighted_recent_sentiment",
]
HIGH_CONFIDENCE_ANNOTATION_FEATURE_COLUMNS = [
    "high_confidence_event_count_20s",
    "high_confidence_max_recent_strength",
    "high_confidence_weighted_recent_sentiment",
]
HIGH_QUALITY_ANNOTATION_FEATURE_COLUMNS = [
    "high_quality_annotation_count_20s",
    "high_quality_positive_count_20s",
    "high_quality_negative_mixed_count_20s",
    "high_quality_max_recent_strength",
    "high_quality_weighted_recent_sentiment",
    "high_quality_coverage_available",
]
MATERIAL_NEGATIVE_MIXED_ANNOTATION_FEATURE_COLUMNS = [
    "material_negative_mixed_event_count_20s",
    "material_negative_mixed_regulator_index_count_20s",
    "days_since_latest_material_negative_mixed_event",
    "material_negative_mixed_max_recent_strength",
    "material_negative_mixed_weighted_recent_sentiment",
    "material_negative_mixed_coverage_available",
]
COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS = [
    "annotation_event_active_0_5s",
    "annotation_event_active_6_20s",
    "annotation_event_active_21_60s",
    "days_since_latest_material_event",
    "days_since_latest_negative_mixed_event",
    "exponential_decay_sentiment_20s",
    "exponential_decay_sentiment_60s",
    "material_event_decay_60s",
]
COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS = [
    "source_quality_weighted_sentiment_decay",
    "informativeness_weighted_event_decay",
    "quality_weighted_event_decay_60s",
    "negative_mixed_quality_weighted_decay_60s",
    "official_regulator_index_weighted_event_decay_60s",
    "routine_downweighted_event_decay_60s",
]
COMPACT_ANNOTATION_FEATURE_COLUMNS = [
    *COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS,
    *COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS,
]
QUALITY_ANNOTATION_FEATURE_COLUMNS = [
    *HIGH_SIGNAL_ANNOTATION_FEATURE_COLUMNS,
    *NON_SEC_ANNOTATION_FEATURE_COLUMNS,
    *NEGATIVE_MIXED_ANNOTATION_FEATURE_COLUMNS,
    *HIGH_CONFIDENCE_ANNOTATION_FEATURE_COLUMNS,
    *HIGH_QUALITY_ANNOTATION_FEATURE_COLUMNS,
    *MATERIAL_NEGATIVE_MIXED_ANNOTATION_FEATURE_COLUMNS,
    *COMPACT_ANNOTATION_FEATURE_COLUMNS,
]
ALL_ANNOTATION_FEATURE_COLUMNS = [*ANNOTATION_FEATURE_COLUMNS, *QUALITY_ANNOTATION_FEATURE_COLUMNS]
ANNOTATION_FEATURE_SET_NAMES = [
    "technical_core",
    "annotation_features_only",
    "technical_core_plus_annotations",
    "annotation_high_signal_only",
    "technical_core_plus_high_signal_annotations",
    "technical_core_plus_non_sec_events",
    "technical_core_plus_negative_mixed_events",
    "technical_core_plus_negative_mixed_material_events",
    "technical_core_plus_high_confidence_events",
    "technical_core_plus_high_quality_annotations",
    "annotation_compact_decay",
    "annotation_compact_weighted",
    "technical_core_plus_annotation_compact_decay",
    "technical_core_plus_annotation_compact_weighted",
]
SENTIMENT_WEIGHTS = {
    "positive": 1.0,
    "negative": -1.0,
    "mixed": 0.0,
    "neutral": 0.0,
    "unknown": 0.0,
}
SOURCE_QUALITY_WEIGHTS = {
    "official_company": 1.0,
    "regulator": 1.1,
    "exchange_or_index_provider": 1.0,
    "sec_archive": 0.8,
    "credible_news": 0.7,
    "manual_note": 0.5,
    "unknown": 0.3,
}
INFORMATIVENESS_WEIGHTS = {
    "material_high": 1.0,
    "material_medium": 0.7,
    "routine_low": 0.2,
    "duplicate_theme": 0.1,
    "low_specificity": 0.1,
    "unknown": 0.3,
}


@dataclass(frozen=True)
class AnnotationArtifactInfo:
    path: Path
    dataset_id: int | None
    created_at: str
    summary: str


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str, indent=2)


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _clean_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _clean_json(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [_clean_json(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return _clean_json(value.item())
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    try:
        if value is not None and not isinstance(value, (str, bytes, list, dict, tuple)) and pd.isna(value):
            return None
    except Exception:
        pass
    return value


def _as_date(value: Any) -> date:
    return pd.to_datetime(value).date()


def _as_utc(value: Any) -> pd.Timestamp:
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        return pd.Timestamp.min.tz_localize(UTC)
    return parsed


def _trading_sessions_between(start: date, end: date) -> int | None:
    if end < start:
        return None
    days = trading_days_between(start, end)
    if not days:
        return None
    return max(0, len(days) - 1)


def _normalized_theme_key(value: Any) -> str:
    text = re.sub(r"\baccession\s+\d+\b", " ", str(value or "").lower())
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _apply_annotation_quality_flags(frame: pd.DataFrame) -> pd.DataFrame:
    """Classify research annotations for model-only feature filtering.

    These flags are deterministic and intentionally conservative. They are not
    written to active catalysts and have no scanner scoring effect.
    """
    if frame.empty:
        return frame.copy()
    output = frame.copy()
    output["event_type"] = output["event_type"].fillna("other").astype(str).str.lower()
    output["sentiment_label"] = output["sentiment_label"].fillna("unknown").astype(str).str.lower()
    for column in ["source_url", "evidence_text", "source", "title", "summary"]:
        if column not in output.columns:
            output[column] = ""
        output[column] = output[column].fillna("").astype(str)

    strength = pd.to_numeric(output["strength"], errors="coerce").fillna(0).clip(lower=0, upper=10)
    confidence = pd.to_numeric(output["confidence"], errors="coerce").fillna(0).clip(lower=0, upper=1)
    has_source_url = output["source_url"].str.strip().ne("")
    has_evidence = output["evidence_text"].str.len().ge(20)
    high_confidence_source = confidence.ge(0.70) & has_source_url & has_evidence

    routine_sec = (
        output["event_type"].eq("sec_filing")
        & output["sentiment_label"].isin(["neutral", "unknown"])
        & strength.le(4)
    )
    low_specificity_neutral = (
        output["sentiment_label"].isin(["neutral", "unknown"])
        & strength.le(4)
        & (
            output["summary"].str.len().lt(80)
            | output["title"].str.lower().str.contains(r"\broutine\b|\bmetadata\b|\bboilerplate\b", regex=True, na=False)
        )
    )
    material_event_type = output["event_type"].isin(
        [
            "corporate_action",
            "product_launch",
            "guidance_update",
            "legal_regulatory",
            "macro_sensitive",
            "management_change",
            "partnership",
            "financing",
            "news",
            "earnings",
        ]
    )
    negative_or_risk = output["sentiment_label"].isin(["negative", "mixed"]) | output["event_type"].isin(["legal_regulatory", "financing"])
    non_sec_material = output["event_type"].ne("sec_filing") & material_event_type & high_confidence_source & strength.ge(5)
    routine_low_signal = routine_sec | low_specificity_neutral

    theme_key = output["title"].map(_normalized_theme_key)
    theme_cluster_size = output.assign(_theme_key=theme_key).groupby(
        ["ticker", "event_type", "sentiment_label", "_theme_key"], dropna=False
    )["_theme_key"].transform("size")

    output["high_confidence_source"] = high_confidence_source.astype(bool)
    output["routine_low_signal"] = routine_low_signal.astype(bool)
    output["non_sec_material_event"] = non_sec_material.astype(bool)
    output["negative_or_risk_event"] = negative_or_risk.astype(bool)
    output["low_specificity_neutral"] = low_specificity_neutral.astype(bool)
    output["duplicate_theme_cluster"] = theme_cluster_size.gt(1).astype(bool)
    output["high_signal_event"] = (
        high_confidence_source
        & ~routine_low_signal
        & (non_sec_material | negative_or_risk | strength.ge(6) | output["sentiment_label"].isin(["positive", "negative", "mixed"]))
    ).astype(bool)
    return enrich_quality_frame(output)


def _annotation_rows_for_features(db_path: str | Path) -> pd.DataFrame:
    frame = list_annotations(db_path, limit=None)
    if frame.empty:
        return pd.DataFrame(
            columns=[
                "ticker",
                "event_date_parsed",
                "available_at_parsed",
                "sentiment_label",
                "strength",
                "confidence",
            ]
        )
    output = frame.copy()
    output["ticker"] = output["ticker"].astype(str).str.upper()
    output["event_date_parsed"] = pd.to_datetime(output["event_date"], errors="coerce").dt.date
    output["available_at_parsed"] = pd.to_datetime(output["available_at"], utc=True, errors="coerce")
    output["sentiment_label"] = output["sentiment_label"].fillna("unknown").astype(str).str.lower()
    output["strength"] = pd.to_numeric(output["strength"], errors="coerce").fillna(0).clip(lower=0, upper=10).astype(float)
    output["confidence"] = pd.to_numeric(output["confidence"], errors="coerce").fillna(0).clip(lower=0, upper=1).astype(float)
    output = output.dropna(subset=["event_date_parsed", "available_at_parsed"])
    return _apply_annotation_quality_flags(output)


def _empty_annotation_feature_row() -> dict[str, float]:
    row = {column: 0.0 for column in ALL_ANNOTATION_FEATURE_COLUMNS}
    for column in [
        "days_since_latest_positive_annotation",
        "days_since_latest_negative_annotation",
        "days_since_latest_negative_mixed_annotation",
        "days_since_latest_material_negative_mixed_event",
        "days_since_latest_material_event",
        "days_since_latest_negative_mixed_event",
    ]:
        row[column] = np.nan
    return row


def _weighted_sentiment(recent: pd.DataFrame) -> float:
    if recent.empty:
        return 0.0
    recent_strength = pd.to_numeric(recent.get("strength"), errors="coerce").fillna(0).astype(float)
    weights = recent["sentiment_label"].map(SENTIMENT_WEIGHTS).fillna(0.0)
    weighted_base = (recent_strength * pd.to_numeric(recent.get("confidence"), errors="coerce").fillna(0)).astype(float)
    denominator = float(weighted_base.abs().sum()) if not weighted_base.empty else 0.0
    return float((weights * weighted_base).sum() / denominator) if denominator > 0 else 0.0


def _max_strength(recent: pd.DataFrame) -> float:
    if recent.empty:
        return 0.0
    return float(pd.to_numeric(recent.get("strength"), errors="coerce").fillna(0).max())


def _decay_weights(frame: pd.DataFrame, half_life_sessions: float) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    sessions = pd.to_numeric(frame.get("sessions_since"), errors="coerce").fillna(999.0).clip(lower=0)
    return pd.Series(np.exp(-np.log(2.0) * sessions / float(half_life_sessions)), index=frame.index).astype(float)


def _source_quality_weights(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.get("source_quality", pd.Series("unknown", index=frame.index)).map(SOURCE_QUALITY_WEIGHTS).fillna(0.3).astype(float)


def _informativeness_weights(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return frame.get("informativeness", pd.Series("unknown", index=frame.index)).map(INFORMATIVENESS_WEIGHTS).fillna(0.3).astype(float)


def _quality_weights(frame: pd.DataFrame) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)
    return (_source_quality_weights(frame) * _informativeness_weights(frame)).astype(float)


def _decayed_sentiment(frame: pd.DataFrame, half_life_sessions: float, weight_kind: str = "none") -> float:
    if frame.empty:
        return 0.0
    sentiment = frame["sentiment_label"].map(SENTIMENT_WEIGHTS).fillna(0.0).astype(float)
    strength = pd.to_numeric(frame.get("strength"), errors="coerce").fillna(0.0).clip(lower=0, upper=10).astype(float) / 10.0
    confidence = pd.to_numeric(frame.get("confidence"), errors="coerce").fillna(0.0).clip(lower=0, upper=1).astype(float)
    weights = _decay_weights(frame, half_life_sessions) * strength * confidence
    if weight_kind == "source_quality":
        weights = weights * _source_quality_weights(frame)
    elif weight_kind == "informativeness":
        weights = weights * _informativeness_weights(frame)
    elif weight_kind == "quality":
        weights = weights * _quality_weights(frame)
    denominator = float(weights.abs().sum())
    return float((sentiment * weights).sum() / denominator) if denominator > 0 else 0.0


def _decayed_event_score(frame: pd.DataFrame, half_life_sessions: float, weight_kind: str = "none") -> float:
    if frame.empty:
        return 0.0
    weights = _decay_weights(frame, half_life_sessions)
    if weight_kind == "source_quality":
        weights = weights * _source_quality_weights(frame)
    elif weight_kind == "informativeness":
        weights = weights * _informativeness_weights(frame)
    elif weight_kind == "quality":
        weights = weights * _quality_weights(frame)
    return float(weights.sum())


def derive_annotation_features(db_path: str | Path, metadata: pd.DataFrame) -> pd.DataFrame:
    """Derive point-in-time research-only annotation features for dataset rows.

    An annotation can activate only after both its `available_at` timestamp and
    its event date are known at the snapshot. This prevents future event labels
    from becoming historical model inputs.
    """
    annotations = _annotation_rows_for_features(db_path)
    if annotations.empty:
        return pd.DataFrame(0.0, index=metadata.index, columns=ALL_ANNOTATION_FEATURE_COLUMNS)

    by_ticker = {
        ticker: group.sort_values(["available_at_parsed", "event_date_parsed", "annotation_id" if "annotation_id" in group.columns else "ticker"]).copy()
        for ticker, group in annotations.groupby("ticker", dropna=False)
    }
    rows: list[dict[str, float]] = []
    for _, row in metadata.iterrows():
        ticker = str(row.get("ticker") or "").upper()
        snapshot_date = _as_date(row.get("trading_date"))
        as_of = _as_utc(row.get("as_of_timestamp") or snapshot_date)
        ticker_events = by_ticker.get(ticker)
        if ticker_events is None or ticker_events.empty:
            rows.append(_empty_annotation_feature_row())
            continue

        available = ticker_events[
            (ticker_events["available_at_parsed"].le(as_of))
            & (ticker_events["event_date_parsed"].map(lambda value: value <= snapshot_date))
        ].copy()
        if available.empty:
            rows.append(_empty_annotation_feature_row())
            continue

        available["sessions_since"] = available["event_date_parsed"].map(lambda value: _trading_sessions_between(value, snapshot_date))
        available = available.dropna(subset=["sessions_since"])
        recent = available[available["sessions_since"].le(20)].copy()
        recent_60 = available[available["sessions_since"].le(60)].copy()
        recent_0_5 = available[available["sessions_since"].between(0, 5, inclusive="both")].copy()
        recent_6_20 = available[available["sessions_since"].between(6, 20, inclusive="both")].copy()
        recent_21_60 = available[available["sessions_since"].between(21, 60, inclusive="both")].copy()
        positives = available[available["sentiment_label"].eq("positive")]
        negatives = available[available["sentiment_label"].eq("negative")]
        recent_positive = recent[recent["sentiment_label"].eq("positive")]
        recent_negative = recent[recent["sentiment_label"].eq("negative")]
        high_signal_recent = recent[recent["high_signal_event"].astype(bool)]
        non_sec_recent = recent[recent["event_type"].ne("sec_filing")]
        negative_mixed_recent = recent[recent["sentiment_label"].isin(["negative", "mixed"]) | recent["negative_or_risk_event"].astype(bool)]
        high_confidence_recent = recent[recent["high_confidence_source"].astype(bool)]
        high_quality_recent = recent[
            recent["source_quality"].isin(["official_company", "regulator", "exchange_or_index_provider", "credible_news"])
            & recent["informativeness"].isin(["material_high", "material_medium"])
            & recent["high_confidence_source"].astype(bool)
        ]
        material_negative_mixed_recent = recent[
            recent["source_quality"].isin(["official_company", "regulator", "exchange_or_index_provider", "credible_news", "sec_archive"])
            & recent["informativeness"].isin(["material_high", "material_medium"])
            & (recent["sentiment_label"].isin(["negative", "mixed"]) | recent["negative_or_risk_event"].astype(bool))
        ]
        negative_mixed_available = available[
            available["sentiment_label"].isin(["negative", "mixed"]) | available["negative_or_risk_event"].astype(bool)
        ]
        material_negative_mixed_available = available[
            available["source_quality"].isin(["official_company", "regulator", "exchange_or_index_provider", "credible_news", "sec_archive"])
            & available["informativeness"].isin(["material_high", "material_medium"])
            & (available["sentiment_label"].isin(["negative", "mixed"]) | available["negative_or_risk_event"].astype(bool))
        ]
        material_available = available[
            available["informativeness"].isin(["material_high", "material_medium"])
            & ~available["routine_low_signal"].astype(bool)
            & ~available["low_specificity_neutral"].astype(bool)
        ]
        material_recent_60 = recent_60[
            recent_60["informativeness"].isin(["material_high", "material_medium"])
            & ~recent_60["routine_low_signal"].astype(bool)
            & ~recent_60["low_specificity_neutral"].astype(bool)
        ]
        negative_mixed_recent_60 = recent_60[
            recent_60["sentiment_label"].isin(["negative", "mixed"]) | recent_60["negative_or_risk_event"].astype(bool)
        ]
        official_regulator_index_recent_60 = recent_60[
            recent_60["source_quality"].isin(["official_company", "regulator", "exchange_or_index_provider"])
        ]
        routine_recent_60 = recent_60[
            recent_60["routine_low_signal"].astype(bool)
            | recent_60["low_specificity_neutral"].astype(bool)
            | recent_60["informativeness"].isin(["routine_low", "duplicate_theme", "low_specificity"])
        ]

        rows.append(
            {
                "recent_positive_annotation_count_20s": float(len(recent_positive)),
                "recent_negative_annotation_count_20s": float(len(recent_negative)),
                "days_since_latest_positive_annotation": float(positives["sessions_since"].min()) if not positives.empty else np.nan,
                "days_since_latest_negative_annotation": float(negatives["sessions_since"].min()) if not negatives.empty else np.nan,
                "max_recent_annotation_strength": _max_strength(recent),
                "weighted_recent_annotation_sentiment": _weighted_sentiment(recent),
                "annotation_coverage_available": 1.0,
                "high_signal_annotation_count_20s": float(len(high_signal_recent)),
                "high_signal_positive_count_20s": float(high_signal_recent["sentiment_label"].eq("positive").sum()) if not high_signal_recent.empty else 0.0,
                "high_signal_negative_mixed_count_20s": float(high_signal_recent["sentiment_label"].isin(["negative", "mixed"]).sum()) if not high_signal_recent.empty else 0.0,
                "high_signal_max_recent_strength": _max_strength(high_signal_recent),
                "high_signal_weighted_recent_sentiment": _weighted_sentiment(high_signal_recent),
                "high_signal_coverage_available": float(not high_signal_recent.empty),
                "non_sec_event_count_20s": float(len(non_sec_recent)),
                "non_sec_positive_count_20s": float(non_sec_recent["sentiment_label"].eq("positive").sum()) if not non_sec_recent.empty else 0.0,
                "non_sec_negative_mixed_count_20s": float(non_sec_recent["sentiment_label"].isin(["negative", "mixed"]).sum()) if not non_sec_recent.empty else 0.0,
                "non_sec_max_recent_strength": _max_strength(non_sec_recent),
                "non_sec_weighted_recent_sentiment": _weighted_sentiment(non_sec_recent),
                "negative_mixed_event_count_20s": float(len(negative_mixed_recent)),
                "days_since_latest_negative_mixed_annotation": (
                    float(negative_mixed_available["sessions_since"].min()) if not negative_mixed_available.empty else np.nan
                ),
                "negative_mixed_max_recent_strength": _max_strength(negative_mixed_recent),
                "negative_mixed_weighted_recent_sentiment": _weighted_sentiment(negative_mixed_recent),
                "high_confidence_event_count_20s": float(len(high_confidence_recent)),
                "high_confidence_max_recent_strength": _max_strength(high_confidence_recent),
                "high_confidence_weighted_recent_sentiment": _weighted_sentiment(high_confidence_recent),
                "high_quality_annotation_count_20s": float(len(high_quality_recent)),
                "high_quality_positive_count_20s": float(high_quality_recent["sentiment_label"].eq("positive").sum()) if not high_quality_recent.empty else 0.0,
                "high_quality_negative_mixed_count_20s": float(high_quality_recent["sentiment_label"].isin(["negative", "mixed"]).sum()) if not high_quality_recent.empty else 0.0,
                "high_quality_max_recent_strength": _max_strength(high_quality_recent),
                "high_quality_weighted_recent_sentiment": _weighted_sentiment(high_quality_recent),
                "high_quality_coverage_available": float(not high_quality_recent.empty),
                "material_negative_mixed_event_count_20s": float(len(material_negative_mixed_recent)),
                "material_negative_mixed_regulator_index_count_20s": (
                    float(material_negative_mixed_recent["source_quality"].isin(["regulator", "exchange_or_index_provider"]).sum())
                    if not material_negative_mixed_recent.empty
                    else 0.0
                ),
                "days_since_latest_material_negative_mixed_event": (
                    float(material_negative_mixed_available["sessions_since"].min()) if not material_negative_mixed_available.empty else np.nan
                ),
                "material_negative_mixed_max_recent_strength": _max_strength(material_negative_mixed_recent),
                "material_negative_mixed_weighted_recent_sentiment": _weighted_sentiment(material_negative_mixed_recent),
                "material_negative_mixed_coverage_available": float(not material_negative_mixed_recent.empty),
                "annotation_event_active_0_5s": float(len(recent_0_5)),
                "annotation_event_active_6_20s": float(len(recent_6_20)),
                "annotation_event_active_21_60s": float(len(recent_21_60)),
                "days_since_latest_material_event": (
                    float(material_available["sessions_since"].min()) if not material_available.empty else np.nan
                ),
                "days_since_latest_negative_mixed_event": (
                    float(negative_mixed_available["sessions_since"].min()) if not negative_mixed_available.empty else np.nan
                ),
                "exponential_decay_sentiment_20s": _decayed_sentiment(recent, half_life_sessions=5.0),
                "exponential_decay_sentiment_60s": _decayed_sentiment(recent_60, half_life_sessions=20.0),
                "material_event_decay_60s": _decayed_event_score(material_recent_60, half_life_sessions=20.0, weight_kind="informativeness"),
                "source_quality_weighted_sentiment_decay": _decayed_sentiment(
                    recent_60, half_life_sessions=20.0, weight_kind="source_quality"
                ),
                "informativeness_weighted_event_decay": _decayed_event_score(
                    recent_60, half_life_sessions=20.0, weight_kind="informativeness"
                ),
                "quality_weighted_event_decay_60s": _decayed_event_score(
                    recent_60, half_life_sessions=20.0, weight_kind="quality"
                ),
                "negative_mixed_quality_weighted_decay_60s": _decayed_event_score(
                    negative_mixed_recent_60, half_life_sessions=20.0, weight_kind="quality"
                ),
                "official_regulator_index_weighted_event_decay_60s": _decayed_event_score(
                    official_regulator_index_recent_60, half_life_sessions=20.0, weight_kind="source_quality"
                ),
                "routine_downweighted_event_decay_60s": _decayed_event_score(
                    routine_recent_60, half_life_sessions=20.0, weight_kind="informativeness"
                ),
            }
        )

    return pd.DataFrame(rows, index=metadata.index, columns=ALL_ANNOTATION_FEATURE_COLUMNS).replace([np.inf, -np.inf], np.nan)


def build_annotation_model_frame(training: TrainingDataset, db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    derived = derive_annotation_features(db_path, training.metadata)
    technical_core = [column for column in technical_core_columns(training.feature_columns) if column in training.X.columns]
    model_frame = pd.concat([training.X[technical_core].copy(), derived], axis=1)
    feature_sets = {
        "technical_core": technical_core,
        "annotation_features_only": list(ANNOTATION_FEATURE_COLUMNS),
        "technical_core_plus_annotations": [*technical_core, *ANNOTATION_FEATURE_COLUMNS],
        "annotation_high_signal_only": list(HIGH_SIGNAL_ANNOTATION_FEATURE_COLUMNS),
        "technical_core_plus_high_signal_annotations": [*technical_core, *HIGH_SIGNAL_ANNOTATION_FEATURE_COLUMNS],
        "technical_core_plus_non_sec_events": [*technical_core, *NON_SEC_ANNOTATION_FEATURE_COLUMNS],
        "technical_core_plus_negative_mixed_events": [*technical_core, *NEGATIVE_MIXED_ANNOTATION_FEATURE_COLUMNS],
        "technical_core_plus_negative_mixed_material_events": [*technical_core, *MATERIAL_NEGATIVE_MIXED_ANNOTATION_FEATURE_COLUMNS],
        "technical_core_plus_high_confidence_events": [*technical_core, *HIGH_CONFIDENCE_ANNOTATION_FEATURE_COLUMNS],
        "technical_core_plus_high_quality_annotations": [*technical_core, *HIGH_QUALITY_ANNOTATION_FEATURE_COLUMNS],
        "annotation_compact_decay": list(COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS),
        "annotation_compact_weighted": list(COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS),
        "technical_core_plus_annotation_compact_decay": [*technical_core, *COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS],
        "technical_core_plus_annotation_compact_weighted": [*technical_core, *COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS],
    }
    return model_frame, derived, feature_sets


def _date_mask(metadata: pd.DataFrame, dates: list[Any]) -> pd.Series:
    date_set = {pd.to_datetime(value).date() for value in dates}
    return pd.to_datetime(metadata["trading_date"]).dt.date.isin(date_set)


def _annotation_active_rate(frame: pd.DataFrame) -> float:
    if frame.empty:
        return 0.0
    return float(pd.to_numeric(frame["annotation_coverage_available"], errors="coerce").fillna(0).gt(0).mean())


def _feature_active_counts(derived: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for column in ALL_ANNOTATION_FEATURE_COLUMNS:
        numeric = pd.to_numeric(derived[column], errors="coerce")
        rows.append(
            {
                "feature": column,
                "active_count": int(numeric.fillna(0).ne(0).sum()),
                "active_rate": float(numeric.fillna(0).ne(0).mean()),
                "missing_rate": float(numeric.isna().mean()),
                "mean": None if pd.isna(numeric.mean()) else float(numeric.mean()),
            }
        )
    return rows


def _source_quality_signal_coverage(annotations: pd.DataFrame, metadata: pd.DataFrame) -> list[dict[str, Any]]:
    if annotations.empty or metadata.empty or "source_quality" not in annotations.columns:
        return [
            {
                "source_quality": category,
                "rows": int(len(metadata)),
                "active_rows_20s": 0,
                "active_rate_20s": 0.0,
            }
            for category in SOURCE_QUALITY_CATEGORIES
        ]
    prepared = metadata.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper()
    prepared["snapshot_date"] = pd.to_datetime(prepared["trading_date"], errors="coerce").dt.date
    prepared["as_of"] = pd.to_datetime(prepared["as_of_timestamp"], utc=True, errors="coerce")
    session_dates = sorted(date_value for date_value in prepared["snapshot_date"].dropna().unique())
    session_index = {date_value: index for index, date_value in enumerate(session_dates)}
    if not session_dates:
        return [
            {
                "source_quality": category,
                "rows": int(len(metadata)),
                "active_rows_20s": 0,
                "active_rate_20s": 0.0,
            }
            for category in SOURCE_QUALITY_CATEGORIES
        ]
    annotations = annotations.copy()
    annotations["_event_session_index"] = annotations["event_date_parsed"].map(
        lambda value: bisect_left(session_dates, value) if pd.notna(value) else np.nan
    )
    by_ticker = {
        ticker: group.sort_values(["available_at_parsed", "event_date_parsed"]).copy()
        for ticker, group in annotations.groupby("ticker", dropna=False)
    }
    active_by_category = {category: 0 for category in SOURCE_QUALITY_CATEGORIES}
    for _, row in prepared.iterrows():
        ticker_events = by_ticker.get(str(row.get("ticker") or "").upper())
        if ticker_events is None or ticker_events.empty:
            continue
        snapshot_date = row["snapshot_date"]
        as_of = row["as_of"]
        if pd.isna(as_of) or pd.isna(snapshot_date):
            continue
        snapshot_index = session_index.get(snapshot_date)
        if snapshot_index is None:
            continue
        available = ticker_events[
            ticker_events["available_at_parsed"].le(as_of)
            & ticker_events["_event_session_index"].le(snapshot_index)
            & (snapshot_index - ticker_events["_event_session_index"]).le(20)
        ]
        if available.empty:
            continue
        for category in set(available["source_quality"].astype(str)):
            if category in active_by_category:
                active_by_category[category] += 1
            else:
                active_by_category["unknown"] += 1
    total_rows = int(len(prepared))
    return [
        {
            "source_quality": category,
            "rows": total_rows,
            "active_rows_20s": int(active_by_category.get(category, 0)),
            "active_rate_20s": float(active_by_category.get(category, 0) / total_rows) if total_rows else 0.0,
        }
        for category in SOURCE_QUALITY_CATEGORIES
    ]


def _annotation_db_summary(db_path: str | Path) -> dict[str, Any]:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        rows = pd.read_sql_query("SELECT * FROM research_event_annotations", conn)
    if rows.empty:
        return {
            "annotation_count": 0,
            "by_ticker": [],
            "by_event_type": [],
            "sentiment_distribution": [],
            "quality_flags": {},
            "source_quality_distribution": [],
            "informativeness_distribution": [],
            "missing_strength_or_confidence": {"zero_strength": 0, "zero_confidence": 0},
        }
    quality = _apply_annotation_quality_flags(rows.copy())
    quality_summary = quality_distribution(quality)
    by_ticker = rows.groupby("ticker").size().reset_index(name="annotation_count").sort_values("ticker").to_dict("records")
    by_event = rows.groupby("event_type").size().reset_index(name="annotation_count").sort_values("event_type").to_dict("records")
    sentiment = rows.groupby("sentiment_label").size().reset_index(name="annotation_count").sort_values("sentiment_label").to_dict("records")
    source = rows.groupby("source").size().reset_index(name="annotation_count").sort_values("annotation_count", ascending=False).head(25).to_dict("records")
    quality_flags = {
        flag: int(quality[flag].astype(bool).sum())
        for flag in [
            "high_signal_event",
            "routine_low_signal",
            "non_sec_material_event",
            "negative_or_risk_event",
            "high_confidence_source",
            "low_specificity_neutral",
            "duplicate_theme_cluster",
        ]
    }
    return {
        "annotation_count": int(len(rows)),
        "by_ticker": by_ticker,
        "by_event_type": by_event,
        "by_source_top25": source,
        "sentiment_distribution": sentiment,
        "source_quality_distribution": quality_summary["source_quality_distribution"],
        "informativeness_distribution": quality_summary["informativeness_distribution"],
        "source_quality_summary": quality_summary,
        "quality_flags": quality_flags,
        "sec_heavy_concentration": {
            "sec_annotation_count": int(quality["event_type"].astype(str).str.lower().eq("sec_filing").sum()),
            "routine_sec_heavy_count": int(quality_summary["routine_sec_heavy_count"]),
            "material_non_sec_count": int(quality_summary["material_non_sec_count"]),
        },
        "neutral_routine_concentration": {
            "neutral_or_unknown_count": int(quality["sentiment_label"].astype(str).str.lower().isin(["neutral", "unknown"]).sum()),
            "low_specificity_neutral_count": int(quality_summary["low_specificity_neutral_count"]),
            "routine_low_count": int(quality["informativeness"].eq("routine_low").sum()),
        },
        "missing_strength_or_confidence": {
            "zero_strength": int(pd.to_numeric(rows["strength"], errors="coerce").fillna(0).eq(0).sum()),
            "zero_confidence": int(pd.to_numeric(rows["confidence"], errors="coerce").fillna(0).eq(0).sum()),
        },
    }


def annotation_density_by_fold(
    derived: pd.DataFrame,
    metadata: pd.DataFrame,
    target_series: pd.Series,
    horizon_sessions: int,
    n_folds: int = 3,
) -> list[dict[str, Any]]:
    folds = make_walk_forward_splits(
        metadata,
        target_series,
        horizon_sessions=horizon_sessions,
        n_folds=n_folds,
        purge_sessions=horizon_sessions,
        embargo_sessions=horizon_sessions,
    )
    rows: list[dict[str, Any]] = []
    for split in folds:
        for role, dates in [("train", split.train_dates), ("eval", split.eval_dates)]:
            subset = derived.loc[_date_mask(metadata, dates)]
            rows.append(
                {
                    "fold_name": split.fold_name,
                    "split_name": split.split_name,
                    "role": role,
                    "rows": int(len(subset)),
                    "annotation_active_rate": _annotation_active_rate(subset),
                    "positive_active_rows": int(pd.to_numeric(subset["recent_positive_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                    "negative_active_rows": int(pd.to_numeric(subset["recent_negative_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                    "high_signal_active_rows": int(pd.to_numeric(subset["high_signal_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                    "non_sec_active_rows": int(pd.to_numeric(subset["non_sec_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                    "negative_mixed_active_rows": int(pd.to_numeric(subset["negative_mixed_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                    "high_quality_active_rows": int(pd.to_numeric(subset["high_quality_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                    "material_negative_mixed_active_rows": int(
                        pd.to_numeric(subset["material_negative_mixed_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()
                    ),
                    "compact_decay_active_rows": int(
                        subset[COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS]
                        .apply(pd.to_numeric, errors="coerce")
                        .fillna(0)
                        .ne(0)
                        .any(axis=1)
                        .sum()
                    )
                    if set(COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS).issubset(subset.columns)
                    else 0,
                    "compact_weighted_active_rows": int(
                        subset[COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS]
                        .apply(pd.to_numeric, errors="coerce")
                        .fillna(0)
                        .ne(0)
                        .any(axis=1)
                        .sum()
                    )
                    if set(COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS).issubset(subset.columns)
                    else 0,
                }
            )
    return rows


def build_annotation_coverage_audit(
    db_path: str | Path,
    dataset_id: int,
    target_name: str = RAW_TARGET_5_SESSION,
) -> dict[str, Any]:
    definition = get_target_definition(target_name)
    training = load_training_dataset(db_path, dataset_id, definition.base_label_column)
    target_series = precompute_target_series(definition, training.y, training.X, training.metadata)
    valid = target_series.notna()
    metadata = training.metadata.loc[valid].copy()
    derived = derive_annotation_features(db_path, metadata)
    warnings: list[str] = []
    fold_density_status = "available"
    try:
        density = annotation_density_by_fold(
            derived,
            metadata,
            target_series.loc[valid],
            horizon_sessions=definition.horizon_sessions,
        )
    except ValueError as exc:
        density = []
        fold_density_status = "unavailable"
        warnings.append(f"Fold-level annotation density unavailable: {exc}")
    annotation_rows = _annotation_rows_for_features(db_path)
    annotation_db_summary = _annotation_db_summary(db_path)
    source_quality_signal_coverage = _source_quality_signal_coverage(annotation_rows, metadata)
    joined = pd.concat([metadata[["ticker", "trading_date"]].reset_index(drop=True), derived.reset_index(drop=True)], axis=1)
    by_ticker: list[dict[str, Any]] = []
    for ticker, group in joined.groupby("ticker", dropna=False):
        by_ticker.append(
            {
                "ticker": str(ticker),
                "rows": int(len(group)),
                "annotation_active_rate": _annotation_active_rate(group),
                "positive_active_rows": int(pd.to_numeric(group["recent_positive_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                "negative_active_rows": int(pd.to_numeric(group["recent_negative_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                "high_signal_active_rows": int(pd.to_numeric(group["high_signal_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                "non_sec_active_rows": int(pd.to_numeric(group["non_sec_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                "negative_mixed_active_rows": int(pd.to_numeric(group["negative_mixed_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                "high_quality_active_rows": int(pd.to_numeric(group["high_quality_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
                "material_negative_mixed_active_rows": int(
                    pd.to_numeric(group["material_negative_mixed_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()
                ),
                "compact_decay_active_rows": int(
                    group[COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .ne(0)
                    .any(axis=1)
                    .sum()
                ),
                "compact_weighted_active_rows": int(
                    group[COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS]
                    .apply(pd.to_numeric, errors="coerce")
                    .fillna(0)
                    .ne(0)
                    .any(axis=1)
                    .sum()
                ),
            }
        )
    sparse_tickers = [row["ticker"] for row in by_ticker if float(row["annotation_active_rate"]) < 0.01]
    artifact = {
        "annotation_feature_version": ANNOTATION_FEATURE_VERSION,
        "artifact_type": "annotation_coverage",
        "created_at": _now_iso(),
        "dataset_id": int(dataset_id),
        "target": target_metadata(definition),
        "row_count": int(len(derived)),
        "derived_feature_count": int(derived.shape[1]),
        "feature_sets": {
            "annotation_features_only": {"columns": list(ANNOTATION_FEATURE_COLUMNS), "column_count": len(ANNOTATION_FEATURE_COLUMNS)},
            "technical_core_plus_annotations": {"columns": [*technical_core_columns(training.feature_columns), *ANNOTATION_FEATURE_COLUMNS]},
            "annotation_high_signal_only": {
                "columns": list(HIGH_SIGNAL_ANNOTATION_FEATURE_COLUMNS),
                "column_count": len(HIGH_SIGNAL_ANNOTATION_FEATURE_COLUMNS),
            },
            "technical_core_plus_high_signal_annotations": {
                "columns": [*technical_core_columns(training.feature_columns), *HIGH_SIGNAL_ANNOTATION_FEATURE_COLUMNS],
            },
            "technical_core_plus_non_sec_events": {
                "columns": [*technical_core_columns(training.feature_columns), *NON_SEC_ANNOTATION_FEATURE_COLUMNS],
            },
            "technical_core_plus_negative_mixed_events": {
                "columns": [*technical_core_columns(training.feature_columns), *NEGATIVE_MIXED_ANNOTATION_FEATURE_COLUMNS],
            },
            "technical_core_plus_negative_mixed_material_events": {
                "columns": [*technical_core_columns(training.feature_columns), *MATERIAL_NEGATIVE_MIXED_ANNOTATION_FEATURE_COLUMNS],
            },
            "technical_core_plus_high_confidence_events": {
                "columns": [*technical_core_columns(training.feature_columns), *HIGH_CONFIDENCE_ANNOTATION_FEATURE_COLUMNS],
            },
            "technical_core_plus_high_quality_annotations": {
                "columns": [*technical_core_columns(training.feature_columns), *HIGH_QUALITY_ANNOTATION_FEATURE_COLUMNS],
            },
            "annotation_compact_decay": {
                "columns": list(COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS),
                "column_count": len(COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS),
            },
            "annotation_compact_weighted": {
                "columns": list(COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS),
                "column_count": len(COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS),
            },
            "technical_core_plus_annotation_compact_decay": {
                "columns": [*technical_core_columns(training.feature_columns), *COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS],
            },
            "technical_core_plus_annotation_compact_weighted": {
                "columns": [*technical_core_columns(training.feature_columns), *COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS],
            },
        },
        "active_observation_counts": _feature_active_counts(derived),
        "coverage_by_ticker": sorted(by_ticker, key=lambda item: item["ticker"]),
        "coverage_by_fold": density,
        "source_quality_signal_coverage": source_quality_signal_coverage,
        "fold_density_status": fold_density_status,
        "sparse_tickers": sparse_tickers,
        "annotation_db_summary": annotation_db_summary,
        "warnings": warnings,
        "summary": {
            "annotation_rows": int(annotation_db_summary["annotation_count"]),
            "labeled_row_count": int(len(metadata)),
            "fold_density_status": fold_density_status,
            "rows_with_annotation_coverage": int(pd.to_numeric(derived["annotation_coverage_available"], errors="coerce").fillna(0).gt(0).sum()),
            "rows_with_high_signal_coverage": int(pd.to_numeric(derived["high_signal_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
            "rows_with_non_sec_event_coverage": int(pd.to_numeric(derived["non_sec_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
            "rows_with_negative_mixed_event_coverage": int(pd.to_numeric(derived["negative_mixed_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
            "rows_with_material_negative_mixed_event_coverage": int(
                pd.to_numeric(derived["material_negative_mixed_event_count_20s"], errors="coerce").fillna(0).gt(0).sum()
            ),
            "rows_with_high_quality_coverage": int(pd.to_numeric(derived["high_quality_annotation_count_20s"], errors="coerce").fillna(0).gt(0).sum()),
            "rows_with_compact_decay_coverage": int(
                derived[COMPACT_DECAY_ANNOTATION_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).ne(0).any(axis=1).sum()
            ),
            "rows_with_compact_weighted_coverage": int(
                derived[COMPACT_WEIGHTED_ANNOTATION_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0).ne(0).any(axis=1).sum()
            ),
            "annotation_active_rate": _annotation_active_rate(derived),
            "low_specificity_neutral_count": int(annotation_db_summary.get("source_quality_summary", {}).get("low_specificity_neutral_count", 0)),
            "routine_sec_heavy_count": int(annotation_db_summary.get("source_quality_summary", {}).get("routine_sec_heavy_count", 0)),
            "material_non_sec_count": int(annotation_db_summary.get("source_quality_summary", {}).get("material_non_sec_count", 0)),
            "scanner_scoring_effect": 0,
            "research_only": True,
        },
    }
    return _clean_json(artifact)


def write_annotation_artifact(artifact: dict[str, Any], output_dir: str | Path) -> Path:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    dataset_id = artifact.get("dataset_id", "unknown")
    path = output / f"phase2d6a_annotation_coverage_dataset{dataset_id}_{timestamp}.json"
    path.write_text(_json_dumps(artifact), encoding="utf-8")
    return path


def load_annotation_artifact(path: str | Path) -> dict[str, Any]:
    return _json_loads(Path(path).read_text(encoding="utf-8"), {}) or {}


def list_annotation_artifacts(output_dir: str | Path) -> list[AnnotationArtifactInfo]:
    output = Path(output_dir)
    if not output.exists():
        return []
    artifacts: list[AnnotationArtifactInfo] = []
    for path in sorted(output.glob("phase2d6a_annotation_coverage_dataset*.json"), reverse=True):
        data = load_annotation_artifact(path)
        summary = data.get("summary", {}) if isinstance(data, dict) else {}
        artifacts.append(
            AnnotationArtifactInfo(
                path=path,
                dataset_id=data.get("dataset_id") if isinstance(data, dict) else None,
                created_at=str(data.get("created_at") or "") if isinstance(data, dict) else "",
                summary=f"active_rate={summary.get('annotation_active_rate', 0)} rows={summary.get('rows_with_annotation_coverage', 0)}",
            )
        )
    return artifacts


def run_annotation_baseline_suite(
    db_path: str | Path,
    dataset_id: int,
    output_dir: str | Path,
    target_names: list[str] | None = None,
) -> tuple[dict[str, Any], Path, list[Any]]:
    from src.modeling.runner import run_single_baseline_model

    target_names = target_names or [
        RAW_TARGET_5_SESSION,
        "label_5_session_excess_return_winsorized_q01_q99",
    ]
    coverage = build_annotation_coverage_audit(db_path, dataset_id)
    coverage_path = write_annotation_artifact(coverage, output_dir)
    training = load_training_dataset(db_path, dataset_id, RAW_TARGET_5_SESSION)
    model_frame, _derived, feature_sets = build_annotation_model_frame(training, db_path)
    summaries: list[Any] = []
    for target_name in target_names:
        definition = get_target_definition(target_name)
        for feature_set_name, columns in feature_sets.items():
            for model_name in definition.allowed_models:
                summaries.append(
                    run_single_baseline_model(
                        db_path,
                        dataset_id=dataset_id,
                        target_column=target_name,
                        feature_set_name=feature_set_name,
                        model_name=model_name,
                        feature_columns_override=columns,
                        feature_frame_override=model_frame,
                        feature_set_metadata={
                            "source": "phase2d6a_research_annotation_features",
                            "coverage_artifact": str(coverage_path),
                            "column_count": len(columns),
                            "annotation_feature_version": ANNOTATION_FEATURE_VERSION,
                            "quality_filtered": feature_set_name not in {"technical_core", "annotation_features_only", "technical_core_plus_annotations"},
                            "research_only": True,
                            "scanner_scoring_effect": 0,
                            "active_catalyst_table_modified": False,
                        },
                        phase="2D-6A",
                    )
                )
    return coverage, coverage_path, summaries
