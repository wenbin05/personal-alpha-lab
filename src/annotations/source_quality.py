from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


SOURCE_QUALITY_CATEGORIES = (
    "official_company",
    "regulator",
    "exchange_or_index_provider",
    "sec_archive",
    "credible_news",
    "manual_note",
    "unknown",
)

EVENT_INFORMATIVENESS_LABELS = (
    "material_high",
    "material_medium",
    "routine_low",
    "duplicate_theme",
    "low_specificity",
)

SOURCE_QUALITY_VERSION = "source_quality_v1"

SOURCE_QUALITY_ALIASES = {
    "company_ir": "official_company",
    "company_ir_press_release": "official_company",
    "company_newsroom": "official_company",
    "investor_relations": "official_company",
    "official_regulator": "regulator",
    "government_regulator": "regulator",
    "index_provider": "exchange_or_index_provider",
    "exchange_provider": "exchange_or_index_provider",
    "sec": "sec_archive",
    "sec_edgar": "sec_archive",
    "sec_local": "sec_archive",
    "manual": "manual_note",
    "manual_csv": "manual_note",
    "csv_manual": "manual_note",
    "manual_demo": "manual_note",
    "local_earnings_events": "manual_note",
}


@dataclass(frozen=True)
class QualityClassification:
    source_quality: str
    informativeness: str
    reason: str
    duplicate_theme_key: str | None = None


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    try:
        parsed = json.loads(str(value))
        return parsed
    except Exception:
        return default


def _metadata_value(row: pd.Series | dict[str, Any], key: str) -> Any:
    for column in ("provider_metadata", "provider_metadata_json"):
        value = row.get(column) if isinstance(row, dict) else row.get(column)
        if isinstance(value, dict):
            if key in value:
                return value[key]
        else:
            parsed = _json_loads(value, {})
            if isinstance(parsed, dict) and key in parsed:
                return parsed[key]
    return None


def _tag_values(row: pd.Series | dict[str, Any]) -> set[str]:
    tags = row.get("tags") if isinstance(row, dict) else row.get("tags")
    if tags is None:
        tags = _json_loads(row.get("tags_json") if isinstance(row, dict) else row.get("tags_json"), [])
    if isinstance(tags, str):
        tags = [part.strip() for part in tags.replace(";", ",").split(",")]
    if not isinstance(tags, list):
        tags = []
    return {str(tag).strip().lower() for tag in tags if str(tag).strip()}


def _explicit_label(row: pd.Series | dict[str, Any], key: str, allowed: tuple[str, ...]) -> str | None:
    direct = row.get(key) if isinstance(row, dict) else row.get(key)
    candidate = direct if direct not in (None, "") else _metadata_value(row, key)
    if candidate in (None, ""):
        tag_prefix = f"{key}:"
        for tag in _tag_values(row):
            if tag.startswith(tag_prefix):
                candidate = tag.split(":", 1)[1]
                break
    normalized = str(candidate or "").strip().lower()
    if key == "source_quality":
        normalized = SOURCE_QUALITY_ALIASES.get(normalized, normalized)
    return normalized if normalized in allowed else None


def _contains_any(text: str, patterns: tuple[str, ...]) -> bool:
    return any(pattern in text for pattern in patterns)


def _normalized_text(*values: Any) -> str:
    text = " ".join(str(value or "") for value in values).lower()
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def infer_source_quality(row: pd.Series | dict[str, Any]) -> tuple[str, str]:
    explicit = _explicit_label(row, "source_quality", SOURCE_QUALITY_CATEGORIES)
    if explicit:
        return explicit, "explicit source_quality field"
    tag_aliases = _tag_values(row)
    for tag in tag_aliases:
        canonical = SOURCE_QUALITY_ALIASES.get(tag, tag)
        if canonical in SOURCE_QUALITY_CATEGORIES:
            return canonical, "source-quality tag"
    source = _normalized_text(row.get("source") if isinstance(row, dict) else row.get("source"))
    url = re.sub(r"\s+", " ", str(row.get("source_url") if isinstance(row, dict) else row.get("source_url") or "").lower())
    combined = f"{source} {url}"
    if "sec.gov/archives" in combined or "sec edgar" in combined or source in {"sec_archive", "sec_local"}:
        return "sec_archive", "SEC source or URL"
    if _contains_any(combined, ("investor relations", "newsroom", "company_ir", "investors.", "ir.", "press release")):
        return "official_company", "company IR/newsroom source"
    if _contains_any(
        combined,
        (
            "justice.gov",
            "department of justice",
            "sec.gov/news",
            "ftc.gov",
            "fda.gov",
            "cftc.gov",
            "consumerfinance.gov",
            "nhtsa.gov",
            "nhtsa",
            "regulator",
        ),
    ):
        return "regulator", "regulator source"
    if _contains_any(combined, ("spglobal.com", "nasdaq.com", "nyse.com", "index provider", "s&p", "s&p dow jones")):
        return "exchange_or_index_provider", "exchange or index-provider source"
    if _contains_any(combined, ("reuters", "apnews", "associated press", "wsj", "bloomberg", "cnbc", "marketwatch", "financial times")):
        return "credible_news", "credible public news source"
    if _contains_any(combined, ("manual", "csv_import", "csv_manual", "manual_demo", "demo", "local_earnings_events", "local earnings")):
        return "manual_note", "manual or CSV source"
    return "unknown", "source quality could not be inferred"


def _duplicate_theme_key(row: pd.Series | dict[str, Any]) -> str | None:
    explicit = row.get("duplicate_theme_key") if isinstance(row, dict) else row.get("duplicate_theme_key")
    if explicit not in (None, ""):
        return str(explicit).strip().lower()
    metadata_value = _metadata_value(row, "duplicate_theme_key")
    if metadata_value not in (None, ""):
        return str(metadata_value).strip().lower()
    tags = _tag_values(row)
    if "duplicate_theme_cluster" in tags:
        ticker = str(row.get("ticker") if isinstance(row, dict) else row.get("ticker") or "").upper()
        event_type = str(row.get("event_type") if isinstance(row, dict) else row.get("event_type") or "other").lower()
        title = _normalized_text(row.get("title") if isinstance(row, dict) else row.get("title"))[:40]
        return "|".join(part for part in [ticker, event_type, title] if part)
    return None


def infer_informativeness(row: pd.Series | dict[str, Any]) -> tuple[str, str, str | None]:
    explicit = _explicit_label(row, "informativeness", EVENT_INFORMATIVENESS_LABELS)
    duplicate_key = _duplicate_theme_key(row)
    if explicit:
        return explicit, "explicit informativeness field", duplicate_key
    tags = _tag_values(row)
    event_type = str(row.get("event_type") if isinstance(row, dict) else row.get("event_type") or "").lower()
    sentiment = str(row.get("sentiment_label") if isinstance(row, dict) else row.get("sentiment_label") or "unknown").lower()
    strength = pd.to_numeric(row.get("strength") if isinstance(row, dict) else row.get("strength"), errors="coerce")
    confidence = pd.to_numeric(row.get("confidence") if isinstance(row, dict) else row.get("confidence"), errors="coerce")
    source_quality, _ = infer_source_quality(row)
    text = _normalized_text(
        row.get("title") if isinstance(row, dict) else row.get("title"),
        row.get("summary") if isinstance(row, dict) else row.get("summary"),
        row.get("evidence_text") if isinstance(row, dict) else row.get("evidence_text"),
    )
    if duplicate_key or "duplicate_theme_cluster" in tags:
        return "duplicate_theme", "duplicate-theme tag or key", duplicate_key
    if "routine_low_signal" in tags or (
        event_type == "sec_filing"
        and sentiment in {"neutral", "unknown"}
        and not _contains_any(text, ("offering", "investigation", "guidance", "material", "merger", "acquisition"))
    ):
        return "routine_low", "routine SEC/neutral event", duplicate_key
    if "low_specificity_neutral" in tags or (
        sentiment in {"neutral", "unknown"}
        and (pd.isna(strength) or float(strength) <= 2)
        and len(text) < 180
    ):
        return "low_specificity", "neutral/unknown row with weak specificity", duplicate_key
    if event_type in {"legal_regulatory", "financing", "corporate_action", "guidance_update"}:
        return "material_high", "event type is normally material", duplicate_key
    if "high_signal_event" in tags or (
        source_quality in {"official_company", "regulator", "exchange_or_index_provider", "credible_news"}
        and not pd.isna(strength)
        and not pd.isna(confidence)
        and float(strength) >= 6
        and float(confidence) >= 0.65
    ):
        return "material_high", "high-signal tags or strong sourced event", duplicate_key
    if event_type in {"earnings", "product_launch", "partnership", "management_change", "news", "analyst"}:
        return "material_medium", "moderately informative event type", duplicate_key
    return "low_specificity", "informativeness could not be established", duplicate_key


def classify_event_quality(row: pd.Series | dict[str, Any]) -> QualityClassification:
    source_quality, source_reason = infer_source_quality(row)
    informativeness, info_reason, duplicate_key = infer_informativeness(row)
    return QualityClassification(
        source_quality=source_quality,
        informativeness=informativeness,
        reason=f"{source_reason}; {info_reason}",
        duplicate_theme_key=duplicate_key,
    )


def enrich_quality_frame(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame.copy() if frame is not None else pd.DataFrame()
    output = frame.copy()
    classifications = [classify_event_quality(row) for _, row in output.iterrows()]
    output["source_quality"] = [item.source_quality for item in classifications]
    output["informativeness"] = [item.informativeness for item in classifications]
    output["quality_reason"] = [item.reason for item in classifications]
    output["duplicate_theme_key"] = [item.duplicate_theme_key for item in classifications]
    return output


def quality_distribution(frame: pd.DataFrame) -> dict[str, Any]:
    enriched = enrich_quality_frame(frame)
    if enriched.empty:
        return {
            "source_quality_distribution": [],
            "informativeness_distribution": [],
            "low_specificity_neutral_count": 0,
            "routine_sec_heavy_count": 0,
            "material_non_sec_count": 0,
            "duplicate_theme_count": 0,
        }
    sentiment = (
        enriched["sentiment_label"].astype(str).str.lower()
        if "sentiment_label" in enriched.columns
        else pd.Series("", index=enriched.index, dtype=object)
    )
    event_type = (
        enriched["event_type"].astype(str).str.lower()
        if "event_type" in enriched.columns
        else pd.Series("", index=enriched.index, dtype=object)
    )
    low_specificity_neutral = enriched[
        enriched["informativeness"].eq("low_specificity")
        & sentiment.isin(["neutral", "unknown"])
    ]
    routine_sec = enriched[enriched["informativeness"].eq("routine_low") & event_type.eq("sec_filing")]
    material_non_sec = enriched[
        enriched["informativeness"].isin(["material_high", "material_medium"])
        & ~event_type.eq("sec_filing")
    ]
    return {
        "source_quality_distribution": enriched.groupby("source_quality").size().reset_index(name="count").sort_values("source_quality").to_dict("records"),
        "informativeness_distribution": enriched.groupby("informativeness").size().reset_index(name="count").sort_values("informativeness").to_dict("records"),
        "low_specificity_neutral_count": int(len(low_specificity_neutral)),
        "routine_sec_heavy_count": int(len(routine_sec)),
        "material_non_sec_count": int(len(material_non_sec)),
        "duplicate_theme_count": int(enriched["informativeness"].eq("duplicate_theme").sum()),
    }
