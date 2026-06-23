from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from src.extractions.models import (
    DETECTED_EVENT_TYPES,
    DOCUMENT_RELEVANCE_LABELS,
    EVIDENCE_SUFFICIENCY_LABELS,
    EXTRACTION_PROVIDERS,
    EXTRACTION_SENTIMENTS,
    EXTRACTION_TYPES,
    REVIEW_STATUSES,
    TIME_HORIZONS,
    LLMExtraction,
)


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, set, dict)):
        return False
    try:
        return bool(pd.isna(value))
    except Exception:
        return False


def clamp_int(value: Any, minimum: int, maximum: int, default: int = 0) -> int:
    try:
        if is_missing(value):
            return default
        parsed = int(round(float(value)))
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def clamp_float(value: Any, minimum: float, maximum: float, default: float = 0.0) -> float:
    try:
        if is_missing(value):
            return default
        parsed = float(value)
    except Exception:
        parsed = default
    return max(minimum, min(maximum, parsed))


def optional_int(value: Any) -> int | None:
    try:
        if is_missing(value):
            return None
        text = str(value).strip()
        if not text or text.lower() in {"none", "nan", "<na>"}:
            return None
        return int(float(text))
    except Exception:
        return None


def enum_or_default(value: Any, allowed: list[str], default: str) -> str:
    text = str(value or "").strip()
    return text if text in allowed else default


def safe_json_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            decoded = json.loads(text)
        except json.JSONDecodeError:
            return [text]
        return safe_json_list(decoded)
    if isinstance(value, (list, tuple, set)):
        cleaned: list[str] = []
        for item in value:
            if isinstance(item, dict):
                rendered = json.dumps(item, sort_keys=True, default=str)
            else:
                rendered = str(item)
            rendered = rendered.strip()
            if rendered:
                cleaned.append(rendered)
        return cleaned
    return [str(value).strip()] if str(value).strip() else []


def json_list_dumps(value: Any) -> str:
    return json.dumps(safe_json_list(value), ensure_ascii=False)


def parse_datetime(value: Any) -> datetime | None:
    if value is None or value == "":
        return None
    try:
        return pd.to_datetime(value).to_pydatetime()
    except Exception:
        return None


def normalize_extraction_payload(payload: dict[str, Any] | None) -> dict[str, Any]:
    payload = dict(payload or {})
    now = datetime.now(UTC)
    review_status = enum_or_default(payload.get("review_status"), REVIEW_STATUSES, "pending_review")
    reviewed_at = parse_datetime(payload.get("reviewed_at"))
    if review_status == "pending_review":
        reviewed_at = None

    return {
        "extraction_id": payload.get("extraction_id"),
        "document_id": clamp_int(payload.get("document_id"), 0, 2_147_483_647, 0),
        "catalyst_id": optional_int(payload.get("catalyst_id")),
        "ticker": str(payload.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN",
        "provider": enum_or_default(payload.get("provider"), EXTRACTION_PROVIDERS, "fallback"),
        "model_name": str(payload.get("model_name") or "").strip() or None,
        "extraction_type": enum_or_default(
            payload.get("extraction_type"),
            EXTRACTION_TYPES,
            "general_document_review",
        ),
        "event_type_detected": enum_or_default(payload.get("event_type_detected"), DETECTED_EVENT_TYPES, "unknown"),
        "sentiment_label": enum_or_default(payload.get("sentiment_label"), EXTRACTION_SENTIMENTS, "unknown"),
        "catalyst_strength": clamp_int(payload.get("catalyst_strength"), 0, 10, 0),
        "risk_severity": clamp_int(payload.get("risk_severity"), 0, 10, 0),
        "confidence": clamp_float(payload.get("confidence"), 0.0, 1.0, 0.0),
        "document_relevance": enum_or_default(
            payload.get("document_relevance"),
            DOCUMENT_RELEVANCE_LABELS,
            "unknown",
        ),
        "evidence_sufficiency": enum_or_default(
            payload.get("evidence_sufficiency"),
            EVIDENCE_SUFFICIENCY_LABELS,
            "unknown",
        ),
        "time_horizon": enum_or_default(payload.get("time_horizon"), TIME_HORIZONS, "unknown"),
        "key_positive_points": safe_json_list(payload.get("key_positive_points")),
        "key_risks": safe_json_list(payload.get("key_risks")),
        "evidence_snippets": safe_json_list(payload.get("evidence_snippets")),
        "short_summary": str(payload.get("short_summary") or "").strip(),
        "detailed_summary": str(payload.get("detailed_summary") or "").strip(),
        "proposed_score_effect": clamp_int(payload.get("proposed_score_effect"), -15, 10, 0),
        "review_status": review_status,
        "reviewer_note": str(payload.get("reviewer_note") or "").strip(),
        "reviewed_at": reviewed_at,
        "created_at": parse_datetime(payload.get("created_at")) or now,
        "updated_at": parse_datetime(payload.get("updated_at")) or now,
        "raw_llm_response_json": str(payload.get("raw_llm_response_json") or "").strip() or None,
        "prompt_version": str(payload.get("prompt_version") or "fallback_v1").strip() or "fallback_v1",
        "extraction_warnings": str(payload.get("extraction_warnings") or "").strip(),
    }


def extraction_from_payload(payload: dict[str, Any] | None) -> LLMExtraction:
    return LLMExtraction(**normalize_extraction_payload(payload))
