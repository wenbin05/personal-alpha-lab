from __future__ import annotations

from typing import Any, Literal

from src.extractions.validation import clamp_float, clamp_int, enum_or_default, safe_json_list


NO_VALID_EVIDENCE_WARNING = "No valid exact evidence returned by provider."

ReviewReadiness = Literal["ready_for_review", "needs_evidence", "insufficient_document"]


def source_quality_overrides(source_text: str) -> tuple[dict[str, str], list[str]]:
    text = str(source_text or "").lower()
    overrides: dict[str, str] = {}
    warnings: list[str] = []

    injection_markers = [
        "ignore previous instructions",
        "ignore all previous instructions",
        "recommend buying",
        "strong buy rating",
        "must output",
    ]
    noisy_markers = [
        "forum post",
        "unsupported claims",
        "anonymous users",
        "heard rumors",
        "no verifiable evidence",
        "no source link",
        "will moon",
    ]

    if any(marker in text for marker in injection_markers):
        overrides["document_relevance"] = "irrelevant"
        overrides["evidence_sufficiency"] = "insufficient"
        warnings.append("Embedded prompt-injection or recommendation-seeking language was detected and ignored.")
    elif any(marker in text for marker in noisy_markers):
        overrides["document_relevance"] = "uncertain"
        overrides["evidence_sufficiency"] = "insufficient"
        warnings.append("Source text appears noisy, speculative, or unsupported; interpretation confidence was capped.")
    elif 0 < len(text.strip()) < 120:
        overrides["document_relevance"] = "uncertain"
        overrides["evidence_sufficiency"] = "limited"
        warnings.append("Source text is short; interpretation confidence was capped.")

    return overrides, warnings


def apply_quality_calibration(payload: dict[str, Any], submitted_text: str = "") -> tuple[dict[str, Any], list[str]]:
    calibrated = dict(payload or {})
    warnings: list[str] = []

    overrides, source_warnings = source_quality_overrides(submitted_text)
    warnings.extend(source_warnings)
    for key, value in overrides.items():
        current = str(calibrated.get(key) or "unknown")
        if current in {"unknown", "relevant", "sufficient"} or key == "document_relevance" and value == "irrelevant":
            calibrated[key] = value

    evidence = safe_json_list(calibrated.get("evidence_snippets"))
    document_relevance = enum_or_default(
        calibrated.get("document_relevance"),
        ["relevant", "uncertain", "irrelevant", "unknown"],
        "unknown",
    )
    evidence_sufficiency = enum_or_default(
        calibrated.get("evidence_sufficiency"),
        ["sufficient", "limited", "insufficient", "unknown"],
        "unknown",
    )

    if evidence_sufficiency == "sufficient" and not evidence:
        evidence_sufficiency = "limited"
        calibrated["evidence_sufficiency"] = "limited"
        warnings.append("Evidence sufficiency downgraded because no valid exact evidence snippets remain.")

    if document_relevance == "irrelevant" or evidence_sufficiency == "insufficient":
        calibrated["confidence"] = min(clamp_float(calibrated.get("confidence"), 0.0, 1.0, 0.0), 0.39)
        calibrated["catalyst_strength"] = min(clamp_int(calibrated.get("catalyst_strength"), 0, 10, 0), 1)
        calibrated["proposed_score_effect"] = 0
        if str(calibrated.get("sentiment_label")) not in {"neutral", "unknown"}:
            calibrated["sentiment_label"] = "unknown"
        warnings.append("Insufficient or irrelevant source document: confidence capped and proposed score effect set to 0.")
    elif document_relevance == "uncertain" or evidence_sufficiency == "limited":
        calibrated["confidence"] = min(clamp_float(calibrated.get("confidence"), 0.0, 1.0, 0.0), 0.74)
        warnings.append("Limited or uncertain source support: confidence capped at 0.74.")

    return calibrated, list(dict.fromkeys(warnings))


def classify_review_readiness(extraction: dict[str, Any]) -> ReviewReadiness:
    evidence = safe_json_list(extraction.get("evidence_snippets"))
    proposed_effect = clamp_int(extraction.get("proposed_score_effect"), -15, 10, 0)
    document_relevance = str(extraction.get("document_relevance") or "unknown")
    evidence_sufficiency = str(extraction.get("evidence_sufficiency") or "unknown")

    if document_relevance == "irrelevant" or evidence_sufficiency == "insufficient":
        return "insufficient_document"
    if proposed_effect != 0 and not evidence:
        return "needs_evidence"
    return "ready_for_review"


def approval_requirements_met(
    extraction: dict[str, Any],
    reviewer_note: str,
    override_not_ready: bool = False,
) -> tuple[bool, str]:
    readiness = classify_review_readiness(extraction)
    if readiness == "ready_for_review":
        return True, ""
    if not override_not_ready:
        return False, f"Approval requires explicit override because readiness is {readiness}."
    if not str(reviewer_note or "").strip():
        return False, f"Approval override for {readiness} requires a reviewer note."
    return True, ""
