from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from src.documents.repository import get_document_by_id
from src.extractions.fallback_extractor import run_fallback_extraction
from src.extractions.models import EXTRACTION_TYPES, REVIEW_STATUSES
from src.extractions.openai_provider import OpenAIExtractionProvider, OpenAIProviderError
from src.extractions.quality import approval_requirements_met, classify_review_readiness
from src.extractions.repository import (
    approve_extraction,
    get_extraction_by_id,
    insert_extraction,
    list_extractions_by_document_id,
    supersede_extraction,
)


MIN_USABLE_TEXT_CHARS = 40
PREVIEW_TRUNCATION_CHARS = 3_000


@dataclass
class DocumentReadiness:
    can_run: bool
    warnings: list[str] = field(default_factory=list)
    cleaned_text: str = ""


@dataclass
class ExtractionRunResult:
    extraction_id: int | None = None
    superseded_ids: list[int] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    blocked: bool = False


@dataclass
class ReviewActionResult:
    changed: bool
    message: str = ""


def document_readiness(document: dict[str, Any] | None) -> DocumentReadiness:
    if not document:
        return DocumentReadiness(can_run=False, warnings=["Source document is missing or was deleted."])

    cleaned_text = str(document.get("cleaned_text") or "").strip()
    raw_text = str(document.get("raw_text") or "").strip()
    text = cleaned_text or raw_text
    warnings: list[str] = []

    if not text:
        warnings.append("Source document has no usable text for extraction.")
    if text and len(text) < MIN_USABLE_TEXT_CHARS:
        warnings.append("Source document text is unusually short; fallback extraction will be low context.")
    if len(cleaned_text) > PREVIEW_TRUNCATION_CHARS:
        warnings.append("Source text preview is truncated in the UI; full cleaned text remains stored locally.")
    parsing_status = str(document.get("parsing_status") or "")
    if parsing_status in {"failed", "not_attempted"}:
        warnings.append(f"Document parsing status is {parsing_status}.")

    can_run = bool(text) and parsing_status != "failed"
    return DocumentReadiness(can_run=can_run, warnings=warnings, cleaned_text=text)


def pending_extractions_for_document(db_path: str | Path, document_id: int) -> pd.DataFrame:
    df = list_extractions_by_document_id(db_path, document_id, limit=100)
    if df.empty:
        return df
    return df[df["review_status"].eq("pending_review")].copy()


def approve_extraction_with_readiness(
    db_path: str | Path,
    extraction_id: int,
    reviewer_note: str = "",
    override_not_ready: bool = False,
) -> ReviewActionResult:
    extraction = get_extraction_by_id(db_path, extraction_id)
    if not extraction:
        return ReviewActionResult(changed=False, message="Extraction was not found.")
    if extraction.get("review_status") != "pending_review":
        return ReviewActionResult(changed=False, message="This extraction is no longer pending and was not changed.")

    allowed, reason = approval_requirements_met(extraction, reviewer_note, override_not_ready=override_not_ready)
    if not allowed:
        return ReviewActionResult(changed=False, message=reason)
    if approve_extraction(db_path, extraction_id, reviewer_note):
        readiness = classify_review_readiness(extraction)
        suffix = " Scanner scoring was not changed."
        if readiness != "ready_for_review":
            suffix = f" Approved with explicit {readiness} override. Scanner scoring was not changed."
        return ReviewActionResult(changed=True, message=f"Extraction #{extraction_id} approved.{suffix}")
    return ReviewActionResult(changed=False, message="This extraction is no longer pending and was not changed.")


def create_fallback_extraction_for_document(
    db_path: str | Path,
    document_id: int,
    extraction_type: str = "general_document_review",
    supersede_existing: bool = False,
) -> ExtractionRunResult:
    document = get_document_by_id(db_path, document_id)
    readiness = document_readiness(document)
    if not readiness.can_run:
        return ExtractionRunResult(warnings=readiness.warnings, blocked=True)

    existing_pending = pending_extractions_for_document(db_path, document_id)
    if not existing_pending.empty and not supersede_existing:
        return ExtractionRunResult(
            warnings=[
                "This document already has a pending extraction. "
                "Choose the explicit supersede option before creating another one."
            ],
            blocked=True,
        )

    extraction_type = extraction_type if extraction_type in EXTRACTION_TYPES else "general_document_review"
    extraction = run_fallback_extraction(document).model_copy(update={"extraction_type": extraction_type})
    extraction_id = insert_extraction(db_path, extraction)

    superseded_ids: list[int] = []
    if supersede_existing and not existing_pending.empty:
        for _, row in existing_pending.iterrows():
            old_id = int(row["extraction_id"])
            if supersede_extraction(db_path, old_id, f"Superseded by fallback extraction #{extraction_id}."):
                superseded_ids.append(old_id)

    return ExtractionRunResult(
        extraction_id=extraction_id,
        superseded_ids=superseded_ids,
        warnings=readiness.warnings,
    )


def create_openai_extraction_for_document(
    db_path: str | Path,
    document_id: int,
    settings: Any | None = None,
    extraction_type: str = "general_document_review",
    supersede_existing: bool = False,
    provider: OpenAIExtractionProvider | None = None,
) -> ExtractionRunResult:
    document = get_document_by_id(db_path, document_id)
    readiness = document_readiness(document)
    if not readiness.can_run:
        return ExtractionRunResult(warnings=readiness.warnings, blocked=True)

    existing_pending = pending_extractions_for_document(db_path, document_id)
    if not existing_pending.empty and not supersede_existing:
        return ExtractionRunResult(
            warnings=[
                "This document already has a pending extraction. "
                "Choose the explicit supersede option before creating another one."
            ],
            blocked=True,
        )

    extraction_type = extraction_type if extraction_type in EXTRACTION_TYPES else "general_document_review"
    provider = provider or OpenAIExtractionProvider.from_settings(settings)
    try:
        extraction = provider.extract(document, extraction_type=extraction_type)
    except OpenAIProviderError as exc:
        return ExtractionRunResult(warnings=[str(exc)], blocked=True)

    extraction_id = insert_extraction(db_path, extraction)

    superseded_ids: list[int] = []
    if supersede_existing and not existing_pending.empty:
        for _, row in existing_pending.iterrows():
            old_id = int(row["extraction_id"])
            if supersede_extraction(db_path, old_id, f"Superseded by OpenAI extraction #{extraction_id}."):
                superseded_ids.append(old_id)

    return ExtractionRunResult(
        extraction_id=extraction_id,
        superseded_ids=superseded_ids,
        warnings=readiness.warnings,
    )


def filter_extractions(
    df: pd.DataFrame,
    ticker: str = "All",
    statuses: list[str] | None = None,
    extraction_types: list[str] | None = None,
    providers: list[str] | None = None,
    document_id: int | None = None,
) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df

    filtered = df.copy()
    if ticker != "All" and "ticker" in filtered.columns:
        filtered = filtered[filtered["ticker"].str.upper().eq(ticker.upper())]
    if statuses:
        valid_statuses = [status for status in statuses if status in REVIEW_STATUSES]
        if valid_statuses and "review_status" in filtered.columns:
            filtered = filtered[filtered["review_status"].isin(valid_statuses)]
    if extraction_types and "extraction_type" in filtered.columns:
        filtered = filtered[filtered["extraction_type"].isin(extraction_types)]
    if providers and "provider" in filtered.columns:
        filtered = filtered[filtered["provider"].isin(providers)]
    if document_id is not None and "document_id" in filtered.columns:
        filtered = filtered[filtered["document_id"].eq(int(document_id))]
    return filtered.sort_values(["created_at", "extraction_id"], ascending=[False, False])


def enrich_extractions_with_documents(extractions: pd.DataFrame, documents: pd.DataFrame) -> pd.DataFrame:
    if extractions is None or extractions.empty:
        return pd.DataFrame() if extractions is None else extractions
    if documents is None or documents.empty:
        enriched = extractions.copy()
        enriched["document_title"] = ""
        enriched["document_type"] = ""
        enriched["document_source"] = ""
        return enriched

    doc_cols = [
        "document_id",
        "title",
        "document_type",
        "source",
        "source_url",
        "published_at",
        "parsing_status",
        "warnings",
        "cleaned_text",
    ]
    available_cols = [column for column in doc_cols if column in documents.columns]
    doc_lookup = documents[available_cols].copy()
    rename = {
        "title": "document_title",
        "document_type": "document_type",
        "source": "document_source",
        "source_url": "document_source_url",
        "published_at": "document_published_at",
        "parsing_status": "document_parsing_status",
        "warnings": "document_warnings",
        "cleaned_text": "document_cleaned_text",
    }
    doc_lookup = doc_lookup.rename(columns=rename)
    return extractions.merge(doc_lookup, how="left", on="document_id")


def extraction_queue_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    display = pd.DataFrame() if df is None else df.copy()
    if not display.empty:
        display["review_readiness"] = display.apply(lambda row: classify_review_readiness(row.to_dict()), axis=1)
    columns = [
        "created_at",
        "ticker",
        "document_title",
        "extraction_type",
        "event_type_detected",
        "sentiment_label",
        "catalyst_strength",
        "risk_severity",
        "confidence",
        "document_relevance",
        "evidence_sufficiency",
        "proposed_score_effect",
        "review_readiness",
        "provider",
        "review_status",
    ]
    if df is None or display.empty:
        return pd.DataFrame(columns=columns)
    for column in columns:
        if column not in display.columns:
            display[column] = ""
    return display[columns]
