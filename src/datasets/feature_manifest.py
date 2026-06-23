from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


FEATURE_CONTRACT_VERSION = "feature_manifest_v1"
MANIFEST_METADATA_KEY = "__manifest_metadata__"

IDENTIFIER_COLUMNS = {"dataset_id", "snapshot_id", "ticker"}
METADATA_COLUMNS = {"trading_date", "as_of_timestamp"}

SEC_AUDIT_COLUMNS = {
    "sec_amendment_count_30s",
    "sec_feature_excluded_count_30s",
    "sec_feature_policy_version",
    "sec_max_feature_eligible_filings_single_day_30s",
    "sec_max_raw_filings_single_day_30s_audit",
    "sec_needs_review_filing_flag",
    "sec_recent_form4_count",
    "sec_recent_s1_s3_424b_flag",
    "sec_unknown_classification_count_30s",
    "sec_unknown_event_days_7s",
    "sec_unknown_event_days_30s",
    "sec_unknown_event_days_90s",
    "sec_unknown_present_30s",
}

AUDIT_COLUMNS = {
    "available_catalyst_ids",
    "catalyst_warnings",
    "regime_warnings",
}


@dataclass(frozen=True)
class FeatureRoleSets:
    model_features: list[str]
    audit_columns: list[str]
    label_columns: list[str]
    identifier_columns: list[str]
    metadata_columns: list[str]
    manifest: dict[str, dict[str, str]]


def _is_label(column: str) -> bool:
    return column.startswith("label_")


def _is_identifier(column: str) -> bool:
    return column in IDENTIFIER_COLUMNS


def _is_metadata(column: str) -> bool:
    return column in METADATA_COLUMNS


def _is_sec_audit_column(column: str) -> bool:
    if not column.startswith("sec_"):
        return False
    if column in SEC_AUDIT_COLUMNS:
        return True
    if "_filing_count_" in column:
        return True
    if column.startswith("sec_unknown_"):
        return True
    return False


def _role_for_column(column: str) -> tuple[str, str]:
    if _is_identifier(column):
        return "identifier", "Dataset identifier column."
    if _is_metadata(column):
        return "metadata", "Dataset timing metadata; excluded from default model inputs."
    if _is_label(column):
        return "label", "Outcome label column; selectable as y only."
    if column in AUDIT_COLUMNS:
        return "audit", "Traceability or warning field; excluded from default model inputs."
    if _is_sec_audit_column(column):
        return "audit", "SEC audit or volume/workflow field; excluded from default model inputs."
    return "model_feature", "Model-ready point-in-time feature."


def build_feature_manifest(columns: Iterable[str]) -> FeatureRoleSets:
    manifest: dict[str, dict[str, str]] = {
        MANIFEST_METADATA_KEY: {
            "role": "metadata",
            "reason": "Feature-role manifest metadata; not a dataset column.",
            "contract_version": FEATURE_CONTRACT_VERSION,
            "policy_version": FEATURE_CONTRACT_VERSION,
        }
    }
    grouped = {
        "model_feature": [],
        "audit": [],
        "label": [],
        "identifier": [],
        "metadata": [],
    }
    for raw_column in columns:
        column = str(raw_column)
        role, reason = _role_for_column(column)
        manifest[column] = {"role": role, "reason": reason, "contract_version": FEATURE_CONTRACT_VERSION}
        grouped[role].append(column)
    return FeatureRoleSets(
        model_features=sorted(grouped["model_feature"]),
        audit_columns=sorted(grouped["audit"]),
        label_columns=sorted(grouped["label"]),
        identifier_columns=sorted(grouped["identifier"]),
        metadata_columns=sorted(grouped["metadata"]),
        manifest=manifest,
    )


def model_feature_columns_from_frame(frame: pd.DataFrame) -> list[str]:
    if frame is None or frame.empty:
        return []
    return build_feature_manifest(frame.columns).model_features


def role_sets_from_frame(frame: pd.DataFrame) -> FeatureRoleSets:
    columns = [] if frame is None else list(frame.columns)
    return build_feature_manifest(columns)
