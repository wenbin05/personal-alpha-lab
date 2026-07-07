from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.annotations.models import normalize_tags
from src.annotations.news_events import json_loads
from src.annotations.source_quality import (
    EVENT_INFORMATIVENESS_LABELS,
    SOURCE_QUALITY_CATEGORIES,
    SOURCE_QUALITY_VERSION,
    classify_event_quality,
    enrich_quality_frame,
)
from src.data import storage


NORMALIZATION_VERSION = "annotation_quality_normalization_v1"


@dataclass(frozen=True)
class QualityNormalizationResult:
    artifact: dict[str, Any]
    annotation_updates: int
    candidate_updates: int


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _strip_quality_tags(tags: list[str]) -> list[str]:
    prefixes = ("source_quality:", "informativeness:", "duplicate_theme_key:", "provider_name:")
    return [tag for tag in normalize_tags(tags) if not any(tag.startswith(prefix) for prefix in prefixes)]


def _tag_value(tags: list[str], prefix: str, allowed: tuple[str, ...] | None = None) -> str:
    for tag in normalize_tags(tags):
        if tag.startswith(prefix):
            value = tag.split(":", 1)[1]
            if allowed is None or value in allowed:
                return value
    return "unknown"


def _metadata_value(value: Any, key: str, allowed: tuple[str, ...] | None = None) -> str:
    metadata = value if isinstance(value, dict) else json_loads(value, {})
    if not isinstance(metadata, dict):
        return "unknown"
    candidate = str(metadata.get(key) or "").strip().lower()
    if not candidate:
        return "unknown"
    if allowed is not None and candidate not in allowed:
        return "unknown"
    return candidate


def _provider_name_from_row(row: dict[str, Any]) -> str:
    provider = str(row.get("provider") or "").strip()
    if provider:
        return provider
    source = str(row.get("source") or "").strip()
    if source:
        return re.sub(r"[^a-z0-9]+", "_", source.lower()).strip("_") or "manual"
    return "manual"


def _theme_key(row: dict[str, Any]) -> str | None:
    title = re.sub(r"[^a-z0-9]+", " ", str(row.get("title") or "").lower())
    title = re.sub(r"\s+", " ", title).strip()
    if len(title) < 12:
        return None
    return "|".join(
        [
            str(row.get("ticker") or "").upper(),
            str(row.get("event_type") or "other").lower(),
            str(row.get("sentiment_label") or "unknown").lower(),
            title[:72],
        ]
    )


def _distribution(rows: list[dict[str, Any]], column: str, allowed: tuple[str, ...]) -> list[dict[str, Any]]:
    counter = Counter(str(row.get(column) or "unknown") if str(row.get(column) or "") in allowed else "unknown" for row in rows)
    return [{"value": key, "count": int(counter.get(key, 0))} for key in sorted(set(allowed) | set(counter))]


def _classify_rows(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    enriched = enrich_quality_frame(frame)
    rows = enriched.to_dict("records")
    theme_counts = Counter(filter(None, (_theme_key(row) for row in rows)))
    for row in rows:
        key = _theme_key(row)
        if key and theme_counts[key] > 1:
            row["duplicate_theme_key"] = key
            if row.get("informativeness") != "routine_low":
                row["informativeness"] = "duplicate_theme"
                row["quality_reason"] = f"{row.get('quality_reason', '')}; duplicate theme cluster"
    return rows


def _load_tables(db_path: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        annotations = pd.read_sql_query("SELECT * FROM research_event_annotations ORDER BY annotation_id", conn)
        candidates = pd.read_sql_query("SELECT * FROM research_event_candidates ORDER BY candidate_id", conn)
    if not annotations.empty:
        annotations["tags"] = annotations["tags_json"].map(lambda value: json_loads(value, []))
    if not candidates.empty:
        candidates["tags"] = candidates["tags_json"].map(lambda value: json_loads(value, []))
        candidates["provider_metadata"] = candidates["provider_metadata_json"].map(lambda value: json_loads(value, {}))
    return annotations, candidates


def normalize_annotation_quality_metadata(db_path: str | Path, artifact_path: str | Path | None = None) -> QualityNormalizationResult:
    """Persist deterministic quality taxonomy metadata for research-only rows.

    Existing schemas are intentionally preserved: annotations store canonical
    labels in tags, and candidates store them in provider metadata plus tags.
    """

    annotations, candidates = _load_tables(db_path)
    annotation_before = [
        {
            "source_quality": _tag_value(row.get("tags") or [], "source_quality:", SOURCE_QUALITY_CATEGORIES),
            "informativeness": _tag_value(row.get("tags") or [], "informativeness:", EVENT_INFORMATIVENESS_LABELS),
        }
        for row in annotations.to_dict("records")
    ]
    candidate_before = [
        {
            "source_quality": _metadata_value(row.get("provider_metadata"), "source_quality", SOURCE_QUALITY_CATEGORIES),
            "informativeness": _metadata_value(row.get("provider_metadata"), "informativeness", EVENT_INFORMATIVENESS_LABELS),
        }
        for row in candidates.to_dict("records")
    ]

    classified_annotations = _classify_rows(annotations)
    classified_candidates = _classify_rows(candidates)
    annotation_updates = 0
    candidate_updates = 0
    examples: list[dict[str, Any]] = []
    now = _now_iso()

    with storage.connect(db_path) as conn:
        for row in classified_annotations:
            tags = _strip_quality_tags(row.get("tags") or json_loads(row.get("tags_json"), []))
            source_quality = str(row.get("source_quality") or "unknown")
            informativeness = str(row.get("informativeness") or "low_specificity")
            if source_quality != "unknown":
                tags.append(f"source_quality:{source_quality}")
            tags.append(f"informativeness:{informativeness}")
            duplicate_key = row.get("duplicate_theme_key")
            if duplicate_key:
                tags.append(f"duplicate_theme_key:{duplicate_key}")
            provider_name = _provider_name_from_row(row)
            if provider_name:
                tags.append(f"provider_name:{provider_name}")
            normalized_tags = normalize_tags(tags)
            old_tags = normalize_tags(json_loads(row.get("tags_json"), []))
            if normalized_tags != old_tags:
                conn.execute(
                    """
                    UPDATE research_event_annotations
                    SET tags_json = ?, updated_at = ?
                    WHERE annotation_id = ?
                    """,
                    (_json_dumps(normalized_tags), now, int(row["annotation_id"])),
                )
                annotation_updates += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "table": "research_event_annotations",
                            "id": int(row["annotation_id"]),
                            "ticker": row.get("ticker"),
                            "title": row.get("title"),
                            "source_quality": source_quality,
                            "informativeness": informativeness,
                            "reason": row.get("quality_reason"),
                        }
                    )

        for row in classified_candidates:
            tags = _strip_quality_tags(row.get("tags") or json_loads(row.get("tags_json"), []))
            source_quality = str(row.get("source_quality") or "unknown")
            informativeness = str(row.get("informativeness") or "low_specificity")
            if source_quality != "unknown":
                tags.append(f"source_quality:{source_quality}")
            tags.append(f"informativeness:{informativeness}")
            duplicate_key = row.get("duplicate_theme_key")
            if duplicate_key:
                tags.append(f"duplicate_theme_key:{duplicate_key}")
            provider_name = _provider_name_from_row(row)
            metadata = dict(row.get("provider_metadata") or json_loads(row.get("provider_metadata_json"), {}) or {})
            metadata.update(
                {
                    "source_quality": source_quality,
                    "informativeness": informativeness,
                    "provider_name": provider_name,
                    "quality_reason": row.get("quality_reason"),
                    "source_quality_version": SOURCE_QUALITY_VERSION,
                    "normalization_version": NORMALIZATION_VERSION,
                }
            )
            if duplicate_key:
                metadata["duplicate_theme_key"] = duplicate_key
            normalized_tags = normalize_tags(tags)
            old_tags = normalize_tags(json_loads(row.get("tags_json"), []))
            old_metadata = row.get("provider_metadata") or json_loads(row.get("provider_metadata_json"), {}) or {}
            if normalized_tags != old_tags or metadata != old_metadata:
                conn.execute(
                    """
                    UPDATE research_event_candidates
                    SET tags_json = ?, provider_metadata_json = ?, updated_at = ?
                    WHERE candidate_id = ?
                    """,
                    (_json_dumps(normalized_tags), _json_dumps(metadata), now, int(row["candidate_id"])),
                )
                candidate_updates += 1
                if len(examples) < 12:
                    examples.append(
                        {
                            "table": "research_event_candidates",
                            "id": int(row["candidate_id"]),
                            "ticker": row.get("ticker"),
                            "title": row.get("title"),
                            "source_quality": source_quality,
                            "informativeness": informativeness,
                            "reason": row.get("quality_reason"),
                        }
                    )

    after_annotations, after_candidates = _load_tables(db_path)
    annotation_after = _classify_rows(after_annotations)
    candidate_after = _classify_rows(after_candidates)
    after_rows = [*annotation_after, *candidate_after]
    before_rows = [*annotation_before, *candidate_before]
    artifact = {
        "artifact_type": "annotation_quality_normalization",
        "created_at": now,
        "normalization_version": NORMALIZATION_VERSION,
        "source_quality_version": SOURCE_QUALITY_VERSION,
        "rules_used": [
            "Canonical source_quality and informativeness are inferred only from explicit fields, provider metadata, tags, source/source_url, event type, sentiment, strength, confidence, title, summary, and evidence text.",
            "Legacy aliases such as official_regulator, company_ir, sec_local, and local_earnings_events are mapped to canonical taxonomy labels.",
            "Annotations persist taxonomy labels as tags_json because the table has no provider metadata column.",
            "Candidates persist taxonomy labels in provider_metadata_json and tags_json.",
            "Unclear source quality remains unknown; unknown rows are reported and are not promoted into scanner scoring.",
            "Duplicate theme keys are assigned only for deterministic same ticker/event/sentiment/normalized-title clusters.",
        ],
        "row_counts": {
            "annotations": int(len(annotations)),
            "candidates": int(len(candidates)),
            "combined": int(len(annotations) + len(candidates)),
        },
        "rows_updated": {
            "annotations": int(annotation_updates),
            "candidates": int(candidate_updates),
            "combined": int(annotation_updates + candidate_updates),
        },
        "before_source_quality_distribution": _distribution(before_rows, "source_quality", SOURCE_QUALITY_CATEGORIES),
        "after_source_quality_distribution": _distribution(after_rows, "source_quality", SOURCE_QUALITY_CATEGORIES),
        "before_informativeness_distribution": _distribution(before_rows, "informativeness", EVENT_INFORMATIVENESS_LABELS),
        "after_informativeness_distribution": _distribution(after_rows, "informativeness", EVENT_INFORMATIVENESS_LABELS),
        "unknown_rows_remaining": {
            "source_quality": int(sum(1 for row in after_rows if row.get("source_quality") == "unknown")),
            "informativeness": int(sum(1 for row in after_rows if row.get("informativeness") in {"unknown", "low_specificity"})),
        },
        "examples": examples,
        "scanner_scoring_effect": 0,
        "research_only": True,
    }
    if artifact_path:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(_json_dumps(artifact), encoding="utf-8")
    return QualityNormalizationResult(artifact=artifact, annotation_updates=annotation_updates, candidate_updates=candidate_updates)
