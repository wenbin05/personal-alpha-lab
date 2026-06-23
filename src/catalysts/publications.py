from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.catalysts.models import EVENT_TYPES, SENTIMENT_LABELS, CatalystEvent
from src.catalysts.proposals import get_proposal_by_id, list_proposals_by_ticker
from src.catalysts.repository import CATALYST_COLUMNS, _event_row, list_catalysts_by_ticker, make_dedupe_key
from src.data import storage
from src.documents.repository import get_document_by_id
from src.extractions.repository import get_extraction_by_id
from src.extractions.validation import clamp_float, clamp_int, safe_json_list
from src.features.catalyst import get_catalyst_features


PUBLICATION_COLUMNS = [
    "publication_id",
    "proposal_id",
    "extraction_id",
    "document_id",
    "catalyst_id",
    "publication_action",
    "publication_status",
    "before_snapshot_json",
    "after_snapshot_json",
    "proposal_snapshot_json",
    "catalyst_component_before",
    "catalyst_component_after",
    "catalyst_component_delta",
    "publisher_note",
    "published_at",
    "reverted_at",
    "revert_note",
    "created_at",
    "updated_at",
]

PUBLICATION_STATUSES = ["published", "reverted", "superseded"]
PUBLICATION_ACTIONS = ["create_new", "update_existing"]

DEFAULT_UPDATE_FIELDS = [
    "event_type",
    "title",
    "summary",
    "sentiment_label",
    "catalyst_strength",
    "confidence",
]

OPTIONAL_UPDATE_FIELDS = ["event_date", "source", "source_url"]
ALL_UPDATE_FIELDS = [*DEFAULT_UPDATE_FIELDS, *OPTIONAL_UPDATE_FIELDS]


@dataclass
class PublicationEligibility:
    eligible: bool
    reasons: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


@dataclass
class PublicationPreview:
    eligible: bool
    reasons: list[str]
    warnings: list[str]
    proposal: dict[str, Any] | None = None
    extraction: dict[str, Any] | None = None
    document: dict[str, Any] | None = None
    target_catalyst: dict[str, Any] | None = None
    proposed_catalyst: dict[str, Any] | None = None
    before_snapshot: dict[str, Any] | None = None
    after_snapshot: dict[str, Any] | None = None
    field_diff: pd.DataFrame = field(default_factory=pd.DataFrame)
    catalyst_component_before: float = 0.0
    catalyst_component_after: float = 0.0
    catalyst_component_delta: float = 0.0
    before_features: dict[str, Any] = field(default_factory=dict)
    after_features: dict[str, Any] = field(default_factory=dict)


@dataclass
class PublicationActionResult:
    changed: bool = False
    publication_id: int | None = None
    catalyst_id: int | None = None
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def create_publication_table(db_path: str | Path) -> None:
    storage.init_db(db_path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _json_loads(value: Any) -> Any:
    if value is None or value == "":
        return None
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return str(value)


def _snapshot_hash(snapshot: dict[str, Any] | None) -> str:
    return hashlib.sha256(_json_dumps(snapshot or {}).encode("utf-8")).hexdigest()


def _empty_publications_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=PUBLICATION_COLUMNS)


def _decode_publication_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    decoded = df.copy()
    for column in ["before_snapshot_json", "after_snapshot_json", "proposal_snapshot_json"]:
        decoded[column.replace("_json", "")] = decoded[column].apply(_json_loads)
    return decoded


def _query_publications(
    db_path: str | Path,
    where: str = "",
    params: tuple[Any, ...] = (),
    limit: int | None = None,
) -> pd.DataFrame:
    create_publication_table(db_path)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    sql = f"""
        SELECT {", ".join(PUBLICATION_COLUMNS)}
        FROM catalyst_publications
        {where}
        ORDER BY datetime(created_at) DESC, publication_id DESC
        {limit_clause}
    """
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return _empty_publications_frame()
    return _decode_publication_frame(df)


def get_publication_by_id(db_path: str | Path, publication_id: int) -> dict[str, Any] | None:
    create_publication_table(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute(
            f"SELECT {', '.join(PUBLICATION_COLUMNS)} FROM catalyst_publications WHERE publication_id = ?",
            (int(publication_id),),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    for column in ["before_snapshot_json", "after_snapshot_json", "proposal_snapshot_json"]:
        data[column.replace("_json", "")] = _json_loads(data.get(column))
    return data


def list_publications_by_proposal_id(db_path: str | Path, proposal_id: int, limit: int | None = 100) -> pd.DataFrame:
    return _query_publications(db_path, "WHERE proposal_id = ?", (int(proposal_id),), limit)


def list_publications_by_catalyst_id(db_path: str | Path, catalyst_id: int, limit: int | None = 100) -> pd.DataFrame:
    return _query_publications(db_path, "WHERE catalyst_id = ?", (int(catalyst_id),), limit)


def list_publications_by_ticker(db_path: str | Path, ticker: str, limit: int | None = 200) -> pd.DataFrame:
    create_publication_table(db_path)
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(
            f"""
            SELECT p.{", p.".join(PUBLICATION_COLUMNS)}
            FROM catalyst_publications p
            JOIN catalyst_proposals cp ON cp.proposal_id = p.proposal_id
            WHERE cp.ticker = ?
            ORDER BY datetime(p.created_at) DESC, p.publication_id DESC
            {"" if limit is None else f"LIMIT {int(limit)}"}
            """,
            conn,
            params=(ticker.upper().strip(),),
        )
    if df.empty:
        return _empty_publications_frame()
    return _decode_publication_frame(df)


def active_publication_for_proposal(db_path: str | Path, proposal_id: int) -> dict[str, Any] | None:
    create_publication_table(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute(
            f"""
            SELECT {", ".join(PUBLICATION_COLUMNS)}
            FROM catalyst_publications
            WHERE proposal_id = ? AND publication_status = 'published'
            ORDER BY publication_id DESC
            LIMIT 1
            """,
            (int(proposal_id),),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    for column in ["before_snapshot_json", "after_snapshot_json", "proposal_snapshot_json"]:
        data[column.replace("_json", "")] = _json_loads(data.get(column))
    return data


def _get_catalyst_by_id(db_path: str | Path, catalyst_id: int) -> dict[str, Any] | None:
    create_publication_table(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute(
            f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?",
            (int(catalyst_id),),
        ).fetchone()
    if row is None:
        return None
    return dict(row)


def _catalyst_frame_from_rows(rows: list[dict[str, Any]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=CATALYST_COLUMNS)
    return pd.DataFrame(rows)


def _net_catalyst_component(features: dict[str, Any]) -> float:
    return round(float(features.get("catalyst_score", 0) or 0) + float(features.get("catalyst_penalty", 0) or 0), 2)


def _score_catalyst_rows(ticker: str, rows: list[dict[str, Any]]) -> tuple[float, dict[str, Any]]:
    features = get_catalyst_features(ticker, _catalyst_frame_from_rows(rows))
    return _net_catalyst_component(features), features


def _event_type_for_active_catalyst(proposed_event_type: str) -> str:
    if proposed_event_type in EVENT_TYPES:
        return proposed_event_type
    if proposed_event_type in {"dilution", "insider_activity", "product_launch", "guidance_update"}:
        return "corporate_action"
    return "other"


def _sentiment_for_active_catalyst(proposed_sentiment: str) -> str:
    return proposed_sentiment if proposed_sentiment in SENTIMENT_LABELS else "unknown"


def _event_date_for_active_catalyst(value: Any) -> date:
    try:
        if value is None or pd.isna(value):
            return datetime.now(UTC).date()
        return pd.to_datetime(value).date()
    except Exception:
        return datetime.now(UTC).date()


def _published_confidence(proposal: dict[str, Any], extraction: dict[str, Any]) -> float:
    confidence = clamp_float(proposal.get("proposed_confidence"), 0.0, 1.0, 0.0)
    extraction_confidence = clamp_float(extraction.get("confidence"), 0.0, 1.0, 0.0)
    confidence = min(confidence, extraction_confidence)
    if proposal.get("evidence_sufficiency") == "limited":
        confidence = min(confidence, 0.60)
    return round(confidence, 4)


def _raw_payload_dict(value: Any) -> dict[str, Any]:
    decoded = _json_loads(value)
    if isinstance(decoded, dict):
        return decoded
    if decoded:
        return {"previous_raw_payload_json": decoded}
    return {}


def _provenance_payload(
    proposal: dict[str, Any],
    extraction: dict[str, Any],
    document: dict[str, Any],
    publication_id: int | None,
    selected_update_fields: list[str] | None = None,
    existing_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload = dict(existing_payload or {})
    publication_ref = {
        "publication_id": publication_id,
        "proposal_id": int(proposal["proposal_id"]),
        "extraction_id": int(extraction["extraction_id"]),
        "document_id": int(document["document_id"]),
        "provider": extraction.get("provider"),
        "model_name": extraction.get("model_name"),
        "prompt_version": extraction.get("prompt_version"),
        "evidence_snippets": safe_json_list(proposal.get("evidence_snippets")),
        "document_source_url": document.get("source_url"),
        "selected_update_fields": selected_update_fields or [],
        "grounding_note": "Grounded evidence proves traceability to source text, not source factual truth.",
        "manually_reviewed": True,
        "llm_supported": True,
    }
    payload["llm_supported"] = True
    payload["manually_reviewed"] = True
    payload["latest_publication_id"] = publication_id
    history = payload.get("llm_publication_history")
    if not isinstance(history, list):
        history = []
    history.append(publication_ref)
    payload["llm_publication_history"] = history
    return payload


def _proposal_snapshot(proposal: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in proposal.items()
        if key not in {"evidence_snippets_json"}
    }


def _active_event_row_from_proposal(
    proposal: dict[str, Any],
    extraction: dict[str, Any],
    document: dict[str, Any],
    publication_id: int | None = None,
) -> dict[str, Any]:
    event = CatalystEvent(
        ticker=proposal["ticker"],
        event_date=_event_date_for_active_catalyst(proposal.get("proposed_event_date")),
        event_type=_event_type_for_active_catalyst(str(proposal.get("proposed_event_type") or "unknown")),
        title=str(proposal.get("proposed_title") or "").strip(),
        summary=str(proposal.get("proposed_summary") or "").strip(),
        source="llm_supported",
        source_url=str(proposal.get("proposed_source_url") or document.get("source_url") or "").strip() or None,
        sentiment_label=_sentiment_for_active_catalyst(str(proposal.get("proposed_sentiment") or "unknown")),
        catalyst_strength=clamp_int(proposal.get("proposed_strength"), 0, 10, 0),
        confidence=_published_confidence(proposal, extraction),
        is_manual=False,
        raw_payload_json=_json_dumps(_provenance_payload(proposal, extraction, document, publication_id)),
    )
    row = _event_row(event, make_dedupe_key(event))
    # LLM-supported catalysts become available when the reviewed publication is
    # created, not on the document's historical event date. Keep the in-memory
    # snapshot aligned with the row persisted by publication inserts.
    row["available_at"] = row.get("available_at") or row.get("created_at")
    return row


def _selected_update_fields(fields: list[str] | None) -> list[str]:
    if fields is None:
        return list(DEFAULT_UPDATE_FIELDS)
    return [field for field in fields if field in ALL_UPDATE_FIELDS]


def _updates_from_proposal(
    proposal: dict[str, Any],
    extraction: dict[str, Any],
    document: dict[str, Any],
    selected_update_fields: list[str] | None,
    publication_id: int | None,
    before_snapshot: dict[str, Any],
) -> dict[str, Any]:
    selected = _selected_update_fields(selected_update_fields)
    candidate = _active_event_row_from_proposal(proposal, extraction, document, publication_id)
    field_map = {
        "event_date": "event_date",
        "event_type": "event_type",
        "title": "title",
        "summary": "summary",
        "source": "source",
        "source_url": "source_url",
        "sentiment_label": "sentiment_label",
        "catalyst_strength": "catalyst_strength",
        "confidence": "confidence",
    }
    updates = {field_map[field]: candidate[field_map[field]] for field in selected if field in field_map}
    updates["raw_payload_json"] = _json_dumps(
        _provenance_payload(
            proposal,
            extraction,
            document,
            publication_id,
            selected,
            existing_payload=_raw_payload_dict(before_snapshot.get("raw_payload_json")),
        )
    )
    return updates


def _apply_updates_to_snapshot(snapshot: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    updated = dict(snapshot)
    updated.update(updates)
    updated["updated_at"] = _now_iso()
    return updated


def _diff_display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return _json_dumps(value)
    return str(value)


def _field_diff(before: dict[str, Any] | None, after: dict[str, Any] | None) -> pd.DataFrame:
    before = before or {}
    after = after or {}
    keys = sorted(set(before) | set(after))
    rows = [
        {
            "field": key,
            "before": _diff_display_value(before.get(key)),
            "after": _diff_display_value(after.get(key)),
            "changed": before.get(key) != after.get(key),
        }
        for key in keys
        if before.get(key) != after.get(key)
    ]
    return pd.DataFrame(rows, columns=["field", "before", "after", "changed"])


def _load_publication_context(db_path: str | Path, proposal_id: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any] | None]:
    proposal = get_proposal_by_id(db_path, proposal_id)
    if not proposal:
        return None, None, None
    extraction = get_extraction_by_id(db_path, int(proposal.get("extraction_id") or 0))
    document = get_document_by_id(db_path, int(proposal.get("document_id") or 0))
    return proposal, extraction, document


def evaluate_publication_eligibility(
    db_path: str | Path,
    proposal_id: int,
    selected_update_fields: list[str] | None = None,
) -> PublicationEligibility:
    proposal, extraction, document = _load_publication_context(db_path, proposal_id)
    reasons: list[str] = []
    warnings: list[str] = []
    if not proposal:
        return PublicationEligibility(False, ["Proposal was not found."])
    if not extraction:
        reasons.append("Linked extraction was not found.")
    if not document:
        reasons.append("Linked source document was not found.")
    if reasons:
        return PublicationEligibility(False, reasons)

    assert extraction is not None and document is not None
    if extraction.get("review_status") != "approved":
        reasons.append("Extraction must be approved before publication.")
    if proposal.get("proposal_status") != "reviewed_ready":
        reasons.append("Proposal must be reviewed_ready before publication.")
    if active_publication_for_proposal(db_path, proposal_id):
        reasons.append("This proposal already has an active publication.")

    if proposal.get("document_relevance") != "relevant":
        reasons.append("Document relevance must be relevant.")
    if proposal.get("evidence_sufficiency") not in {"sufficient", "limited"}:
        reasons.append("Evidence sufficiency must be sufficient or limited.")
    if proposal.get("evidence_sufficiency") == "limited":
        warnings.append("Limited evidence caps published confidence at 0.60.")

    sentiment = str(proposal.get("proposed_sentiment") or "unknown")
    evidence = safe_json_list(proposal.get("evidence_snippets"))
    if sentiment not in {"neutral", "unknown"} and not evidence:
        reasons.append("Non-neutral proposals require at least one validated exact evidence snippet.")

    tickers = {
        str(proposal.get("ticker") or "").upper(),
        str(extraction.get("ticker") or "").upper(),
        str(document.get("ticker") or "").upper(),
    }
    if len(tickers) != 1:
        reasons.append("Ticker must match across proposal, extraction, and source document.")

    target_id = proposal.get("target_catalyst_id")
    if proposal.get("proposal_type") == "update_existing":
        if target_id is None or pd.isna(target_id):
            reasons.append("Update-existing proposals require a target catalyst.")
        else:
            target = _get_catalyst_by_id(db_path, int(target_id))
            if not target:
                reasons.append("Target catalyst was not found.")
            elif str(target.get("ticker") or "").upper() != str(proposal.get("ticker") or "").upper():
                reasons.append("Target catalyst ticker must match proposal ticker.")
    elif target_id is not None and not pd.isna(target_id):
        target = _get_catalyst_by_id(db_path, int(target_id))
        if target and str(target.get("ticker") or "").upper() != str(proposal.get("ticker") or "").upper():
            reasons.append("Target catalyst ticker must match proposal ticker.")

    selected = _selected_update_fields(selected_update_fields)
    if proposal.get("proposal_type") == "update_existing" and not selected:
        reasons.append("At least one update field must be selected.")

    return PublicationEligibility(not reasons, reasons, warnings)


def build_publication_preview(
    db_path: str | Path,
    proposal_id: int,
    selected_update_fields: list[str] | None = None,
) -> PublicationPreview:
    proposal, extraction, document = _load_publication_context(db_path, proposal_id)
    eligibility = evaluate_publication_eligibility(db_path, proposal_id, selected_update_fields)
    if not proposal or not extraction or not document:
        return PublicationPreview(False, eligibility.reasons, eligibility.warnings, proposal, extraction, document)

    ticker = str(proposal.get("ticker") or "").upper()
    current_events = list_catalysts_by_ticker(db_path, ticker, limit=500)
    current_rows = current_events.to_dict("records") if not current_events.empty else []
    before_component, before_features = _score_catalyst_rows(ticker, current_rows)

    before_snapshot: dict[str, Any] | None = None
    after_snapshot: dict[str, Any] | None = None
    target_catalyst: dict[str, Any] | None = None
    proposed_catalyst: dict[str, Any] | None = None
    after_rows = list(current_rows)

    if proposal.get("proposal_type") == "update_existing":
        target_id = int(proposal.get("target_catalyst_id") or 0)
        target_catalyst = _get_catalyst_by_id(db_path, target_id)
        before_snapshot = dict(target_catalyst or {})
        if target_catalyst:
            updates = _updates_from_proposal(proposal, extraction, document, selected_update_fields, None, before_snapshot)
            after_snapshot = _apply_updates_to_snapshot(target_catalyst, updates)
            proposed_catalyst = after_snapshot
            after_rows = [after_snapshot if int(row.get("id") or 0) == target_id else row for row in after_rows]
    else:
        proposed_catalyst = _active_event_row_from_proposal(proposal, extraction, document, publication_id=None)
        proposed_catalyst["id"] = "preview"
        before_snapshot = {}
        after_snapshot = proposed_catalyst
        after_rows.append(proposed_catalyst)

    after_component, after_features = _score_catalyst_rows(ticker, after_rows)
    return PublicationPreview(
        eligible=eligibility.eligible,
        reasons=eligibility.reasons,
        warnings=eligibility.warnings,
        proposal=proposal,
        extraction=extraction,
        document=document,
        target_catalyst=target_catalyst,
        proposed_catalyst=proposed_catalyst,
        before_snapshot=before_snapshot,
        after_snapshot=after_snapshot,
        field_diff=_field_diff(before_snapshot, after_snapshot),
        catalyst_component_before=before_component,
        catalyst_component_after=after_component,
        catalyst_component_delta=round(after_component - before_component, 2),
        before_features=before_features,
        after_features=after_features,
    )


def _insert_publication_row(
    conn,
    proposal: dict[str, Any],
    catalyst_id: int,
    before_snapshot: dict[str, Any] | None,
    after_snapshot: dict[str, Any],
    component_before: float,
    component_after: float,
    publisher_note: str,
) -> int:
    now = _now_iso()
    conn.execute(
        """
        INSERT INTO catalyst_publications (
            proposal_id, extraction_id, document_id, catalyst_id, publication_action,
            publication_status, before_snapshot_json, after_snapshot_json, proposal_snapshot_json,
            catalyst_component_before, catalyst_component_after, catalyst_component_delta,
            publisher_note, published_at, reverted_at, revert_note, created_at, updated_at
        )
        VALUES (?, ?, ?, ?, ?, 'published', ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, ?, ?)
        """,
        (
            int(proposal["proposal_id"]),
            int(proposal["extraction_id"]),
            int(proposal["document_id"]),
            int(catalyst_id),
            proposal.get("proposal_type") or "create_new",
            _json_dumps(before_snapshot or {}),
            _json_dumps(after_snapshot),
            _json_dumps(_proposal_snapshot(proposal)),
            float(component_before),
            float(component_after),
            round(float(component_after) - float(component_before), 2),
            publisher_note.strip(),
            now,
            now,
            now,
        ),
    )
    return int(conn.execute("SELECT last_insert_rowid() AS publication_id").fetchone()["publication_id"])


def _insert_active_catalyst(conn, row: dict[str, Any]) -> int:
    conn.execute(
        """
        INSERT INTO catalysts (
            ticker, event_date, event_time, event_type, title, summary, source, source_url,
            sentiment_label, catalyst_strength, confidence, is_manual, available_at, created_at, updated_at,
            raw_payload_json, dedupe_key
        )
        VALUES (
            :ticker, :event_date, :event_time, :event_type, :title, :summary, :source, :source_url,
            :sentiment_label, :catalyst_strength, :confidence, :is_manual, :available_at, :created_at, :updated_at,
            :raw_payload_json, :dedupe_key
        )
        """,
        row,
    )
    return int(conn.execute("SELECT last_insert_rowid() AS catalyst_id").fetchone()["catalyst_id"])


def _update_active_catalyst(conn, catalyst_id: int, updates: dict[str, Any]) -> None:
    row = dict(updates)
    row["updated_at"] = _now_iso()
    row["id"] = int(catalyst_id)
    assignments = ", ".join(f"{key} = :{key}" for key in row if key != "id")
    conn.execute(f"UPDATE catalysts SET {assignments} WHERE id = :id", row)


def publish_proposal(
    db_path: str | Path,
    proposal_id: int,
    publisher_note: str,
    selected_update_fields: list[str] | None = None,
    *,
    _simulate_failure_after_mutation: bool = False,
) -> PublicationActionResult:
    if not str(publisher_note or "").strip():
        return PublicationActionResult(False, message="Publisher note is required.")
    preview = build_publication_preview(db_path, proposal_id, selected_update_fields)
    if not preview.eligible:
        return PublicationActionResult(False, message="Publication is blocked.", warnings=preview.reasons)
    assert preview.proposal is not None and preview.extraction is not None and preview.document is not None
    proposal = preview.proposal
    extraction = preview.extraction
    document = preview.document

    create_publication_table(db_path)
    try:
        with storage.connect(db_path) as conn:
            conn.execute("BEGIN")
            existing_publication = conn.execute(
                """
                SELECT publication_id
                FROM catalyst_publications
                WHERE proposal_id = ? AND publication_status = 'published'
                LIMIT 1
                """,
                (int(proposal_id),),
            ).fetchone()
            if existing_publication is not None:
                raise ValueError("This proposal already has an active publication.")
            if proposal.get("proposal_type") == "update_existing":
                catalyst_id = int(proposal.get("target_catalyst_id") or 0)
                before_row = conn.execute(
                    f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?",
                    (catalyst_id,),
                ).fetchone()
                if before_row is None:
                    raise ValueError("Target catalyst was not found.")
                before_snapshot = dict(before_row)
                selected = _selected_update_fields(selected_update_fields)
                temp_updates = _updates_from_proposal(proposal, extraction, document, selected, None, before_snapshot)
                temp_after = _apply_updates_to_snapshot(before_snapshot, temp_updates)
                after_component = preview.catalyst_component_after
                publication_id = _insert_publication_row(
                    conn,
                    proposal,
                    catalyst_id,
                    before_snapshot,
                    temp_after,
                    preview.catalyst_component_before,
                    after_component,
                    publisher_note,
                )
                updates = _updates_from_proposal(proposal, extraction, document, selected, publication_id, before_snapshot)
                _update_active_catalyst(conn, catalyst_id, updates)
                if _simulate_failure_after_mutation:
                    raise RuntimeError("Simulated publication failure.")
                after_snapshot = dict(conn.execute(
                    f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?",
                    (catalyst_id,),
                ).fetchone())
                conn.execute(
                    """
                    UPDATE catalyst_publications
                    SET after_snapshot_json = ?, updated_at = ?
                    WHERE publication_id = ?
                    """,
                    (_json_dumps(after_snapshot), _now_iso(), publication_id),
                )
            else:
                event_row = _active_event_row_from_proposal(proposal, extraction, document, publication_id=None)
                catalyst_id = _insert_active_catalyst(conn, event_row)
                if _simulate_failure_after_mutation:
                    raise RuntimeError("Simulated publication failure.")
                temp_after = dict(conn.execute(
                    f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?",
                    (catalyst_id,),
                ).fetchone())
                publication_id = _insert_publication_row(
                    conn,
                    proposal,
                    catalyst_id,
                    {},
                    temp_after,
                    preview.catalyst_component_before,
                    preview.catalyst_component_after,
                    publisher_note,
                )
                payload = _provenance_payload(proposal, extraction, document, publication_id)
                conn.execute(
                    "UPDATE catalysts SET raw_payload_json = ?, updated_at = ? WHERE id = ?",
                    (_json_dumps(payload), _now_iso(), catalyst_id),
                )
                after_snapshot = dict(conn.execute(
                    f"SELECT {', '.join(CATALYST_COLUMNS)} FROM catalysts WHERE id = ?",
                    (catalyst_id,),
                ).fetchone())
                conn.execute(
                    """
                    UPDATE catalyst_publications
                    SET after_snapshot_json = ?, updated_at = ?
                    WHERE publication_id = ?
                    """,
                    (_json_dumps(after_snapshot), _now_iso(), publication_id),
                )

            conn.execute(
                "UPDATE source_documents SET catalyst_id = ?, updated_at = ? WHERE document_id = ?",
                (catalyst_id, _now_iso(), int(document["document_id"])),
            )
            conn.execute(
                "UPDATE llm_extractions SET catalyst_id = ?, updated_at = ? WHERE extraction_id = ?",
                (catalyst_id, _now_iso(), int(extraction["extraction_id"])),
            )
            conn.commit()
    except Exception as exc:
        return PublicationActionResult(False, message=f"Publication failed: {exc}")

    return PublicationActionResult(
        True,
        publication_id=publication_id,
        catalyst_id=catalyst_id,
        message=f"Published proposal #{proposal_id} into active catalyst #{catalyst_id}.",
    )


def _publication_conflict(current: dict[str, Any] | None, expected: dict[str, Any] | None) -> bool:
    if current is None:
        return True
    return _snapshot_hash(current) != _snapshot_hash(expected)


def revert_publication(db_path: str | Path, publication_id: int, revert_note: str) -> PublicationActionResult:
    if not str(revert_note or "").strip():
        return PublicationActionResult(False, message="Revert note is required.")
    publication = get_publication_by_id(db_path, publication_id)
    if not publication:
        return PublicationActionResult(False, message="Publication was not found.")
    if publication.get("publication_status") != "published":
        return PublicationActionResult(False, message="Only currently published records can be reverted.")

    catalyst_id = int(publication["catalyst_id"])
    current = _get_catalyst_by_id(db_path, catalyst_id)
    expected_after = publication.get("after_snapshot")
    if _publication_conflict(current, expected_after):
        return PublicationActionResult(
            False,
            catalyst_id=catalyst_id,
            message="Automatic reversal blocked: active catalyst changed after publication.",
            warnings=["Resolve the catalyst manually; the publication audit record was not changed."],
        )

    before = publication.get("before_snapshot") or {}
    now = _now_iso()
    create_publication_table(db_path)
    with storage.connect(db_path) as conn:
        conn.execute("BEGIN")
        if publication.get("publication_action") == "create_new":
            conn.execute("DELETE FROM catalysts WHERE id = ?", (catalyst_id,))
        else:
            restore = {key: before.get(key) for key in CATALYST_COLUMNS if key != "id"}
            restore["id"] = catalyst_id
            assignments = ", ".join(f"{key} = :{key}" for key in restore if key != "id")
            conn.execute(f"UPDATE catalysts SET {assignments} WHERE id = :id", restore)
        conn.execute(
            """
            UPDATE catalyst_publications
            SET publication_status = 'reverted',
                reverted_at = ?,
                revert_note = ?,
                updated_at = ?
            WHERE publication_id = ?
            """,
            (now, revert_note.strip(), now, int(publication_id)),
        )
        conn.commit()
    return PublicationActionResult(
        True,
        publication_id=int(publication_id),
        catalyst_id=catalyst_id,
        message=f"Reverted publication #{publication_id}.",
    )


def publication_summary_by_catalyst(db_path: str | Path, catalyst_ids: list[int]) -> dict[int, dict[str, Any]]:
    create_publication_table(db_path)
    ids = [int(catalyst_id) for catalyst_id in catalyst_ids if pd.notna(catalyst_id)]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT catalyst_id,
                   COUNT(*) AS publication_count,
                   SUM(CASE WHEN publication_status = 'published' THEN 1 ELSE 0 END) AS active_publication_count,
                   MAX(publication_id) AS latest_publication_id
            FROM catalyst_publications
            WHERE catalyst_id IN ({placeholders})
            GROUP BY catalyst_id
            """,
            ids,
        ).fetchall()
    return {
        int(row["catalyst_id"]): {
            "publication_count": int(row["publication_count"] or 0),
            "active_publication_count": int(row["active_publication_count"] or 0),
            "latest_publication_id": int(row["latest_publication_id"]) if row["latest_publication_id"] is not None else None,
        }
        for row in rows
    }


def is_llm_supported_catalyst(row: dict[str, Any] | pd.Series) -> bool:
    payload = _raw_payload_dict(row.get("raw_payload_json") if hasattr(row, "get") else None)
    return bool(payload.get("llm_supported"))


def llm_publication_id_from_catalyst(row: dict[str, Any] | pd.Series) -> int | None:
    payload = _raw_payload_dict(row.get("raw_payload_json") if hasattr(row, "get") else None)
    try:
        publication_id = payload.get("latest_publication_id")
        return int(publication_id) if publication_id is not None else None
    except Exception:
        return None


def publication_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "publication_id",
        "proposal_id",
        "catalyst_id",
        "publication_action",
        "publication_status",
        "catalyst_component_before",
        "catalyst_component_after",
        "catalyst_component_delta",
        "published_at",
        "reverted_at",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    display = df.copy()
    for column in columns:
        if column not in display.columns:
            display[column] = ""
    return display[columns]
