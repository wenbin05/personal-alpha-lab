from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from src.catalysts.models import EVENT_TYPES
from src.catalysts.repository import list_catalysts_by_ticker
from src.data import storage
from src.documents.repository import get_document_by_id
from src.extractions.quality import ReviewReadiness, classify_review_readiness
from src.extractions.repository import get_extraction_by_id
from src.extractions.validation import (
    clamp_float,
    clamp_int,
    enum_or_default,
    json_list_dumps,
    optional_int,
    safe_json_list,
)


ProposalType = Literal["create_new", "update_existing"]
ProposalStatus = Literal["draft", "reviewed_ready", "rejected", "superseded"]
LinkStatus = Literal["active", "unlinked"]

PROPOSAL_TYPES = ["create_new", "update_existing"]
PROPOSAL_STATUSES = ["draft", "reviewed_ready", "rejected", "superseded"]
LINK_STATUSES = ["active", "unlinked"]

PROPOSAL_EVENT_TYPES = [
    "earnings",
    "sec_filing",
    "news",
    "analyst",
    "manual_note",
    "corporate_action",
    "dilution",
    "insider_activity",
    "product_launch",
    "guidance_update",
    "legal_regulatory",
    "macro_sensitive",
    "other",
    "unknown",
]

PROPOSAL_SENTIMENTS = ["positive", "neutral", "negative", "mixed", "unknown"]

PROPOSAL_COLUMNS = [
    "proposal_id",
    "extraction_id",
    "document_id",
    "ticker",
    "target_catalyst_id",
    "proposal_type",
    "proposed_event_type",
    "proposed_event_date",
    "proposed_title",
    "proposed_summary",
    "proposed_sentiment",
    "proposed_strength",
    "proposed_confidence",
    "proposed_source",
    "proposed_source_url",
    "evidence_snippets_json",
    "risk_severity",
    "document_relevance",
    "evidence_sufficiency",
    "proposal_status",
    "reviewer_note",
    "initiated_by",
    "created_at",
    "updated_at",
    "reviewed_at",
]

LINK_COLUMNS = [
    "link_id",
    "extraction_id",
    "document_id",
    "ticker",
    "catalyst_id",
    "link_status",
    "reviewer_note",
    "initiated_by",
    "created_at",
    "updated_at",
    "unlinked_at",
]


class CatalystProposal(BaseModel):
    proposal_id: int | None = None
    extraction_id: int
    document_id: int
    ticker: str
    target_catalyst_id: int | None = None
    proposal_type: ProposalType = "create_new"
    proposed_event_type: str = "unknown"
    proposed_event_date: date | None = None
    proposed_title: str
    proposed_summary: str = ""
    proposed_sentiment: str = "unknown"
    proposed_strength: int = Field(default=0, ge=0, le=10)
    proposed_confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    proposed_source: str = "unknown"
    proposed_source_url: str | None = None
    evidence_snippets: list[str] = Field(default_factory=list)
    risk_severity: int = Field(default=0, ge=0, le=10)
    document_relevance: str = "unknown"
    evidence_sufficiency: str = "unknown"
    proposal_status: ProposalStatus = "draft"
    reviewer_note: str = ""
    initiated_by: str = "local_user"
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    reviewed_at: datetime | None = None

    @field_validator("ticker")
    @classmethod
    def normalize_ticker(cls, value: str) -> str:
        cleaned = (value or "").strip().upper()
        if not cleaned:
            raise ValueError("Ticker is required.")
        return cleaned

    @field_validator("proposed_title")
    @classmethod
    def clean_required_text(cls, value: str) -> str:
        cleaned = (value or "").strip()
        if not cleaned:
            raise ValueError("Proposal title is required.")
        return cleaned

    @field_validator(
        "proposed_event_type",
        "proposed_summary",
        "proposed_sentiment",
        "proposed_source",
        "reviewer_note",
        "initiated_by",
        "document_relevance",
        "evidence_sufficiency",
    )
    @classmethod
    def clean_text(cls, value: str | None) -> str:
        return (value or "").strip()

    @field_validator("evidence_snippets", mode="before")
    @classmethod
    def coerce_evidence(cls, value: object) -> list[str]:
        return safe_json_list(value)


@dataclass
class ProposalActionResult:
    changed: bool = False
    proposal_id: int | None = None
    link_id: int | None = None
    message: str = ""
    warnings: list[str] | None = None


def create_proposal_tables(db_path: str | Path) -> None:
    storage.init_db(db_path)


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _dt_iso(value: datetime | date | str | None) -> str | None:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.isoformat(timespec="seconds")
    if isinstance(value, date):
        return value.isoformat()
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    if getattr(parsed, "hour", 0) or getattr(parsed, "minute", 0) or getattr(parsed, "second", 0):
        return parsed.to_pydatetime().isoformat(timespec="seconds")
    return parsed.date().isoformat()


def _date_iso(value: Any) -> str | None:
    if value is None or value == "":
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date().isoformat()


def _parse_date(value: Any) -> date | None:
    if value is None or value == "":
        return None
    try:
        parsed = pd.to_datetime(value)
    except Exception:
        return None
    if pd.isna(parsed):
        return None
    return parsed.date()


def _proposal_row(proposal: CatalystProposal | dict[str, Any]) -> dict[str, Any]:
    if isinstance(proposal, CatalystProposal):
        payload = proposal.model_dump()
    else:
        payload = dict(proposal or {})

    now = _now_iso()
    proposal_type = enum_or_default(payload.get("proposal_type"), PROPOSAL_TYPES, "create_new")
    target_catalyst_id = optional_int(payload.get("target_catalyst_id"))
    if target_catalyst_id is not None:
        proposal_type = "update_existing"

    return {
        "extraction_id": clamp_int(payload.get("extraction_id"), 0, 2_147_483_647, 0),
        "document_id": clamp_int(payload.get("document_id"), 0, 2_147_483_647, 0),
        "ticker": str(payload.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN",
        "target_catalyst_id": target_catalyst_id,
        "proposal_type": proposal_type,
        "proposed_event_type": enum_or_default(
            payload.get("proposed_event_type"),
            PROPOSAL_EVENT_TYPES,
            "unknown",
        ),
        "proposed_event_date": _date_iso(payload.get("proposed_event_date")),
        "proposed_title": str(payload.get("proposed_title") or "Untitled catalyst proposal").strip(),
        "proposed_summary": str(payload.get("proposed_summary") or "").strip(),
        "proposed_sentiment": enum_or_default(payload.get("proposed_sentiment"), PROPOSAL_SENTIMENTS, "unknown"),
        "proposed_strength": clamp_int(payload.get("proposed_strength"), 0, 10, 0),
        "proposed_confidence": clamp_float(payload.get("proposed_confidence"), 0.0, 1.0, 0.0),
        "proposed_source": str(payload.get("proposed_source") or "unknown").strip() or "unknown",
        "proposed_source_url": str(payload.get("proposed_source_url") or "").strip() or None,
        "evidence_snippets_json": json_list_dumps(
            payload.get("evidence_snippets", payload.get("evidence_snippets_json"))
        ),
        "risk_severity": clamp_int(payload.get("risk_severity"), 0, 10, 0),
        "document_relevance": enum_or_default(
            payload.get("document_relevance"),
            ["relevant", "uncertain", "irrelevant", "unknown"],
            "unknown",
        ),
        "evidence_sufficiency": enum_or_default(
            payload.get("evidence_sufficiency"),
            ["sufficient", "limited", "insufficient", "unknown"],
            "unknown",
        ),
        "proposal_status": enum_or_default(payload.get("proposal_status"), PROPOSAL_STATUSES, "draft"),
        "reviewer_note": str(payload.get("reviewer_note") or "").strip(),
        "initiated_by": str(payload.get("initiated_by") or "local_user").strip() or "local_user",
        "created_at": _dt_iso(payload.get("created_at")) or now,
        "updated_at": _dt_iso(payload.get("updated_at")) or now,
        "reviewed_at": _dt_iso(payload.get("reviewed_at")),
    }


def _decode_proposal_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    decoded = df.copy()
    decoded["evidence_snippets"] = decoded["evidence_snippets_json"].apply(safe_json_list)
    return decoded


def _empty_proposals_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=[*PROPOSAL_COLUMNS, "evidence_snippets"])
    return frame


def _empty_links_frame() -> pd.DataFrame:
    return pd.DataFrame(columns=LINK_COLUMNS)


def _document_event_date(document: dict[str, Any] | None) -> date | None:
    if not document:
        return None
    return _parse_date(document.get("published_at")) or _parse_date(document.get("created_at"))


def _proposal_title(extraction: dict[str, Any], document: dict[str, Any] | None) -> str:
    ticker = str(extraction.get("ticker") or "").upper()
    event_type = str(extraction.get("event_type_detected") or "unknown").replace("_", " ")
    doc_title = str((document or {}).get("title") or "").strip()
    base = f"{ticker} {event_type} proposal"
    if doc_title:
        base = f"{base}: {doc_title}"
    return base[:180]


def map_extraction_to_proposal(
    extraction: dict[str, Any],
    document: dict[str, Any] | None,
    target_catalyst_id: int | None = None,
    reviewer_note: str = "",
) -> CatalystProposal:
    """Create a deterministic, non-scoring proposal object from reviewed extraction output."""
    target_id = optional_int(target_catalyst_id)
    return CatalystProposal(
        extraction_id=int(extraction.get("extraction_id") or 0),
        document_id=int(extraction.get("document_id") or 0),
        ticker=str(extraction.get("ticker") or "UNKNOWN"),
        target_catalyst_id=target_id,
        proposal_type="update_existing" if target_id is not None else "create_new",
        proposed_event_type=enum_or_default(
            extraction.get("event_type_detected"),
            PROPOSAL_EVENT_TYPES,
            "unknown",
        ),
        proposed_event_date=_document_event_date(document),
        proposed_title=_proposal_title(extraction, document),
        proposed_summary=str(extraction.get("short_summary") or "").strip(),
        proposed_sentiment=enum_or_default(extraction.get("sentiment_label"), PROPOSAL_SENTIMENTS, "unknown"),
        proposed_strength=clamp_int(extraction.get("catalyst_strength"), 0, 10, 0),
        proposed_confidence=clamp_float(extraction.get("confidence"), 0.0, 1.0, 0.0),
        proposed_source=str((document or {}).get("source") or extraction.get("provider") or "unknown"),
        proposed_source_url=str((document or {}).get("source_url") or "").strip() or None,
        evidence_snippets=safe_json_list(extraction.get("evidence_snippets")),
        risk_severity=clamp_int(extraction.get("risk_severity"), 0, 10, 0),
        document_relevance=enum_or_default(
            extraction.get("document_relevance"),
            ["relevant", "uncertain", "irrelevant", "unknown"],
            "unknown",
        ),
        evidence_sufficiency=enum_or_default(
            extraction.get("evidence_sufficiency"),
            ["sufficient", "limited", "insufficient", "unknown"],
            "unknown",
        ),
        proposal_status="draft",
        reviewer_note=reviewer_note,
    )


def proposal_requirements_met(
    extraction: dict[str, Any] | None,
    reviewer_note: str = "",
    override_weak_readiness: bool = False,
) -> tuple[bool, str, ReviewReadiness | None]:
    if not extraction:
        return False, "Extraction was not found.", None
    if extraction.get("review_status") != "approved":
        return False, "Only approved extractions can create catalyst proposals or links.", None

    readiness = classify_review_readiness(extraction)
    if readiness == "ready_for_review":
        return True, "", readiness
    if not override_weak_readiness:
        return False, f"Proposal requires explicit override because readiness is {readiness}.", readiness
    if not str(reviewer_note or "").strip():
        return False, f"Proposal override for {readiness} requires a reviewer note.", readiness
    return True, "", readiness


def insert_proposal(db_path: str | Path, proposal: CatalystProposal | dict[str, Any]) -> int:
    create_proposal_tables(db_path)
    row = _proposal_row(proposal)
    with storage.connect(db_path) as conn:
        conn.execute(
            """
            INSERT INTO catalyst_proposals (
                extraction_id, document_id, ticker, target_catalyst_id, proposal_type,
                proposed_event_type, proposed_event_date, proposed_title, proposed_summary,
                proposed_sentiment, proposed_strength, proposed_confidence, proposed_source,
                proposed_source_url, evidence_snippets_json, risk_severity, document_relevance,
                evidence_sufficiency, proposal_status, reviewer_note, initiated_by,
                created_at, updated_at, reviewed_at
            )
            VALUES (
                :extraction_id, :document_id, :ticker, :target_catalyst_id, :proposal_type,
                :proposed_event_type, :proposed_event_date, :proposed_title, :proposed_summary,
                :proposed_sentiment, :proposed_strength, :proposed_confidence, :proposed_source,
                :proposed_source_url, :evidence_snippets_json, :risk_severity, :document_relevance,
                :evidence_sufficiency, :proposal_status, :reviewer_note, :initiated_by,
                :created_at, :updated_at, :reviewed_at
            )
            """,
            row,
        )
        inserted = conn.execute("SELECT last_insert_rowid() AS proposal_id").fetchone()
    return int(inserted["proposal_id"])


def get_proposal_by_id(db_path: str | Path, proposal_id: int) -> dict[str, Any] | None:
    create_proposal_tables(db_path)
    with storage.connect(db_path) as conn:
        row = conn.execute(
            f"SELECT {', '.join(PROPOSAL_COLUMNS)} FROM catalyst_proposals WHERE proposal_id = ?",
            (int(proposal_id),),
        ).fetchone()
    if row is None:
        return None
    data = dict(row)
    data["evidence_snippets"] = safe_json_list(data.get("evidence_snippets_json"))
    return data


def update_proposal(db_path: str | Path, proposal_id: int, updates: dict[str, Any]) -> bool:
    create_proposal_tables(db_path)
    allowed = {
        "target_catalyst_id",
        "proposal_type",
        "proposed_event_type",
        "proposed_event_date",
        "proposed_title",
        "proposed_summary",
        "proposed_sentiment",
        "proposed_strength",
        "proposed_confidence",
        "proposed_source",
        "proposed_source_url",
        "evidence_snippets",
        "evidence_snippets_json",
        "risk_severity",
        "document_relevance",
        "evidence_sufficiency",
        "proposal_status",
        "reviewer_note",
        "reviewed_at",
    }
    current = get_proposal_by_id(db_path, proposal_id)
    if not current:
        return False
    cleaned = {key: value for key, value in updates.items() if key in allowed}
    if not cleaned:
        return False
    row = _proposal_row({**current, **cleaned})
    row["updated_at"] = _now_iso()
    if row["proposal_status"] in {"reviewed_ready", "rejected", "superseded"} and not row.get("reviewed_at"):
        row["reviewed_at"] = _now_iso()
    if row["proposal_status"] == "draft":
        row["reviewed_at"] = None
    assignments = ", ".join(f"{key} = :{key}" for key in row if key not in {"extraction_id", "document_id", "ticker", "created_at"})
    row["proposal_id"] = int(proposal_id)
    with storage.connect(db_path) as conn:
        result = conn.execute(
            f"UPDATE catalyst_proposals SET {assignments} WHERE proposal_id = :proposal_id",
            row,
        )
        return result.rowcount > 0


def set_proposal_status(
    db_path: str | Path,
    proposal_id: int,
    status: str,
    reviewer_note: str = "",
) -> bool:
    if status not in PROPOSAL_STATUSES:
        return False
    return update_proposal(
        db_path,
        proposal_id,
        {
            "proposal_status": status,
            "reviewer_note": reviewer_note,
            "reviewed_at": _now_iso() if status != "draft" else None,
        },
    )


def _query_proposals(
    db_path: str | Path,
    where: str = "",
    params: tuple[Any, ...] = (),
    limit: int | None = None,
) -> pd.DataFrame:
    create_proposal_tables(db_path)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    sql = f"""
        SELECT {", ".join(PROPOSAL_COLUMNS)}
        FROM catalyst_proposals
        {where}
        ORDER BY datetime(created_at) DESC, proposal_id DESC
        {limit_clause}
    """
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    if df.empty:
        return _empty_proposals_frame()
    return _decode_proposal_frame(df)


def list_recent_proposals(db_path: str | Path, limit: int | None = 500) -> pd.DataFrame:
    return _query_proposals(db_path, limit=limit)


def list_proposals_by_ticker(db_path: str | Path, ticker: str, limit: int | None = 200) -> pd.DataFrame:
    return _query_proposals(db_path, "WHERE ticker = ?", (ticker.upper().strip(),), limit)


def list_proposals_by_extraction_id(db_path: str | Path, extraction_id: int, limit: int | None = 100) -> pd.DataFrame:
    return _query_proposals(db_path, "WHERE extraction_id = ?", (int(extraction_id),), limit)


def list_proposals_by_target_catalyst_id(db_path: str | Path, catalyst_id: int, limit: int | None = 100) -> pd.DataFrame:
    return _query_proposals(db_path, "WHERE target_catalyst_id = ?", (int(catalyst_id),), limit)


def _catalyst_row_by_id(db_path: str | Path, ticker: str, catalyst_id: int) -> dict[str, Any] | None:
    catalysts = list_catalysts_by_ticker(db_path, ticker, limit=500)
    if catalysts.empty:
        return None
    matched = catalysts[catalysts["id"].astype(int).eq(int(catalyst_id))]
    if matched.empty:
        return None
    return matched.iloc[0].to_dict()


def create_proposal_from_extraction(
    db_path: str | Path,
    extraction_id: int,
    target_catalyst_id: int | None = None,
    reviewer_note: str = "",
    override_weak_readiness: bool = False,
) -> ProposalActionResult:
    extraction = get_extraction_by_id(db_path, extraction_id)
    allowed, reason, readiness = proposal_requirements_met(extraction, reviewer_note, override_weak_readiness)
    if not allowed:
        return ProposalActionResult(changed=False, message=reason, warnings=[reason])
    assert extraction is not None
    document = get_document_by_id(db_path, int(extraction.get("document_id") or 0))
    if not document:
        return ProposalActionResult(changed=False, message="Source document was not found.", warnings=["Source document was not found."])

    target_id = optional_int(target_catalyst_id)
    if target_id is not None and _catalyst_row_by_id(db_path, extraction["ticker"], target_id) is None:
        message = "Target catalyst must exist and have the same ticker as the approved extraction."
        return ProposalActionResult(changed=False, message=message, warnings=[message])

    proposal = map_extraction_to_proposal(extraction, document, target_id, reviewer_note)
    proposal_id = insert_proposal(db_path, proposal)
    suffix = "" if readiness == "ready_for_review" else f" Created with explicit {readiness} override."
    return ProposalActionResult(
        changed=True,
        proposal_id=proposal_id,
        message=f"Created non-scoring catalyst proposal #{proposal_id}.{suffix}",
        warnings=[],
    )


def _link_row(row: dict[str, Any]) -> dict[str, Any]:
    now = _now_iso()
    return {
        "extraction_id": clamp_int(row.get("extraction_id"), 0, 2_147_483_647, 0),
        "document_id": clamp_int(row.get("document_id"), 0, 2_147_483_647, 0),
        "ticker": str(row.get("ticker") or "UNKNOWN").strip().upper() or "UNKNOWN",
        "catalyst_id": clamp_int(row.get("catalyst_id"), 0, 2_147_483_647, 0),
        "link_status": enum_or_default(row.get("link_status"), LINK_STATUSES, "active"),
        "reviewer_note": str(row.get("reviewer_note") or "").strip(),
        "initiated_by": str(row.get("initiated_by") or "local_user").strip() or "local_user",
        "created_at": _dt_iso(row.get("created_at")) or now,
        "updated_at": _dt_iso(row.get("updated_at")) or now,
        "unlinked_at": _dt_iso(row.get("unlinked_at")),
    }


def link_extraction_to_catalyst(
    db_path: str | Path,
    extraction_id: int,
    catalyst_id: int,
    reviewer_note: str = "",
) -> ProposalActionResult:
    extraction = get_extraction_by_id(db_path, extraction_id)
    allowed, reason, _ = proposal_requirements_met(extraction, reviewer_note, override_weak_readiness=True)
    if not allowed:
        return ProposalActionResult(changed=False, message=reason, warnings=[reason])
    assert extraction is not None
    catalyst = _catalyst_row_by_id(db_path, extraction["ticker"], catalyst_id)
    if catalyst is None:
        message = "Target catalyst must exist and have the same ticker as the approved extraction."
        return ProposalActionResult(changed=False, message=message, warnings=[message])
    if not str(reviewer_note or "").strip():
        message = "Linking an extraction to an existing catalyst requires a reviewer note."
        return ProposalActionResult(changed=False, message=message, warnings=[message])

    row = _link_row(
        {
            "extraction_id": extraction["extraction_id"],
            "document_id": extraction["document_id"],
            "ticker": extraction["ticker"],
            "catalyst_id": catalyst_id,
            "reviewer_note": reviewer_note,
        }
    )
    create_proposal_tables(db_path)
    with storage.connect(db_path) as conn:
        existing = conn.execute(
            """
            SELECT link_id
            FROM extraction_catalyst_links
            WHERE extraction_id = ? AND catalyst_id = ? AND link_status = 'active'
            ORDER BY link_id DESC
            LIMIT 1
            """,
            (int(extraction_id), int(catalyst_id)),
        ).fetchone()
        if existing is not None:
            return ProposalActionResult(
                changed=False,
                link_id=int(existing["link_id"]),
                message=f"Extraction is already actively linked to catalyst #{catalyst_id}.",
            )
        conn.execute(
            """
            INSERT INTO extraction_catalyst_links (
                extraction_id, document_id, ticker, catalyst_id, link_status,
                reviewer_note, initiated_by, created_at, updated_at, unlinked_at
            )
            VALUES (
                :extraction_id, :document_id, :ticker, :catalyst_id, :link_status,
                :reviewer_note, :initiated_by, :created_at, :updated_at, :unlinked_at
            )
            """,
            row,
        )
        inserted = conn.execute("SELECT last_insert_rowid() AS link_id").fetchone()
    link_id = int(inserted["link_id"])
    return ProposalActionResult(
        changed=True,
        link_id=link_id,
        message=f"Linked extraction #{extraction_id} to catalyst #{catalyst_id}. Active catalyst fields were not changed.",
    )


def unlink_extraction_catalyst_link(db_path: str | Path, link_id: int, reviewer_note: str = "") -> ProposalActionResult:
    if not str(reviewer_note or "").strip():
        return ProposalActionResult(changed=False, message="Unlinking requires a reviewer note.")
    create_proposal_tables(db_path)
    now = _now_iso()
    with storage.connect(db_path) as conn:
        current = conn.execute(
            "SELECT link_id, link_status FROM extraction_catalyst_links WHERE link_id = ?",
            (int(link_id),),
        ).fetchone()
        if current is None:
            return ProposalActionResult(changed=False, message="Link was not found.")
        if current["link_status"] != "active":
            return ProposalActionResult(changed=False, message="Link is already inactive.")
        result = conn.execute(
            """
            UPDATE extraction_catalyst_links
            SET link_status = 'unlinked',
                reviewer_note = ?,
                updated_at = ?,
                unlinked_at = ?
            WHERE link_id = ?
            """,
            (reviewer_note.strip(), now, now, int(link_id)),
        )
    return ProposalActionResult(changed=result.rowcount > 0, message=f"Unlinked extraction-catalyst link #{link_id}.")


def _query_links(
    db_path: str | Path,
    where: str = "",
    params: tuple[Any, ...] = (),
    limit: int | None = None,
) -> pd.DataFrame:
    create_proposal_tables(db_path)
    limit_clause = "" if limit is None else f" LIMIT {int(limit)}"
    sql = f"""
        SELECT {", ".join(LINK_COLUMNS)}
        FROM extraction_catalyst_links
        {where}
        ORDER BY datetime(created_at) DESC, link_id DESC
        {limit_clause}
    """
    with storage.connect(db_path) as conn:
        df = pd.read_sql_query(sql, conn, params=params)
    return _empty_links_frame() if df.empty else df


def list_links_by_ticker(db_path: str | Path, ticker: str, limit: int | None = 200) -> pd.DataFrame:
    return _query_links(db_path, "WHERE ticker = ?", (ticker.upper().strip(),), limit)


def list_links_by_extraction_id(db_path: str | Path, extraction_id: int, limit: int | None = 100) -> pd.DataFrame:
    return _query_links(db_path, "WHERE extraction_id = ?", (int(extraction_id),), limit)


def list_links_by_catalyst_id(db_path: str | Path, catalyst_id: int, limit: int | None = 100) -> pd.DataFrame:
    return _query_links(db_path, "WHERE catalyst_id = ?", (int(catalyst_id),), limit)


def link_summary_by_catalyst(db_path: str | Path, catalyst_ids: list[int]) -> dict[int, dict[str, int]]:
    create_proposal_tables(db_path)
    ids = [int(catalyst_id) for catalyst_id in catalyst_ids if pd.notna(catalyst_id)]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT catalyst_id,
                   COUNT(*) AS total_links,
                   SUM(CASE WHEN link_status = 'active' THEN 1 ELSE 0 END) AS active_links
            FROM extraction_catalyst_links
            WHERE catalyst_id IN ({placeholders})
            GROUP BY catalyst_id
            """,
            ids,
        ).fetchall()
    return {
        int(row["catalyst_id"]): {
            "total_links": int(row["total_links"] or 0),
            "active_links": int(row["active_links"] or 0),
        }
        for row in rows
    }


def proposal_summary_by_catalyst(db_path: str | Path, catalyst_ids: list[int]) -> dict[int, dict[str, int]]:
    create_proposal_tables(db_path)
    ids = [int(catalyst_id) for catalyst_id in catalyst_ids if pd.notna(catalyst_id)]
    if not ids:
        return {}
    placeholders = ",".join(["?"] * len(ids))
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT target_catalyst_id,
                   COUNT(*) AS proposal_count,
                   SUM(CASE WHEN proposal_status = 'reviewed_ready' THEN 1 ELSE 0 END) AS reviewed_ready_count,
                   SUM(CASE WHEN proposal_status = 'draft' THEN 1 ELSE 0 END) AS draft_count
            FROM catalyst_proposals
            WHERE target_catalyst_id IN ({placeholders})
            GROUP BY target_catalyst_id
            """,
            ids,
        ).fetchall()
    return {
        int(row["target_catalyst_id"]): {
            "proposal_count": int(row["proposal_count"] or 0),
            "reviewed_ready_count": int(row["reviewed_ready_count"] or 0),
            "draft_count": int(row["draft_count"] or 0),
        }
        for row in rows
    }


def proposal_summary_by_ticker(db_path: str | Path, tickers: list[str]) -> dict[str, dict[str, int]]:
    create_proposal_tables(db_path)
    cleaned = [ticker.upper().strip() for ticker in tickers if str(ticker).strip()]
    if not cleaned:
        return {}
    placeholders = ",".join(["?"] * len(cleaned))
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            f"""
            SELECT ticker,
                   COUNT(*) AS proposal_count,
                   SUM(CASE WHEN proposal_status = 'draft' THEN 1 ELSE 0 END) AS draft_count,
                   SUM(CASE WHEN proposal_status = 'reviewed_ready' THEN 1 ELSE 0 END) AS reviewed_ready_count,
                   SUM(CASE WHEN proposal_status = 'rejected' THEN 1 ELSE 0 END) AS rejected_count,
                   SUM(CASE WHEN proposal_status = 'superseded' THEN 1 ELSE 0 END) AS superseded_count
            FROM catalyst_proposals
            WHERE ticker IN ({placeholders})
            GROUP BY ticker
            """,
            cleaned,
        ).fetchall()
    return {
        str(row["ticker"]): {
            "proposal_count": int(row["proposal_count"] or 0),
            "draft_count": int(row["draft_count"] or 0),
            "reviewed_ready_count": int(row["reviewed_ready_count"] or 0),
            "rejected_count": int(row["rejected_count"] or 0),
            "superseded_count": int(row["superseded_count"] or 0),
        }
        for row in rows
    }


def proposal_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "created_at",
        "ticker",
        "proposal_type",
        "proposal_status",
        "proposed_event_type",
        "proposed_event_date",
        "proposed_title",
        "proposed_sentiment",
        "proposed_strength",
        "proposed_confidence",
        "target_catalyst_id",
        "extraction_id",
        "document_id",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    display = df.copy()
    for column in columns:
        if column not in display.columns:
            display[column] = ""
    return display[columns]


def link_display_frame(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "created_at",
        "ticker",
        "link_status",
        "extraction_id",
        "catalyst_id",
        "document_id",
        "reviewer_note",
        "unlinked_at",
    ]
    if df is None or df.empty:
        return pd.DataFrame(columns=columns)
    display = df.copy()
    for column in columns:
        if column not in display.columns:
            display[column] = ""
    return display[columns]


def proposal_score_contribution() -> int:
    return 0


def active_catalyst_event_types() -> list[str]:
    return list(EVENT_TYPES)
