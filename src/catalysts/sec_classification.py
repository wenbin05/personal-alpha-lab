from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.data import storage


SEC_CLASSIFIER_VERSION = "sec_classifier_v2"
SEC_FEATURE_POLICY_VERSION = "sec_feature_policy_v3"

SEC_CLASSIFICATIONS = {
    "core_periodic",
    "current_event",
    "ownership",
    "equity_financing",
    "debt_financing",
    "structured_note",
    "registration_or_prospectus_other",
    "amendment",
    "unknown",
}

FEATURE_ELIGIBLE_CLASSIFICATIONS = {
    "core_periodic",
    "current_event",
    "ownership",
    "equity_financing",
    "debt_financing",
    "structured_note",
    "registration_or_prospectus_other",
    "amendment",
}

SEC_FEATURE_POLICY = {
    "policy_version": SEC_FEATURE_POLICY_VERSION,
    "classifier_version": SEC_CLASSIFIER_VERSION,
    "eligible_categories": sorted(FEATURE_ELIGIBLE_CLASSIFICATIONS),
    "excluded_classifications": ["unknown"],
    "unknown_classification_handling": "retained for audit, excluded from curated feature counts",
    "aggregation_rules": {
        "windows_sessions": [7, 30, 90],
        "primary_unit": "unique filing days by classification",
        "raw_counts": "retained only in *_audit feature columns",
        "ownership": "aggregated by issuer/date/reporting person when reporting owner is available; otherwise by issuer/date",
        "structured_note": "tracked separately and excluded from equity financing flags",
        "model_contract": "feature_columns_json contains model_feature columns only; filing volume and workflow fields are audit-only",
    },
}

STRUCTURED_NOTE_TERMS = (
    "structured note",
    "market-linked",
    "market linked",
    "pricing supplement",
    "preliminary pricing supplement",
    "auto-callable",
    "autocallable",
    "buffer note",
    "accelerated return",
    "contingent income",
    "trigger performance",
    "leveraged return",
)

EQUITY_TERMS = (
    "common stock",
    "ordinary shares",
    "class a common",
    "equity offering",
    "at-the-market",
    "at the market",
    "atm offering",
    "public offering of shares",
    "shares of common",
    "share offering",
    "underwritten offering",
)

DEBT_TERMS = (
    "senior notes",
    "subordinated notes",
    "debt securities",
    "fixed rate notes",
    "floating rate notes",
    "medium-term notes",
    "medium term notes",
    "notes due",
    "bond offering",
)


@dataclass(frozen=True)
class SecClassificationResult:
    classification: str
    classification_reason: str
    feature_eligible: bool
    exclusion_reason: str | None = None
    classifier_version: str = SEC_CLASSIFIER_VERSION


def _now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _json_loads(value: Any, default: Any = None) -> Any:
    if value is None or value == "":
        return default
    if isinstance(value, (dict, list)):
        return value
    try:
        return json.loads(str(value))
    except Exception:
        return default


def _payload_text(payload: dict[str, Any], row: dict[str, Any] | pd.Series | None = None) -> str:
    row = row or {}
    text_parts: list[str] = [
        str(row.get("title", "") or ""),
        str(row.get("summary", "") or ""),
        str(payload.get("form", "") or ""),
        str(payload.get("primaryDocDescription", "") or ""),
        str(payload.get("primaryDocument", "") or ""),
        str(payload.get("primaryDocDescription", "") or ""),
    ]
    for key in [
        "documentDescriptions",
        "filingIndexDocumentDescriptions",
        "primaryDocDescriptions",
        "docDescriptions",
    ]:
        value = payload.get(key)
        if isinstance(value, list):
            text_parts.extend(str(item or "") for item in value)
        elif value:
            text_parts.append(str(value))
    documents = payload.get("documents")
    if isinstance(documents, list):
        for document in documents:
            if isinstance(document, dict):
                text_parts.append(str(document.get("description", "") or ""))
                text_parts.append(str(document.get("document", "") or ""))
    return " ".join(text_parts).lower()


def _raw_payload_hash(raw_payload_json: Any) -> str:
    return hashlib.sha256(str(raw_payload_json or "").encode("utf-8")).hexdigest()


def _raw_form(row: dict[str, Any] | pd.Series) -> str:
    payload = _json_loads(row.get("raw_payload_json"), {}) or {}
    form = str(payload.get("form") or "").upper().strip()
    if not form:
        title = str(row.get("title", "") or "").upper()
        if title.startswith("SEC "):
            form = title.replace("SEC ", "", 1).split(" FILING", 1)[0].strip()
    return form


def _is_amendment(form: str) -> bool:
    return form.upper().strip().endswith("/A")


def _base_form(form: str) -> str:
    return form.upper().strip().replace("/A", "")


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(term in text for term in terms)


def classify_sec_filing(row: dict[str, Any] | pd.Series) -> SecClassificationResult:
    """Classify SEC metadata conservatively without interpreting sentiment."""
    payload = _json_loads(row.get("raw_payload_json"), {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    form = _raw_form(row)
    base_form = _base_form(form)
    text = _payload_text(payload, row)

    if not form:
        return SecClassificationResult(
            "unknown",
            "Missing SEC form in raw payload and title.",
            False,
            "missing_form",
        )

    if _is_amendment(form):
        return SecClassificationResult("amendment", f"{form} is an amended filing.", True)

    if base_form in {"10-K", "10-Q"}:
        return SecClassificationResult("core_periodic", f"{form} is a periodic report.", True)
    if base_form == "8-K":
        return SecClassificationResult("current_event", f"{form} is a current report.", True)
    if base_form in {"3", "4", "5"}:
        return SecClassificationResult(
            "ownership",
            f"Form {form} reports ownership/insider activity; sentiment remains neutral.",
            True,
        )

    structured_hint = _contains_any(text, STRUCTURED_NOTE_TERMS) or (
        base_form.startswith("424B") and "note" in text
    )
    equity_hint = _contains_any(text, EQUITY_TERMS)
    if structured_hint and not equity_hint:
        return SecClassificationResult(
            "structured_note",
            "Filing text/description indicates structured or note-linked securities.",
            True,
        )
    if equity_hint:
        return SecClassificationResult(
            "equity_financing",
            "Filing text/description references equity or common-stock offering language.",
            True,
        )
    if _contains_any(text, DEBT_TERMS):
        return SecClassificationResult(
            "debt_financing",
            "Filing text/description references debt or note offering language.",
            True,
        )

    if base_form in {"S-1", "S-3"}:
        return SecClassificationResult(
            "registration_or_prospectus_other",
            f"{form} is a registration statement without deterministic equity/debt classification.",
            True,
        )

    if base_form.startswith("424B"):
        return SecClassificationResult(
            "unknown",
            "424B filing lacks enough deterministic description to classify financing type.",
            False,
            "ambiguous_424b",
        )

    return SecClassificationResult(
        "unknown",
        f"Unsupported or ambiguous SEC form {form}.",
        False,
        "unsupported_or_ambiguous_form",
    )


def create_sec_classification_table(db_path: str | Path) -> None:
    storage.init_db(db_path)


def classification_row_from_catalyst(row: dict[str, Any] | pd.Series) -> dict[str, Any]:
    result = classify_sec_filing(row)
    payload = _json_loads(row.get("raw_payload_json"), {}) or {}
    if not isinstance(payload, dict):
        payload = {}
    now = _now_iso()
    return {
        "catalyst_id": int(row.get("id")),
        "ticker": str(row.get("ticker") or "").upper(),
        "accession_number": str(payload.get("accessionNumber") or "").strip() or None,
        "form": _raw_form(row) or None,
        "classification": result.classification,
        "classification_reason": result.classification_reason,
        "classifier_version": result.classifier_version,
        "feature_eligible": 1 if result.feature_eligible else 0,
        "exclusion_reason": result.exclusion_reason,
        "classified_at": now,
        "raw_payload_hash": _raw_payload_hash(row.get("raw_payload_json")),
        "created_at": now,
        "updated_at": now,
    }


def upsert_sec_classification(db_path: str | Path, row: dict[str, Any] | pd.Series) -> None:
    create_sec_classification_table(db_path)
    payload = classification_row_from_catalyst(row)
    with storage.connect(db_path) as conn:
        _upsert_payloads(conn, [payload])


def _upsert_payloads(conn: Any, payloads: list[dict[str, Any]]) -> None:
    if not payloads:
        return
    conn.executemany(
        """
        INSERT INTO sec_filing_classifications (
            catalyst_id, ticker, accession_number, form, classification, classification_reason,
            classifier_version, feature_eligible, exclusion_reason, classified_at,
            raw_payload_hash, created_at, updated_at
        )
        VALUES (
            :catalyst_id, :ticker, :accession_number, :form, :classification, :classification_reason,
            :classifier_version, :feature_eligible, :exclusion_reason, :classified_at,
            :raw_payload_hash, :created_at, :updated_at
        )
        ON CONFLICT(catalyst_id) DO UPDATE SET
            ticker = excluded.ticker,
            accession_number = excluded.accession_number,
            form = excluded.form,
            classification = excluded.classification,
            classification_reason = excluded.classification_reason,
            classifier_version = excluded.classifier_version,
            feature_eligible = excluded.feature_eligible,
            exclusion_reason = excluded.exclusion_reason,
            classified_at = excluded.classified_at,
            raw_payload_hash = excluded.raw_payload_hash,
            updated_at = excluded.updated_at
        """,
        payloads,
    )


def classify_ticker_sec_filings(db_path: str | Path, ticker: str, force: bool = False) -> dict[str, int]:
    create_sec_classification_table(db_path)
    ticker = ticker.upper().strip()
    with storage.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT c.*, s.classifier_version, s.raw_payload_hash AS existing_raw_payload_hash
            FROM catalysts c
            LEFT JOIN sec_filing_classifications s ON s.catalyst_id = c.id
            WHERE c.ticker = ?
              AND c.event_type = 'sec_filing'
              AND c.source = 'SEC EDGAR'
            """,
            (ticker,),
        ).fetchall()
    classified = 0
    payloads: list[dict[str, Any]] = []
    for sqlite_row in rows:
        row = dict(sqlite_row)
        existing_hash = row.pop("existing_raw_payload_hash", None)
        existing_version = row.pop("classifier_version", None)
        if (
            force
            or existing_version != SEC_CLASSIFIER_VERSION
            or existing_hash != _raw_payload_hash(row.get("raw_payload_json"))
        ):
            payloads.append(classification_row_from_catalyst(row))
            classified += 1
    if payloads:
        with storage.connect(db_path) as conn:
            _upsert_payloads(conn, payloads)
    return {"ticker": ticker, "classified": classified}


def classify_ticker_sec_filings_safe(db_path: str | Path, ticker: str, force: bool = False) -> dict[str, int]:
    return classify_ticker_sec_filings(db_path, ticker, force=force)


def list_sec_classifications_by_ticker(db_path: str | Path, ticker: str) -> pd.DataFrame:
    create_sec_classification_table(db_path)
    with storage.connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT *
            FROM sec_filing_classifications
            WHERE ticker = ?
            ORDER BY catalyst_id
            """,
            conn,
            params=(ticker.upper().strip(),),
        )


def sec_classification_summary(db_path: str | Path, tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    create_sec_classification_table(db_path)
    where = ""
    params: list[Any] = []
    if tickers:
        clean = [ticker.upper().strip() for ticker in tickers if ticker and ticker.strip()]
        if clean:
            where = f"WHERE ticker IN ({','.join(['?'] * len(clean))})"
            params = clean
    with storage.connect(db_path) as conn:
        by_category = pd.read_sql_query(
            f"""
            SELECT classification, feature_eligible, COUNT(*) AS filings
            FROM sec_filing_classifications
            {where}
            GROUP BY classification, feature_eligible
            ORDER BY filings DESC
            """,
            conn,
            params=params,
        )
        by_ticker = pd.read_sql_query(
            f"""
            SELECT ticker,
                   COUNT(*) AS raw_filings,
                   SUM(CASE WHEN feature_eligible = 1 THEN 1 ELSE 0 END) AS feature_eligible_filings,
                   SUM(CASE WHEN classification = 'structured_note' THEN 1 ELSE 0 END) AS structured_note_filings,
                   SUM(CASE WHEN classification = 'unknown' THEN 1 ELSE 0 END) AS unknown_filings,
                   COUNT(DISTINCT classified_at) AS classification_batches
            FROM sec_filing_classifications
            {where}
            GROUP BY ticker
            ORDER BY raw_filings DESC
            """,
            conn,
            params=params,
        )
        exclusions = pd.read_sql_query(
            f"""
            SELECT COALESCE(exclusion_reason, '') AS exclusion_reason, COUNT(*) AS filings
            FROM sec_filing_classifications
            {where}
            GROUP BY exclusion_reason
            ORDER BY filings DESC
            """,
            conn,
            params=params,
        )
    if not by_ticker.empty:
        total = float(by_ticker["raw_filings"].sum() or 1)
        eligible_total = float(by_ticker["feature_eligible_filings"].sum() or 1)
        by_ticker["raw_concentration"] = by_ticker["raw_filings"] / total
        by_ticker["eligible_concentration"] = by_ticker["feature_eligible_filings"] / eligible_total
    return {"by_category": by_category, "by_ticker": by_ticker, "exclusions": exclusions}
