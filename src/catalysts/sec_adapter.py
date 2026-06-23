from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
import gzip
import zlib
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from src.catalysts.models import CatalystEvent
from src.data import storage
from src.documents.models import SourceDocument
from src.documents.repository import build_source_document
from src.documents.text_cleaning import join_warnings


SEC_TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik}.json"
SEC_SUBMISSIONS_FILE_URL = "https://data.sec.gov/submissions/{name}"
SEC_ARCHIVE_URL = "https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{document}"
SUPPORTED_FORMS = {"10-K", "10-Q", "8-K", "S-1", "S-3", "4", "10-K/A", "10-Q/A", "8-K/A", "S-1/A", "S-3/A"}
IMPORTANT_REVIEW_FORMS = {"8-K", "8-K/A", "S-1", "S-1/A", "S-3", "S-3/A", "4"}
DEFAULT_MAX_FILING_BYTES = 2_000_000
MIN_REQUEST_INTERVAL_SECONDS = 0.12


@dataclass
class ProviderResult:
    events: list[CatalystEvent] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilingTextResult:
    document: SourceDocument | None = None
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class SecFilingMetadata:
    ticker: str
    cik: str
    company_name: str
    form: str
    filing_date: date | None
    report_date: date | None
    acceptance_datetime: datetime | None
    accession_number: str
    primary_document: str
    source_url: str | None
    is_amended: bool
    raw_payload: dict[str, Any]


class SecFilingsProvider:
    name = "sec_edgar_submissions"

    def __init__(
        self,
        user_agent: str | None = None,
        timeout: int = 10,
        db_path: str | Path | None = None,
        min_interval_seconds: float = MIN_REQUEST_INTERVAL_SECONDS,
        use_cache: bool = True,
    ) -> None:
        self.user_agent = user_agent or os.getenv("SEC_USER_AGENT")
        self.timeout = timeout
        self.db_path = Path(db_path) if db_path is not None else None
        self.min_interval_seconds = max(0.0, float(min_interval_seconds))
        self.use_cache = use_cache
        self._ticker_map: dict[str, str] | None = None
        self._ticker_names: dict[str, str] = {}
        self._last_request_at = 0.0
        self.request_stats = {"downloaded": 0, "cache_hits": 0, "errors": 0, "retries": 0}

    def user_agent_warning(self) -> str | None:
        if not self.user_agent or "@" not in self.user_agent or "example.com" in self.user_agent:
            return (
                "SEC_USER_AGENT is required for SEC requests and should identify this local app "
                "plus a real contact email, for example 'personal-alpha-lab/0.1 you@example.com'."
            )
        return None

    def _cache_key(self, url: str) -> str:
        import hashlib

        return hashlib.sha256(url.encode("utf-8")).hexdigest()

    def _cache_get(self, url: str) -> Any | None:
        if not self.use_cache or self.db_path is None:
            return None
        storage.init_db(self.db_path)
        with storage.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT response_json FROM sec_response_cache WHERE cache_key = ? AND response_json IS NOT NULL",
                (self._cache_key(url),),
            ).fetchone()
        if row is None:
            return None
        self.request_stats["cache_hits"] += 1
        return json.loads(row["response_json"])

    def _cache_set(self, url: str, payload: Any, status_code: int = 200, error: str | None = None) -> None:
        if not self.use_cache or self.db_path is None:
            return
        storage.init_db(self.db_path)
        now = datetime.now(UTC).isoformat(timespec="seconds")
        with storage.connect(self.db_path) as conn:
            conn.execute(
                """
                INSERT INTO sec_response_cache (
                    cache_key, url, response_json, status_code, error, fetched_at, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    response_json = excluded.response_json,
                    status_code = excluded.status_code,
                    error = excluded.error,
                    fetched_at = excluded.fetched_at,
                    updated_at = excluded.updated_at
                """,
                (
                    self._cache_key(url),
                    url,
                    None if payload is None else json.dumps(payload, default=str),
                    status_code,
                    error,
                    now,
                    now,
                    now,
                ),
            )

    def _throttle(self) -> None:
        elapsed = time.monotonic() - self._last_request_at
        if elapsed < self.min_interval_seconds:
            time.sleep(self.min_interval_seconds - elapsed)
        self._last_request_at = time.monotonic()

    def _fetch_json(self, url: str) -> Any:
        cached = self._cache_get(url)
        if cached is not None:
            return cached
        warning = self.user_agent_warning()
        if warning:
            raise ValueError(warning)
        self._throttle()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": str(self.user_agent),
                "Accept-Encoding": "gzip, deflate",
                "Accept": "application/json",
            },
        )
        attempts = 0
        while True:
            attempts += 1
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    raw_payload = response.read()
                    encoding = str(response.headers.get("Content-Encoding", "") or "").lower()
                    if "gzip" in encoding or raw_payload.startswith(b"\x1f\x8b"):
                        raw_payload = gzip.decompress(raw_payload)
                    elif "deflate" in encoding:
                        raw_payload = zlib.decompress(raw_payload)
                    payload = json.loads(raw_payload.decode("utf-8"))
                    self.request_stats["downloaded"] += 1
                    self._cache_set(url, payload, status_code=getattr(response, "status", 200))
                    return payload
            except urllib.error.HTTPError as exc:
                if exc.code in {429, 500, 502, 503, 504} and attempts < 3:
                    self.request_stats["retries"] += 1
                    time.sleep(min(2.0, attempts * 0.5))
                    continue
                self.request_stats["errors"] += 1
                self._cache_set(url, None, status_code=exc.code, error=str(exc))
                raise

    def _fetch_bytes(self, url: str, max_bytes: int = DEFAULT_MAX_FILING_BYTES) -> tuple[bytes, bool]:
        warning = self.user_agent_warning()
        if warning:
            raise ValueError(warning)
        self._throttle()
        request = urllib.request.Request(
            url,
            headers={
                "User-Agent": str(self.user_agent),
                "Accept": "text/html,text/plain,*/*",
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = response.read(max_bytes + 1)
        oversized = len(payload) > max_bytes
        return payload[:max_bytes], oversized

    def ticker_to_cik(self, ticker: str) -> str | None:
        ticker = ticker.upper().strip()
        if self._ticker_map is None:
            raw = self._fetch_json(SEC_TICKER_MAP_URL)
            self._ticker_map = {
                str(record.get("ticker", "")).upper(): str(record.get("cik_str", "")).zfill(10)
                for record in raw.values()
            }
            self._ticker_names = {
                str(record.get("ticker", "")).upper(): str(record.get("title", ""))
                for record in raw.values()
            }
        return self._ticker_map.get(ticker)

    def _company_name(self, ticker: str) -> str:
        if self._ticker_map is None:
            self.ticker_to_cik(ticker)
        return self._ticker_names.get(ticker.upper().strip(), "")

    def _submission_payloads(self, cik: str) -> list[tuple[str, dict[str, Any]]]:
        root_url = SEC_SUBMISSIONS_URL.format(cik=cik)
        root = self._fetch_json(root_url)
        payloads = [(root_url, root)]
        for item in (root or {}).get("filings", {}).get("files", []) or []:
            name = str(item.get("name") or "").strip()
            if not name:
                continue
            url = SEC_SUBMISSIONS_FILE_URL.format(name=name)
            payloads.append((url, self._fetch_json(url)))
        return payloads

    @staticmethod
    def _parse_date(value: Any) -> date | None:
        try:
            if value is None or value == "":
                return None
            parsed = pd.to_datetime(value, errors="coerce")
            if pd.isna(parsed):
                return None
            return parsed.date()
        except Exception:
            return None

    @staticmethod
    def _parse_acceptance(value: Any) -> datetime | None:
        try:
            if value is None or value == "":
                return None
            text = str(value)
            parsed = pd.to_datetime(text, errors="coerce", utc=False)
            if pd.isna(parsed):
                return None
            dt = parsed.to_pydatetime()
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=UTC)
            return dt.astimezone(UTC)
        except Exception:
            return None

    @staticmethod
    def _supported_form(form: str) -> bool:
        form = form.upper().strip()
        return form in SUPPORTED_FORMS or form.startswith("424B")

    @staticmethod
    def _form_needs_review(form: str) -> bool:
        form = form.upper().strip()
        return form in IMPORTANT_REVIEW_FORMS or form.startswith("424B")

    def _metadata_from_recent(
        self,
        ticker: str,
        cik: str,
        company_name: str,
        recent: dict[str, list[Any]],
        idx: int,
    ) -> SecFilingMetadata | None:
        def value(column: str) -> Any:
            values = recent.get(column, [])
            return values[idx] if idx < len(values) else None

        form = str(value("form") or "").upper().strip()
        if not self._supported_form(form):
            return None
        filing_date = self._parse_date(value("filingDate"))
        report_date = self._parse_date(value("reportDate"))
        acceptance = self._parse_acceptance(value("acceptanceDateTime"))
        accession = str(value("accessionNumber") or "").strip()
        primary_document = str(value("primaryDocument") or "").strip()
        source_url = None
        if accession and primary_document:
            source_url = SEC_ARCHIVE_URL.format(
                cik=str(int(cik)),
                accession=accession.replace("-", ""),
                document=primary_document,
            )
        raw_payload = {
            "cik": cik,
            "companyName": company_name,
            "form": form,
            "filingDate": None if filing_date is None else filing_date.isoformat(),
            "reportDate": None if report_date is None else report_date.isoformat(),
            "acceptanceDateTime": None if acceptance is None else acceptance.isoformat(timespec="seconds"),
            "accessionNumber": accession,
            "primaryDocument": primary_document,
            "primaryDocDescription": value("primaryDocDescription"),
        }
        return SecFilingMetadata(
            ticker=ticker,
            cik=cik,
            company_name=company_name,
            form=form,
            filing_date=filing_date,
            report_date=report_date,
            acceptance_datetime=acceptance,
            accession_number=accession,
            primary_document=primary_document,
            source_url=source_url,
            is_amended=form.endswith("/A"),
            raw_payload=raw_payload,
        )

    def list_historical_filings(self, ticker: str, start_date: date, end_date: date) -> tuple[list[SecFilingMetadata], list[str]]:
        ticker = ticker.upper().strip()
        warnings: list[str] = []
        cik = self.ticker_to_cik(ticker)
        if not cik:
            return [], [f"SEC CIK mapping unavailable for {ticker}."]
        company_name = self._company_name(ticker)
        try:
            payloads = self._submission_payloads(cik)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            return [], [f"SEC historical metadata unavailable for {ticker}: {exc}"]

        seen: set[tuple[str, str, str]] = set()
        filings: list[SecFilingMetadata] = []
        for _, payload in payloads:
            recent = (payload or {}).get("filings", {}).get("recent", {})
            forms = recent.get("form", []) or []
            for idx in range(len(forms)):
                metadata = self._metadata_from_recent(ticker, cik, company_name, recent, idx)
                if metadata is None:
                    continue
                available_date = metadata.acceptance_datetime.date() if metadata.acceptance_datetime else metadata.filing_date
                if available_date is None:
                    warnings.append(f"{ticker}: filing {metadata.accession_number or metadata.form} lacks acceptance and filing date.")
                    continue
                if available_date < start_date or available_date > end_date:
                    continue
                key = (metadata.accession_number, metadata.form, metadata.primary_document)
                if key in seen:
                    continue
                seen.add(key)
                filings.append(metadata)
        filings.sort(key=lambda item: item.acceptance_datetime or datetime.combine(item.filing_date or start_date, datetime.min.time(), UTC))
        return filings, warnings

    def filing_to_event(self, filing: SecFilingMetadata) -> CatalystEvent:
        needs_review = self._form_needs_review(filing.form)
        title_suffix = " (Needs Review)" if needs_review else ""
        event_date = filing.filing_date or (filing.acceptance_datetime.date() if filing.acceptance_datetime else datetime.now(UTC).date())
        summary = (
            f"{filing.ticker} filed SEC form {filing.form}. "
            "Metadata only; review the filing manually before assigning sentiment."
        )
        if filing.report_date:
            summary += f" Report period: {filing.report_date.isoformat()}."
        return CatalystEvent(
            ticker=filing.ticker,
            event_date=event_date,
            event_time=filing.acceptance_datetime.strftime("%H:%M:%S") if filing.acceptance_datetime else None,
            event_type="sec_filing",
            title=f"SEC {filing.form} filing{title_suffix}",
            summary=summary,
            source="SEC EDGAR",
            source_url=filing.source_url,
            sentiment_label="unknown",
            catalyst_strength=0,
            confidence=0.7 if filing.acceptance_datetime else 0.45,
            is_manual=False,
            available_at=filing.acceptance_datetime,
            created_at=datetime.now(UTC),
            updated_at=datetime.now(UTC),
            raw_payload_json=json.dumps(
                {
                    **filing.raw_payload,
                    "provider": self.name,
                    "needsReview": needs_review,
                    "isAmended": filing.is_amended,
                    "secDedupeKey": f"{filing.accession_number}|{filing.form}|{filing.primary_document}",
                },
                default=str,
            ),
        )

    def fetch_recent_filings(self, ticker: str, limit: int = 20) -> ProviderResult:
        ticker = ticker.upper().strip()
        warnings: list[str] = []
        try:
            cik = self.ticker_to_cik(ticker)
            if not cik:
                return ProviderResult(warnings=[f"SEC CIK mapping unavailable for {ticker}."])
            submission = self._fetch_json(SEC_SUBMISSIONS_URL.format(cik=cik))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            return ProviderResult(warnings=[f"SEC metadata unavailable for {ticker}: {exc}"])

        recent = (submission or {}).get("filings", {}).get("recent", {})
        if not recent:
            return ProviderResult(warnings=[f"SEC provider returned no recent filings for {ticker}."])

        events: list[CatalystEvent] = []

        forms = recent.get("form", [])
        company_name = self._company_name(ticker)
        for idx, _ in enumerate(forms):
            metadata = self._metadata_from_recent(ticker, cik, company_name, recent, idx)
            if metadata is None:
                continue
            events.append(self.filing_to_event(metadata))
            if len(events) >= limit:
                break

        if not events:
            warnings.append(f"No supported SEC filing types found recently for {ticker}.")
        return ProviderResult(events=events, warnings=warnings, metadata=dict(self.request_stats))

    def fetch_historical_filing_events(self, ticker: str, start_date: date, end_date: date) -> ProviderResult:
        warning = self.user_agent_warning()
        if warning:
            return ProviderResult(warnings=[warning], metadata=dict(self.request_stats))
        try:
            filings, warnings = self.list_historical_filings(ticker, start_date, end_date)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as exc:
            return ProviderResult(warnings=[f"SEC historical metadata unavailable for {ticker}: {exc}"], metadata=dict(self.request_stats))
        events = [self.filing_to_event(filing) for filing in filings]
        metadata = {
            **self.request_stats,
            "filings_seen": len(filings),
            "forms": _form_counts(filings),
            "missing_acceptance_timestamps": sum(1 for filing in filings if filing.acceptance_datetime is None),
        }
        if not events and not warnings:
            warnings.append(f"No supported SEC filings found for {ticker} in requested range.")
        return ProviderResult(events=events, warnings=warnings, metadata=metadata)

    def fetch_filing_text_document(
        self,
        catalyst_row: Mapping[str, Any],
        max_bytes: int = DEFAULT_MAX_FILING_BYTES,
    ) -> FilingTextResult:
        ticker = str(catalyst_row.get("ticker") or "").upper().strip()
        catalyst_id_value = catalyst_row.get("id")
        catalyst_id = None
        try:
            catalyst_id_text = str(catalyst_id_value).strip()
            if catalyst_id_text and catalyst_id_text.lower() not in {"none", "nan", "<na>"}:
                catalyst_id = int(float(catalyst_id_text))
        except (TypeError, ValueError):
            catalyst_id = None
        source_url = str(catalyst_row.get("source_url") or "").strip() or None
        title = str(catalyst_row.get("title") or "SEC filing text").strip()
        event_date = catalyst_row.get("event_date")
        raw_payload_json = catalyst_row.get("raw_payload_json")
        raw_payload: dict[str, Any] = {}
        if raw_payload_json:
            try:
                raw_payload = json.loads(str(raw_payload_json))
            except json.JSONDecodeError:
                raw_payload = {"raw_payload_json": str(raw_payload_json)}

        accession_number = raw_payload.get("accessionNumber")
        filing_type = raw_payload.get("form")
        warnings: list[str] = []
        raw_text = ""
        parsing_status = "not_attempted"

        if not ticker:
            warnings.append("SEC filing text fetch skipped because ticker is missing.")
        if not source_url:
            warnings.append("SEC filing source URL is unavailable; text cannot be fetched.")

        if source_url and ticker:
            try:
                payload, oversized = self._fetch_bytes(source_url, max_bytes=max_bytes)
                raw_text = payload.decode("utf-8", errors="replace")
                parsing_status = "partial" if oversized else "success"
                if oversized:
                    warnings.append(
                        f"SEC filing exceeded {max_bytes:,} bytes; stored the first {max_bytes:,} bytes only."
                    )
            except (urllib.error.URLError, TimeoutError, OSError, ValueError) as exc:
                parsing_status = "failed"
                warnings.append(f"SEC filing text unavailable from {source_url}: {exc}")
        elif warnings:
            parsing_status = "failed"

        try:
            document = build_source_document(
                ticker=ticker or "UNKNOWN",
                catalyst_id=catalyst_id,
                document_type="sec_filing",
                source="SEC",
                source_url=source_url,
                accession_number=accession_number,
                filing_type=filing_type,
                title=title,
                published_at=event_date,
                raw_text=raw_text,
                parsing_status=parsing_status,
                warnings=join_warnings(warnings),
                raw_payload_json=json.dumps(
                    {
                        "provider": self.name,
                        "source_url": source_url,
                        "catalyst_raw_payload": raw_payload,
                    },
                    default=str,
                ),
            )
        except Exception as exc:
            return FilingTextResult(warnings=[*warnings, f"Could not build SEC source document: {exc}"])

        return FilingTextResult(document=document, warnings=warnings)


def _form_counts(filings: list[SecFilingMetadata]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for filing in filings:
        counts[filing.form] = counts.get(filing.form, 0) + 1
    return counts
