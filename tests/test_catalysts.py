from __future__ import annotations

import gzip
import json
import sqlite3
from datetime import UTC, date, datetime, timedelta

import pandas as pd

from src.catalysts.earnings_adapter import YFinanceEarningsProvider
from src.catalysts.models import CatalystEvent
from src.catalysts.repository import (
    delete_catalyst,
    insert_catalyst,
    list_catalysts_by_ticker,
    list_recent_catalysts,
)
from src.catalysts.sec_adapter import ProviderResult, SecFilingsProvider
from src.catalysts.sec_backfill import create_sec_backfill_run, list_sec_backfill_items, process_sec_backfill_run
from src.catalysts.sec_classification import (
    SEC_CLASSIFIER_VERSION,
    classify_sec_filing,
    classify_ticker_sec_filings,
    list_sec_classifications_by_ticker,
)
from src.data import storage
from src.features.catalyst import get_catalyst_features
from src.scoring.score_engine import score_ticker_from_features


def base_features() -> dict:
    return {
        "has_data": True,
        "data_quality": "ok",
        "last_price": 50.0,
        "ret_20d": 0.12,
        "ret_60d": 0.25,
        "above_50d_ma": True,
        "above_200d_ma": True,
        "relative_strength_20d": 0.05,
        "relative_strength_60d": 0.08,
        "volume_ratio_20d": 1.8,
        "avg_dollar_volume_20d": 50_000_000,
        "avg_dollar_volume_ok": True,
        "liquidity_score_raw": 1.0,
        "liquidity_label": "Acceptable",
        "distance_20d_ma": 0.04,
        "volatility_20d": 0.35,
    }


def event(
    sentiment: str,
    strength: int = 8,
    confidence: float = 0.8,
    event_type: str = "manual_note",
    title: str = "Manual catalyst",
) -> CatalystEvent:
    return CatalystEvent(
        ticker="AAA",
        event_date=datetime.now(UTC).date(),
        event_type=event_type,
        title=title,
        summary="Test catalyst",
        source="manual" if event_type == "manual_note" else "test",
        sentiment_label=sentiment,
        catalyst_strength=strength,
        confidence=confidence,
        is_manual=event_type == "manual_note",
    )


def test_catalyst_table_insert_list_delete(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    catalyst_id = insert_catalyst(db_path, event("positive"))

    by_ticker = list_catalysts_by_ticker(db_path, "AAA")
    recent = list_recent_catalysts(db_path)

    assert catalyst_id > 0
    assert len(by_ticker) == 1
    assert len(recent) == 1
    assert delete_catalyst(db_path, catalyst_id) is True
    assert list_catalysts_by_ticker(db_path, "AAA").empty


def test_catalyst_available_at_migration_handles_existing_database(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE catalysts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                event_date TEXT NOT NULL,
                event_time TEXT,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                source TEXT,
                source_url TEXT,
                sentiment_label TEXT DEFAULT 'unknown',
                catalyst_strength INTEGER DEFAULT 0,
                confidence REAL DEFAULT 0,
                is_manual INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                raw_payload_json TEXT,
                dedupe_key TEXT NOT NULL UNIQUE
            )
            """
        )
        conn.execute(
            """
            INSERT INTO catalysts (
                ticker, event_date, event_type, title, created_at, updated_at, dedupe_key
            )
            VALUES ('AAA', '2024-01-01', 'manual_note', 'Legacy catalyst', '2024-01-02T00:00:00+00:00',
                    '2024-01-02T00:00:00+00:00', 'legacy-key')
            """
        )

    storage.init_db(db_path)

    with sqlite3.connect(db_path) as conn:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(catalysts)").fetchall()}
        indexes = {row[1] for row in conn.execute("PRAGMA index_list(catalysts)").fetchall()}
        available_at = conn.execute("SELECT available_at FROM catalysts WHERE ticker = 'AAA'").fetchone()[0]

    assert "available_at" in columns
    assert "idx_catalysts_available" in indexes
    assert available_at == "2024-01-02T00:00:00+00:00"


def test_catalyst_deduplicates_events(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    first = insert_catalyst(db_path, event("positive"))
    second = insert_catalyst(db_path, event("positive"))

    assert first == second
    assert len(list_catalysts_by_ticker(db_path, "AAA")) == 1


def test_no_catalyst_is_neutral() -> None:
    catalyst_features = get_catalyst_features("AAA", pd.DataFrame())
    result = score_ticker_from_features("AAA", base_features(), {"regime": "Risk-On"}, catalyst_features)

    assert catalyst_features["catalyst_score"] == 0
    assert result["breakdown"]["catalyst"] == 0
    assert not any(p["name"] == "negative_catalyst" for p in result["penalties"])


def test_positive_catalyst_boosts_conservatively() -> None:
    df = pd.DataFrame([event("positive", strength=9, confidence=0.8).model_dump()])
    catalyst_features = get_catalyst_features("AAA", df)
    result = score_ticker_from_features("AAA", base_features(), {"regime": "Risk-On"}, catalyst_features)

    assert catalyst_features["catalyst_score"] == 7.2
    assert result["breakdown"]["catalyst"] == 7.2
    assert result["score"] <= 100


def test_negative_catalyst_penalizes() -> None:
    df = pd.DataFrame([event("negative", strength=8, confidence=0.9).model_dump()])
    catalyst_features = get_catalyst_features("AAA", df)
    result = score_ticker_from_features("AAA", base_features(), {"regime": "Risk-On"}, catalyst_features)

    assert catalyst_features["catalyst_penalty"] < 0
    assert any(p["name"] == "negative_catalyst" for p in result["penalties"])


def test_low_confidence_catalyst_has_reduced_effect() -> None:
    df = pd.DataFrame([event("positive", strength=10, confidence=0.2).model_dump()])
    catalyst_features = get_catalyst_features("AAA", df)

    assert catalyst_features["catalyst_score"] == 2.0


def test_sec_needs_review_filing_is_warning_not_auto_negative() -> None:
    filing = event("unknown", strength=2, confidence=0.7, event_type="sec_filing", title="SEC 8-K filing (Needs Review)")
    df = pd.DataFrame([filing.model_dump()])
    catalyst_features = get_catalyst_features("AAA", df)
    result = score_ticker_from_features("AAA", base_features(), {"regime": "Risk-On"}, catalyst_features)

    assert catalyst_features["catalyst_penalty"] == 0
    assert catalyst_features["needs_review"]
    assert not any(p["name"] == "negative_catalyst" for p in result["penalties"])


def test_stale_catalyst_does_not_score() -> None:
    old_event = event("positive", strength=10, confidence=1.0)
    old_event.event_date = date.today() - timedelta(days=200)
    df = pd.DataFrame([old_event.model_dump()])
    catalyst_features = get_catalyst_features("AAA", df)

    assert catalyst_features["catalyst_score"] == 0
    assert "none are recent enough" in catalyst_features["catalyst_warnings"][0]


def test_sec_adapter_fails_gracefully(monkeypatch) -> None:
    provider = SecFilingsProvider()

    def fail(_: str):
        raise OSError("offline")

    monkeypatch.setattr(provider, "_fetch_json", fail)
    result = provider.fetch_recent_filings("AAA")

    assert result.events == []
    assert result.warnings


def sec_row(form: str, title: str = "SEC filing", summary: str = "", payload: dict | None = None) -> dict:
    raw = {"form": form, **(payload or {})}
    return {
        "id": 1,
        "ticker": "AAA",
        "title": title,
        "summary": summary,
        "raw_payload_json": json.dumps(raw),
    }


def test_sec_classifier_core_current_and_ownership_forms() -> None:
    assert classify_sec_filing(sec_row("10-K")).classification == "core_periodic"
    assert classify_sec_filing(sec_row("10-Q")).classification == "core_periodic"
    assert classify_sec_filing(sec_row("8-K")).classification == "current_event"
    form4 = classify_sec_filing(sec_row("4"))
    assert form4.classification == "ownership"
    assert form4.feature_eligible is True
    assert "sentiment remains neutral" in form4.classification_reason


def test_sec_classifier_structured_note_equity_offering_and_ambiguous_424b() -> None:
    structured = classify_sec_filing(
        sec_row("424B2", payload={"primaryDocDescription": "Market-linked structured notes due 2028"})
    )
    pricing_supplement = classify_sec_filing(
        sec_row("424B2", payload={"primaryDocDescription": "PRELIMINARY PRICING SUPPLEMENT"})
    )
    equity = classify_sec_filing(
        sec_row("424B5", payload={"primaryDocDescription": "Public offering of shares of common stock"})
    )
    ambiguous = classify_sec_filing(sec_row("424B2", payload={"primaryDocDescription": "Prospectus supplement"}))

    assert structured.classification == "structured_note"
    assert pricing_supplement.classification == "structured_note"
    assert equity.classification == "equity_financing"
    assert ambiguous.classification == "unknown"
    assert ambiguous.feature_eligible is False
    assert ambiguous.exclusion_reason == "ambiguous_424b"


def test_sec_classifier_amendments_are_explicit() -> None:
    amended = classify_sec_filing(sec_row("10-Q/A"))

    assert amended.classification == "amendment"
    assert amended.feature_eligible is True
    assert amended.classifier_version == SEC_CLASSIFIER_VERSION


def test_sec_classification_layer_preserves_raw_catalyst_rows(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    raw_payload = '{"form":"424B5","primaryDocDescription":"Public offering of shares of common stock","accessionNumber":"000-test"}'
    catalyst_id = insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="AAA",
            event_date=date(2024, 1, 10),
            event_type="sec_filing",
            title="SEC 424B5 filing",
            summary="Metadata only.",
            source="SEC EDGAR",
            sentiment_label="unknown",
            catalyst_strength=0,
            confidence=0.7,
            raw_payload_json=raw_payload,
        ),
    )

    result = classify_ticker_sec_filings(db_path, "AAA")
    classifications = list_sec_classifications_by_ticker(db_path, "AAA")
    catalysts = list_catalysts_by_ticker(db_path, "AAA", limit=None)

    assert result["classified"] == 1
    assert int(classifications.iloc[0]["catalyst_id"]) == catalyst_id
    assert classifications.iloc[0]["classification"] == "equity_financing"
    assert catalysts.iloc[0]["raw_payload_json"] == raw_payload
    assert catalysts.iloc[0]["sentiment_label"] == "unknown"


def test_sec_adapter_decodes_gzip_json(monkeypatch, tmp_path) -> None:
    payload = {"0": {"ticker": "AAA", "cik_str": 1234, "title": "AAA Corp"}}
    compressed = gzip.compress(json.dumps(payload).encode("utf-8"))

    class FakeResponse:
        status = 200
        headers = {"Content-Encoding": "gzip"}

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return compressed

    def fake_urlopen(*_, **__):
        return FakeResponse()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    provider = SecFilingsProvider(
        user_agent="personal-alpha-lab-test test@alpha.test",
        db_path=tmp_path / "alpha_lab.db",
        min_interval_seconds=0,
    )

    assert provider._fetch_json("https://www.sec.gov/files/company_tickers.json") == payload
    assert provider.request_stats["downloaded"] == 1


def test_sec_historical_adapter_uses_files_and_dedupes_accessions() -> None:
    provider = SecFilingsProvider(user_agent="personal-alpha-lab-test test@alpha.test", min_interval_seconds=0)
    payloads = {
        "https://www.sec.gov/files/company_tickers.json": {
            "0": {"ticker": "AAA", "cik_str": 1234, "title": "AAA Corp"}
        },
        "https://data.sec.gov/submissions/CIK0000001234.json": {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "filingDate": ["2024-01-10"],
                    "reportDate": ["2024-01-09"],
                    "acceptanceDateTime": ["2024-01-10T21:00:00.000Z"],
                    "accessionNumber": ["0000001234-24-000001"],
                    "primaryDocument": ["aaa-8k.htm"],
                    "primaryDocDescription": ["8-K"],
                },
                "files": [{"name": "CIK0000001234-submissions-001.json"}],
            }
        },
        "https://data.sec.gov/submissions/CIK0000001234-submissions-001.json": {
            "filings": {
                "recent": {
                    "form": ["8-K", "10-Q/A"],
                    "filingDate": ["2024-01-10", "2023-11-01"],
                    "reportDate": ["2024-01-09", "2023-09-30"],
                    "acceptanceDateTime": ["2024-01-10T21:00:00.000Z", "2023-11-01T20:30:00.000Z"],
                    "accessionNumber": ["0000001234-24-000001", "0000001234-23-000010"],
                    "primaryDocument": ["aaa-8k.htm", "aaa-10qa.htm"],
                    "primaryDocDescription": ["8-K", "10-Q/A"],
                }
            }
        },
    }

    provider._fetch_json = lambda url: payloads[url]  # type: ignore[method-assign]

    filings, warnings = provider.list_historical_filings("AAA", date(2023, 1, 1), date(2024, 12, 31))

    assert warnings == []
    assert len(filings) == 2
    assert {filing.form for filing in filings} == {"8-K", "10-Q/A"}
    assert sum(1 for filing in filings if filing.accession_number == "0000001234-24-000001") == 1
    assert any(filing.is_amended for filing in filings)


def test_sec_historical_adapter_missing_cik() -> None:
    provider = SecFilingsProvider(user_agent="personal-alpha-lab-test test@alpha.test", min_interval_seconds=0)
    provider._fetch_json = lambda url: {}  # type: ignore[method-assign]

    filings, warnings = provider.list_historical_filings("MISSING", date(2024, 1, 1), date(2024, 12, 31))

    assert filings == []
    assert any("CIK mapping unavailable" in warning for warning in warnings)


def test_sec_backfill_is_idempotent_and_metadata_only_does_not_change_score(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    event_row = CatalystEvent(
        ticker="AAA",
        event_date=date(2024, 1, 10),
        event_type="sec_filing",
        title="SEC 8-K filing (Needs Review)",
        summary="Metadata only.",
        source="SEC EDGAR",
        sentiment_label="unknown",
        catalyst_strength=0,
        confidence=0.7,
        available_at=datetime(2024, 1, 10, 21, 0, tzinfo=UTC),
        raw_payload_json='{"form":"8-K","acceptanceDateTime":"2024-01-10T21:00:00+00:00","accessionNumber":"0000000000-24-000001"}',
    )

    class FakeProvider:
        name = "fake_sec"

        def fetch_historical_filing_events(self, ticker: str, start_date: date, end_date: date) -> ProviderResult:
            return ProviderResult(events=[event_row], metadata={"downloaded": 0, "cache_hits": 1, "forms": {"8-K": 1}})

    before = score_ticker_from_features("AAA", base_features(), {"regime": "Risk-On"}, {"catalyst_score": 0, "catalyst_penalty": 0, "has_catalyst": False})
    run_id = create_sec_backfill_run(db_path, ["AAA"], date(2024, 1, 1), date(2024, 12, 31))
    first = process_sec_backfill_run(db_path, run_id, provider=FakeProvider())
    second_run = create_sec_backfill_run(db_path, ["AAA"], date(2024, 1, 1), date(2024, 12, 31))
    second = process_sec_backfill_run(db_path, second_run, provider=FakeProvider())
    catalysts = list_catalysts_by_ticker(db_path, "AAA", limit=None)
    catalyst_features = get_catalyst_features("AAA", catalysts, as_of_date=date(2024, 1, 10))
    after = score_ticker_from_features("AAA", base_features(), {"regime": "Risk-On"}, catalyst_features)
    items = list_sec_backfill_items(db_path, second_run)

    assert first.events_inserted == 1
    assert second.events_inserted == 0
    assert second.duplicates_skipped == 1
    assert int(items.iloc[0]["duplicates_skipped"]) == 1
    assert catalyst_features["catalyst_score"] == 0
    assert catalyst_features["catalyst_penalty"] == 0
    assert after["score"] == before["score"]


def test_earnings_adapter_fails_gracefully(monkeypatch) -> None:
    provider = YFinanceEarningsProvider()

    def fail(_: str):
        raise RuntimeError("offline")

    monkeypatch.setattr(provider, "_get_ticker", fail)
    result = provider.fetch_earnings_events("AAA")

    assert result.events == []
    assert result.warnings
