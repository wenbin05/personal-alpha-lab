from __future__ import annotations

import sqlite3
from datetime import date

import pandas as pd

from src.catalysts.models import CatalystEvent
from src.catalysts.repository import insert_catalyst
from src.catalysts.sec_adapter import SecFilingsProvider
from src.data import storage
from src.documents.csv_import import parse_document_import_frame
from src.documents.repository import (
    build_source_document,
    delete_document,
    get_document_by_id,
    insert_document,
    link_document_to_catalyst,
    list_documents_by_catalyst_id,
    list_documents_by_ticker,
    unlink_document_from_catalyst,
)
from src.documents.text_cleaning import clean_text, compute_text_hash, preview_text, text_quality_warnings
from src.validation.debug import document_availability_warnings


def long_text() -> str:
    return "This is a sufficiently long source document for deterministic storage and future extraction testing."


def test_documents_table_insert_list_get_delete(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    document = build_source_document("AAPL", "manual_text", "Manual note", long_text(), source="manual")
    document_id = insert_document(db_path, document)

    stored = get_document_by_id(db_path, document_id)
    by_ticker = list_documents_by_ticker(db_path, "AAPL")

    assert document_id > 0
    assert stored is not None
    assert stored["ticker"] == "AAPL"
    assert len(by_ticker) == 1
    assert delete_document(db_path, document_id) is True
    assert list_documents_by_ticker(db_path, "AAPL").empty


def test_document_deduplicates_by_hash_and_source_url(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    first = build_source_document(
        "AAPL",
        "news_article",
        "Same article",
        long_text(),
        source="manual",
        source_url="https://example.com/article",
    )
    second = build_source_document(
        "AAPL",
        "news_article",
        "Same article again",
        long_text(),
        source="manual",
        source_url="https://example.com/article",
    )

    first_id = insert_document(db_path, first)
    second_id = insert_document(db_path, second)

    assert first_id == second_id
    assert len(list_documents_by_ticker(db_path, "AAPL")) == 1


def test_document_deduplication_is_ticker_scoped_and_uses_title_date_fallback(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    shared_text = long_text()
    aapl_id = insert_document(
        db_path,
        build_source_document(
            "AAPL",
            "company_ir_press_release",
            "Shared wording",
            shared_text,
            source="company_ir_press_release",
            published_at="2024-01-03",
        ),
    )
    msft_id = insert_document(
        db_path,
        build_source_document(
            "MSFT",
            "company_ir_press_release",
            "Shared wording",
            shared_text,
            source="company_ir_press_release",
            published_at="2024-01-03",
        ),
    )
    title_date_duplicate_id = insert_document(
        db_path,
        build_source_document(
            "AAPL",
            "company_ir_press_release",
            "Shared wording",
            "Different text on the same titled release date.",
            source="company_ir_press_release",
            published_at="2024-01-03",
        ),
    )

    assert aapl_id != msft_id
    assert title_date_duplicate_id == aapl_id
    assert len(list_documents_by_ticker(db_path, "AAPL")) == 1
    assert len(list_documents_by_ticker(db_path, "MSFT")) == 1


def test_legacy_global_document_hash_constraint_migrates_without_losing_rows(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            """
            CREATE TABLE source_documents (
                document_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                catalyst_id INTEGER,
                document_type TEXT NOT NULL,
                source TEXT NOT NULL,
                source_url TEXT,
                accession_number TEXT,
                filing_type TEXT,
                title TEXT NOT NULL,
                published_at TEXT,
                raw_text TEXT,
                cleaned_text TEXT,
                text_hash TEXT NOT NULL UNIQUE,
                parsing_status TEXT NOT NULL,
                warnings TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                raw_payload_json TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO source_documents (
                ticker, document_type, source, title, raw_text, cleaned_text, text_hash,
                parsing_status, created_at, updated_at
            ) VALUES ('AAPL', 'manual_text', 'manual', 'Legacy', 'legacy text', 'legacy text',
                      'legacy-hash', 'success', '2024-01-01T00:00:00+00:00', '2024-01-01T00:00:00+00:00')
            """
        )

    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        original = conn.execute("SELECT ticker, text_hash FROM source_documents").fetchall()
        table_sql = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = 'source_documents'"
        ).fetchone()[0]
        unique_index = conn.execute(
            "SELECT COUNT(*) FROM sqlite_master WHERE type = 'index' AND name = 'idx_source_documents_unique_ticker_hash'"
        ).fetchone()[0]

    assert [(row["ticker"], row["text_hash"]) for row in original] == [("AAPL", "legacy-hash")]
    assert "text_hash TEXT NOT NULL UNIQUE" not in table_sql
    assert unique_index == 1


def test_text_cleaning_hash_preview_and_warnings() -> None:
    raw = "<html><body><p>Hello&nbsp;world</p><script>bad()</script><p>Second line</p></body></html>"
    cleaned = clean_text(raw)

    assert "Hello world" in cleaned
    assert "bad()" not in cleaned
    assert compute_text_hash(cleaned) == compute_text_hash(cleaned)
    assert preview_text("a" * 100, limit=10).endswith("[preview truncated]")
    assert text_quality_warnings("", "")[0] == "Raw text is empty."


def test_csv_import_validates_and_builds_optional_catalyst() -> None:
    df = pd.DataFrame(
        [
            {
                "ticker": "AMD",
                "document_type": "news_article",
                "title": "Imported news",
                "published_at": "2026-01-15",
                "source": "example",
                "source_url": "https://example.com/amd",
                "text": long_text(),
                "sentiment_label": "positive",
                "catalyst_strength": 6,
                "confidence": 0.7,
            },
            {
                "ticker": "AMD",
                "document_type": "news_article",
                "title": "Duplicate imported news",
                "published_at": "2026-01-15",
                "source": "example",
                "source_url": "https://example.com/amd",
                "text": long_text(),
            },
            {"ticker": "", "text": "missing ticker"},
            {"ticker": "AMD", "document_type": "unsupported", "text": long_text()},
        ]
    )

    result = parse_document_import_frame(df)

    assert len(result.rows) == 2
    assert result.rows[0].document.ticker == "AMD"
    assert result.rows[0].catalyst is not None
    assert result.rows[0].catalyst.sentiment_label == "positive"
    assert len(result.errors) == 2
    assert any("duplicate" in warning for warning in result.warnings)


def test_linking_document_to_catalyst(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    catalyst_id = insert_catalyst(
        db_path,
        CatalystEvent(
            ticker="NVDA",
            event_date=date(2026, 1, 5),
            event_type="manual_note",
            title="Manual catalyst",
            summary="Test",
            source="manual",
            is_manual=True,
        ),
    )
    document_id = insert_document(db_path, build_source_document("NVDA", "manual_text", "Linked note", long_text()))

    assert link_document_to_catalyst(db_path, document_id, catalyst_id) is True
    linked = list_documents_by_catalyst_id(db_path, catalyst_id)
    assert len(linked) == 1
    assert unlink_document_from_catalyst(db_path, document_id) is True
    assert list_documents_by_catalyst_id(db_path, catalyst_id).empty


def test_sec_text_fetcher_fails_gracefully(monkeypatch) -> None:
    provider = SecFilingsProvider()

    def fail(_: str, max_bytes: int = 2_000_000):
        raise OSError("offline")

    monkeypatch.setattr(provider, "_fetch_bytes", fail)
    result = provider.fetch_filing_text_document(
        {
            "id": 1,
            "ticker": "AAPL",
            "event_date": date(2026, 1, 10),
            "title": "SEC 8-K filing",
            "source_url": "https://www.sec.gov/example.txt",
            "raw_payload_json": '{"accessionNumber": "0000000000-26-000001", "form": "8-K"}',
        }
    )

    assert result.document is not None
    assert result.document.parsing_status == "failed"
    assert result.warnings


def test_validation_document_warnings_include_missing_and_sec_unlinked() -> None:
    catalyst_events = pd.DataFrame(
        [
            {
                "id": 7,
                "ticker": "AAPL",
                "event_type": "sec_filing",
                "title": "SEC 8-K filing",
            }
        ]
    )

    warnings = document_availability_warnings(pd.DataFrame(), catalyst_events)
    warning_names = {warning["name"] for warning in warnings}

    assert "no_source_documents" in warning_names
    assert "sec_catalyst_without_source_text" in warning_names
