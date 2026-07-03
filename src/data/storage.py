from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


OHLCV_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


def connect(db_path: str | Path) -> sqlite3.Connection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS ohlcv_cache (
                ticker TEXT NOT NULL,
                date TEXT NOT NULL,
                open REAL,
                high REAL,
                low REAL,
                close REAL,
                adj_close REAL,
                volume REAL,
                PRIMARY KEY (ticker, date)
            );

            CREATE TABLE IF NOT EXISTS scan_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT NOT NULL,
                run_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                score REAL,
                label TEXT,
                payload_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS trade_journal (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                ticker TEXT NOT NULL,
                direction TEXT NOT NULL,
                entry_date TEXT,
                entry_price REAL,
                stop_loss REAL,
                target_price REAL,
                position_size REAL,
                thesis TEXT,
                exit_date TEXT,
                exit_price REAL,
                result TEXT,
                lessons_learned TEXT
            );

            CREATE TABLE IF NOT EXISTS manual_catalysts (
                ticker TEXT PRIMARY KEY,
                note TEXT,
                catalyst_score REAL DEFAULT 0,
                updated_at TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS catalysts (
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
                available_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                raw_payload_json TEXT,
                dedupe_key TEXT NOT NULL UNIQUE
            );

            CREATE INDEX IF NOT EXISTS idx_catalysts_ticker_date
                ON catalysts (ticker, event_date DESC);

            CREATE INDEX IF NOT EXISTS idx_catalysts_event_type
                ON catalysts (event_type);

            CREATE INDEX IF NOT EXISTS idx_catalysts_ticker_source_event
                ON catalysts (ticker, source, event_type, id);

            CREATE TABLE IF NOT EXISTS sec_filing_classifications (
                catalyst_id INTEGER PRIMARY KEY,
                ticker TEXT NOT NULL,
                accession_number TEXT,
                form TEXT,
                classification TEXT NOT NULL,
                classification_reason TEXT NOT NULL,
                classifier_version TEXT NOT NULL,
                feature_eligible INTEGER NOT NULL DEFAULT 0,
                exclusion_reason TEXT,
                classified_at TEXT NOT NULL,
                raw_payload_hash TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sec_filing_classifications_ticker
                ON sec_filing_classifications (ticker, classification, feature_eligible);

            CREATE INDEX IF NOT EXISTS idx_sec_filing_classifications_accession
                ON sec_filing_classifications (accession_number);

            CREATE TABLE IF NOT EXISTS source_documents (
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
            );

            CREATE INDEX IF NOT EXISTS idx_source_documents_ticker_date
                ON source_documents (ticker, published_at DESC);

            CREATE INDEX IF NOT EXISTS idx_source_documents_catalyst
                ON source_documents (catalyst_id);

            CREATE INDEX IF NOT EXISTS idx_source_documents_source_url
                ON source_documents (source_url);

            CREATE TABLE IF NOT EXISTS llm_extractions (
                extraction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                document_id INTEGER NOT NULL,
                catalyst_id INTEGER,
                ticker TEXT NOT NULL,
                provider TEXT NOT NULL,
                model_name TEXT,
                extraction_type TEXT NOT NULL,
                event_type_detected TEXT NOT NULL,
                sentiment_label TEXT NOT NULL,
                catalyst_strength INTEGER NOT NULL,
                risk_severity INTEGER NOT NULL,
                confidence REAL NOT NULL,
                document_relevance TEXT NOT NULL DEFAULT 'unknown',
                evidence_sufficiency TEXT NOT NULL DEFAULT 'unknown',
                time_horizon TEXT NOT NULL,
                key_positive_points TEXT NOT NULL,
                key_risks TEXT NOT NULL,
                evidence_snippets TEXT NOT NULL,
                short_summary TEXT,
                detailed_summary TEXT,
                proposed_score_effect INTEGER NOT NULL,
                review_status TEXT NOT NULL DEFAULT 'pending_review',
                reviewer_note TEXT,
                reviewed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                raw_llm_response_json TEXT,
                prompt_version TEXT,
                extraction_warnings TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_llm_extractions_ticker
                ON llm_extractions (ticker, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_llm_extractions_document
                ON llm_extractions (document_id);

            CREATE INDEX IF NOT EXISTS idx_llm_extractions_catalyst
                ON llm_extractions (catalyst_id);

            CREATE INDEX IF NOT EXISTS idx_llm_extractions_review_status
                ON llm_extractions (review_status);

            CREATE TABLE IF NOT EXISTS catalyst_proposals (
                proposal_id INTEGER PRIMARY KEY AUTOINCREMENT,
                extraction_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                target_catalyst_id INTEGER,
                proposal_type TEXT NOT NULL,
                proposed_event_type TEXT NOT NULL,
                proposed_event_date TEXT,
                proposed_title TEXT NOT NULL,
                proposed_summary TEXT,
                proposed_sentiment TEXT NOT NULL,
                proposed_strength INTEGER NOT NULL,
                proposed_confidence REAL NOT NULL,
                proposed_source TEXT,
                proposed_source_url TEXT,
                evidence_snippets_json TEXT NOT NULL,
                risk_severity INTEGER NOT NULL,
                document_relevance TEXT NOT NULL DEFAULT 'unknown',
                evidence_sufficiency TEXT NOT NULL DEFAULT 'unknown',
                proposal_status TEXT NOT NULL DEFAULT 'draft',
                reviewer_note TEXT,
                initiated_by TEXT NOT NULL DEFAULT 'local_user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                reviewed_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_catalyst_proposals_ticker
                ON catalyst_proposals (ticker, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_catalyst_proposals_extraction
                ON catalyst_proposals (extraction_id);

            CREATE INDEX IF NOT EXISTS idx_catalyst_proposals_document
                ON catalyst_proposals (document_id);

            CREATE INDEX IF NOT EXISTS idx_catalyst_proposals_target
                ON catalyst_proposals (target_catalyst_id);

            CREATE INDEX IF NOT EXISTS idx_catalyst_proposals_status
                ON catalyst_proposals (proposal_status);

            CREATE TABLE IF NOT EXISTS extraction_catalyst_links (
                link_id INTEGER PRIMARY KEY AUTOINCREMENT,
                extraction_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                catalyst_id INTEGER NOT NULL,
                link_status TEXT NOT NULL DEFAULT 'active',
                reviewer_note TEXT,
                initiated_by TEXT NOT NULL DEFAULT 'local_user',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                unlinked_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_extraction_catalyst_links_ticker
                ON extraction_catalyst_links (ticker, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_extraction_catalyst_links_extraction
                ON extraction_catalyst_links (extraction_id);

            CREATE INDEX IF NOT EXISTS idx_extraction_catalyst_links_catalyst
                ON extraction_catalyst_links (catalyst_id);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_extraction_catalyst_links_active_unique
                ON extraction_catalyst_links (extraction_id, catalyst_id)
                WHERE link_status = 'active';

            CREATE TABLE IF NOT EXISTS catalyst_publications (
                publication_id INTEGER PRIMARY KEY AUTOINCREMENT,
                proposal_id INTEGER NOT NULL,
                extraction_id INTEGER NOT NULL,
                document_id INTEGER NOT NULL,
                catalyst_id INTEGER NOT NULL,
                publication_action TEXT NOT NULL,
                publication_status TEXT NOT NULL DEFAULT 'published',
                before_snapshot_json TEXT,
                after_snapshot_json TEXT NOT NULL,
                proposal_snapshot_json TEXT NOT NULL,
                catalyst_component_before REAL NOT NULL DEFAULT 0,
                catalyst_component_after REAL NOT NULL DEFAULT 0,
                catalyst_component_delta REAL NOT NULL DEFAULT 0,
                publisher_note TEXT NOT NULL,
                published_at TEXT NOT NULL,
                reverted_at TEXT,
                revert_note TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_catalyst_publications_proposal
                ON catalyst_publications (proposal_id);

            CREATE INDEX IF NOT EXISTS idx_catalyst_publications_extraction
                ON catalyst_publications (extraction_id);

            CREATE INDEX IF NOT EXISTS idx_catalyst_publications_document
                ON catalyst_publications (document_id);

            CREATE INDEX IF NOT EXISTS idx_catalyst_publications_catalyst
                ON catalyst_publications (catalyst_id);

            CREATE INDEX IF NOT EXISTS idx_catalyst_publications_status
                ON catalyst_publications (publication_status);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_catalyst_publications_active_proposal
                ON catalyst_publications (proposal_id)
                WHERE publication_status = 'published';

            CREATE TABLE IF NOT EXISTS catalyst_revisions (
                revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                catalyst_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                before_snapshot_json TEXT,
                after_snapshot_json TEXT,
                effective_timestamp TEXT NOT NULL,
                recorded_timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_catalyst_revisions_catalyst
                ON catalyst_revisions (catalyst_id, effective_timestamp);

            CREATE INDEX IF NOT EXISTS idx_catalyst_revisions_ticker
                ON catalyst_revisions (ticker, effective_timestamp);

            CREATE INDEX IF NOT EXISTS idx_catalyst_revisions_ticker_action
                ON catalyst_revisions (ticker, action, catalyst_id, effective_timestamp);

            CREATE TABLE IF NOT EXISTS sec_response_cache (
                cache_key TEXT PRIMARY KEY,
                url TEXT NOT NULL,
                response_json TEXT,
                status_code INTEGER,
                error TEXT,
                fetched_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sec_response_cache_url
                ON sec_response_cache (url);

            CREATE TABLE IF NOT EXISTS sec_backfill_runs (
                sec_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                universe_snapshot_json TEXT NOT NULL,
                requested_start_date TEXT NOT NULL,
                requested_end_date TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                total_tickers INTEGER NOT NULL DEFAULT 0,
                completed_tickers INTEGER NOT NULL DEFAULT 0,
                failed_tickers INTEGER NOT NULL DEFAULT 0,
                events_inserted INTEGER NOT NULL DEFAULT 0,
                duplicates_skipped INTEGER NOT NULL DEFAULT 0,
                provider TEXT,
                warnings_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_sec_backfill_runs_status
                ON sec_backfill_runs (status, started_at DESC);

            CREATE TABLE IF NOT EXISTS sec_backfill_items (
                sec_item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                sec_run_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                status TEXT NOT NULL,
                filings_seen INTEGER NOT NULL DEFAULT 0,
                events_inserted INTEGER NOT NULL DEFAULT 0,
                duplicates_skipped INTEGER NOT NULL DEFAULT 0,
                first_acceptance_at TEXT,
                last_acceptance_at TEXT,
                error TEXT,
                warning TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(sec_run_id, ticker)
            );

            CREATE INDEX IF NOT EXISTS idx_sec_backfill_items_run_status
                ON sec_backfill_items (sec_run_id, status);

            CREATE TABLE IF NOT EXISTS earnings_events (
                earnings_event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                fiscal_period_end TEXT,
                announced_at TEXT,
                available_at TEXT NOT NULL,
                timing TEXT NOT NULL DEFAULT 'unknown',
                eps_estimate REAL,
                eps_actual REAL,
                eps_surprise REAL,
                eps_surprise_percent REAL,
                revenue_estimate REAL,
                revenue_actual REAL,
                revenue_surprise_percent REAL,
                currency TEXT,
                provider TEXT NOT NULL,
                provider_event_id TEXT,
                fetched_at TEXT NOT NULL,
                raw_payload_json TEXT,
                raw_payload_hash TEXT NOT NULL,
                data_quality_status TEXT NOT NULL,
                warnings TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                dedupe_key TEXT NOT NULL UNIQUE
            );

            CREATE INDEX IF NOT EXISTS idx_earnings_events_ticker_available
                ON earnings_events (ticker, available_at);

            CREATE INDEX IF NOT EXISTS idx_earnings_events_ticker_announced
                ON earnings_events (ticker, announced_at);

            CREATE INDEX IF NOT EXISTS idx_earnings_events_provider_event
                ON earnings_events (provider, provider_event_id);

            CREATE TABLE IF NOT EXISTS earnings_event_revisions (
                revision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                earnings_event_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                action TEXT NOT NULL,
                before_snapshot_json TEXT,
                after_snapshot_json TEXT,
                effective_timestamp TEXT NOT NULL,
                recorded_timestamp TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_earnings_revisions_event
                ON earnings_event_revisions (earnings_event_id, effective_timestamp);

            CREATE TABLE IF NOT EXISTS earnings_response_cache (
                cache_key TEXT PRIMARY KEY,
                provider TEXT NOT NULL,
                ticker TEXT NOT NULL,
                response_json TEXT,
                status TEXT NOT NULL,
                error TEXT,
                fetched_at TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_earnings_response_cache_ticker
                ON earnings_response_cache (provider, ticker);

            CREATE TABLE IF NOT EXISTS research_event_annotations (
                annotation_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                event_date TEXT NOT NULL,
                available_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                sentiment_label TEXT NOT NULL DEFAULT 'unknown',
                strength INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0,
                source TEXT NOT NULL DEFAULT 'manual',
                source_url TEXT,
                title TEXT,
                summary TEXT,
                evidence_text TEXT,
                tags_json TEXT NOT NULL DEFAULT '[]',
                research_only INTEGER NOT NULL DEFAULT 1,
                scanner_scoring_effect INTEGER NOT NULL DEFAULT 0,
                dedupe_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_research_event_annotations_ticker_available
                ON research_event_annotations (ticker, available_at);

            CREATE INDEX IF NOT EXISTS idx_research_event_annotations_ticker_event
                ON research_event_annotations (ticker, event_date);

            CREATE INDEX IF NOT EXISTS idx_research_event_annotations_type
                ON research_event_annotations (event_type, sentiment_label);

            CREATE TABLE IF NOT EXISTS research_event_candidates (
                candidate_id INTEGER PRIMARY KEY AUTOINCREMENT,
                ticker TEXT NOT NULL,
                event_date TEXT NOT NULL,
                available_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                title TEXT NOT NULL,
                summary TEXT,
                source TEXT NOT NULL,
                source_url TEXT,
                evidence_text TEXT,
                sentiment_label TEXT NOT NULL DEFAULT 'unknown',
                strength INTEGER NOT NULL DEFAULT 0,
                confidence REAL NOT NULL DEFAULT 0,
                tags_json TEXT NOT NULL DEFAULT '[]',
                provider TEXT NOT NULL DEFAULT 'csv_manual',
                provider_metadata_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'staged',
                duplicate_of_annotation_id INTEGER,
                duplicate_of_candidate_id INTEGER,
                duplicate_reason TEXT,
                rejection_reason TEXT,
                normalized_title TEXT NOT NULL,
                evidence_text_hash TEXT,
                dedupe_key TEXT NOT NULL UNIQUE,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                reviewed_at TEXT,
                imported_annotation_id INTEGER
            );

            CREATE INDEX IF NOT EXISTS idx_research_event_candidates_status
                ON research_event_candidates (status, ticker, available_at);

            CREATE INDEX IF NOT EXISTS idx_research_event_candidates_ticker_event
                ON research_event_candidates (ticker, event_date);

            CREATE INDEX IF NOT EXISTS idx_research_event_candidates_source_url
                ON research_event_candidates (source_url);

            CREATE TABLE IF NOT EXISTS dataset_builds (
                dataset_id INTEGER PRIMARY KEY AUTOINCREMENT,
                version TEXT NOT NULL,
                build_timestamp TEXT NOT NULL,
                requested_start_date TEXT NOT NULL,
                requested_end_date TEXT NOT NULL,
                ticker_universe_json TEXT NOT NULL,
                feature_columns_json TEXT NOT NULL,
                label_definitions_json TEXT NOT NULL,
                row_count INTEGER NOT NULL DEFAULT 0,
                data_hash TEXT NOT NULL,
                warnings_json TEXT NOT NULL,
                export_path TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_dataset_builds_timestamp
                ON dataset_builds (build_timestamp DESC);

            CREATE TABLE IF NOT EXISTS feature_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                trading_date TEXT NOT NULL,
                as_of_timestamp TEXT NOT NULL,
                feature_version TEXT NOT NULL,
                market_regime_json TEXT NOT NULL,
                technical_json TEXT NOT NULL,
                relative_strength_json TEXT NOT NULL,
                volume_liquidity_json TEXT NOT NULL,
                catalyst_json TEXT NOT NULL,
                llm_supported_json TEXT NOT NULL,
                data_quality_json TEXT NOT NULL,
                features_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_feature_snapshots_dataset
                ON feature_snapshots (dataset_id, ticker, trading_date);

            CREATE INDEX IF NOT EXISTS idx_feature_snapshots_ticker_date
                ON feature_snapshots (ticker, trading_date);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_feature_snapshots_dataset_ticker_date
                ON feature_snapshots (dataset_id, ticker, trading_date);

            CREATE TABLE IF NOT EXISTS outcome_labels (
                label_id INTEGER PRIMARY KEY AUTOINCREMENT,
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                entry_date TEXT NOT NULL,
                horizon TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_date TEXT NOT NULL,
                exit_price REAL NOT NULL,
                forward_return REAL NOT NULL,
                spy_forward_return REAL,
                excess_return REAL,
                label_available_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_outcome_labels_snapshot
                ON outcome_labels (snapshot_id);

            CREATE INDEX IF NOT EXISTS idx_outcome_labels_ticker_horizon
                ON outcome_labels (ticker, horizon);

            CREATE UNIQUE INDEX IF NOT EXISTS idx_outcome_labels_snapshot_horizon
                ON outcome_labels (snapshot_id, horizon);

            CREATE TABLE IF NOT EXISTS backfill_runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                version TEXT NOT NULL,
                universe_snapshot_json TEXT NOT NULL,
                requested_start_date TEXT NOT NULL,
                requested_end_date TEXT NOT NULL,
                status TEXT NOT NULL,
                started_at TEXT NOT NULL,
                completed_at TEXT,
                total_tickers INTEGER NOT NULL DEFAULT 0,
                completed_tickers INTEGER NOT NULL DEFAULT 0,
                failed_tickers INTEGER NOT NULL DEFAULT 0,
                generated_rows INTEGER NOT NULL DEFAULT 0,
                provider TEXT,
                price_adjustment TEXT,
                warnings_json TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_backfill_runs_status
                ON backfill_runs (status, started_at DESC);

            CREATE TABLE IF NOT EXISTS backfill_items (
                item_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                status TEXT NOT NULL,
                rows_generated INTEGER NOT NULL DEFAULT 0,
                first_date TEXT,
                last_date TEXT,
                expected_snapshots INTEGER NOT NULL DEFAULT 0,
                generated_snapshots INTEGER NOT NULL DEFAULT 0,
                completed_labels_1_session INTEGER NOT NULL DEFAULT 0,
                completed_labels_5_session INTEGER NOT NULL DEFAULT 0,
                completed_labels_20_session INTEGER NOT NULL DEFAULT 0,
                error TEXT,
                warning TEXT,
                metadata_json TEXT NOT NULL DEFAULT '{}',
                started_at TEXT,
                completed_at TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                UNIQUE(run_id, ticker)
            );

            CREATE INDEX IF NOT EXISTS idx_backfill_items_run_status
                ON backfill_items (run_id, status);

            CREATE TABLE IF NOT EXISTS model_runs (
                model_run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                dataset_id INTEGER NOT NULL,
                dataset_hash TEXT,
                target_column TEXT NOT NULL,
                target_horizon TEXT NOT NULL,
                task TEXT NOT NULL,
                feature_set_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                model_version TEXT NOT NULL,
                config_json TEXT NOT NULL,
                split_config_json TEXT NOT NULL,
                feature_columns_json TEXT NOT NULL,
                status TEXT NOT NULL,
                warnings_json TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL,
                completed_at TEXT,
                updated_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_model_runs_dataset
                ON model_runs (dataset_id, created_at DESC);

            CREATE INDEX IF NOT EXISTS idx_model_runs_model
                ON model_runs (target_horizon, feature_set_name, model_name, created_at DESC);

            CREATE TABLE IF NOT EXISTS model_fold_metrics (
                metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_run_id INTEGER NOT NULL,
                fold_name TEXT NOT NULL,
                split_name TEXT NOT NULL,
                train_start_date TEXT,
                train_end_date TEXT,
                eval_start_date TEXT,
                eval_end_date TEXT,
                train_rows INTEGER NOT NULL,
                eval_rows INTEGER NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_model_fold_metrics_run
                ON model_fold_metrics (model_run_id, fold_name);

            CREATE TABLE IF NOT EXISTS model_final_metrics (
                final_metric_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_run_id INTEGER NOT NULL,
                split_name TEXT NOT NULL,
                metrics_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_model_final_metrics_run
                ON model_final_metrics (model_run_id);

            CREATE TABLE IF NOT EXISTS model_predictions (
                prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_run_id INTEGER NOT NULL,
                snapshot_id INTEGER NOT NULL,
                ticker TEXT NOT NULL,
                snapshot_date TEXT NOT NULL,
                target_horizon TEXT NOT NULL,
                split_name TEXT NOT NULL,
                fold_name TEXT NOT NULL,
                y_true REAL,
                y_pred REAL,
                y_pred_label INTEGER,
                y_score REAL,
                feature_set_name TEXT NOT NULL,
                model_name TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_model_predictions_run
                ON model_predictions (model_run_id, split_name, fold_name);

            CREATE INDEX IF NOT EXISTS idx_model_predictions_snapshot
                ON model_predictions (snapshot_id);
            """
        )
        _ensure_column(conn, "llm_extractions", "document_relevance", "TEXT NOT NULL DEFAULT 'unknown'")
        _ensure_column(conn, "llm_extractions", "evidence_sufficiency", "TEXT NOT NULL DEFAULT 'unknown'")
        _ensure_column(conn, "catalyst_proposals", "initiated_by", "TEXT NOT NULL DEFAULT 'local_user'")
        _ensure_column(conn, "extraction_catalyst_links", "initiated_by", "TEXT NOT NULL DEFAULT 'local_user'")
        _ensure_column(conn, "catalysts", "available_at", "TEXT")
        _ensure_column(conn, "dataset_builds", "audit_columns_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "dataset_builds", "label_columns_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "dataset_builds", "identifier_columns_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "dataset_builds", "metadata_columns_json", "TEXT NOT NULL DEFAULT '[]'")
        _ensure_column(conn, "dataset_builds", "feature_manifest_json", "TEXT NOT NULL DEFAULT '{}'")
        conn.execute("UPDATE catalysts SET available_at = created_at WHERE available_at IS NULL OR available_at = ''")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_catalysts_available ON catalysts (ticker, available_at)")


def _ensure_column(conn: sqlite3.Connection, table: str, column: str, definition: str) -> None:
    columns = {row["name"] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    if column not in columns:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {definition}")


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    normalized = df.copy()
    if isinstance(normalized.index, pd.DatetimeIndex):
        normalized = normalized.reset_index()
        index_column = normalized.columns[0]
        if index_column not in {"Date", "Datetime", "date"}:
            normalized = normalized.rename(columns={index_column: "date"})

    rename = {
        "Date": "date",
        "Datetime": "date",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Adj Close": "adj_close",
        "Adj_Close": "adj_close",
        "Volume": "volume",
    }
    normalized = normalized.rename(columns=rename)
    if "adj_close" not in normalized.columns and "close" in normalized.columns:
        normalized["adj_close"] = normalized["close"]

    for col in OHLCV_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = pd.NA

    normalized = normalized[OHLCV_COLUMNS]
    normalized["date"] = pd.to_datetime(normalized["date"]).dt.tz_localize(None).dt.date.astype(str)
    numeric_cols = [c for c in OHLCV_COLUMNS if c != "date"]
    normalized[numeric_cols] = normalized[numeric_cols].apply(pd.to_numeric, errors="coerce")
    normalized = normalized.dropna(subset=["date", "close"]).drop_duplicates("date")
    return normalized.sort_values("date").reset_index(drop=True)


def upsert_ohlcv(db_path: str | Path, ticker: str, df: pd.DataFrame) -> None:
    init_db(db_path)
    normalized = normalize_ohlcv(df)
    if normalized.empty:
        return

    rows = [
        (
            ticker.upper(),
            row.date,
            row.open,
            row.high,
            row.low,
            row.close,
            row.adj_close,
            row.volume,
        )
        for row in normalized.itertuples(index=False)
    ]

    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR REPLACE INTO ohlcv_cache
            (ticker, date, open, high, low, close, adj_close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def load_ohlcv(db_path: str | Path, ticker: str) -> pd.DataFrame:
    init_db(db_path)
    try:
        with connect(db_path) as conn:
            df = pd.read_sql_query(
                """
                SELECT date, open, high, low, close, adj_close, volume
                FROM ohlcv_cache
                WHERE ticker = ?
                ORDER BY date ASC
                """,
                conn,
                params=(ticker.upper(),),
                parse_dates=["date"],
            )
    except sqlite3.OperationalError:
        init_db(db_path)
        return pd.DataFrame()
    if df.empty:
        return df
    return df.set_index("date")


def clear_ohlcv_cache(db_path: str | Path) -> None:
    init_db(db_path)
    with connect(db_path) as conn:
        conn.execute("DELETE FROM ohlcv_cache")


def save_scan_results(db_path: str | Path, rows: list[dict[str, Any]]) -> str:
    init_db(db_path)
    run_id = datetime.now(UTC).strftime("%Y%m%d%H%M%S")
    run_at = datetime.now(UTC).isoformat(timespec="seconds")
    payload_rows = [
        (
            run_id,
            run_at,
            row.get("ticker", ""),
            float(row.get("score", 0) or 0),
            row.get("label", ""),
            json.dumps(row, default=str),
        )
        for row in rows
    ]
    with connect(db_path) as conn:
        conn.executemany(
            """
            INSERT INTO scan_results (run_id, run_at, ticker, score, label, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            payload_rows,
        )
    return run_id


def load_latest_scan_results(db_path: str | Path) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        latest = conn.execute("SELECT run_id FROM scan_results ORDER BY id DESC LIMIT 1").fetchone()
        if latest is None:
            return pd.DataFrame()
        rows = conn.execute(
            "SELECT payload_json FROM scan_results WHERE run_id = ? ORDER BY score DESC",
            (latest["run_id"],),
        ).fetchall()
    payloads = [json.loads(row["payload_json"]) for row in rows]
    return pd.DataFrame(payloads)


def load_manual_catalysts(db_path: str | Path) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query(
            """
            SELECT ticker, note, catalyst_score, updated_at
            FROM manual_catalysts
            ORDER BY updated_at DESC
            """,
            conn,
        )


def upsert_manual_catalyst(db_path: str | Path, ticker: str, note: str, catalyst_score: float) -> None:
    init_db(db_path)
    score = max(0.0, min(10.0, float(catalyst_score)))
    with connect(db_path) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO manual_catalysts (ticker, note, catalyst_score, updated_at)
            VALUES (?, ?, ?, ?)
            """,
            (
                ticker.upper(),
                note.strip(),
                score,
                datetime.now(UTC).isoformat(timespec="seconds"),
            ),
        )
    try:
        from src.catalysts.models import CatalystEvent
        from src.catalysts.repository import insert_catalyst

        insert_catalyst(
            db_path,
            CatalystEvent(
                ticker=ticker.upper(),
                event_date=datetime.now(UTC).date(),
                event_type="manual_note",
                title="Manual catalyst note",
                summary=note.strip(),
                source="manual_legacy",
                sentiment_label="unknown",
                catalyst_strength=int(round(score)),
                confidence=1.0,
                is_manual=True,
            ),
        )
    except Exception:
        # Legacy helper should never fail because the richer event bridge failed.
        pass


def add_trade(db_path: str | Path, trade: dict[str, Any]) -> None:
    init_db(db_path)
    fields = [
        "ticker",
        "direction",
        "entry_date",
        "entry_price",
        "stop_loss",
        "target_price",
        "position_size",
        "thesis",
        "exit_date",
        "exit_price",
        "result",
        "lessons_learned",
    ]
    values = [trade.get(field) for field in fields]
    with connect(db_path) as conn:
        conn.execute(
            f"""
            INSERT INTO trade_journal
            (created_at, {", ".join(fields)})
            VALUES ({", ".join(["?"] * (len(fields) + 1))})
            """,
            [datetime.now(UTC).isoformat(timespec="seconds"), *values],
        )


def load_trades(db_path: str | Path) -> pd.DataFrame:
    init_db(db_path)
    with connect(db_path) as conn:
        return pd.read_sql_query(
            "SELECT * FROM trade_journal ORDER BY id DESC",
            conn,
        )
