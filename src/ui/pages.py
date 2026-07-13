from __future__ import annotations

import json
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from src.annotations.csv_import import parse_annotation_import_frame
from src.annotations.document_coverage import build_document_coverage_audit
from src.annotations.models import ANNOTATION_EVENT_TYPES, ANNOTATION_SENTIMENT_LABELS, ResearchEventAnnotation
from src.annotations.news_csv_provider import parse_candidate_import_frame, parse_company_ir_press_release_frame
from src.annotations.news_repository import (
    accept_candidate,
    build_candidate_ingestion_artifact,
    candidate_counts_by_status,
    create_or_link_candidate_document,
    import_accepted_candidates,
    list_candidates,
    reject_candidate,
    stage_candidates,
)
from src.annotations.provider_registry import build_provider_readiness_report
from src.annotations.repository import annotation_counts_by_ticker, bulk_insert_annotations, insert_annotation, list_annotations
from src.annotations.source_quality import enrich_quality_frame, quality_distribution
from src.alerts.alert_formatter import format_alert, suggested_watch_action
from src.backtest.simple_backtester import (
    backtest_mean_reversion_strategy,
    backtest_momentum_breakout_strategy,
    backtest_moving_average_strategy,
    backtest_top_score_strategy,
)
from src.catalysts.earnings_adapter import YFinanceEarningsProvider
from src.catalysts.models import EVENT_TYPES, SENTIMENT_LABELS, CatalystEvent
from src.catalysts.news_adapter import PlaceholderNewsProvider
from src.catalysts.repository import (
    bulk_insert_catalysts,
    catalyst_display_frame,
    delete_catalyst,
    insert_catalyst,
    list_catalysts_by_ticker,
    list_recent_catalysts,
)
from src.catalysts.proposals import (
    PROPOSAL_EVENT_TYPES,
    PROPOSAL_SENTIMENTS,
    PROPOSAL_STATUSES,
    create_proposal_from_extraction,
    link_display_frame,
    link_extraction_to_catalyst,
    link_summary_by_catalyst,
    list_links_by_catalyst_id,
    list_links_by_extraction_id,
    list_links_by_ticker,
    list_proposals_by_extraction_id,
    list_proposals_by_target_catalyst_id,
    list_proposals_by_ticker,
    list_recent_proposals,
    map_extraction_to_proposal,
    proposal_display_frame,
    proposal_score_contribution,
    proposal_summary_by_catalyst,
    set_proposal_status,
    unlink_extraction_catalyst_link,
    update_proposal,
)
from src.catalysts.publications import (
    ALL_UPDATE_FIELDS,
    DEFAULT_UPDATE_FIELDS,
    build_publication_preview,
    is_llm_supported_catalyst,
    list_publications_by_catalyst_id,
    list_publications_by_proposal_id,
    list_publications_by_ticker,
    llm_publication_id_from_catalyst,
    publication_display_frame,
    publication_summary_by_catalyst,
    publish_proposal,
    revert_publication,
)
from src.catalysts.sec_adapter import SecFilingsProvider
from src.catalysts.sec_backfill import (
    create_sec_backfill_run,
    list_sec_backfill_items,
    list_sec_backfill_runs,
    process_sec_backfill_run,
    retry_failed_sec_items,
)
from src.catalysts.sec_classification import (
    SEC_FEATURE_POLICY,
    SEC_FEATURE_POLICY_VERSION,
    classify_ticker_sec_filings_safe,
    sec_classification_summary,
)
from src.config import Settings
from src.data import market_data, storage
from src.data.universe import load_universe
from src.datasets.builder import DEFAULT_HORIZONS, build_point_in_time_dataset
from src.datasets.backfill import (
    create_backfill_run,
    dataset_sufficiency_report,
    list_backfill_items,
    list_backfill_runs,
    process_backfill_run,
    retry_failed_items,
)
from src.datasets.repository import flatten_saved_dataset, list_dataset_builds
from src.datasets.feature_manifest import role_sets_from_frame
from src.datasets.splits import assign_chronological_splits
from src.documents.csv_import import parse_document_import_frame
from src.documents.models import DOCUMENT_SOURCES, DOCUMENT_TYPES, PARSING_STATUSES
from src.documents.repository import (
    build_source_document,
    delete_document,
    document_counts_by_catalyst,
    document_display_frame,
    get_document_by_id,
    insert_document,
    link_document_to_catalyst,
    list_documents_by_catalyst_id,
    list_documents_by_ticker,
    list_recent_documents,
    unlink_document_from_catalyst,
)
from src.documents.text_cleaning import preview_text
from src.earnings.backfill import backfill_earnings_events
from src.earnings.csv_import import parse_earnings_import_frame
from src.earnings.repository import bulk_insert_earnings_events, earnings_coverage_report, list_earnings_by_ticker
from src.earnings.yfinance_provider import YFinanceHistoricalEarningsProvider
from src.extractions.models import EXTRACTION_PROVIDERS, EXTRACTION_TYPES, REVIEW_STATUSES
from src.extractions.openai_provider import openai_provider_status
from src.extractions.prompting import prepare_document_text
from src.extractions.quality import classify_review_readiness
from src.extractions.repository import (
    extraction_summary_by_document,
    list_pending_review_extractions,
    list_recent_extractions,
    list_reviewed_extractions,
    reject_extraction,
    supersede_extraction,
)
from src.extractions.review_workflow import (
    approve_extraction_with_readiness,
    create_fallback_extraction_for_document,
    create_openai_extraction_for_document,
    document_readiness,
    enrich_extractions_with_documents,
    extraction_queue_display_frame,
    filter_extractions,
    pending_extractions_for_document,
)
from src.features.regime import classify_market_regime
from src.modeling.feature_sets import FEATURE_SET_NAMES, feature_set_definitions
from src.modeling.artifacts import list_model_artifacts
from src.modeling.feature_quality import (
    build_feature_quality_audit,
    feature_set_quality_rows,
    list_feature_quality_artifacts,
    load_feature_quality_artifact,
    write_feature_quality_artifact,
)
from src.modeling.event_feature_redesign import (
    build_event_coverage_audit,
    build_event_timing_audit,
    event_feature_set_rows,
    list_event_redesign_artifacts,
    load_event_artifact,
    write_event_artifact,
)
from src.modeling.annotation_features import (
    build_annotation_coverage_audit,
    list_annotation_artifacts,
    load_annotation_artifact,
    run_annotation_baseline_suite,
    write_annotation_artifact,
)
from src.modeling.diagnostics import (
    build_model_diagnostics,
    list_diagnostic_artifacts,
    load_diagnostics_artifact,
    write_diagnostics_artifact,
)
from src.modeling.evaluation_regime import get_dataset_evaluation_regime
from src.modeling.holdout_maturity import assess_holdout_maturity, build_holdout_extension_plan
from src.modeling.shadow_predictions import list_shadow_prediction_runs, list_shadow_predictions, shadow_status_report
from src.modeling.repository import (
    list_model_final_metrics,
    list_model_fold_metrics,
    list_model_predictions,
    list_model_runs,
)
from src.modeling.runner import latest_accepted_dataset_id, run_single_baseline_model
from src.modeling.targets import RAW_TARGET_5_SESSION, get_target_definition, target_options, target_metadata
from src.research.llm_placeholder import summarize_ticker_features
from src.research.news_placeholder import get_news_placeholder
from src.scoring.risk_rules import calculate_position_size, stop_candidates
from src.scoring.score_engine import flatten_score_result, score_ticker
from src.ui.charts import equity_curve_chart, price_volume_chart, score_breakdown_chart
from src.validation.debug import load_validation_report


def fmt_pct(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        return f"{float(value):.1%}"
    except Exception:
        return "n/a"


def fmt_money(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return "n/a"
        value = float(value)
    except Exception:
        return "n/a"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:.1f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:.1f}M"
    return f"${value:,.0f}"


def dataframe_for_streamlit(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a display-only copy with nested object cells stringified."""
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame
    display = frame.copy()
    for column in display.columns:
        dtype_name = str(display[column].dtype)
        if dtype_name not in {"object", "str", "string"} and not dtype_name.startswith("string"):
            continue
        if display[column].map(lambda value: isinstance(value, (list, dict, tuple, set))).any():
            display[column] = display[column].map(
                lambda value: json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
                if isinstance(value, (list, dict, tuple, set))
                else value
            )
    return display


def _manual_catalyst_event(
    ticker: str,
    event_date: date,
    event_type: str,
    title: str,
    summary: str,
    sentiment_label: str,
    catalyst_strength: int,
    confidence: float,
    source_url: str | None = None,
) -> CatalystEvent:
    return CatalystEvent(
        ticker=ticker,
        event_date=event_date,
        event_type=event_type,
        title=title,
        summary=summary,
        source="manual",
        source_url=source_url.strip() if source_url else None,
        sentiment_label=sentiment_label,
        catalyst_strength=int(catalyst_strength),
        confidence=float(confidence),
        is_manual=True,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )


def _show_catalyst_events(df: pd.DataFrame, title: str = "Catalysts", allow_manual_delete: bool = False, settings: Settings | None = None) -> None:
    st.subheader(title)
    if df is None or df.empty:
        st.info("No catalyst events found.")
        return

    display = catalyst_display_frame(df)
    document_counts: dict[int, int] = {}
    link_counts: dict[int, dict[str, int]] = {}
    proposal_counts: dict[int, dict[str, int]] = {}
    publication_counts: dict[int, dict[str, Any]] = {}
    if settings is not None and "id" in df.columns:
        catalyst_ids = [int(value) for value in df["id"].dropna().tolist()]
        document_counts = document_counts_by_catalyst(settings.database_file, catalyst_ids)
        link_counts = link_summary_by_catalyst(settings.database_file, catalyst_ids)
        proposal_counts = proposal_summary_by_catalyst(settings.database_file, catalyst_ids)
        publication_counts = publication_summary_by_catalyst(settings.database_file, catalyst_ids)
        display["linked documents"] = [
            document_counts.get(int(value), 0) if pd.notna(value) else 0 for value in df["id"].tolist()
        ]
        display["source text?"] = display["linked documents"].map(lambda count: "yes" if count else "no")
        display["linked extractions"] = [
            link_counts.get(int(value), {}).get("active_links", 0) if pd.notna(value) else 0
            for value in df["id"].tolist()
        ]
        display["proposals"] = [
            proposal_counts.get(int(value), {}).get("proposal_count", 0) if pd.notna(value) else 0
            for value in df["id"].tolist()
        ]
        display["llm_supported"] = [
            "yes" if is_llm_supported_catalyst(row) else "no" for _, row in df.iterrows()
        ]
        display["active publications"] = [
            publication_counts.get(int(value), {}).get("active_publication_count", 0) if pd.notna(value) else 0
            for value in df["id"].tolist()
        ]
    st.dataframe(display, width="stretch", hide_index=True)
    for row_idx, row in df.head(25).iterrows():
        manual_label = "manual" if bool(row.get("is_manual")) else "system"
        with st.expander(f"{row.get('ticker')} | {row.get('event_date')} | {row.get('event_type')} | {row.get('title')}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Sentiment", row.get("sentiment_label", "unknown"))
            c2.metric("Strength", row.get("catalyst_strength", 0))
            c3.metric("Confidence", f"{float(row.get('confidence', 0) or 0):.0%}")
            c4.metric("Source", manual_label)
            if is_llm_supported_catalyst(row):
                publication_id = llm_publication_id_from_catalyst(row)
                st.success(
                    "LLM-supported, manually reviewed active catalyst"
                    + (f" | publication #{publication_id}" if publication_id else "")
                )
                st.caption("Grounded evidence proves traceability to source text, not factual truth.")
            st.write(row.get("summary") or "No summary provided.")
            source_value = row.get("source_url")
            source_url = "" if source_value is None or pd.isna(source_value) else str(source_value).strip()
            catalyst_id = int(row["id"]) if pd.notna(row.get("id")) else None
            if source_url:
                st.link_button("Open Source", source_url, key=f"catalyst_source_{catalyst_id or row_idx}")
            if settings is not None and catalyst_id is not None:
                linked_docs = list_documents_by_catalyst_id(settings.database_file, catalyst_id, limit=10)
                st.write(f"Linked source documents: {len(linked_docs)}")
                if linked_docs.empty:
                    st.info("No source text stored for this catalyst yet.")
                else:
                    st.dataframe(document_display_frame(linked_docs), width="stretch", hide_index=True)
                linked_extractions = list_links_by_catalyst_id(settings.database_file, catalyst_id, limit=25)
                active_links = linked_extractions[linked_extractions["link_status"].eq("active")] if not linked_extractions.empty else linked_extractions
                st.write(f"Linked approved extractions: {len(active_links)} active / {len(linked_extractions)} total")
                if not linked_extractions.empty:
                    st.dataframe(link_display_frame(linked_extractions), width="stretch", hide_index=True)
                catalyst_proposals = list_proposals_by_target_catalyst_id(settings.database_file, catalyst_id, limit=25)
                st.write(f"Review-only proposals targeting this catalyst: {len(catalyst_proposals)}")
                if not catalyst_proposals.empty:
                    st.caption("These proposals are not active catalysts and do not affect scanner scoring.")
                    st.dataframe(proposal_display_frame(catalyst_proposals), width="stretch", hide_index=True)
                publications = list_publications_by_catalyst_id(settings.database_file, catalyst_id, limit=25)
                st.write(f"Publication/reversal history: {len(publications)}")
                if not publications.empty:
                    st.dataframe(publication_display_frame(publications), width="stretch", hide_index=True)
                if row.get("event_type") == "sec_filing":
                    if st.button("Fetch / store SEC filing text", key=f"fetch_sec_text_{catalyst_id}"):
                        _fetch_and_store_sec_text(settings, row, key_suffix=f"catalyst_{catalyst_id}")
            if allow_manual_delete and settings is not None and bool(row.get("is_manual")):
                if st.button("Delete manual catalyst", key=f"delete_catalyst_{row.get('id')}"):
                    delete_catalyst(settings.database_file, int(row["id"]))
                    st.success("Manual catalyst deleted. Refresh the page to update the table.")


def _fetch_and_store_sec_text(settings: Settings, catalyst_row: pd.Series | dict[str, Any], key_suffix: str = "") -> None:
    with st.spinner("Fetching SEC filing text..."):
        result = SecFilingsProvider().fetch_filing_text_document(catalyst_row)
        for warning in result.warnings:
            st.warning(warning)
        if result.document is None:
            st.error("SEC filing text could not be stored.")
            return
        document_id = insert_document(settings.database_file, result.document)
        if result.document.catalyst_id:
            link_document_to_catalyst(settings.database_file, document_id, result.document.catalyst_id)
        status = result.document.parsing_status
        st.success(f"Stored source document #{document_id} with parsing status: {status}.")


def _catalyst_select_options(catalysts: pd.DataFrame) -> tuple[list[str], dict[str, int | None]]:
    options = ["None"]
    mapping: dict[str, int | None] = {"None": None}
    if catalysts is None or catalysts.empty:
        return options, mapping
    for _, row in catalysts.head(100).iterrows():
        if pd.isna(row.get("id")):
            continue
        label = f"{int(row['id'])} | {row.get('event_date')} | {row.get('event_type')} | {row.get('title')}"
        options.append(label)
        mapping[label] = int(row["id"])
    return options, mapping


def _show_documents(
    df: pd.DataFrame,
    title: str = "Source Documents",
    settings: Settings | None = None,
    allow_delete: bool = False,
    allow_link_controls: bool = False,
    show_extraction_status: bool = False,
) -> None:
    st.subheader(title)
    if df is None or df.empty:
        st.info("No stored source documents found.")
        return

    display = document_display_frame(df)
    extraction_summary: dict[int, dict[str, Any]] = {}
    if settings is not None and show_extraction_status:
        document_ids = [int(value) for value in df["document_id"].dropna().tolist()]
        extraction_summary = extraction_summary_by_document(settings.database_file, document_ids)
        display["extractions"] = [
            extraction_summary.get(int(value), {}).get("extraction_count", 0) if pd.notna(value) else 0
            for value in df["document_id"].tolist()
        ]
        display["pending reviews"] = [
            extraction_summary.get(int(value), {}).get("pending_count", 0) if pd.notna(value) else 0
            for value in df["document_id"].tolist()
        ]
        display["latest review status"] = [
            (extraction_summary.get(int(value), {}).get("latest_review_status") or "none") if pd.notna(value) else "none"
            for value in df["document_id"].tolist()
        ]
        st.caption("Use the LLM Review page to run fallback or explicitly confirmed OpenAI extraction and manually approve/reject results.")
    st.dataframe(display, width="stretch", hide_index=True)
    for _, row in df.head(25).iterrows():
        doc_id = int(row["document_id"])
        with st.expander(f"{row.get('ticker')} | {row.get('published_at') or 'no date'} | {row.get('document_type')} | {row.get('title')}"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Parsing status", row.get("parsing_status", "n/a"))
            c2.metric("Source", row.get("source", "n/a"))
            c3.metric("Raw length", len(str(row.get("raw_text") or "")))
            c4.metric("Cleaned length", len(str(row.get("cleaned_text") or "")))
            source_value = row.get("source_url")
            source_url = "" if source_value is None or pd.isna(source_value) else str(source_value).strip()
            if source_url:
                st.link_button("Open Source", source_url, key=f"document_source_{doc_id}")
            st.write("Warnings:", row.get("warnings") or "None")
            st.write("Linked catalyst:", "None" if pd.isna(row.get("catalyst_id")) else int(row["catalyst_id"]))
            if show_extraction_status:
                summary = extraction_summary.get(doc_id, {})
                st.write(
                    "Extraction review:",
                    f"{summary.get('extraction_count', 0)} extraction(s), "
                    f"{summary.get('pending_count', 0)} pending, "
                    f"latest status {summary.get('latest_review_status') or 'none'}.",
                )
                st.info("Open the LLM Review page to inspect source text next to extraction results.")
            with st.expander("Cleaned Text Preview"):
                st.text(preview_text(str(row.get("cleaned_text") or ""), limit=3_000) or "No cleaned text available.")
            if settings is not None and allow_link_controls:
                if pd.notna(row.get("catalyst_id")):
                    if st.button("Unlink catalyst", key=f"unlink_document_{doc_id}"):
                        unlink_document_from_catalyst(settings.database_file, doc_id)
                        st.success("Document unlinked. Refresh the page to update the table.")
                else:
                    link_id = st.number_input(
                        "Catalyst ID to link",
                        min_value=0,
                        value=0,
                        step=1,
                        key=f"link_document_id_{doc_id}",
                    )
                    if st.button("Link catalyst", key=f"link_document_{doc_id}") and link_id > 0:
                        link_document_to_catalyst(settings.database_file, doc_id, int(link_id))
                        st.success("Document linked. Refresh the page to update the table.")
            if settings is not None and allow_delete:
                if st.button("Delete document", key=f"delete_document_{doc_id}"):
                    delete_document(settings.database_file, doc_id)
                    st.success("Document deleted. Refresh the page to update the table.")


def _cached_coverage(db_path: Any, tickers: list[str]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        df = storage.load_ohlcv(db_path, ticker)
        if df.empty:
            rows.append({"ticker": ticker, "start": None, "end": None, "rows": 0})
            continue
        rows.append(
            {
                "ticker": ticker,
                "start": pd.to_datetime(df.index.min()).date().isoformat(),
                "end": pd.to_datetime(df.index.max()).date().isoformat(),
                "rows": int(len(df)),
            }
        )
    return pd.DataFrame(rows)


def _load_regime_histories(settings: Settings, refresh: bool = False) -> dict[str, pd.DataFrame]:
    return market_data.get_histories(
        ["SPY", "QQQ", "IWM", "^VIX"],
        settings.database_file,
        settings.market_data_provider,
        settings.default_history_period,
        refresh=refresh,
    )


def _metrics_grid(metrics: dict[str, Any], reporting: dict[str, Any] | None = None) -> None:
    reporting = reporting or {}
    labels = [
        ("Total Return", "total_return", fmt_pct),
        ("Annualized Return", "annualized_return", fmt_pct),
        ("Max Drawdown", "max_drawdown", fmt_pct),
        ("Sharpe", "sharpe_ratio", lambda x: f"{x:.2f}"),
        (reporting.get("win_rate_label", "Win Rate"), "win_rate", fmt_pct),
        (reporting.get("average_win_label", "Average Win"), "average_win", fmt_pct),
        (reporting.get("average_loss_label", "Average Loss"), "average_loss", fmt_pct),
        (reporting.get("trade_count_label", "Trades"), "number_of_trades", lambda x: f"{int(x)}"),
    ]
    cols = st.columns(4)
    for idx, (label, key, formatter) in enumerate(labels):
        value = metrics.get(key, 0)
        try:
            rendered = formatter(value)
        except Exception:
            rendered = "n/a"
        cols[idx % 4].metric(label, rendered)


def market_regime_page(settings: Settings) -> None:
    st.header("Market Regime Dashboard")
    st.caption("Research and paper trading only. Not financial advice.")
    refresh = st.button("Refresh latest market data", key="regime_refresh")
    with st.spinner("Loading SPY, QQQ, IWM, and VIX data..."):
        histories = _load_regime_histories(settings, refresh=refresh)
    regime = classify_market_regime(histories)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Regime", regime["regime"])
    c2.metric("QQQ/SPY 20D RS", fmt_pct(regime.get("qqq_spy_rs_20")))
    c3.metric("IWM/SPY 20D RS", fmt_pct(regime.get("iwm_spy_rs_20")))
    c4.metric("VIX", "n/a" if regime.get("vix") is None else f"{regime['vix']:.1f}")

    st.subheader("Classification Explanation")
    if regime.get("warnings"):
        for warning in regime["warnings"]:
            st.warning(warning)
    st.caption(f"Regime confidence: {regime.get('confidence', 'unknown')}")
    for reason in regime["reasons"]:
        st.write(f"- {reason}")

    tabs = st.tabs(["SPY", "QQQ", "IWM"])
    for tab, ticker in zip(tabs, ["SPY", "QQQ", "IWM"], strict=False):
        with tab:
            df = histories.get(ticker, pd.DataFrame())
            st.plotly_chart(price_volume_chart(df, f"{ticker} Trend"), width="stretch", key=f"regime_price_{ticker}")


def catalyst_center_page(settings: Settings) -> None:
    st.header("Catalyst Center")
    st.caption(
        "Provider-agnostic catalyst/event log. Manual notes work without paid APIs. Reviewed LLM extractions do not affect catalysts yet."
    )

    universe = load_universe(settings.universe_file)
    tickers = universe["ticker"].tolist() if not universe.empty else []
    ticker_options = ["All", *tickers]
    today = datetime.now(UTC).date()

    with st.expander("Add Manual Catalyst", expanded=True):
        with st.form("manual_catalyst_center_form"):
            c1, c2, c3 = st.columns(3)
            ticker = c1.selectbox("Ticker", tickers if tickers else ["SPY"], index=0)
            event_date = c2.date_input("Event date", value=today)
            event_type = c3.selectbox("Event type", EVENT_TYPES, index=EVENT_TYPES.index("manual_note"))
            title = st.text_input("Title", value="")
            summary = st.text_area("Summary / thesis", value="")
            c4, c5, c6 = st.columns(3)
            sentiment = c4.selectbox("Sentiment", SENTIMENT_LABELS, index=SENTIMENT_LABELS.index("unknown"))
            strength = c5.slider("Catalyst strength", min_value=0, max_value=10, value=0, step=1)
            confidence = c6.slider("Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05)
            source_url = st.text_input("Source URL optional", value="")
            submitted = st.form_submit_button("Save Manual Catalyst")

        if submitted:
            try:
                event = _manual_catalyst_event(
                    ticker,
                    event_date,
                    event_type,
                    title,
                    summary,
                    sentiment,
                    strength,
                    confidence,
                    source_url,
                )
                insert_catalyst(settings.database_file, event)
                st.success("Manual catalyst saved.")
            except Exception as exc:
                st.error(f"Could not save catalyst: {exc}")

    st.subheader("Provider Refresh")
    c1, c2, c3 = st.columns(3)
    refresh_ticker = c1.selectbox("Refresh ticker", tickers if tickers else ["SPY"], index=0)
    if c2.button("Refresh SEC Metadata"):
        with st.spinner(f"Fetching SEC metadata for {refresh_ticker}..."):
            result = SecFilingsProvider().fetch_recent_filings(refresh_ticker)
            if result.events:
                ids = bulk_insert_catalysts(settings.database_file, result.events)
                st.success(f"Stored {len(set(ids))} SEC catalyst event(s).")
            for warning in result.warnings:
                st.warning(warning)
            if not result.events and not result.warnings:
                st.info("No SEC events returned.")
    if c3.button("Refresh Earnings Data"):
        with st.spinner(f"Fetching earnings metadata for {refresh_ticker}..."):
            result = YFinanceEarningsProvider().fetch_earnings_events(refresh_ticker)
            if result.events:
                ids = bulk_insert_catalysts(settings.database_file, result.events)
                st.success(f"Stored {len(set(ids))} earnings catalyst event(s).")
            for warning in result.warnings:
                st.warning(warning)
            if not result.events and not result.warnings:
                st.info("No earnings events returned.")

    news_result = PlaceholderNewsProvider().get_recent_news(refresh_ticker, today - timedelta(days=7), today)
    for warning in news_result.warnings:
        st.info(warning)
    st.caption("SEC metadata uses the free EDGAR submissions API when available. Earnings uses yfinance best-effort metadata.")

    st.subheader("Recent Catalyst Events")
    c1, c2, c3, c4 = st.columns([1, 1, 1, 2])
    ticker_filter = c1.selectbox("Ticker filter", ticker_options, index=0)
    event_filter = c2.multiselect("Event type filter", EVENT_TYPES, default=[])
    sentiment_filter = c3.multiselect("Sentiment filter", SENTIMENT_LABELS, default=[])
    date_range = c4.date_input("Date range", value=(today - timedelta(days=90), today + timedelta(days=45)))
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = today - timedelta(days=90), today + timedelta(days=45)

    catalysts = list_recent_catalysts(settings.database_file, days=365, limit=1000)
    if not catalysts.empty:
        catalysts = catalysts[
            (pd.to_datetime(catalysts["event_date"]).dt.date >= start_date)
            & (pd.to_datetime(catalysts["event_date"]).dt.date <= end_date)
        ]
        if ticker_filter != "All":
            catalysts = catalysts[catalysts["ticker"].str.upper() == ticker_filter.upper()]
        if event_filter:
            catalysts = catalysts[catalysts["event_type"].isin(event_filter)]
        if sentiment_filter:
            catalysts = catalysts[catalysts["sentiment_label"].isin(sentiment_filter)]

    _show_catalyst_events(catalysts, "Catalyst Events", allow_manual_delete=True, settings=settings)

    st.divider()
    proposals = list_recent_proposals(settings.database_file, limit=1000)
    if not proposals.empty:
        if ticker_filter != "All":
            proposals = proposals[proposals["ticker"].str.upper() == ticker_filter.upper()]
        status_filter = st.multiselect("Proposal status filter", PROPOSAL_STATUSES, default=[], key="catalyst_center_proposal_status")
        if status_filter:
            proposals = proposals[proposals["proposal_status"].isin(status_filter)]
    _show_catalyst_proposals(settings, proposals, "Review-Only Catalyst Proposals", allow_edit=False, key_prefix="catalyst_center")


def documents_text_page(settings: Settings) -> None:
    st.header("Documents / Text")
    st.caption(
        "Local source-text ingestion for fallback or optional OpenAI extraction review. No options data, ML, or broker execution is used here."
    )

    universe = load_universe(settings.universe_file)
    tickers = universe["ticker"].tolist() if not universe.empty else ["SPY"]
    today = datetime.now(UTC).date()

    st.subheader("Add Manual Text")
    manual_ticker = st.selectbox("Ticker for manual text", tickers, index=0, key="manual_document_ticker")
    manual_catalysts = list_catalysts_by_ticker(settings.database_file, manual_ticker, limit=100)
    catalyst_options, catalyst_mapping = _catalyst_select_options(manual_catalysts)
    with st.form("manual_source_document_form"):
        c1, c2, c3 = st.columns(3)
        document_type = c1.selectbox("Document type", DOCUMENT_TYPES, index=DOCUMENT_TYPES.index("manual_text"))
        source = c2.selectbox("Source", DOCUMENT_SOURCES, index=DOCUMENT_SOURCES.index("manual"))
        published_at = c3.date_input("Published date", value=today)
        title = st.text_input("Title", value="")
        source_url = st.text_input("Source URL optional", value="")
        linked_catalyst_label = st.selectbox("Linked catalyst optional", catalyst_options)
        raw_text = st.text_area("Raw text", value="", height=220)
        submitted = st.form_submit_button("Save Source Document")

    if submitted:
        if not raw_text.strip():
            st.error("Raw text is required.")
        else:
            try:
                document = build_source_document(
                    ticker=manual_ticker,
                    document_type=document_type,
                    source=source,
                    title=title,
                    published_at=published_at,
                    source_url=source_url.strip() or None,
                    catalyst_id=catalyst_mapping.get(linked_catalyst_label),
                    raw_text=raw_text,
                )
                document_id = insert_document(settings.database_file, document)
                st.success(f"Stored source document #{document_id}.")
            except Exception as exc:
                st.error(f"Could not store source document: {exc}")

    st.subheader("CSV Import")
    st.caption(
        "Supported columns include ticker, document_type, title, published_at, source, source_url, text, "
        "sentiment_label, catalyst_strength, and confidence. Catalyst columns optionally create linked catalyst events."
    )
    uploaded = st.file_uploader("Upload document/news CSV", type=["csv"])
    if uploaded is not None and st.button("Import CSV Documents", key="import_csv_documents"):
        try:
            csv_df = pd.read_csv(uploaded)
            import_result = parse_document_import_frame(csv_df)
            for warning in import_result.warnings:
                st.warning(warning)
            for error in import_result.errors:
                st.error(error)
            stored_docs = 0
            stored_catalysts = 0
            for row in import_result.rows:
                document = row.document
                if row.catalyst is not None:
                    catalyst_id = insert_catalyst(settings.database_file, row.catalyst)
                    document = document.model_copy(update={"catalyst_id": catalyst_id})
                    stored_catalysts += 1
                insert_document(settings.database_file, document)
                stored_docs += 1
            if stored_docs:
                st.success(f"Imported {stored_docs} document(s) and {stored_catalysts} linked catalyst event(s).")
        except Exception as exc:
            st.error(f"Could not import CSV: {exc}")

    st.subheader("Fetch SEC Filing Text")
    sec_events = list_recent_catalysts(settings.database_file, days=365, limit=1000)
    if not sec_events.empty:
        sec_events = sec_events[sec_events["event_type"].eq("sec_filing")]
    if sec_events.empty:
        st.info("No SEC filing catalyst metadata found yet. Use Catalyst Center to refresh SEC metadata first.")
    else:
        sec_labels: list[str] = []
        sec_mapping: dict[str, pd.Series] = {}
        for _, row in sec_events.head(200).iterrows():
            label = f"{int(row['id'])} | {row.get('ticker')} | {row.get('event_date')} | {row.get('title')}"
            sec_labels.append(label)
            sec_mapping[label] = row
        selected_sec = st.selectbox("Recent SEC filing catalyst", sec_labels)
        if st.button("Fetch / store selected SEC filing text", key="documents_fetch_sec_text"):
            _fetch_and_store_sec_text(settings, sec_mapping[selected_sec], key_suffix="documents_page")

    st.subheader("Stored Documents")
    c1, c2, c3, c4 = st.columns(4)
    ticker_filter = c1.selectbox("Ticker filter", ["All", *tickers], index=0)
    type_filter = c2.multiselect("Document type", DOCUMENT_TYPES, default=[])
    source_filter = c3.multiselect("Source", DOCUMENT_SOURCES, default=[])
    status_filter = c4.multiselect("Parsing status", PARSING_STATUSES, default=[])
    search = st.text_input("Search title/text preview", value="")

    documents = list_recent_documents(settings.database_file, limit=500)
    if not documents.empty:
        if ticker_filter != "All":
            documents = documents[documents["ticker"].str.upper().eq(ticker_filter.upper())]
        if type_filter:
            documents = documents[documents["document_type"].isin(type_filter)]
        if source_filter:
            documents = documents[documents["source"].isin(source_filter)]
        if status_filter:
            documents = documents[documents["parsing_status"].isin(status_filter)]
        if search.strip():
            needle = search.strip().casefold()
            haystack = (
                documents["title"].fillna("").str.casefold()
                + " "
                + documents["cleaned_text"].fillna("").str.casefold().str.slice(0, 5_000)
            )
            documents = documents[haystack.str.contains(needle, regex=False)]

    _show_documents(
        documents,
        "Stored Source Documents",
        settings=settings,
        allow_delete=True,
        allow_link_controls=True,
        show_extraction_status=True,
    )


def _document_options(documents: pd.DataFrame) -> tuple[list[str], dict[str, int]]:
    options: list[str] = []
    mapping: dict[str, int] = {}
    if documents is None or documents.empty:
        return options, mapping
    for _, row in documents.iterrows():
        document_id = int(row["document_id"])
        label = (
            f"{document_id} | {row.get('ticker')} | {row.get('document_type')} | "
            f"{row.get('published_at') or 'no date'} | {row.get('title')}"
        )
        options.append(label)
        mapping[label] = document_id
    return options, mapping


def _show_list_block(title: str, values: Any, empty_text: str) -> None:
    st.write(f"**{title}**")
    if not values:
        st.caption(empty_text)
        return
    if isinstance(values, str):
        values = [values]
    for value in values:
        st.write(f"- {value}")


def _show_extraction_details(
    settings: Settings,
    row: pd.Series,
    allow_review_actions: bool,
    key_prefix: str,
) -> None:
    extraction_id = int(row["extraction_id"])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Sentiment", row.get("sentiment_label", "unknown"))
    c2.metric("Catalyst Strength", int(row.get("catalyst_strength", 0) or 0))
    c3.metric("Risk Severity", int(row.get("risk_severity", 0) or 0))
    c4.metric("Confidence", f"{float(row.get('confidence', 0) or 0):.0%}")

    st.write("**Short summary**")
    st.write(row.get("short_summary") or "No short summary.")
    st.write("**Detailed summary**")
    st.write(row.get("detailed_summary") or "No detailed summary.")

    c5, c6, c7 = st.columns(3)
    c5.metric("Detected Event", row.get("event_type_detected", "unknown"))
    c6.metric("Proposed Score Effect", int(row.get("proposed_score_effect", 0) or 0))
    c7.metric("Review Status", row.get("review_status", "unknown"))

    readiness = classify_review_readiness(row.to_dict() if hasattr(row, "to_dict") else dict(row))
    q1, q2, q3 = st.columns(3)
    q1.metric("Document Relevance", row.get("document_relevance", "unknown"))
    q2.metric("Evidence Sufficiency", row.get("evidence_sufficiency", "unknown"))
    q3.metric("Review Readiness", readiness)
    if readiness == "needs_evidence":
        st.warning("This extraction has a non-zero proposed effect but no valid exact evidence.")
    elif readiness == "insufficient_document":
        st.warning("This extraction is based on insufficient or irrelevant source material.")

    _show_list_block("Extracted positive points", row.get("key_positive_points"), "No positive points extracted.")
    _show_list_block("Extracted risks", row.get("key_risks"), "No risks extracted.")
    _show_list_block("Evidence snippets", row.get("evidence_snippets"), "No evidence snippets extracted.")

    if row.get("extraction_warnings"):
        st.warning(str(row.get("extraction_warnings")))
    st.caption(
        f"Provider: {row.get('provider', 'unknown')} | Model: {row.get('model_name') or 'none'} | "
        f"Prompt version: {row.get('prompt_version') or 'n/a'}"
    )
    if pd.notna(row.get("catalyst_id")):
        st.write(f"Linked catalyst ID: {int(row['catalyst_id'])}")
    else:
        st.write("Linked catalyst ID: none")

    st.write("**Source document**")
    document_title = row.get("document_title")
    has_document = document_title is not None and not pd.isna(document_title) and bool(str(document_title).strip())
    if has_document:
        st.write(
            f"{document_title} | {row.get('document_type') or 'unknown type'} | "
            f"{row.get('document_source') or 'unknown source'} | {row.get('document_published_at') or 'no date'}"
        )
        if row.get("document_warnings"):
            st.info(f"Document warnings: {row.get('document_warnings')}")
        with st.expander("Cleaned Source Text Preview"):
            st.text(preview_text(str(row.get("document_cleaned_text") or ""), limit=3_000) or "No cleaned text available.")
    else:
        st.warning("Source document metadata is unavailable. It may have been deleted.")

    if not allow_review_actions and row.get("review_status") == "approved":
        _show_approved_extraction_proposal_controls(settings, row, key_prefix=key_prefix)

    if allow_review_actions and row.get("review_status") == "pending_review":
        reviewer_note = st.text_area("Reviewer note", key=f"{key_prefix}_note_{extraction_id}")
        override_not_ready = False
        if readiness != "ready_for_review":
            override_not_ready = st.checkbox(
                f"Override {readiness} and allow approval",
                value=False,
                key=f"{key_prefix}_override_{extraction_id}",
            )
            st.caption("Approval override requires this checkbox and a non-empty reviewer note. Scoring remains unchanged.")
        a1, a2, a3 = st.columns(3)
        if a1.button("Approve", key=f"{key_prefix}_approve_{extraction_id}"):
            result = approve_extraction_with_readiness(
                settings.database_file,
                extraction_id,
                reviewer_note=reviewer_note,
                override_not_ready=override_not_ready,
            )
            if result.changed:
                st.session_state["llm_review_flash"] = result.message
                st.rerun()
            else:
                st.warning(result.message)
        if a2.button("Reject", key=f"{key_prefix}_reject_{extraction_id}"):
            if reject_extraction(settings.database_file, extraction_id, reviewer_note):
                st.session_state["llm_review_flash"] = f"Extraction #{extraction_id} rejected."
                st.rerun()
            else:
                st.warning("This extraction is no longer pending and was not changed.")
        if a3.button("Supersede", key=f"{key_prefix}_supersede_{extraction_id}"):
            if supersede_extraction(settings.database_file, extraction_id, reviewer_note):
                st.session_state["llm_review_flash"] = f"Extraction #{extraction_id} marked superseded."
                st.rerun()
            else:
                st.warning("This extraction is no longer pending and was not changed.")
    elif allow_review_actions:
        st.info("Reviewed records are read-only in this phase.")


def _date_input_value(value: Any) -> date:
    try:
        if value is None or pd.isna(value):
            return datetime.now(UTC).date()
        return pd.to_datetime(value).date()
    except Exception:
        return datetime.now(UTC).date()


def _show_proposal_details(settings: Settings, row: pd.Series, allow_edit: bool, key_prefix: str) -> None:
    proposal_id = int(row["proposal_id"])
    st.warning("This proposal is not an active catalyst and does not affect scanner scoring.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", row.get("proposal_status", "draft"))
    c2.metric("Type", row.get("proposal_type", "create_new"))
    c3.metric("Strength", int(row.get("proposed_strength", 0) or 0))
    c4.metric("Confidence", f"{float(row.get('proposed_confidence', 0) or 0):.0%}")
    st.write(f"**Proposed title:** {row.get('proposed_title')}")
    st.write(f"**Proposed summary:** {row.get('proposed_summary') or 'No proposed summary.'}")
    st.write(
        f"Event: {row.get('proposed_event_type')} | Date: {row.get('proposed_event_date') or 'n/a'} | "
        f"Sentiment: {row.get('proposed_sentiment')}"
    )
    st.write(
        f"Extraction #{int(row.get('extraction_id') or 0)} | Document #{int(row.get('document_id') or 0)} | "
        f"Target catalyst: {row.get('target_catalyst_id') if pd.notna(row.get('target_catalyst_id')) else 'none'}"
    )
    _show_list_block("Exact evidence", row.get("evidence_snippets"), "No exact evidence stored with this proposal.")
    st.write(
        f"Risk severity: {int(row.get('risk_severity', 0) or 0)} | "
        f"Document relevance: {row.get('document_relevance', 'unknown')} | "
        f"Evidence sufficiency: {row.get('evidence_sufficiency', 'unknown')}"
    )
    if row.get("reviewer_note"):
        st.info(f"Reviewer note: {row.get('reviewer_note')}")
    source_url = "" if row.get("proposed_source_url") is None or pd.isna(row.get("proposed_source_url")) else str(row.get("proposed_source_url")).strip()
    if source_url:
        st.link_button("Open Proposal Source", source_url, key=f"{key_prefix}_proposal_source_{proposal_id}")

    publications = list_publications_by_proposal_id(settings.database_file, proposal_id, limit=25)
    st.write(f"Publication history: {len(publications)}")
    if not publications.empty:
        st.dataframe(publication_display_frame(publications), width="stretch", hide_index=True)
        for _, publication in publications.iterrows():
            if publication.get("publication_status") == "published":
                pub_id = int(publication["publication_id"])
                with st.expander(f"Revert publication #{pub_id}"):
                    st.warning("Reversal is blocked if the active catalyst changed after publication.")
                    revert_note = st.text_area("Required revert reason", key=f"{key_prefix}_revert_note_{pub_id}")
                    if st.button("Revert Publication", key=f"{key_prefix}_revert_publication_{pub_id}"):
                        result = revert_publication(settings.database_file, pub_id, revert_note)
                        if result.changed:
                            st.session_state["llm_review_flash"] = result.message
                            st.rerun()
                        else:
                            st.warning(result.message)
                            for warning in result.warnings:
                                st.warning(warning)

    if not allow_edit:
        return

    with st.form(f"{key_prefix}_edit_proposal_{proposal_id}"):
        e1, e2, e3 = st.columns(3)
        event_type = e1.selectbox(
            "Proposed event type",
            PROPOSAL_EVENT_TYPES,
            index=PROPOSAL_EVENT_TYPES.index(row.get("proposed_event_type")) if row.get("proposed_event_type") in PROPOSAL_EVENT_TYPES else PROPOSAL_EVENT_TYPES.index("unknown"),
            key=f"{key_prefix}_event_type_{proposal_id}",
        )
        event_date = e2.date_input(
            "Proposed event date",
            value=_date_input_value(row.get("proposed_event_date")),
            key=f"{key_prefix}_event_date_{proposal_id}",
        )
        sentiment = e3.selectbox(
            "Proposed sentiment",
            PROPOSAL_SENTIMENTS,
            index=PROPOSAL_SENTIMENTS.index(row.get("proposed_sentiment")) if row.get("proposed_sentiment") in PROPOSAL_SENTIMENTS else PROPOSAL_SENTIMENTS.index("unknown"),
            key=f"{key_prefix}_sentiment_{proposal_id}",
        )
        title = st.text_input("Proposed title", value=str(row.get("proposed_title") or ""), key=f"{key_prefix}_title_{proposal_id}")
        summary = st.text_area("Proposed summary", value=str(row.get("proposed_summary") or ""), key=f"{key_prefix}_summary_{proposal_id}")
        s1, s2, s3 = st.columns(3)
        strength = s1.slider("Proposed strength", 0, 10, int(row.get("proposed_strength", 0) or 0), key=f"{key_prefix}_strength_{proposal_id}")
        confidence = s2.slider(
            "Proposed confidence",
            0.0,
            1.0,
            float(row.get("proposed_confidence", 0) or 0),
            step=0.05,
            key=f"{key_prefix}_confidence_{proposal_id}",
        )
        risk = s3.slider("Risk severity", 0, 10, int(row.get("risk_severity", 0) or 0), key=f"{key_prefix}_risk_{proposal_id}")
        source = st.text_input("Proposed source", value=str(row.get("proposed_source") or ""), key=f"{key_prefix}_source_{proposal_id}")
        source_url_value = "" if row.get("proposed_source_url") is None or pd.isna(row.get("proposed_source_url")) else str(row.get("proposed_source_url"))
        source_url = st.text_input("Proposed source URL", value=source_url_value, key=f"{key_prefix}_source_url_{proposal_id}")
        note = st.text_area("Reviewer note", value=str(row.get("reviewer_note") or ""), key=f"{key_prefix}_proposal_note_{proposal_id}")
        updated = st.form_submit_button("Save Proposal Edits")

    if updated:
        changed = update_proposal(
            settings.database_file,
            proposal_id,
            {
                "proposed_event_type": event_type,
                "proposed_event_date": event_date,
                "proposed_title": title,
                "proposed_summary": summary,
                "proposed_sentiment": sentiment,
                "proposed_strength": strength,
                "proposed_confidence": confidence,
                "proposed_source": source,
                "proposed_source_url": source_url,
                "risk_severity": risk,
                "reviewer_note": note,
            },
        )
        st.session_state["llm_review_flash"] = "Proposal edits saved." if changed else "Proposal was not changed."
        st.rerun()

    status_note = st.text_input("Status transition note", key=f"{key_prefix}_proposal_status_note_{proposal_id}")
    s1, s2, s3 = st.columns(3)
    if s1.button("Mark Reviewed Ready", key=f"{key_prefix}_proposal_ready_{proposal_id}"):
        if set_proposal_status(settings.database_file, proposal_id, "reviewed_ready", status_note):
            st.session_state["llm_review_flash"] = f"Proposal #{proposal_id} marked reviewed_ready. Scoring was not changed."
            st.rerun()
    if s2.button("Reject Proposal", key=f"{key_prefix}_proposal_reject_{proposal_id}"):
        if set_proposal_status(settings.database_file, proposal_id, "rejected", status_note):
            st.session_state["llm_review_flash"] = f"Proposal #{proposal_id} rejected."
            st.rerun()
    if s3.button("Supersede Proposal", key=f"{key_prefix}_proposal_supersede_{proposal_id}"):
        if set_proposal_status(settings.database_file, proposal_id, "superseded", status_note):
            st.session_state["llm_review_flash"] = f"Proposal #{proposal_id} superseded."
            st.rerun()

    st.divider()
    st.subheader("Controlled Active Catalyst Publication")
    st.caption("Nothing publishes automatically. Publication uses the existing catalyst scoring layer only.")
    selected_fields: list[str] | None = None
    if row.get("proposal_type") == "update_existing":
        selected_fields = st.multiselect(
            "Fields to update on target catalyst",
            ALL_UPDATE_FIELDS,
            default=DEFAULT_UPDATE_FIELDS,
            key=f"{key_prefix}_publish_fields_{proposal_id}",
        )
        st.caption("Event date and source URL are not selected by default.")
    preview = build_publication_preview(settings.database_file, proposal_id, selected_update_fields=selected_fields)
    if preview.reasons:
        for reason in preview.reasons:
            st.error(reason)
    if preview.warnings:
        for warning in preview.warnings:
            st.warning(warning)
    if preview.proposal is not None:
        p1, p2, p3 = st.columns(3)
        p1.metric("Catalyst Before", f"{preview.catalyst_component_before:.2f}")
        p2.metric("Catalyst After", f"{preview.catalyst_component_after:.2f}")
        p3.metric("Catalyst-Only Delta", f"{preview.catalyst_component_delta:.2f}")
        st.write("Field-level before/after diff")
        if preview.field_diff.empty:
            st.info("No active catalyst field changes in the current preview.")
        else:
            st.dataframe(preview.field_diff, width="stretch", hide_index=True)
        with st.expander("Publication Provenance Preview"):
            st.write("Source document")
            if preview.document:
                st.json(
                    {
                        "document_id": preview.document.get("document_id"),
                        "ticker": preview.document.get("ticker"),
                        "title": preview.document.get("title"),
                        "source": preview.document.get("source"),
                        "source_url": preview.document.get("source_url"),
                    }
                )
            st.write("Approved extraction")
            if preview.extraction:
                st.json(
                    {
                        "extraction_id": preview.extraction.get("extraction_id"),
                        "provider": preview.extraction.get("provider"),
                        "model_name": preview.extraction.get("model_name"),
                        "review_status": preview.extraction.get("review_status"),
                        "confidence": preview.extraction.get("confidence"),
                    }
                )
            _show_list_block("Exact evidence", (preview.proposal or {}).get("evidence_snippets"), "No exact evidence available.")
            if preview.target_catalyst:
                st.write("Target catalyst")
                st.json({key: preview.target_catalyst.get(key) for key in ["id", "ticker", "event_date", "title", "sentiment_label", "catalyst_strength", "confidence", "source_url"]})
    publish_note = st.text_area("Required publisher note", key=f"{key_prefix}_publisher_note_{proposal_id}")
    confirm_publish = st.checkbox(
        "I confirm this reviewed-ready proposal should publish into active catalysts and affect scoring only through existing catalyst logic.",
        key=f"{key_prefix}_confirm_publish_{proposal_id}",
    )
    publish_disabled = not preview.eligible or not confirm_publish or not publish_note.strip()
    if st.button(
        "Publish Active Catalyst",
        key=f"{key_prefix}_publish_{proposal_id}",
        disabled=publish_disabled,
    ):
        result = publish_proposal(
            settings.database_file,
            proposal_id,
            publisher_note=publish_note,
            selected_update_fields=selected_fields,
        )
        if result.changed:
            st.session_state["llm_review_flash"] = result.message
            st.rerun()
        else:
            st.error(result.message)
            for warning in result.warnings:
                st.warning(warning)


def _show_catalyst_proposals(settings: Settings, proposals: pd.DataFrame, title: str, allow_edit: bool = False, key_prefix: str = "proposals") -> None:
    st.subheader(title)
    st.caption("Review-only catalyst proposals are separate from active catalyst records and have zero scoring effect.")
    if proposals is None or proposals.empty:
        st.info("No catalyst proposals found.")
        return
    st.dataframe(proposal_display_frame(proposals), width="stretch", hide_index=True)
    for _, row in proposals.head(50).iterrows():
        with st.expander(
            f"Proposal #{int(row['proposal_id'])} | {row.get('ticker')} | "
            f"{row.get('proposal_status')} | {row.get('proposed_title')}"
        ):
            _show_proposal_details(settings, row, allow_edit=allow_edit, key_prefix=key_prefix)


def _show_approved_extraction_proposal_controls(settings: Settings, row: pd.Series, key_prefix: str) -> None:
    if row.get("review_status") != "approved":
        return

    extraction_id = int(row["extraction_id"])
    ticker = str(row.get("ticker") or "").upper()
    extraction = row.to_dict()
    document = get_document_by_id(settings.database_file, int(row.get("document_id") or 0))
    readiness = classify_review_readiness(extraction)

    st.divider()
    st.subheader("Review-Only Catalyst Proposal")
    st.warning("This workflow creates a non-scoring proposal or audit link only. It does not create/update active catalysts.")
    if readiness != "ready_for_review":
        st.warning(f"Extraction readiness is {readiness}. Proposal creation requires explicit override and a reviewer note.")
    if document is None:
        st.error("Source document is missing, so a proposal cannot be created.")
        return

    existing_proposals = list_proposals_by_extraction_id(settings.database_file, extraction_id, limit=25)
    if not existing_proposals.empty:
        st.write(f"Existing proposal(s) for this extraction: {len(existing_proposals)}")
        st.dataframe(proposal_display_frame(existing_proposals), width="stretch", hide_index=True)

    preview = map_extraction_to_proposal(extraction, document)
    st.write("**Deterministic proposal preview**")
    st.dataframe(
        pd.DataFrame(
            [
                {
                    "ticker": preview.ticker,
                    "event_type": preview.proposed_event_type,
                    "event_date": preview.proposed_event_date,
                    "title": preview.proposed_title,
                    "sentiment": preview.proposed_sentiment,
                    "strength": preview.proposed_strength,
                    "confidence": preview.proposed_confidence,
                    "source": preview.proposed_source,
                    "source_url": preview.proposed_source_url,
                }
            ]
        ),
        width="stretch",
        hide_index=True,
    )
    _show_list_block("Proposal evidence", preview.evidence_snippets, "No exact evidence is available.")

    same_ticker_catalysts = list_catalysts_by_ticker(settings.database_file, ticker, limit=100)
    catalyst_options, catalyst_mapping = _catalyst_select_options(same_ticker_catalysts)
    target_label = st.selectbox(
        "Optional target catalyst for update_existing proposal",
        catalyst_options,
        key=f"{key_prefix}_proposal_target_{extraction_id}",
    )
    target_catalyst_id = catalyst_mapping.get(target_label)
    if target_catalyst_id is not None:
        target = same_ticker_catalysts[same_ticker_catalysts["id"].astype(int).eq(int(target_catalyst_id))]
        if not target.empty:
            st.write("Target catalyst preview")
            st.dataframe(catalyst_display_frame(target), width="stretch", hide_index=True)

    note = st.text_area(
        "Proposal reviewer note",
        key=f"{key_prefix}_proposal_note_{extraction_id}",
        placeholder="Required for weak-readiness overrides.",
    )
    override = False
    if readiness != "ready_for_review":
        override = st.checkbox(
            f"Override {readiness} for non-scoring proposal creation",
            key=f"{key_prefix}_proposal_override_{extraction_id}",
        )
    if st.button("Create Non-Scoring Catalyst Proposal", key=f"{key_prefix}_create_proposal_{extraction_id}"):
        result = create_proposal_from_extraction(
            settings.database_file,
            extraction_id,
            target_catalyst_id=target_catalyst_id,
            reviewer_note=note,
            override_weak_readiness=override,
        )
        if result.changed:
            st.session_state["llm_review_flash"] = result.message
            st.rerun()
        else:
            st.warning(result.message)

    st.subheader("Existing Catalyst Link Audit")
    st.caption("Links are reversible audit records. They do not mutate the target catalyst.")
    links = list_links_by_extraction_id(settings.database_file, extraction_id, limit=50)
    if not links.empty:
        st.dataframe(link_display_frame(links), width="stretch", hide_index=True)
        for _, link_row in links[links["link_status"].eq("active")].iterrows():
            link_id = int(link_row["link_id"])
            unlink_note = st.text_input("Unlink reviewer note", key=f"{key_prefix}_unlink_note_{link_id}")
            if st.button(f"Unlink catalyst #{int(link_row['catalyst_id'])}", key=f"{key_prefix}_unlink_{link_id}"):
                result = unlink_extraction_catalyst_link(settings.database_file, link_id, unlink_note)
                if result.changed:
                    st.session_state["llm_review_flash"] = result.message
                    st.rerun()
                else:
                    st.warning(result.message)
    else:
        st.info("No catalyst links recorded for this extraction.")

    if not same_ticker_catalysts.empty:
        link_label = st.selectbox(
            "Catalyst to link explicitly",
            catalyst_options[1:] if len(catalyst_options) > 1 else catalyst_options,
            key=f"{key_prefix}_link_target_{extraction_id}",
        )
        link_target_id = catalyst_mapping.get(link_label)
        if link_target_id is not None:
            target = same_ticker_catalysts[same_ticker_catalysts["id"].astype(int).eq(int(link_target_id))]
            st.write("Target catalyst before linking")
            st.dataframe(catalyst_display_frame(target), width="stretch", hide_index=True)
            confirm = st.checkbox(
                "Confirm this explicit audit link without changing active catalyst fields",
                key=f"{key_prefix}_link_confirm_{extraction_id}",
            )
            link_note = st.text_area("Link reviewer note", key=f"{key_prefix}_link_note_{extraction_id}")
            if st.button("Link Extraction to Existing Catalyst", key=f"{key_prefix}_link_{extraction_id}"):
                if not confirm:
                    st.warning("Confirm the audit link before saving.")
                else:
                    result = link_extraction_to_catalyst(
                        settings.database_file,
                        extraction_id,
                        int(link_target_id),
                        reviewer_note=link_note,
                    )
                    if result.changed:
                        st.session_state["llm_review_flash"] = result.message
                        st.rerun()
                    else:
                        st.warning(result.message)


def llm_review_page(settings: Settings) -> None:
    st.header("LLM Review")
    openai_status = openai_provider_status(settings)
    st.warning(
        "Review-first extraction workflow. Fallback test mode remains available. "
        "OpenAI extraction is optional, explicit, and every result stays pending review. "
        "Approval does not affect scanner scoring, catalysts, or alerts."
    )
    if openai_status.enabled:
        st.info(f"OpenAI provider configured with model `{openai_status.model}`. No API call happens unless you press the OpenAI run button.")
    else:
        st.info("OpenAI provider disabled: " + " ".join(openai_status.warnings))
    flash = st.session_state.pop("llm_review_flash", None)
    if flash:
        st.success(flash)

    universe = load_universe(settings.universe_file)
    tickers = universe["ticker"].tolist() if not universe.empty else []
    documents_all = list_recent_documents(settings.database_file, limit=1000)

    st.subheader("Run Extraction")
    provider_choice = st.radio(
        "Provider",
        ["Fallback test mode", "OpenAI"],
        horizontal=True,
        key="llm_provider_choice",
    )
    if provider_choice == "Fallback test mode":
        st.caption("Fallback mode is deterministic keyword logic. Document text remains local.")
    else:
        st.caption("OpenAI mode sends the selected document text only after explicit confirmation and button click.")
    if documents_all.empty:
        st.info("No source documents are stored yet. Add text on the Documents / Text page first.")
    else:
        c1, c2 = st.columns([1, 2])
        ticker_filter = c1.selectbox("Document ticker filter", ["All", *tickers], index=0, key="llm_run_ticker_filter")
        run_documents = documents_all.copy()
        if ticker_filter != "All":
            run_documents = run_documents[run_documents["ticker"].str.upper().eq(ticker_filter.upper())]
        options, mapping = _document_options(run_documents)
        if not options:
            st.info("No documents match the selected ticker filter.")
        else:
            selected_label = c2.selectbox("Stored document", options, key="llm_run_document")
            document_id = mapping[selected_label]
            document = get_document_by_id(settings.database_file, document_id)
            readiness = document_readiness(document)

            if document is None:
                st.error("Selected source document is no longer available.")
            else:
                c3, c4, c5, c6 = st.columns(4)
                c3.metric("Ticker", document.get("ticker", "n/a"))
                c4.metric("Type", document.get("document_type", "n/a"))
                c5.metric("Source", document.get("source", "n/a"))
                c6.metric("Parsing", document.get("parsing_status", "n/a"))
                st.write(f"**Title:** {document.get('title') or 'Untitled'}")
                st.write(f"**Published:** {document.get('published_at') or 'n/a'}")
                st.write(f"**Warnings:** {document.get('warnings') or 'None'}")
                for warning in readiness.warnings:
                    if readiness.can_run:
                        st.warning(warning)
                    else:
                        st.error(warning)
                with st.expander("Source Text Preview", expanded=True):
                    st.text(preview_text(readiness.cleaned_text, limit=3_000) or "No usable text available.")

                pending_existing = pending_extractions_for_document(settings.database_file, document_id)
                supersede_existing = False
                if not pending_existing.empty:
                    st.warning(
                        f"This document already has {len(pending_existing)} pending extraction(s). "
                        "Creating another requires explicit supersede."
                    )
                    supersede_existing = st.checkbox(
                        "Create a new pending extraction and mark existing pending extraction(s) as superseded",
                        value=False,
                        key=f"llm_supersede_existing_{document_id}",
                    )

                extraction_type = st.selectbox(
                    "Extraction type",
                    EXTRACTION_TYPES,
                    index=EXTRACTION_TYPES.index("general_document_review"),
                    key="llm_run_extraction_type",
                )
                if provider_choice == "Fallback test mode":
                    if st.button("Run Fallback Extraction", key=f"run_fallback_extraction_{document_id}"):
                        if not readiness.can_run:
                            st.error("Extraction was blocked because the selected document has no usable source text.")
                        elif not pending_existing.empty and not supersede_existing:
                            st.error("Extraction was blocked. Confirm the supersede option before rerunning this document.")
                        else:
                            try:
                                result = create_fallback_extraction_for_document(
                                    settings.database_file,
                                    document_id,
                                    extraction_type=extraction_type,
                                    supersede_existing=supersede_existing,
                                )
                            except Exception as exc:
                                st.error(f"Could not run fallback extraction: {exc}")
                            else:
                                if result.blocked:
                                    for warning in result.warnings:
                                        st.warning(warning)
                                elif result.extraction_id is not None:
                                    superseded = (
                                        f" Superseded older extraction(s): {', '.join(map(str, result.superseded_ids))}."
                                        if result.superseded_ids
                                        else ""
                                    )
                                    st.session_state["llm_review_flash"] = (
                                        f"Stored fallback extraction #{result.extraction_id} as pending_review.{superseded}"
                                    )
                                    st.rerun()
                else:
                    submitted, original_chars, submitted_chars, truncated, input_warnings = prepare_document_text(
                        document,
                        settings.llm.max_input_chars,
                    )
                    o1, o2, o3 = st.columns(3)
                    o1.metric("Configured model", openai_status.model or "not configured")
                    o2.metric("Document chars", f"{original_chars:,}")
                    o3.metric("Submitted chars", f"{submitted_chars:,}")
                    if truncated:
                        st.warning(
                            f"Input will be truncated to {submitted_chars:,} of {original_chars:,} characters. "
                            "Chunking and aggregation are intentionally not implemented yet."
                        )
                    for warning in input_warnings:
                        st.warning(warning)
                    st.error(
                        "Privacy notice: running OpenAI extraction sends the selected document text outside this machine. "
                        "The API key is never displayed or stored by the app."
                    )
                    confirm_external = st.checkbox(
                        "I understand this will send the selected document text to OpenAI for analysis.",
                        key=f"openai_confirm_external_{document_id}",
                    )
                    openai_blocked = (
                        not openai_status.enabled
                        or not readiness.can_run
                        or not confirm_external
                        or (not pending_existing.empty and not supersede_existing)
                    )
                    if not openai_status.enabled:
                        st.warning("OpenAI extraction is disabled until LLM_PROVIDER=openai, OPENAI_API_KEY, and LLM_MODEL are configured.")
                    run_openai = st.button(
                        "Run OpenAI Extraction",
                        key=f"run_openai_extraction_{document_id}",
                        disabled=openai_blocked,
                    )
                    if run_openai:
                        if not readiness.can_run:
                            st.error("Extraction was blocked because the selected document has no usable source text.")
                        elif not pending_existing.empty and not supersede_existing:
                            st.error("Extraction was blocked. Confirm the supersede option before rerunning this document.")
                        elif not confirm_external:
                            st.error("Confirm the external provider privacy notice before running OpenAI extraction.")
                        else:
                            result = create_openai_extraction_for_document(
                                settings.database_file,
                                document_id,
                                settings=settings,
                                extraction_type=extraction_type,
                                supersede_existing=supersede_existing,
                            )
                            if result.blocked:
                                for warning in result.warnings:
                                    st.error(warning)
                            elif result.extraction_id is not None:
                                superseded = (
                                    f" Superseded older extraction(s): {', '.join(map(str, result.superseded_ids))}."
                                    if result.superseded_ids
                                    else ""
                                )
                                st.session_state["llm_review_flash"] = (
                                    f"Stored OpenAI extraction #{result.extraction_id} as pending_review.{superseded}"
                                )
                                st.rerun()

    st.divider()
    st.subheader("Review Queue / History")

    filter_cols = st.columns(5)
    queue_ticker = filter_cols[0].selectbox("Ticker", ["All", *tickers], index=0, key="llm_queue_ticker")
    queue_statuses = filter_cols[1].multiselect("Status", REVIEW_STATUSES, default=[])
    queue_types = filter_cols[2].multiselect("Extraction type", EXTRACTION_TYPES, default=[])
    queue_providers = filter_cols[3].multiselect("Provider", EXTRACTION_PROVIDERS, default=[])
    doc_options, doc_mapping = _document_options(documents_all)
    doc_label = filter_cols[4].selectbox("Document", ["All", *doc_options], index=0, key="llm_queue_document")
    queue_document_id = None if doc_label == "All" else doc_mapping.get(doc_label)

    pending = enrich_extractions_with_documents(list_pending_review_extractions(settings.database_file, limit=500), documents_all)
    pending = filter_extractions(
        pending,
        ticker=queue_ticker,
        statuses=queue_statuses,
        extraction_types=queue_types,
        providers=queue_providers,
        document_id=queue_document_id,
    )

    st.subheader("Pending Review")
    if pending.empty:
        st.info("No pending extractions match the current filters.")
    else:
        st.dataframe(extraction_queue_display_frame(pending), width="stretch", hide_index=True)
        for _, row in pending.iterrows():
            with st.expander(f"Pending #{int(row['extraction_id'])} | {row.get('ticker')} | {row.get('document_title') or 'missing document'}"):
                _show_extraction_details(settings, row, allow_review_actions=True, key_prefix="pending")

    history_statuses = queue_statuses or ["approved", "rejected", "superseded"]
    history = enrich_extractions_with_documents(list_reviewed_extractions(settings.database_file, limit=500), documents_all)
    history = filter_extractions(
        history,
        ticker=queue_ticker,
        statuses=history_statuses,
        extraction_types=queue_types,
        providers=queue_providers,
        document_id=queue_document_id,
    )

    st.subheader("Review History")
    if history.empty:
        st.info("No reviewed extractions match the current filters.")
    else:
        st.dataframe(extraction_queue_display_frame(history), width="stretch", hide_index=True)
        for _, row in history.iterrows():
            with st.expander(f"Reviewed #{int(row['extraction_id'])} | {row.get('review_status')} | {row.get('ticker')}"):
                _show_extraction_details(settings, row, allow_review_actions=False, key_prefix="history")

    st.divider()
    proposals = list_recent_proposals(settings.database_file, limit=500)
    if not proposals.empty:
        if queue_ticker != "All":
            proposals = proposals[proposals["ticker"].str.upper().eq(queue_ticker.upper())]
        proposal_status_filter = st.multiselect("Proposal status", PROPOSAL_STATUSES, default=[], key="llm_review_proposal_status")
        if proposal_status_filter:
            proposals = proposals[proposals["proposal_status"].isin(proposal_status_filter)]
    _show_catalyst_proposals(
        settings,
        proposals,
        "Catalyst Proposals / Link History",
        allow_edit=True,
        key_prefix="llm_review",
    )


def _run_scanner(
    settings: Settings,
    tickers: list[str],
    company_names: dict[str, str],
    lookback_period: str,
    min_price: float,
    min_avg_dollar_volume: float,
    refresh: bool,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, Any]]:
    needed = sorted(set([ticker.upper() for ticker in tickers] + ["SPY", "QQQ", "IWM", "^VIX"]))
    histories: dict[str, pd.DataFrame] = {}
    progress = st.progress(0, text="Downloading or loading cached market data...")
    for idx, ticker in enumerate(needed, start=1):
        try:
            histories[ticker] = market_data.get_history(
                ticker,
                settings.database_file,
                settings.market_data_provider,
                lookback_period,
                refresh=refresh,
            )
        except Exception as exc:
            histories[ticker] = pd.DataFrame()
            st.warning(f"{ticker}: {exc}")
        progress.progress(idx / len(needed), text=f"Loaded {idx}/{len(needed)} tickers")
    progress.empty()

    regime = classify_market_regime(histories)
    spy_df = histories.get("SPY", pd.DataFrame())
    catalyst_events = list_recent_catalysts(settings.database_file, days=120, limit=1000)

    rows: list[dict[str, Any]] = []
    for ticker in tickers:
        result = score_ticker(
            ticker,
            histories.get(ticker, pd.DataFrame()),
            spy_df,
            regime,
            catalyst_events,
            min_price=min_price,
            min_avg_dollar_volume=min_avg_dollar_volume,
        )
        flat = flatten_score_result(result, company_names.get(ticker, ""))
        rows.append(flat)

    rows = sorted(rows, key=lambda item: item.get("alpha_score") or 0, reverse=True)
    if rows:
        storage.save_scan_results(settings.database_file, rows)
    scan_df = pd.DataFrame(rows)
    return scan_df, histories, regime


def scanner_page(settings: Settings) -> None:
    st.header("Daily Opportunity Scanner")
    st.caption("Every score is rule-based and explainable. This is research only, not a trading signal.")

    universe = load_universe(settings.universe_file)
    if universe.empty:
        st.error("Universe is empty. Edit config/universe.csv and add at least one ticker.")
        return

    if st.button("Clear OHLCV cache", key="clear_cache"):
        storage.clear_ohlcv_cache(settings.database_file)
        st.success("OHLCV cache cleared.")

    all_tickers = universe["ticker"].tolist()
    default_count = min(settings.scanner.max_tickers, len(all_tickers))
    with st.form("scanner_controls"):
        selected = st.multiselect("Universe tickers", all_tickers, default=all_tickers[:default_count])
        c1, c2, c3, c4 = st.columns(4)
        lookback_period = c1.selectbox("Lookback", ["1y", "2y", "5y"], index=1)
        max_tickers = c2.number_input("Max tickers", min_value=1, max_value=max(1, len(all_tickers)), value=default_count)
        min_price = c3.number_input("Min price", min_value=0.0, value=float(settings.scanner.min_price), step=1.0)
        min_adv = c4.number_input(
            "Min avg dollar volume",
            min_value=0.0,
            value=float(settings.scanner.min_avg_dollar_volume),
            step=1_000_000.0,
        )
        refresh = st.checkbox("Force refresh market data", value=False)
        run = st.form_submit_button("Run Scanner")

    if not run and "last_scan_df" not in st.session_state:
        st.info("Choose controls and run the scanner. Default universe comes from config/universe.csv.")
        return

    if run:
        tickers = [ticker.upper() for ticker in selected[: int(max_tickers)]]
        if not tickers:
            st.error("Select at least one ticker.")
            return
        company_names = dict(zip(universe["ticker"], universe["name"], strict=False))
        with st.spinner("Scoring universe..."):
            scan_df, histories, regime = _run_scanner(
                settings,
                tickers,
                company_names,
                lookback_period,
                min_price,
                min_adv,
                refresh,
            )
        st.session_state["last_scan_df"] = scan_df
        st.session_state["last_scan_histories"] = histories
        st.session_state["last_scan_regime"] = regime
    else:
        scan_df = st.session_state["last_scan_df"]
        histories = st.session_state.get("last_scan_histories", {})
        regime = st.session_state.get("last_scan_regime", {"regime": "Unknown"})

    if scan_df.empty:
        st.warning("No scan results were produced. Check your internet connection or ticker universe.")
        return

    st.subheader(f"Ranked Watchlist - Regime: {regime.get('regime', 'Unknown')}")
    display_cols = [
        "ticker",
        "company_name",
        "last_price",
        "20d_return",
        "60d_return",
        "above_50d_ma",
        "above_200d_ma",
        "relative_strength_vs_spy",
        "volume_ratio_vs_20d",
        "avg_daily_dollar_volume",
        "catalyst_score",
        "alpha_score",
        "risk_label",
        "reasons",
    ]
    display = scan_df[display_cols].copy()
    st.dataframe(
        display,
        width="stretch",
        hide_index=True,
        column_config={
            "last_price": st.column_config.NumberColumn("Last Price", format="$%.2f"),
            "20d_return": st.column_config.NumberColumn("20D Return", format="%.2f"),
            "60d_return": st.column_config.NumberColumn("60D Return", format="%.2f"),
            "relative_strength_vs_spy": st.column_config.NumberColumn("RS vs SPY", format="%.2f"),
            "volume_ratio_vs_20d": st.column_config.NumberColumn("Vol Ratio", format="%.2f"),
            "avg_daily_dollar_volume": st.column_config.NumberColumn("Avg Dollar Vol", format="$%.0f"),
            "catalyst_score": st.column_config.NumberColumn("Catalyst", format="%.1f"),
            "alpha_score": st.column_config.ProgressColumn("Alpha Score", min_value=0, max_value=100, format="%.1f"),
        },
    )

    st.subheader("Expandable Details")
    for idx, row in scan_df.head(25).iterrows():
        result = row["full_result"]
        ticker = row["ticker"]
        action = suggested_watch_action(result)
        with st.expander(f"{ticker} | {row['alpha_score']}/100 | {row['risk_label']} | {action}"):
            c1, c2 = st.columns([2, 1])
            with c1:
                st.plotly_chart(
                    price_volume_chart(histories.get(ticker, pd.DataFrame()), f"{ticker} Price"),
                    width="stretch",
                    key=f"scanner_price_{ticker}_{idx}",
                )
            with c2:
                st.plotly_chart(
                    score_breakdown_chart(result["breakdown"]),
                    width="stretch",
                    key=f"scanner_score_breakdown_{ticker}_{idx}",
                )
                st.write("Suggested watch action:", action)
                catalyst_features = result.get("catalyst_features", {})
                st.write(f"Catalyst contribution: {result['breakdown'].get('catalyst', 0):.1f}/10")
                if catalyst_features.get("recent_catalysts"):
                    st.write("Recent Catalysts")
                    for event in catalyst_features["recent_catalysts"][:5]:
                        st.write(
                            f"- {event.get('event_date')} {event.get('event_type')}: {event.get('title')} "
                            f"({event.get('sentiment_label')}, confidence {float(event.get('confidence') or 0):.0%})"
                        )
                else:
                    st.write("No catalyst events found for this ticker.")
                ticker_documents = list_documents_by_ticker(settings.database_file, ticker, limit=25)
                st.write(f"Stored source documents: {len(ticker_documents)}")
                if catalyst_features.get("recent_catalysts") and ticker_documents.empty:
                    sec_recent = [
                        event for event in catalyst_features["recent_catalysts"] if event.get("event_type") == "sec_filing"
                    ]
                    if sec_recent:
                        st.info("Recent SEC catalyst metadata exists, but no linked source text is stored yet.")
                if result["penalties"]:
                    st.write("Penalties")
                    for penalty in result["penalties"]:
                        st.write(f"- {penalty['amount']}: {penalty['reason']}")
                st.write("Reasons")
                for reason in result["reasons"][:8]:
                    st.write(f"- {reason}")


def ticker_research_page(settings: Settings) -> None:
    st.header("Ticker Research")
    st.caption("Rule-based summary if no LLM key is configured.")
    ticker = st.text_input("Ticker", value="NVDA").strip().upper()
    refresh = st.checkbox("Force refresh ticker data", value=False)
    if not ticker:
        return

    try:
        ticker_df = market_data.get_history(ticker, settings.database_file, settings.market_data_provider, settings.default_history_period, refresh)
        spy_df = market_data.get_history("SPY", settings.database_file, settings.market_data_provider, settings.default_history_period, refresh)
    except Exception as exc:
        st.error(f"Could not load {ticker}: {exc}")
        return
    if ticker_df.empty:
        st.error(f"No market data found for {ticker}.")
        return

    regime = classify_market_regime(_load_regime_histories(settings, refresh=False))
    catalyst_events = list_catalysts_by_ticker(settings.database_file, ticker, limit=100)
    documents = list_documents_by_ticker(settings.database_file, ticker, limit=100)
    proposals = list_proposals_by_ticker(settings.database_file, ticker, limit=100)
    extraction_links = list_links_by_ticker(settings.database_file, ticker, limit=100)
    publications = list_publications_by_ticker(settings.database_file, ticker, limit=100)
    approved_extractions = list_recent_extractions(settings.database_file, limit=500)
    if not approved_extractions.empty:
        approved_extractions = approved_extractions[
            approved_extractions["ticker"].str.upper().eq(ticker)
            & approved_extractions["review_status"].eq("approved")
        ]
    result = score_ticker(ticker, ticker_df, spy_df, regime, catalyst_events)
    features = result["features"]
    catalyst_features = result.get("catalyst_features", {})
    company_info = market_data.get_company_info(ticker, settings.market_data_provider)
    news = get_news_placeholder(ticker)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Score", f"{result['score']}/100")
    c2.metric("Label", result["label"])
    c3.metric("20D Return", fmt_pct(features.get("ret_20d")))
    c4.metric("Avg Dollar Vol", fmt_money(features.get("avg_dollar_volume_20d")))

    st.plotly_chart(
        price_volume_chart(ticker_df, f"{ticker} Price, Moving Averages, and Volume"),
        width="stretch",
        key=f"research_price_{ticker}",
    )

    earnings_history = list_earnings_by_ticker(settings.database_file, ticker, limit=50)

    tabs = st.tabs(["Feature Breakdown", "Summary", "Catalysts", "Documents", "Earnings", "LLM Proposals", "Risk Helper"])
    with tabs[0]:
        st.json(
            {
                "company_info": company_info,
                "momentum_liquidity_features": features,
                "score_breakdown": result["breakdown"],
                "penalties": result["penalties"],
                "catalyst_features": catalyst_features,
                "historical_earnings_events_count": int(len(earnings_history)),
                "source_documents_count": int(len(documents)),
                "news_placeholder": news,
            }
        )
    with tabs[1]:
        st.write(summarize_ticker_features(ticker, features, result))
        sec_events = catalyst_events[catalyst_events["event_type"] == "sec_filing"] if not catalyst_events.empty else pd.DataFrame()
        summary_bits = [
            f"{ticker} has {int(catalyst_features.get('catalyst_events_count', 0))} recent catalyst event(s).",
            f"Catalyst contribution: {result['breakdown'].get('catalyst', 0):.1f}/10.",
        ]
        if catalyst_features.get("catalyst_penalty", 0) < 0:
            summary_bits.append(f"Catalyst penalty: {catalyst_features['catalyst_penalty']:.1f}.")
        if not earnings_history.empty:
            latest_earnings = earnings_history.sort_values("available_at", ascending=False).iloc[0]
            summary_bits.append(
                f"Stored historical earnings events: {len(earnings_history)}; latest available at {latest_earnings.get('available_at')}."
            )
        else:
            summary_bits.append("No historical earnings events stored for this ticker yet.")
        if not sec_events.empty:
            summary_bits.append(f"Recent SEC filings stored: {len(sec_events)}; review flagged filing metadata manually.")
        else:
            summary_bits.append("No recent SEC filing metadata stored for this ticker.")
        if not documents.empty:
            counts = documents["document_type"].value_counts().to_dict()
            count_text = ", ".join(f"{count} {doc_type}" for doc_type, count in counts.items())
            summary_bits.append(
                f"{ticker} has {len(documents)} stored source document(s): {count_text}. These are available for future extraction."
            )
        else:
            summary_bits.append("No local source documents are stored for this ticker yet.")
        if not approved_extractions.empty or not proposals.empty:
            summary_bits.append(
                f"Approved extraction summaries: {len(approved_extractions)}; non-scoring catalyst proposals: {len(proposals)}."
            )
        st.info(" ".join(summary_bits))
    with tabs[2]:
        _show_catalyst_events(catalyst_events, f"{ticker} Catalyst Events", settings=settings)
        with st.form("ticker_manual_catalyst_form"):
            event_date = st.date_input("Event date", value=datetime.now(UTC).date())
            event_type = st.selectbox("Event type", EVENT_TYPES, index=EVENT_TYPES.index("manual_note"), key="research_catalyst_type")
            title = st.text_input("Title", value="", key="research_catalyst_title")
            note = st.text_area("Summary / thesis", value="")
            c1, c2, c3 = st.columns(3)
            sentiment = c1.selectbox("Sentiment", SENTIMENT_LABELS, index=SENTIMENT_LABELS.index("unknown"), key="research_catalyst_sentiment")
            strength = c2.slider("Catalyst strength", min_value=0, max_value=10, value=0, step=1, key="research_catalyst_strength")
            confidence = c3.slider("Confidence", min_value=0.0, max_value=1.0, value=0.5, step=0.05, key="research_catalyst_confidence")
            source_url = st.text_input("Source URL optional", value="", key="research_catalyst_source")
            submitted = st.form_submit_button("Save Manual Catalyst")
        if submitted:
            try:
                insert_catalyst(
                    settings.database_file,
                    _manual_catalyst_event(
                        ticker,
                        event_date,
                        event_type,
                        title,
                        note,
                        sentiment,
                        strength,
                        confidence,
                        source_url,
                    ),
                )
                st.success("Manual catalyst saved.")
            except Exception as exc:
                st.error(f"Could not save catalyst: {exc}")
    with tabs[3]:
        _show_documents(documents, f"{ticker} Source Documents", settings=settings, allow_link_controls=True)
    with tabs[4]:
        st.warning("Historical earnings events are informational and dataset-facing only; they do not affect scanner scoring.")
        if earnings_history.empty:
            st.info("No historical earnings events stored for this ticker.")
        else:
            display_cols = [
                "available_at",
                "announced_at",
                "fiscal_period_end",
                "timing",
                "eps_estimate",
                "eps_actual",
                "eps_surprise_percent",
                "revenue_estimate",
                "revenue_actual",
                "provider",
                "data_quality_status",
                "warnings",
            ]
            st.dataframe(
                earnings_history[[col for col in display_cols if col in earnings_history.columns]],
                width="stretch",
                hide_index=True,
            )
    with tabs[5]:
        st.warning("Approved extractions and catalyst proposals shown here do not affect scanner scoring.")
        st.metric("LLM proposal score contribution", proposal_score_contribution())
        if approved_extractions.empty:
            st.info("No approved extractions for this ticker.")
        else:
            st.subheader("Approved Extraction Summaries")
            st.dataframe(extraction_queue_display_frame(approved_extractions), width="stretch", hide_index=True)
            for _, row in approved_extractions.head(20).iterrows():
                with st.expander(f"Approved #{int(row['extraction_id'])} | {row.get('event_type_detected')} | {row.get('short_summary')}"):
                    _show_list_block("Evidence", row.get("evidence_snippets"), "No exact evidence stored.")
                    st.write(row.get("detailed_summary") or row.get("short_summary") or "No summary.")
                    st.caption(
                        f"Review status: {row.get('review_status')} | Relevance: {row.get('document_relevance')} | "
                        f"Evidence sufficiency: {row.get('evidence_sufficiency')}"
                    )
        _show_catalyst_proposals(settings, proposals, f"{ticker} Non-Scoring Catalyst Proposals", allow_edit=False, key_prefix=f"research_{ticker}")
        if extraction_links.empty:
            st.info("No extraction-to-catalyst audit links for this ticker.")
        else:
            st.subheader("Extraction-Catalyst Audit Links")
            st.dataframe(link_display_frame(extraction_links), width="stretch", hide_index=True)
        if publications.empty:
            st.info("No publication/reversal audit rows for this ticker.")
        else:
            st.subheader("Publication / Reversal Audit")
            st.dataframe(publication_display_frame(publications), width="stretch", hide_index=True)
    with tabs[6]:
        stops = stop_candidates(features)
        portfolio_size = st.number_input("Portfolio size", min_value=1_000.0, value=float(settings.risk.default_portfolio_size), step=5_000.0)
        risk_pct = st.number_input("Risk per trade %", min_value=0.1, max_value=5.0, value=float(settings.risk.default_risk_per_trade_pct), step=0.1)
        entry = st.number_input("Entry price", min_value=0.01, value=float(features.get("last_price") or 1.0), step=0.5)
        default_stop = stops.get("below_20d_ma") or stops.get("fixed_7_pct") or entry * 0.93
        stop = st.number_input("Stop loss price", min_value=0.01, value=float(default_stop), step=0.5)
        if st.button("Calculate Position Size"):
            try:
                sizing = calculate_position_size(portfolio_size, risk_pct, entry, stop, settings.risk.max_position_pct)
                st.json(sizing)
                st.write("Stop candidates")
                st.json(stops)
            except ValueError as exc:
                st.error(str(exc))


def _render_dataset_evaluation_regime(db_path: Path, dataset_id: int) -> None:
    try:
        regime = get_dataset_evaluation_regime(db_path, int(dataset_id))
    except Exception as exc:
        st.warning(f"Could not load dataset evaluation regime metadata: {exc}")
        return
    if not regime:
        st.info("Evaluation regime: unclassified. Treat this dataset as exploratory until a regime is explicitly assigned.")
        return

    label = str(regime.get("evaluation_regime") or "unclassified")
    strategy = str(regime.get("strategy") or "n/a")
    rationale = str(regime.get("rationale") or "")
    parent = regime.get("parent_dataset_id")
    message = f"Evaluation regime: `{label}` | strategy: `{strategy}`"
    if parent is not None:
        message += f" | parent dataset: `{parent}`"

    if label == "final_holdout":
        st.error(f"{message}. Do not repeatedly evaluate or tune against this dataset.")
    elif label == "holdout_candidate":
        st.warning(f"{message}. Use for protocol validation only until promoted under the holdout workflow.")
    elif label == "exploratory_dev":
        st.warning(f"{message}. Results on this dataset are exploratory/dev evidence, not final robustness proof.")
    else:
        st.info(message)
    if rationale:
        st.caption(rationale)

    try:
        maturity = assess_holdout_maturity(db_path, int(dataset_id))
        extension = build_holdout_extension_plan(db_path, int(dataset_id))
    except Exception as exc:
        st.caption(f"Holdout maturity status unavailable: {exc}")
        return

    labels = maturity.get("label_coverage", {})
    five = labels.get("5_session", {})
    twenty = labels.get("20_session", {})
    readiness = maturity.get("readiness", {})
    sanity_ready = bool(readiness.get("holdout_candidate_sanity_check", {}).get("ready"))
    final_ready = bool(readiness.get("final_holdout_evaluation_5_session", {}).get("ready"))
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("5S labels", int(five.get("label_count", 0) or 0))
    m2.metric("5S coverage", f"{float(five.get('label_coverage_pct', 0) or 0) * 100:.1f}%")
    m3.metric("20S labels", int(twenty.get("label_count", 0) or 0))
    m4.metric("Extension days", int(extension.get("extension_trading_day_count", 0) or 0))
    if label == "holdout_candidate":
        if final_ready:
            st.success("Holdout candidate passes 5-session final maturity gates, but promotion still requires explicit confirmation.")
        elif sanity_ready:
            st.info("Holdout candidate is mature enough for sanity checks, but not final-holdout evaluation.")
        else:
            blockers = readiness.get("holdout_candidate_sanity_check", {}).get("blockers", [])
            st.warning("Holdout candidate is immature; model evaluation remains blocked.")
            if blockers:
                st.caption("Primary blocker: " + str(blockers[0]))


def dataset_lab_page(settings: Settings) -> None:
    st.header("Dataset Lab")
    st.caption(
        "Point-in-time research dataset foundation. Builds use cached OHLCV and reviewed active catalysts only; no model training is performed."
    )
    st.warning(
        "Leakage controls: features are sliced through each snapshot date, LLM-supported catalysts are only available after publication time, "
        "and outcome labels are kept out of feature columns."
    )

    universe = load_universe(settings.universe_file)
    if universe.empty:
        st.error("Universe is empty. Edit config/universe.csv and add at least one ticker.")
        return

    all_tickers = universe["ticker"].tolist()
    default_tickers = [ticker for ticker in ["SPY", "QQQ", "AAPL", "NVDA", "AMD"] if ticker in all_tickers]
    if not default_tickers:
        default_tickers = all_tickers[: min(5, len(all_tickers))]
    corporate_universe = universe[
        ~universe["sector"].astype(str).str.contains("ETF|Benchmark", case=False, na=False)
    ]
    default_sec_tickers = [
        ticker for ticker in ["AAPL", "JPM", "AMZN", "META", "UBER"] if ticker in corporate_universe["ticker"].tolist()
    ]
    if len(default_sec_tickers) < 5:
        default_sec_tickers = corporate_universe["ticker"].head(5).tolist()
    default_earnings_tickers = [
        ticker for ticker in ["AAPL", "JPM", "AMZN", "NVDA", "TSLA"] if ticker in corporate_universe["ticker"].tolist()
    ]
    if len(default_earnings_tickers) < 5:
        default_earnings_tickers = corporate_universe["ticker"].head(5).tolist()

    coverage = _cached_coverage(settings.database_file, sorted(set(all_tickers + ["SPY", "QQQ", "IWM", "^VIX"])))
    valid_coverage = coverage.dropna(subset=["start", "end"])
    latest_cached = pd.to_datetime(valid_coverage["end"]).max().date() if not valid_coverage.empty else datetime.now(UTC).date()
    earliest_cached = pd.to_datetime(valid_coverage["start"]).min().date() if not valid_coverage.empty else latest_cached - timedelta(days=365)
    default_end = max(earliest_cached, latest_cached - timedelta(days=35))
    default_start = max(earliest_cached, default_end - timedelta(days=180))

    with st.expander("Build Point-In-Time Dataset", expanded=True):
        selected_tickers = st.multiselect("Tickers", all_tickers, default=default_tickers)
        c1, c2, c3 = st.columns(3)
        start_date = c1.date_input("Start date", value=default_start, min_value=earliest_cached, max_value=latest_cached)
        end_date = c2.date_input("End date", value=default_end, min_value=earliest_cached, max_value=latest_cached)
        version = c3.text_input("Dataset version", value="pit_research_v1")
        st.caption(
            "Label timing: signal after T close, enter next cached session close, exit N sessions after entry. "
            "The signal-date-to-entry return is not included."
        )
        st.write("Configured horizons:", ", ".join(f"{horizon} session(s)" for horizon in DEFAULT_HORIZONS))
        build = st.button("Build Dataset From Cache", type="primary")

    if start_date > end_date:
        st.error("Start date must be on or before end date.")
        build = False

    if build:
        with st.spinner("Building leakage-resistant point-in-time dataset from local cache..."):
            result = build_point_in_time_dataset(
                settings.database_file,
                selected_tickers,
                start_date,
                end_date,
                output_dir=settings.database_file.parent / "processed",
                version=version.strip() or "pit_research_v1",
            )
        st.session_state["dataset_lab_latest_dataset_id"] = result.dataset_id
        st.session_state["dataset_lab_latest_frame"] = result.dataset_frame.head(500)
        st.success(f"Dataset build #{result.dataset_id} created with {len(result.dataset_frame)} rows.")
        if result.export_path:
            st.info(f"Exported CSV: {result.export_path}")
        for warning in result.warnings[:20]:
            st.warning(warning)
        if len(result.warnings) > 20:
            st.warning(f"{len(result.warnings) - 20} additional warnings omitted from the page; see build metadata.")

    with st.expander("Resumable Historical Backfill", expanded=False):
        st.caption(
            "Backfill runs are tracked per ticker. Existing cache is used first; data is fetched only when the requested range is not covered."
        )
        b1, b2, b3 = st.columns(3)
        if b1.button("Start Backfill Run"):
            run_id = create_backfill_run(
                settings.database_file,
                selected_tickers,
                start_date,
                end_date,
                version=version.strip() or "pit_research_v1",
                provider_name=settings.market_data_provider,
            )
            st.session_state["dataset_lab_backfill_run_id"] = run_id
            st.success(f"Backfill run #{run_id} created.")
        runs = list_backfill_runs(settings.database_file, limit=20)
        selected_run_id = None
        if not runs.empty:
            selected_run_id = b2.selectbox(
                "Run",
                runs["run_id"].astype(int).tolist(),
                index=0,
                format_func=lambda value: f"Run #{value}",
            )
            st.dataframe(
                runs[
                    [
                        "run_id",
                        "dataset_id",
                        "status",
                        "requested_start_date",
                        "requested_end_date",
                        "total_tickers",
                        "completed_tickers",
                        "failed_tickers",
                        "generated_rows",
                        "provider",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        max_tickers = b3.number_input("Tickers per resume step", min_value=1, max_value=50, value=10, step=1)
        if selected_run_id is not None:
            c1, c2 = st.columns(2)
            if c1.button("Resume Backfill"):
                with st.spinner(f"Processing backfill run #{selected_run_id}..."):
                    result = process_backfill_run(
                        settings.database_file,
                        int(selected_run_id),
                        provider_name=settings.market_data_provider,
                        output_dir=settings.database_file.parent / "processed",
                        max_tickers=int(max_tickers),
                    )
                st.session_state["dataset_lab_latest_dataset_id"] = result.dataset_id
                st.success(
                    f"Processed {result.processed_tickers} ticker(s). Completed: {result.completed_tickers}; failed: {result.failed_tickers}."
                )
                if result.export_path:
                    st.info(f"Exported CSV: {result.export_path}")
            if c2.button("Retry Failed Tickers"):
                reset = retry_failed_items(settings.database_file, int(selected_run_id))
                st.info(f"Reset {reset} failed ticker(s) to pending.")
            items = list_backfill_items(settings.database_file, int(selected_run_id))
            if not items.empty:
                st.write("Backfill items")
                st.dataframe(
                    items[
                        [
                            "ticker",
                            "status",
                            "rows_generated",
                            "first_date",
                            "last_date",
                            "expected_snapshots",
                            "generated_snapshots",
                            "completed_labels_1_session",
                            "completed_labels_5_session",
                            "completed_labels_20_session",
                            "warning",
                            "error",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    with st.expander("Historical SEC Filing Backfill", expanded=False):
        st.caption(
            "Fetches SEC filing metadata only. Filings are stored as neutral / needs-review catalyst events with EDGAR acceptance time as availability time. "
            "This does not change catalyst scoring rules and does not fetch full filing text."
        )
        sec_provider_status = SecFilingsProvider(db_path=settings.database_file).user_agent_warning()
        if sec_provider_status:
            st.warning(sec_provider_status)
        sec_tickers = st.multiselect("SEC tickers", corporate_universe["ticker"].tolist(), default=default_sec_tickers)
        s1, s2, s3 = st.columns(3)
        sec_start = s1.date_input("SEC start date", value=start_date, key="sec_backfill_start")
        sec_end = s2.date_input("SEC end date", value=end_date, key="sec_backfill_end")
        sec_step = s3.number_input("SEC tickers per resume step", min_value=1, max_value=10, value=5, step=1)
        if st.button("Start SEC Filing Backfill"):
            sec_run_id = create_sec_backfill_run(settings.database_file, sec_tickers, sec_start, sec_end)
            st.session_state["dataset_lab_sec_run_id"] = sec_run_id
            st.success(f"SEC backfill run #{sec_run_id} created.")

        sec_runs = list_sec_backfill_runs(settings.database_file, limit=20)
        selected_sec_run_id = None
        if not sec_runs.empty:
            selected_sec_run_id = st.selectbox(
                "SEC run",
                sec_runs["sec_run_id"].astype(int).tolist(),
                index=0,
                format_func=lambda value: f"SEC run #{value}",
            )
            st.dataframe(
                sec_runs[
                    [
                        "sec_run_id",
                        "status",
                        "requested_start_date",
                        "requested_end_date",
                        "total_tickers",
                        "completed_tickers",
                        "failed_tickers",
                        "events_inserted",
                        "duplicates_skipped",
                        "provider",
                    ]
                ],
                width="stretch",
                hide_index=True,
            )
        if selected_sec_run_id is not None:
            c1, c2 = st.columns(2)
            if c1.button("Resume SEC Backfill"):
                with st.spinner(f"Processing SEC backfill run #{selected_sec_run_id}..."):
                    result = process_sec_backfill_run(
                        settings.database_file,
                        int(selected_sec_run_id),
                        max_tickers=int(sec_step),
                    )
                st.success(
                    f"Processed {result.processed_tickers} ticker(s). Inserted: {result.events_inserted}; duplicates: {result.duplicates_skipped}; failed: {result.failed_tickers}."
                )
                for warning in result.warnings[:10]:
                    st.warning(warning)
            if c2.button("Retry Failed SEC Tickers"):
                reset = retry_failed_sec_items(settings.database_file, int(selected_sec_run_id))
                st.info(f"Reset {reset} failed SEC ticker(s) to pending.")
            sec_items = list_sec_backfill_items(settings.database_file, int(selected_sec_run_id))
            if not sec_items.empty:
                st.write("SEC backfill items")
                st.dataframe(
                    sec_items[
                        [
                            "ticker",
                            "status",
                            "filings_seen",
                            "events_inserted",
                            "duplicates_skipped",
                            "first_acceptance_at",
                            "last_acceptance_at",
                            "warning",
                            "error",
                        ]
                    ],
                    width="stretch",
                    hide_index=True,
                )

    with st.expander("SEC Classification / Feature Policy", expanded=False):
        st.caption(
            "Raw SEC catalyst rows are preserved. This panel shows the deterministic, versioned classification layer used by point-in-time SEC features."
        )
        st.write(f"Current SEC feature policy: `{SEC_FEATURE_POLICY_VERSION}`")
        st.json(SEC_FEATURE_POLICY, expanded=False)
        policy_tickers = st.multiselect(
            "Classification audit tickers",
            corporate_universe["ticker"].tolist(),
            default=default_sec_tickers,
            key="sec_classification_audit_tickers",
        )
        c1, c2 = st.columns(2)
        if c1.button("Classify / Refresh SEC Metadata Layer"):
            total = 0
            for ticker in policy_tickers:
                result = classify_ticker_sec_filings_safe(settings.database_file, ticker)
                total += int(result.get("classified", 0) or 0)
            st.success(f"Classified or refreshed {total} SEC filing row(s). Raw SEC rows were not modified.")
        summary = sec_classification_summary(settings.database_file, policy_tickers)
        by_ticker = summary["by_ticker"]
        by_category = summary["by_category"]
        exclusions = summary["exclusions"]
        if by_ticker.empty:
            st.info("No classified SEC filings found yet. Run SEC metadata backfill or refresh classifications.")
        else:
            st.write("Raw vs feature-eligible filings by ticker")
            st.dataframe(by_ticker, width="stretch", hide_index=True)
            raw_total = float(by_ticker["raw_filings"].sum() or 1)
            eligible_total = float(by_ticker["feature_eligible_filings"].sum() or 1)
            c1.metric("Top raw ticker concentration", f"{by_ticker['raw_concentration'].max() * 100:.1f}%")
            c2.metric("Top eligible ticker concentration", f"{by_ticker['eligible_concentration'].max() * 100:.1f}%")
            st.caption(
                f"Raw filings: {int(raw_total):,}. Feature-eligible filings: {int(eligible_total):,}. "
                "Dataset features aggregate by category and filing day so raw volume does not become a generic signal."
            )
        if not by_category.empty:
            st.write("Counts by deterministic SEC category")
            st.dataframe(by_category, width="stretch", hide_index=True)
        if not exclusions.empty:
            st.write("Exclusion reasons")
            st.dataframe(exclusions, width="stretch", hide_index=True)

    with st.expander("Historical Earnings Event Backfill", expanded=False):
        st.caption(
            "Best-effort historical earnings events are stored separately from catalysts. They feed point-in-time datasets only and do not change scanner scoring."
        )
        earnings_tickers = st.multiselect(
            "Earnings tickers",
            corporate_universe["ticker"].tolist(),
            default=default_earnings_tickers,
            key="earnings_backfill_tickers",
        )
        e1, e2, e3 = st.columns(3)
        earnings_start = e1.date_input("Earnings start date", value=start_date, key="earnings_start")
        earnings_end = e2.date_input("Earnings end date", value=end_date, key="earnings_end")
        use_earnings_cache = e3.checkbox("Use earnings provider cache", value=True, key="earnings_use_cache")
        if st.button("Backfill Earnings Events"):
            provider = YFinanceHistoricalEarningsProvider(db_path=settings.database_file)
            with st.spinner("Fetching historical earnings metadata with yfinance best-effort coverage..."):
                result = backfill_earnings_events(
                    settings.database_file,
                    earnings_tickers,
                    earnings_start,
                    earnings_end,
                    provider=provider,
                    use_cache=use_earnings_cache,
                )
            st.success(
                f"Earnings backfill complete. Inserted {result.inserted}, updated {result.updated}, duplicates {result.duplicates}, failed tickers {result.failed_tickers}."
            )
            if result.per_ticker:
                st.dataframe(pd.DataFrame(result.per_ticker), width="stretch", hide_index=True)
            for warning in result.warnings[:10]:
                st.warning(warning)
        st.write("Earnings coverage")
        earnings_summary = earnings_coverage_report(settings.database_file, earnings_tickers, earnings_start, earnings_end)
        if earnings_summary.empty:
            st.info("No stored earnings events for selected tickers yet.")
        else:
            st.dataframe(earnings_summary, width="stretch", hide_index=True)
        upload = st.file_uploader(
            "Import earnings CSV",
            type=["csv"],
            key="earnings_csv_import",
            help="Supported columns include ticker, fiscal_period_end, announced_at, available_at, timing, EPS and revenue fields.",
        )
        if upload is not None and st.button("Import Earnings CSV"):
            imported = parse_earnings_import_frame(pd.read_csv(upload))
            counts = bulk_insert_earnings_events(settings.database_file, imported.events)
            st.success(
                f"Imported {len(imported.events)} row(s): inserted {counts.get('inserted', 0)}, updated {counts.get('updated', 0)}, duplicates {counts.get('duplicate', 0)}."
            )
            for error in imported.errors[:10]:
                st.error(error)
            for warning in imported.warnings[:10]:
                st.warning(warning)

    st.subheader("Recent Dataset Builds")
    builds = list_dataset_builds(settings.database_file, limit=20)
    if builds.empty:
        st.info("No dataset builds have been stored yet.")
        selected_dataset_id = None
    else:
        build_display = builds[
            [
                "dataset_id",
                "version",
                "build_timestamp",
                "requested_start_date",
                "requested_end_date",
                "row_count",
                "data_hash",
                "tickers",
                "feature_count",
                "export_path",
                "warnings",
            ]
        ].copy()
        st.dataframe(build_display, width="stretch", hide_index=True)
        selected_dataset_id = st.selectbox(
            "Inspect stored build",
            builds["dataset_id"].astype(int).tolist(),
            index=0,
            format_func=lambda value: f"Dataset #{value}",
        )
        _render_dataset_evaluation_regime(settings.database_file, int(selected_dataset_id))

    frame = st.session_state.get("dataset_lab_latest_frame")
    dataset_id = st.session_state.get("dataset_lab_latest_dataset_id")
    if selected_dataset_id is not None and (frame is None or int(selected_dataset_id) != int(dataset_id or -1)):
        frame = flatten_saved_dataset(settings.database_file, int(selected_dataset_id), limit=500)
        dataset_id = int(selected_dataset_id)
    elif frame is not None and len(frame) > 500:
        frame = frame.head(500)

    if frame is None or frame.empty:
        st.info("Build a dataset or select a stored build to inspect rows, labels, and missingness.")
        st.subheader("Cached OHLCV Coverage")
        st.dataframe(coverage, width="stretch", hide_index=True)
        return

    build_row = None
    if selected_dataset_id is not None and not builds.empty:
        match = builds[builds["dataset_id"].astype(int).eq(int(selected_dataset_id))]
        if not match.empty:
            build_row = match.iloc[0]

    def _role_json(column: str, fallback: list[str]) -> list[str]:
        if build_row is None or column not in build_row:
            return fallback
        try:
            value = json.loads(str(build_row.get(column) or "[]"))
            return value if isinstance(value, list) else fallback
        except Exception:
            return fallback

    fallback_roles = role_sets_from_frame(frame)
    feature_columns = _role_json("feature_columns_json", fallback_roles.model_features)
    audit_columns = _role_json("audit_columns_json", fallback_roles.audit_columns)
    label_columns = _role_json("label_columns_json", fallback_roles.label_columns)
    identifier_columns = _role_json("identifier_columns_json", fallback_roles.identifier_columns)
    metadata_columns = _role_json("metadata_columns_json", fallback_roles.metadata_columns)

    c1, c2, c3, c4 = st.columns(4)
    total_rows_metric = int(build_row.get("row_count", len(frame))) if build_row is not None else len(frame)
    c1.metric("Dataset ID", int(dataset_id or 0))
    c2.metric("Rows", total_rows_metric)
    c3.metric("Model features", len(feature_columns))
    c4.metric("Label columns", len(label_columns))
    c5, c6, c7 = st.columns(3)
    c5.metric("Audit columns", len(audit_columns))
    c6.metric("Identifier columns", len(identifier_columns))
    c7.metric("Metadata columns", len(metadata_columns))

    st.subheader("Data Sufficiency Report")
    report = None
    try:
        report = dataset_sufficiency_report(settings.database_file, int(dataset_id or 0))
        st.json(report["summary"])
        if report.get("warnings"):
            for warning in report["warnings"]:
                st.warning(warning)
        if not report["label_counts"].empty:
            st.write("Label counts by horizon")
            st.dataframe(report["label_counts"], width="stretch", hide_index=True)
        if not report["return_distribution"].empty:
            st.write("Return distribution")
            st.dataframe(report["return_distribution"], width="stretch", hide_index=True)
        if not report["per_ticker"].empty:
            st.write("Per-ticker coverage")
            st.dataframe(report["per_ticker"], width="stretch", hide_index=True)
    except Exception as exc:
        st.warning(f"Could not build sufficiency report: {exc}")

    st.subheader("Coverage")
    if report is not None and not report.get("per_ticker", pd.DataFrame()).empty:
        coverage_frame = report["per_ticker"]
    else:
        coverage_frame = (
            frame.groupby("ticker")["trading_date"]
            .agg(["min", "max", "count"])
            .reset_index()
            .rename(columns={"min": "start", "max": "end", "count": "rows"})
        )
    st.dataframe(coverage_frame, width="stretch", hide_index=True)

    st.subheader("Missingness")
    if report is not None and not report.get("missingness", pd.DataFrame()).empty:
        missingness = report["missingness"]
    else:
        missingness = (
            frame.isna()
            .mean()
            .reset_index()
            .rename(columns={"index": "column", 0: "missing_fraction"})
            .sort_values("missing_fraction", ascending=False)
        )
    st.dataframe(missingness.head(80), width="stretch", hide_index=True)

    st.subheader("Dataset Preview")
    st.caption("Preview is bounded to the first 500 flattened rows; exports are generated only by explicit build/backfill actions.")
    st.dataframe(dataframe_for_streamlit(frame.head(200)), width="stretch", hide_index=True)

    st.subheader("Inspect Individual Snapshot")
    options = [
        f"{row.ticker} | {row.trading_date}"
        for row in frame[["ticker", "trading_date"]].drop_duplicates().itertuples(index=False)
    ]
    selected = st.selectbox("Snapshot", options, index=0)
    selected_ticker, selected_date = [part.strip() for part in selected.split("|", 1)]
    row = frame[(frame["ticker"].eq(selected_ticker)) & (frame["trading_date"].astype(str).eq(selected_date))].iloc[0]
    st.write("Feature timestamp:", row.get("as_of_timestamp"))
    st.write("Available catalyst IDs:", row.get("available_catalyst_ids", "n/a"))
    label_timestamps = {
        column: row[column]
        for column in label_columns
        if column.endswith("_available_at") and column in row and pd.notna(row[column])
    }
    st.write("Label availability timestamps")
    st.json(label_timestamps)
    with st.expander("Snapshot Row"):
        snapshot_display = row.to_frame("value").reset_index().rename(columns={"index": "field"})
        snapshot_display["value"] = snapshot_display["value"].map(
            lambda value: str(value) if isinstance(value, (list, dict)) else ("" if pd.isna(value) else str(value))
        )
        st.dataframe(snapshot_display, width="stretch", hide_index=True)

    st.subheader("Chronological Split Preview")
    unique_dates = sorted(pd.to_datetime(frame["trading_date"]).dt.date.unique())
    if len(unique_dates) >= 3:
        train_default = unique_dates[max(0, int(len(unique_dates) * 0.6) - 1)]
        validation_default = unique_dates[max(0, int(len(unique_dates) * 0.8) - 1)]
        s1, s2, s3 = st.columns(3)
        train_end = s1.date_input("Train end", value=train_default, key="dataset_train_end")
        validation_end = s2.date_input("Validation end", value=validation_default, key="dataset_validation_end")
        gap = s3.number_input("Gap sessions", min_value=0, max_value=20, value=0, step=1)
        split_frame = assign_chronological_splits(frame[["ticker", "trading_date"]], train_end, validation_end, gap_sessions=int(gap))
        split_counts = split_frame.groupby("split").size().reset_index(name="rows")
        st.dataframe(split_counts, width="stretch", hide_index=True)
    else:
        st.info("Need at least three unique snapshot dates to preview chronological splits.")

    st.subheader("Cached OHLCV Coverage")
    st.dataframe(coverage, width="stretch", hide_index=True)


def _metrics_records(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    for _, row in frame.iterrows():
        metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
        if not metrics and row.get("metrics_json"):
            try:
                metrics = json.loads(str(row.get("metrics_json")))
            except Exception:
                metrics = {}
        base = {
            key: row.get(key)
            for key in [
                "fold_name",
                "split_name",
                "train_start_date",
                "train_end_date",
                "eval_start_date",
                "eval_end_date",
                "train_rows",
                "eval_rows",
            ]
            if key in frame.columns
        }
        base.update(metrics)
        rows.append(base)
    return pd.DataFrame(rows)


def shadow_research_page(settings: Settings) -> None:
    st.header("Shadow Research")
    st.warning(
        "Exploratory shadow predictions only. Rankings are not validated alpha, trading recommendations, or scanner inputs."
    )
    artifacts = list_model_artifacts(settings.database_file)
    frozen = artifacts[artifacts["evaluation_regime"].eq("exploratory_shadow")] if not artifacts.empty else pd.DataFrame()
    if frozen.empty:
        st.info("No frozen exploratory shadow artifact is registered.")
        return
    artifact = frozen.sort_values("created_at", ascending=False).iloc[0]
    artifact_id = str(artifact["artifact_id"])
    status = shadow_status_report(settings.database_file, artifact_id=artifact_id)
    c1, c2, c3 = st.columns(3)
    c1.metric("Prediction runs", int(status.get("run_count", 0)))
    c2.metric("Prediction dates", int(status.get("prediction_date_count", 0)))
    c3.metric("Forward sample", str(status.get("sample_status", "insufficient_forward_sample")))
    if status.get("sample_status") == "insufficient_forward_sample":
        prediction_date_count = int(status.get("prediction_date_count", 0))
        noun = "date is" if prediction_date_count == 1 else "dates are"
        st.warning(
            f"{prediction_date_count} prediction {noun} insufficient for performance conclusions; "
            "outcomes are maturity tracking only."
        )
    st.caption(
        f"Artifact: {artifact_id} | feature hash: {str(artifact['feature_manifest_hash'])[:12]}... | "
        "scanner score contribution: 0"
    )
    runs = list_shadow_prediction_runs(settings.database_file, limit=100)
    if runs.empty:
        st.info("No immutable shadow prediction runs have been recorded yet.")
        return
    latest = runs.iloc[0]
    st.subheader(f"Latest Run: {latest['prediction_date']}")
    st.subheader("Outcome Maturity")
    maturity_columns = st.columns(3)
    outcomes_by_horizon = status.get("outcomes_by_horizon", {})
    for column, horizon in zip(maturity_columns, (1, 5, 20), strict=True):
        maturity = outcomes_by_horizon.get(str(horizon), {})
        column.metric(
            f"{horizon}-session",
            f"{int(maturity.get('matured', 0))} matured",
            f"{int(maturity.get('pending', 0))} pending",
        )
    st.caption(
        "SPY outcomes are retained for audit but excluded from cross-sectional IC, ranking, and equity directional metrics."
    )
    warnings = json.loads(str(latest.get("warnings_json") or "[]"))
    if warnings:
        st.warning(" | ".join(str(item) for item in warnings))
    predictions = list_shadow_predictions(settings.database_file, int(latest["run_id"]), limit=500)
    display_columns = [
        "predicted_rank", "ticker", "predicted_value", "predicted_percentile", "data_quality_flags"
    ]
    st.dataframe(
        dataframe_for_streamlit(predictions[[column for column in display_columns if column in predictions.columns]]),
        width="stretch",
        hide_index=True,
    )
    st.caption("Predictions are immutable research records and are never combined with the Daily Scanner score.")


def model_lab_page(settings: Settings) -> None:
    st.header("Model Lab")
    st.warning(
        "Research-only baseline modeling. These runs do not change scanner scoring, catalysts, alerts, or recommendations."
    )
    st.caption(
        "Uses dataset-approved model features only. Labels, audit columns, identifiers, timestamps, hashes, raw JSON, and workflow statuses are excluded from model inputs."
    )

    builds = list_dataset_builds(settings.database_file, limit=100)
    completed = (
        builds[(builds["row_count"].fillna(0).astype(int) > 0) & (builds["data_hash"].astype(str) != "pending")]
        if not builds.empty
        else pd.DataFrame()
    )
    if completed.empty:
        st.info("No completed dataset builds are available yet.")
        return

    try:
        default_dataset = latest_accepted_dataset_id(settings.database_file)
    except Exception:
        default_dataset = int(completed.sort_values("dataset_id", ascending=False).iloc[0]["dataset_id"])
    dataset_ids = completed["dataset_id"].astype(int).tolist()
    default_index = dataset_ids.index(default_dataset) if default_dataset in dataset_ids else 0

    c1, c2, c3, c4 = st.columns(4)
    dataset_id = int(c1.selectbox("Dataset", dataset_ids, index=default_index))
    target_names = target_options()
    target_name = c2.selectbox(
        "Target",
        target_names,
        index=0,
        format_func=lambda name: get_target_definition(name).display_name,
    )
    target_definition = get_target_definition(target_name)
    feature_set_name = c3.selectbox("Feature set", FEATURE_SET_NAMES)
    model_name = c4.selectbox("Model", list(target_definition.allowed_models), index=min(2, len(target_definition.allowed_models) - 1))

    selected_build = completed[completed["dataset_id"].astype(int).eq(dataset_id)].iloc[0]
    st.caption(
        f"Dataset {dataset_id}: {int(selected_build['row_count'])} rows, hash {str(selected_build['data_hash'])[:12]}..., "
        f"version {selected_build['version']}"
    )
    _render_dataset_evaluation_regime(settings.database_file, dataset_id)
    st.json(
        {
            "selected_target": target_metadata(target_definition),
            "target_research_only": True,
            "scanner_scoring_effect": 0,
        },
        expanded=False,
    )

    feature_columns = json.loads(selected_build.get("feature_columns_json") or "[]")
    feature_defs = feature_set_definitions(feature_columns)
    st.dataframe(
        pd.DataFrame(
            [
                {"feature_set": item.name, "columns": len(item.columns), "description": item.description}
                for item in feature_defs
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Baseline Diagnostics", expanded=False):
        st.caption(
            "Read-only signal-quality diagnostics for completed baseline runs. Diagnostics do not alter datasets, scanner scoring, catalysts, or model runs."
        )
        diagnostics_dir = settings.database_file.parent / "processed"
        artifacts = list_diagnostic_artifacts(diagnostics_dir)
        artifact_options = {
            f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in artifacts
        }
        generate_confirm = st.checkbox(
            "I understand diagnostics are read-only and do not affect scanner scoring.",
            key="model_lab_diagnostics_confirm",
        )
        if st.button("Generate Diagnostics Artifact", disabled=not generate_confirm):
            with st.spinner("Analyzing completed model runs and Dataset Lab features..."):
                diagnostics = build_model_diagnostics(settings.database_file, dataset_id)
                artifact_path = write_diagnostics_artifact(diagnostics, diagnostics_dir)
            st.success(f"Saved diagnostics artifact: {artifact_path.name}")
            artifacts = list_diagnostic_artifacts(diagnostics_dir)
            artifact_options = {
                f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in artifacts
            }

        if not artifact_options:
            st.info("No diagnostics artifacts found yet.")
        else:
            selected_artifact = st.selectbox("Diagnostics artifact", list(artifact_options.keys()))
            diagnostics = load_diagnostics_artifact(artifact_options[selected_artifact])
            summary = diagnostics.get("summary", {})
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Dataset", diagnostics.get("dataset_id", "n/a"))
            c2.metric("Rows", int(diagnostics.get("dataset_rows", 0) or 0))
            c3.metric("Runs Used", len(diagnostics.get("model_runs_used", [])))
            c4.metric("Failure Modes", len(summary.get("likely_failure_modes", [])))
            st.write("Likely failure modes:", ", ".join(summary.get("likely_failure_modes", [])) or "n/a")
            diag_tabs = st.tabs(["Targets", "Folds", "Ablations", "Features", "Ticker Errors"])
            with diag_tabs[0]:
                st.dataframe(dataframe_for_streamlit(pd.DataFrame(diagnostics.get("target_distribution", []))), width="stretch", hide_index=True)
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(diagnostics.get("split_distribution_5_session", []))),
                    width="stretch",
                    hide_index=True,
                )
            with diag_tabs[1]:
                fold_cols = [
                    "target_horizon",
                    "feature_set_name",
                    "model_name",
                    "fold_name",
                    "split_name",
                    "eval_start_date",
                    "eval_end_date",
                    "metric_rmse",
                    "metric_oos_r2_vs_train_mean",
                    "metric_spearman_ic",
                    "metric_directional_accuracy",
                    "metric_roc_auc",
                ]
                fold_frame = pd.DataFrame(diagnostics.get("fold_metrics", []))
                st.dataframe(
                    dataframe_for_streamlit(fold_frame[[col for col in fold_cols if col in fold_frame.columns]]),
                    width="stretch",
                    hide_index=True,
                )
            with diag_tabs[2]:
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(diagnostics.get("ablation_diagnostics", []))),
                    width="stretch",
                    hide_index=True,
                )
            with diag_tabs[3]:
                feature_diag = diagnostics.get("feature_diagnostics", {})
                st.dataframe(dataframe_for_streamlit(pd.DataFrame(feature_diag.get("group_missingness", []))), width="stretch", hide_index=True)
                st.dataframe(dataframe_for_streamlit(pd.DataFrame(feature_diag.get("near_constant_features", []))), width="stretch", hide_index=True)
                st.dataframe(dataframe_for_streamlit(pd.DataFrame(feature_diag.get("highly_correlated_pairs", []))), width="stretch", hide_index=True)
            with diag_tabs[4]:
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(diagnostics.get("per_ticker_prediction_error_5_session_technical_ridge", []))),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(dataframe_for_streamlit(pd.DataFrame(diagnostics.get("coverage_by_ticker", []))), width="stretch", hide_index=True)

    with st.expander("Feature Quality / Pruning", expanded=False):
        st.caption(
            "Phase 2D-4 feature-quality artifacts audit missingness, constants, sparse event features, "
            "correlations, outliers, and fold-safe univariate IC. Generated pruned feature sets are research-only "
            "and do not affect scanner scoring."
        )
        quality_dir = settings.database_file.parent / "processed"
        quality_artifacts = list_feature_quality_artifacts(quality_dir)
        quality_options = {
            f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in quality_artifacts
        }
        quality_confirm = st.checkbox(
            "I understand feature-quality analysis is read-only and does not inspect final-test labels for selection.",
            key="model_lab_feature_quality_confirm",
        )
        if st.button("Generate Feature Quality Artifact", disabled=not quality_confirm):
            with st.spinner("Auditing feature quality and generating pruned feature-set definitions..."):
                artifact = build_feature_quality_audit(settings.database_file, dataset_id)
                artifact_path = write_feature_quality_artifact(artifact, quality_dir)
            st.success(f"Saved feature-quality artifact: {artifact_path.name}")
            quality_artifacts = list_feature_quality_artifacts(quality_dir)
            quality_options = {
                f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in quality_artifacts
            }

        if not quality_options:
            st.info("No Phase 2D-4 feature-quality artifacts found yet.")
        else:
            selected_quality = st.selectbox("Feature-quality artifact", list(quality_options.keys()))
            quality = load_feature_quality_artifact(quality_options[selected_quality])
            summary = quality.get("summary", {})
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Features", int(summary.get("feature_count", 0) or 0))
            q2.metric("Near Constant", int(summary.get("near_constant_count", 0) or 0))
            q3.metric("Sparse Events", int(summary.get("sparse_event_feature_count", 0) or 0))
            q4.metric("Corr Pairs", int(summary.get("high_correlation_pair_count", 0) or 0))
            st.write(summary.get("selection_guardrail", "Final-test labels are not used for feature selection."))
            quality_tabs = st.tabs(["Feature Sets", "Groups", "Missingness", "Correlations", "Univariate IC"])
            with quality_tabs[0]:
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(feature_set_quality_rows(quality))),
                    width="stretch",
                    hide_index=True,
                )
            with quality_tabs[1]:
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(quality.get("feature_group_summary", []))),
                    width="stretch",
                    hide_index=True,
                )
            with quality_tabs[2]:
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(quality.get("feature_missingness", [])).head(200)),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(quality.get("sparse_event_features", [])).head(100)),
                    width="stretch",
                    hide_index=True,
                )
            with quality_tabs[3]:
                corr = quality.get("correlation_audit", {})
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(corr.get("redundant_groups", []))),
                    width="stretch",
                    hide_index=True,
                )
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(corr.get("highly_correlated_pairs", [])).head(100)),
                    width="stretch",
                    hide_index=True,
                )
            with quality_tabs[4]:
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(quality.get("univariate_ic", [])).head(200)),
                    width="stretch",
                    hide_index=True,
                )

    with st.expander("Event Feature Coverage / Timing", expanded=False):
        st.caption(
            "Phase 2D-5 artifacts audit SEC, earnings, catalyst, and LLM-supported event coverage/timing. "
            "They are derived from point-in-time dataset features and local event tables only; no provider calls are made."
        )
        event_dir = settings.database_file.parent / "processed"
        event_artifacts = list_event_redesign_artifacts(event_dir)
        event_options = {
            f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in event_artifacts
        }
        event_confirm = st.checkbox(
            "I understand event redesign artifacts are research-only and do not affect scanner scoring.",
            key="model_lab_event_redesign_confirm",
        )
        if st.button("Generate Event Coverage / Timing Artifacts", disabled=not event_confirm):
            with st.spinner("Auditing event feature coverage and timing..."):
                coverage = build_event_coverage_audit(settings.database_file, dataset_id)
                timing = build_event_timing_audit(settings.database_file, dataset_id)
                coverage_path = write_event_artifact(coverage, event_dir)
                timing_path = write_event_artifact(timing, event_dir)
            st.success(f"Saved artifacts: {coverage_path.name}, {timing_path.name}")
            event_artifacts = list_event_redesign_artifacts(event_dir)
            event_options = {
                f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in event_artifacts
            }

        if not event_options:
            st.info("No Phase 2D-5 event redesign artifacts found yet.")
        else:
            selected_event = st.selectbox("Event artifact", list(event_options.keys()))
            event_artifact = load_event_artifact(event_options[selected_event])
            event_type = event_artifact.get("artifact_type", "event")
            summary = event_artifact.get("summary", {})
            e1, e2, e3, e4 = st.columns(4)
            e1.metric("Artifact", event_type)
            e2.metric("Rows", int(event_artifact.get("row_count", 0) or 0))
            e3.metric("Derived Features", int(event_artifact.get("derived_feature_count", 0) or 0))
            e4.metric("Timing Violations", int(summary.get("pre_availability_violation_count", 0) or 0))
            if event_type == "event_coverage":
                st.write("Inactive groups:", ", ".join(summary.get("inactive_groups", [])) or "none")
                event_tabs = st.tabs(["Feature Sets", "Feature Activity", "Ticker Coverage", "Fold Density"])
                with event_tabs[0]:
                    st.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(event_feature_set_rows(event_artifact))),
                        width="stretch",
                        hide_index=True,
                    )
                with event_tabs[1]:
                    st.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(event_artifact.get("active_observation_counts", [])).head(200)),
                        width="stretch",
                        hide_index=True,
                    )
                with event_tabs[2]:
                    st.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(event_artifact.get("coverage_by_ticker", []))),
                        width="stretch",
                        hide_index=True,
                    )
                with event_tabs[3]:
                    st.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(event_artifact.get("event_density_by_fold", []))),
                        width="stretch",
                        hide_index=True,
                    )
            elif event_type == "event_timing":
                timing_tabs = st.tabs(["Summary", "Lag Buckets", "Violations"])
                with timing_tabs[0]:
                    st.json(summary, expanded=False)
                    st.json(event_artifact.get("available_at_missing_counts", {}), expanded=False)
                with timing_tabs[1]:
                    st.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(event_artifact.get("lag_bucket_summary", [])).head(300)),
                        width="stretch",
                        hide_index=True,
                    )
                with timing_tabs[2]:
                    violations = event_artifact.get("pre_availability_violations", {})
                    st.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(violations.get("sec", []))),
                        width="stretch",
                        hide_index=True,
                    )
                    st.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(violations.get("earnings", []))),
                        width="stretch",
                        hide_index=True,
                    )

    with st.expander("Research Event Provider Readiness", expanded=False):
        st.caption(
            "Read-only provider registry for future research-event candidates. This panel makes no provider calls, "
            "does not import candidates, and has zero scanner scoring effect."
        )
        readiness = build_provider_readiness_report()
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Configured", int(readiness.get("configured_provider_count", 0) or 0))
        r2.metric("Enabled", int(readiness.get("enabled_provider_count", 0) or 0))
        r3.metric("API Key Providers", int(readiness.get("requires_api_key_count", 0) or 0))
        r4.metric("Network Calls", "Yes" if readiness.get("network_calls_would_occur") else "No")
        provider_rows = pd.DataFrame(readiness.get("providers", []))
        provider_cols = [
            "provider_name",
            "provider_type",
            "enabled",
            "config_status",
            "requires_api_key",
            "supports_point_in_time_available_at",
            "supports_backfill",
            "source_quality_default",
            "allowed_usage",
            "compliance_notes",
            "next_action_required",
            "network_calls_would_occur",
        ]
        st.dataframe(
            dataframe_for_streamlit(provider_rows[[col for col in provider_cols if col in provider_rows.columns]]),
            width="stretch",
            hide_index=True,
        )
        st.json(readiness.get("guardrails", {}), expanded=False)

        st.divider()
        st.write("Company IR document coverage")
        st.caption(
            "Read-only candidate-to-annotation-to-SourceDocument audit. Workflow priority is not an alpha score, "
            "and this panel never creates documents or runs extraction."
        )
        try:
            document_coverage = build_document_coverage_audit(
                settings.database_file,
                provider="company_ir_press_release",
            )
            coverage_summary = document_coverage.summary
            d1, d2, d3, d4, d5 = st.columns(5)
            d1.metric("IR Candidates", int(coverage_summary.get("total_company_ir_candidates", 0) or 0))
            d2.metric(
                "Linked Documents",
                f"{int(coverage_summary.get('candidates_with_linked_documents', 0) or 0)} "
                f"({float(coverage_summary.get('linked_document_pct', 0.0) or 0.0):.1f}%)",
            )
            d3.metric("Complete", int(coverage_summary.get("complete_documents", 0) or 0))
            d4.metric("Needs Enrichment", int(coverage_summary.get("queue_row_count", 0) or 0))
            d5.metric("Broken Links", int(coverage_summary.get("broken_linkages", 0) or 0))

            coverage_left, coverage_right = st.columns(2)
            with coverage_left:
                st.write("Coverage status")
                st.dataframe(
                    dataframe_for_streamlit(pd.DataFrame(document_coverage.status_counts)),
                    width="stretch",
                    hide_index=True,
                )
            with coverage_right:
                st.write("Top missing-document tickers")
                missing_tickers = pd.DataFrame(document_coverage.top_missing_document_tickers)
                if missing_tickers.empty:
                    st.info("No company IR candidates currently require document enrichment.")
                else:
                    st.dataframe(dataframe_for_streamlit(missing_tickers), width="stretch", hide_index=True)

            queue_path = settings.database_file.parent / "processed" / "phase2e5a_company_ir_enrichment_queue.csv"
            st.code(str(queue_path), language=None)
            st.info(
                "The enrichment queue requires manually supplied source text. No URL is fetched, no text is fabricated, "
                "and no database record is changed by this coverage panel."
            )
        except Exception as exc:
            st.warning(f"Company IR document coverage is unavailable: {exc}")

    with st.expander("Research Event Annotations", expanded=False):
        st.caption(
            "Research-only historical annotations for dataset/model experiments. They do not create active catalysts, "
            "do not change scanner scoring, and are not trading recommendations."
        )
        annotation_dir = settings.database_file.parent / "processed"
        annotation_tabs = st.tabs(["Manual Entry", "CSV Import", "News/Event Candidates", "Coverage / Baselines", "Stored Annotations"])
        universe_tickers = load_universe(settings.universe_file)["ticker"].astype(str).str.upper().tolist()
        with annotation_tabs[0]:
            with st.form("manual_research_annotation_form", clear_on_submit=True):
                a1, a2, a3, a4 = st.columns(4)
                ticker = a1.selectbox("Ticker", universe_tickers, key="annotation_manual_ticker")
                event_date_value = a2.date_input("Event date", value=date.today(), key="annotation_manual_event_date")
                available_date = a3.date_input("Available date", value=event_date_value, key="annotation_manual_available_date")
                available_time = a4.time_input("Available time", value=time(23, 59), key="annotation_manual_available_time")
                b1, b2, b3, b4 = st.columns(4)
                event_type = b1.selectbox("Event type", list(ANNOTATION_EVENT_TYPES), key="annotation_manual_event_type")
                sentiment = b2.selectbox("Sentiment", list(ANNOTATION_SENTIMENT_LABELS), index=list(ANNOTATION_SENTIMENT_LABELS).index("unknown"), key="annotation_manual_sentiment")
                strength = b3.slider("Strength", 0, 10, 0, key="annotation_manual_strength")
                confidence = b4.slider("Confidence", 0.0, 1.0, 0.0, 0.05, key="annotation_manual_confidence")
                title = st.text_input("Title", key="annotation_manual_title")
                source = st.text_input("Source", value="manual", key="annotation_manual_source")
                source_url = st.text_input("Source URL (optional)", key="annotation_manual_source_url")
                summary_text = st.text_area("Summary", key="annotation_manual_summary")
                evidence_text = st.text_area("Evidence text", key="annotation_manual_evidence")
                tags_text = st.text_input("Tags (comma-separated)", key="annotation_manual_tags")
                confirm_annotation = st.checkbox(
                    "I understand this is research-only and has zero scanner scoring effect.",
                    key="annotation_manual_confirm",
                )
                submitted = st.form_submit_button("Add Research Annotation", disabled=not confirm_annotation)
            if submitted:
                available_at = datetime.combine(available_date, available_time, tzinfo=UTC)
                result = insert_annotation(
                    settings.database_file,
                    ResearchEventAnnotation(
                        ticker=ticker,
                        event_date=event_date_value,
                        available_at=available_at,
                        event_type=event_type,
                        sentiment_label=sentiment,
                        strength=strength,
                        confidence=confidence,
                        source=source or "manual",
                        source_url=source_url or None,
                        title=title,
                        summary=summary_text,
                        evidence_text=evidence_text,
                        tags=[part.strip() for part in tags_text.split(",") if part.strip()],
                    ),
                )
                if result.inserted:
                    st.success(f"Stored research annotation #{result.annotation_id}.")
                else:
                    st.info(f"Duplicate annotation already exists as #{result.annotation_id}; no new row inserted.")

        with annotation_tabs[1]:
            st.write(
                "CSV columns: ticker, event_date, available_at, event_type, sentiment_label, strength, confidence, "
                "source, source_url, title, summary, evidence_text, tags."
            )
            upload = st.file_uploader("Upload annotation CSV", type=["csv"], key="annotation_csv_upload")
            csv_confirm = st.checkbox(
                "I understand imported annotations are research-only and do not affect scanner scoring.",
                key="annotation_csv_confirm",
            )
            if upload is not None and csv_confirm and st.button("Import Annotation CSV", key="annotation_csv_import_button"):
                try:
                    import_frame = pd.read_csv(upload)
                    import_result = parse_annotation_import_frame(import_frame)
                    inserted = bulk_insert_annotations(settings.database_file, import_result.annotations)
                    inserted_count = sum(1 for item in inserted if item.inserted)
                    duplicate_count = sum(1 for item in inserted if not item.inserted)
                    st.success(f"Imported {inserted_count} annotations; skipped {duplicate_count} database duplicates.")
                    if import_result.errors:
                        st.warning(f"{len(import_result.errors)} CSV rows were rejected.")
                        st.dataframe(pd.DataFrame([item.__dict__ for item in import_result.errors]), width="stretch", hide_index=True)
                    if import_result.warnings:
                        st.info(f"{len(import_result.warnings)} CSV warnings.")
                        st.dataframe(pd.DataFrame([item.__dict__ for item in import_result.warnings]), width="stretch", hide_index=True)
                except Exception as exc:
                    st.error(f"CSV import failed: {exc}")

        with annotation_tabs[2]:
            st.caption(
                "Provider-style news/event candidate staging. Candidates are review-gated and do not affect models until accepted and imported as research-only annotations."
            )
            st.write(
                "Candidate CSV columns: ticker, event_date, available_at, event_type, title, summary, source, source_url, "
                "evidence_text, sentiment_label, strength, confidence, tags, provider_metadata_json."
            )
            candidate_upload = st.file_uploader("Upload news/event candidate CSV", type=["csv"], key="news_candidate_csv_upload")
            candidate_confirm = st.checkbox(
                "I understand staged candidates are research-only and have zero scanner scoring effect.",
                key="news_candidate_stage_confirm",
            )
            if candidate_upload is not None and candidate_confirm and st.button("Stage Candidate CSV", key="news_candidate_stage_button"):
                try:
                    candidate_frame = pd.read_csv(candidate_upload)
                    candidate_result = parse_candidate_import_frame(candidate_frame)
                    staged = stage_candidates(settings.database_file, candidate_result.candidates)
                    staged_count = sum(1 for item in staged if item.inserted and item.status == "staged")
                    duplicate_count = sum(1 for item in staged if item.status == "duplicate")
                    existing_count = sum(1 for item in staged if not item.inserted)
                    artifact = build_candidate_ingestion_artifact(settings.database_file)
                    artifact_path = annotation_dir / f"news_event_candidate_ingestion_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
                    artifact_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
                    st.success(
                        f"Staged {staged_count} candidate(s); marked {duplicate_count} duplicate(s); skipped {existing_count} existing staged row(s)."
                    )
                    st.caption(f"Candidate artifact saved: {artifact_path.name}")
                    if candidate_result.errors:
                        st.warning(f"{len(candidate_result.errors)} CSV rows were rejected.")
                        st.dataframe(pd.DataFrame([item.__dict__ for item in candidate_result.errors]), width="stretch", hide_index=True)
                    if candidate_result.warnings:
                        st.info(f"{len(candidate_result.warnings)} CSV warnings.")
                        st.dataframe(pd.DataFrame([item.__dict__ for item in candidate_result.warnings]), width="stretch", hide_index=True)
                except Exception as exc:
                    st.error(f"Candidate staging failed: {exc}")

            st.divider()
            st.write("Strict company IR / press-release provider")
            st.caption(
                "Upload only user-supplied company IR, newsroom, or press-release rows. This path requires source_url, "
                "does not fetch URLs, does not crawl, and stages candidates for review only."
            )
            ir_upload = st.file_uploader(
                "Upload company IR / press-release candidate CSV",
                type=["csv"],
                key="company_ir_candidate_csv_upload",
            )
            ir_confirm = st.checkbox(
                "I confirm these are user-supplied company IR / press-release rows and no website discovery should occur.",
                key="company_ir_candidate_stage_confirm",
            )
            if ir_upload is not None and ir_confirm and st.button("Stage Company IR Candidates", key="company_ir_candidate_stage_button"):
                try:
                    ir_frame = pd.read_csv(ir_upload)
                    ir_result = parse_company_ir_press_release_frame(ir_frame)
                    staged = stage_candidates(settings.database_file, ir_result.candidates)
                    staged_count = sum(1 for item in staged if item.inserted and item.status == "staged")
                    duplicate_count = sum(1 for item in staged if item.status == "duplicate")
                    existing_count = sum(1 for item in staged if not item.inserted)
                    artifact = build_candidate_ingestion_artifact(settings.database_file)
                    artifact_path = annotation_dir / f"company_ir_candidate_ingestion_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
                    artifact_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
                    st.success(
                        f"Staged {staged_count} company IR candidate(s); marked {duplicate_count} duplicate(s); "
                        f"skipped {existing_count} existing staged row(s)."
                    )
                    st.caption(f"Company IR candidate artifact saved: {artifact_path.name}")
                    if ir_result.errors:
                        st.warning(f"{len(ir_result.errors)} CSV rows were rejected.")
                        st.dataframe(pd.DataFrame([item.__dict__ for item in ir_result.errors]), width="stretch", hide_index=True)
                    if ir_result.warnings:
                        st.info(f"{len(ir_result.warnings)} CSV warnings.")
                        st.dataframe(pd.DataFrame([item.__dict__ for item in ir_result.warnings]), width="stretch", hide_index=True)
                except Exception as exc:
                    st.error(f"Company IR candidate staging failed: {exc}")

            status_counts = candidate_counts_by_status(settings.database_file)
            if not status_counts.empty:
                st.write("Candidate status counts")
                st.dataframe(dataframe_for_streamlit(status_counts), width="stretch", hide_index=True)

            status_filter = st.selectbox(
                "Candidate status filter",
                ["staged", "accepted", "rejected", "duplicate", "imported", "all"],
                key="news_candidate_status_filter",
            )
            candidate_frame = list_candidates(
                settings.database_file,
                status=None if status_filter == "all" else status_filter,
                limit=300,
            )
            if candidate_frame.empty:
                st.info("No news/event candidates found for this filter.")
            else:
                candidate_frame = candidate_frame.copy()
                candidate_frame["source_document_linked"] = candidate_frame["source_document_id"].notna().map(
                    {True: "yes", False: "no"}
                )
                candidate_quality = quality_distribution(candidate_frame)
                q1, q2, q3 = st.columns(3)
                q1.metric("Low-specificity neutral", int(candidate_quality.get("low_specificity_neutral_count", 0)))
                q2.metric("Routine SEC-heavy", int(candidate_quality.get("routine_sec_heavy_count", 0)))
                q3.metric("Material non-SEC", int(candidate_quality.get("material_non_sec_count", 0)))
                with st.expander("Candidate source-quality summary", expanded=False):
                    cqa, cqb = st.columns(2)
                    cqa.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(candidate_quality.get("source_quality_distribution", []))),
                        width="stretch",
                        hide_index=True,
                    )
                    cqb.dataframe(
                        dataframe_for_streamlit(pd.DataFrame(candidate_quality.get("informativeness_distribution", []))),
                        width="stretch",
                        hide_index=True,
                    )
                candidate_cols = [
                    "candidate_id",
                    "status",
                    "ticker",
                    "event_date",
                    "available_at",
                    "event_type",
                    "sentiment_label",
                    "strength",
                    "confidence",
                    "source",
                    "source_quality",
                    "informativeness",
                    "title",
                    "duplicate_reason",
                    "imported_annotation_id",
                    "source_document_linked",
                    "source_document_id",
                ]
                st.dataframe(
                    dataframe_for_streamlit(candidate_frame[[col for col in candidate_cols if col in candidate_frame.columns]]),
                    width="stretch",
                    hide_index=True,
                )

                reviewable = candidate_frame[candidate_frame["status"].isin(["staged", "accepted", "rejected"])]
                if not reviewable.empty:
                    st.write("Review selected candidate")
                    review_id = int(
                        st.selectbox(
                            "Candidate ID",
                            reviewable["candidate_id"].astype(int).tolist(),
                            format_func=lambda value: f"#{value}",
                            key="news_candidate_review_id",
                        )
                    )
                    selected_candidate = candidate_frame[candidate_frame["candidate_id"].astype(int).eq(review_id)].iloc[0]
                    with st.expander("Candidate details", expanded=False):
                        st.json(
                            {
                                "ticker": selected_candidate.get("ticker"),
                                "event_date": selected_candidate.get("event_date"),
                                "available_at": selected_candidate.get("available_at"),
                                "event_type": selected_candidate.get("event_type"),
                                "sentiment_label": selected_candidate.get("sentiment_label"),
                                "source_url": selected_candidate.get("source_url"),
                                "source_quality": selected_candidate.get("source_quality"),
                                "informativeness": selected_candidate.get("informativeness"),
                                "quality_reason": selected_candidate.get("quality_reason"),
                                "duplicate_theme_key": selected_candidate.get("duplicate_theme_key"),
                                "summary": selected_candidate.get("summary"),
                                "evidence_text": selected_candidate.get("evidence_text"),
                                "provider_metadata": selected_candidate.get("provider_metadata"),
                                "source_document_id": selected_candidate.get("source_document_id"),
                                "source_document_linked": selected_candidate.get("source_document_linked"),
                                "scanner_scoring_effect": 0,
                            },
                            expanded=False,
                        )
                    r1, r2 = st.columns(2)
                    if r1.button("Accept Candidate", key="news_candidate_accept_button"):
                        try:
                            accept_candidate(settings.database_file, review_id)
                            st.success(f"Accepted candidate #{review_id}.")
                        except Exception as exc:
                            st.error(f"Accept failed: {exc}")
                    reject_reason = r2.text_input("Reject reason", key="news_candidate_reject_reason")
                    if r2.button("Reject Candidate", key="news_candidate_reject_button"):
                        try:
                            reject_candidate(settings.database_file, review_id, reason=reject_reason or "Rejected in review.")
                            st.success(f"Rejected candidate #{review_id}.")
                        except Exception as exc:
                            st.error(f"Reject failed: {exc}")

                imported_ir = candidate_frame[
                    candidate_frame["status"].eq("imported")
                    & candidate_frame["provider"].eq("company_ir_press_release")
                    & candidate_frame["source_document_id"].isna()
                ]
                if not imported_ir.empty:
                    with st.expander("Link an imported company IR candidate to SourceDocument", expanded=False):
                        st.caption(
                            "This creates or reuses a local review document only. It does not run fallback/OpenAI extraction, "
                            "create an active catalyst, or affect scanner scoring."
                        )
                        link_candidate_id = int(
                            st.selectbox(
                                "Imported company IR candidate",
                                imported_ir["candidate_id"].astype(int).tolist(),
                                format_func=lambda value: f"#{value}",
                                key="news_candidate_document_link_id",
                            )
                        )
                        link_confirm = st.checkbox(
                            "Create or reuse the local SourceDocument for this imported candidate.",
                            key="news_candidate_document_link_confirm",
                        )
                        if st.button(
                            "Create / Link SourceDocument",
                            disabled=not link_confirm,
                            key="news_candidate_document_link_button",
                        ):
                            try:
                                link_result = create_or_link_candidate_document(
                                    settings.database_file,
                                    link_candidate_id,
                                )
                                if link_result.linked:
                                    action = "Created" if link_result.document_created else "Reused"
                                    st.success(f"{action} SourceDocument #{link_result.source_document_id}.")
                                else:
                                    st.warning(link_result.warning or "Document linkage was not completed.")
                            except Exception as exc:
                                st.error(f"Source-document linkage failed: {exc}")

            import_confirm = st.checkbox(
                "I understand accepted candidates will import only as research annotations and will not create active catalysts.",
                key="news_candidate_import_confirm",
            )
            create_ir_documents = st.checkbox(
                "Create or reuse SourceDocuments for accepted company IR candidates that include source text or evidence.",
                value=False,
                key="news_candidate_import_documents",
                help="Opt-in only. Documents become available in Documents / Text and LLM Review; no extraction runs automatically.",
            )
            if st.button("Import Accepted Candidates", disabled=not import_confirm, key="news_candidate_import_accepted"):
                try:
                    summary = import_accepted_candidates(
                        settings.database_file,
                        create_source_documents=create_ir_documents,
                    )
                    artifact = build_candidate_ingestion_artifact(settings.database_file)
                    artifact_path = annotation_dir / f"news_event_candidate_ingestion_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}.json"
                    artifact_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
                    st.success(
                        f"Imported {summary.imported_count} accepted candidate(s); skipped {summary.skipped_count}. "
                        f"Linked {summary.linked_document_count} SourceDocument(s), including "
                        f"{summary.created_document_count} newly created. "
                        f"Artifact: {artifact_path.name}"
                    )
                    for warning in summary.warnings[:10]:
                        st.warning(warning)
                    st.caption(
                        "Linked documents are for manual review/extraction only. Open Documents / Text to inspect source text, "
                        "then LLM Review to deliberately run an extraction. Scanner scoring remains unchanged."
                    )
                except Exception as exc:
                    st.error(f"Accepted-candidate import failed: {exc}")

        with annotation_tabs[3]:
            st.caption("Coverage artifacts and optional simple baseline reruns use Dataset Lab rows point-in-time; no provider or LLM calls are made.")
            annotation_artifacts = list_annotation_artifacts(annotation_dir)
            artifact_options = {
                f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in annotation_artifacts
            }
            coverage_confirm = st.checkbox(
                "I understand annotation coverage/baselines are research-only and do not affect scanner scoring.",
                key="annotation_coverage_confirm",
            )
            c_left, c_right = st.columns(2)
            if c_left.button("Generate Annotation Coverage Artifact", disabled=not coverage_confirm, key="annotation_generate_coverage"):
                with st.spinner("Building point-in-time annotation coverage artifact..."):
                    coverage = build_annotation_coverage_audit(settings.database_file, dataset_id)
                    artifact_path = write_annotation_artifact(coverage, annotation_dir)
                st.success(f"Saved annotation artifact: {artifact_path.name}")
                annotation_artifacts = list_annotation_artifacts(annotation_dir)
                artifact_options = {
                    f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in annotation_artifacts
                }
            if c_right.button("Run Annotation Baselines", disabled=not coverage_confirm, key="annotation_run_baselines"):
                with st.spinner("Running annotation feature baselines..."):
                    coverage, artifact_path, summaries = run_annotation_baseline_suite(settings.database_file, dataset_id, annotation_dir)
                st.success(f"Stored {len(summaries)} annotation baseline runs. Coverage artifact: {artifact_path.name}")
                annotation_artifacts = list_annotation_artifacts(annotation_dir)
                artifact_options = {
                    f"Dataset {item.dataset_id} | {item.created_at} | {item.path.name}": item.path for item in annotation_artifacts
                }
            if not artifact_options:
                st.info("No Phase 2D-6A annotation artifacts found yet.")
            else:
                selected_annotation_artifact = st.selectbox("Annotation artifact", list(artifact_options.keys()))
                annotation_artifact = load_annotation_artifact(artifact_options[selected_annotation_artifact])
                summary = annotation_artifact.get("summary", {})
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Rows", int(annotation_artifact.get("row_count", 0) or 0))
                m2.metric("Annotation Rows", int(summary.get("annotation_rows", 0) or 0))
                m3.metric("Covered Rows", int(summary.get("rows_with_annotation_coverage", 0) or 0))
                m4.metric("Active Rate", f"{float(summary.get('annotation_active_rate', 0) or 0):.2%}")
                ann_tabs = st.tabs(["Feature Activity", "Ticker Coverage", "Fold Coverage", "DB Summary"])
                with ann_tabs[0]:
                    st.dataframe(dataframe_for_streamlit(pd.DataFrame(annotation_artifact.get("active_observation_counts", []))), width="stretch", hide_index=True)
                with ann_tabs[1]:
                    st.dataframe(dataframe_for_streamlit(pd.DataFrame(annotation_artifact.get("coverage_by_ticker", []))), width="stretch", hide_index=True)
                with ann_tabs[2]:
                    st.dataframe(dataframe_for_streamlit(pd.DataFrame(annotation_artifact.get("coverage_by_fold", []))), width="stretch", hide_index=True)
                with ann_tabs[3]:
                    st.json(annotation_artifact.get("annotation_db_summary", {}), expanded=False)

        with annotation_tabs[4]:
            counts = annotation_counts_by_ticker(settings.database_file)
            if not counts.empty:
                st.dataframe(dataframe_for_streamlit(counts), width="stretch", hide_index=True)
            recent_annotations = list_annotations(settings.database_file, limit=200)
            if recent_annotations.empty:
                st.info("No research-only annotations stored yet.")
            else:
                recent_annotations = enrich_quality_frame(recent_annotations)
                annotation_quality = quality_distribution(recent_annotations)
                aq1, aq2, aq3 = st.columns(3)
                aq1.metric("Low-specificity neutral", int(annotation_quality.get("low_specificity_neutral_count", 0)))
                aq2.metric("Routine SEC-heavy", int(annotation_quality.get("routine_sec_heavy_count", 0)))
                aq3.metric("Material non-SEC", int(annotation_quality.get("material_non_sec_count", 0)))
                display_cols = [
                    "annotation_id",
                    "ticker",
                    "event_date",
                    "available_at",
                    "event_type",
                    "sentiment_label",
                    "strength",
                    "confidence",
                    "source",
                    "source_quality",
                    "informativeness",
                    "title",
                    "research_only",
                    "scanner_scoring_effect",
                ]
                st.dataframe(
                    dataframe_for_streamlit(recent_annotations[[col for col in display_cols if col in recent_annotations.columns]]),
                    width="stretch",
                    hide_index=True,
                )

    with st.expander("Run Selected Baseline", expanded=False):
        st.caption("Runs one leakage-safe chronological/walk-forward baseline and stores metrics/predictions locally.")
        confirm = st.checkbox("I understand this does not affect scanner scoring.", key="model_lab_confirm_run")
        if st.button("Run Baseline Model", disabled=not confirm):
            with st.spinner("Running baseline model..."):
                summary = run_single_baseline_model(
                    settings.database_file,
                    dataset_id=dataset_id,
                    target_column=target_name,
                    feature_set_name=feature_set_name,
                    model_name=model_name,
                )
            st.success(f"Stored model run #{summary.model_run_id}.")
            if summary.warnings:
                st.warning("; ".join(summary.warnings[:5]))

    runs = list_model_runs(settings.database_file, limit=200)
    st.subheader("Available Model Runs")
    if runs.empty:
        st.info("No model runs stored yet.")
        return

    filters = st.columns(4)
    dataset_filter = filters[0].selectbox(
        "Filter dataset",
        ["All", *sorted(runs["dataset_id"].dropna().astype(int).unique().tolist())],
    )
    horizon_filter = filters[1].selectbox(
        "Filter horizon",
        ["All", *sorted(runs["target_horizon"].dropna().astype(str).unique().tolist())],
    )
    feature_filter = filters[2].selectbox(
        "Filter feature set",
        ["All", *sorted(runs["feature_set_name"].dropna().astype(str).unique().tolist())],
    )
    model_filter = filters[3].selectbox(
        "Filter model",
        ["All", *sorted(runs["model_name"].dropna().astype(str).unique().tolist())],
    )

    filtered = runs.copy()
    if dataset_filter != "All":
        filtered = filtered[filtered["dataset_id"].astype(int).eq(int(dataset_filter))]
    if horizon_filter != "All":
        filtered = filtered[filtered["target_horizon"].astype(str).eq(str(horizon_filter))]
    if feature_filter != "All":
        filtered = filtered[filtered["feature_set_name"].astype(str).eq(str(feature_filter))]
    if model_filter != "All":
        filtered = filtered[filtered["model_name"].astype(str).eq(str(model_filter))]

    display_cols = [
        "model_run_id",
        "dataset_id",
        "target_column",
        "target_horizon",
        "feature_set_name",
        "model_name",
        "task",
        "status",
        "created_at",
        "completed_at",
    ]
    st.dataframe(filtered[[col for col in display_cols if col in filtered.columns]], width="stretch", hide_index=True)
    if filtered.empty:
        return

    selected_run_id = int(st.selectbox("Inspect model run", filtered["model_run_id"].astype(int).tolist()))
    run_row = filtered[filtered["model_run_id"].astype(int).eq(selected_run_id)].iloc[0]
    st.caption(
        f"Run #{selected_run_id}: {run_row['model_name']} on {run_row['feature_set_name']} for {run_row['target_horizon']}. "
        "Prediction preview is informational only."
    )

    fold_metrics = _metrics_records(list_model_fold_metrics(settings.database_file, selected_run_id))
    final_metrics = _metrics_records(list_model_final_metrics(settings.database_file, selected_run_id))
    tabs = st.tabs(["Fold Metrics", "Final Test Metrics", "Prediction Preview", "Config"])
    with tabs[0]:
        if fold_metrics.empty:
            st.info("No fold metrics stored.")
        else:
            st.dataframe(fold_metrics, width="stretch", hide_index=True)
    with tabs[1]:
        if final_metrics.empty:
            st.info("No final test metrics stored.")
        else:
            st.dataframe(final_metrics, width="stretch", hide_index=True)
            raw_matches = runs[
                runs["dataset_id"].astype(int).eq(int(run_row["dataset_id"]))
                & runs["target_column"].astype(str).eq(RAW_TARGET_5_SESSION)
                & runs["feature_set_name"].astype(str).eq(str(run_row["feature_set_name"]))
                & runs["model_name"].astype(str).eq(str(run_row["model_name"]))
                & runs["status"].astype(str).eq("completed")
            ]
            if str(run_row["target_column"]) != RAW_TARGET_5_SESSION and not raw_matches.empty:
                raw_run_id = int(raw_matches.sort_values("model_run_id", ascending=False).iloc[0]["model_run_id"])
                raw_metrics = _metrics_records(list_model_final_metrics(settings.database_file, raw_run_id))
                if not raw_metrics.empty:
                    selected_metrics = final_metrics.iloc[0].to_dict()
                    raw_metric_values = raw_metrics.iloc[0].to_dict()
                    comparison_rows = []
                    for metric in [
                        "rmse",
                        "mae",
                        "oos_r2_vs_train_mean",
                        "spearman_ic",
                        "mean_daily_cross_sectional_ic",
                        "directional_accuracy",
                        "balanced_accuracy",
                        "roc_auc",
                    ]:
                        if metric in selected_metrics or metric in raw_metric_values:
                            selected_value = selected_metrics.get(metric)
                            raw_value = raw_metric_values.get(metric)
                            try:
                                delta = float(selected_value) - float(raw_value)
                            except Exception:
                                delta = None
                            comparison_rows.append(
                                {
                                    "metric": metric,
                                    "selected_target": selected_value,
                                    "raw_5_session_baseline": raw_value,
                                    "delta_selected_minus_raw": delta,
                                }
                            )
                    st.write(f"Comparison against raw 5-session run #{raw_run_id}")
                    st.dataframe(pd.DataFrame(comparison_rows), width="stretch", hide_index=True)
    with tabs[2]:
        predictions = list_model_predictions(settings.database_file, selected_run_id, limit=500)
        if predictions.empty:
            st.info("No predictions stored.")
        else:
            preview_cols = [
                "snapshot_id",
                "ticker",
                "snapshot_date",
                "target_horizon",
                "split_name",
                "fold_name",
                "y_true",
                "y_pred",
                "y_pred_label",
                "y_score",
            ]
            st.dataframe(predictions[[col for col in preview_cols if col in predictions.columns]], width="stretch", hide_index=True)
    with tabs[3]:
        st.json(
            {
                "config": json.loads(run_row.get("config_json") or "{}"),
                "split_config": json.loads(run_row.get("split_config_json") or "{}"),
                "feature_columns": json.loads(run_row.get("feature_columns_json") or "[]"),
                "warnings": json.loads(run_row.get("warnings_json") or "[]"),
            }
        )


def validation_debug_page(settings: Settings) -> None:
    st.header("Validation / Debug")
    st.caption("Audit raw data, engineered features, score components, penalties, and data-quality warnings.")

    universe = load_universe(settings.universe_file)
    if universe.empty:
        st.error("Universe is empty. Edit config/universe.csv and add at least one ticker.")
        return

    all_tickers = universe["ticker"].tolist()
    c1, c2, c3 = st.columns([2, 1, 1])
    ticker = c1.selectbox("Ticker", all_tickers, index=0)
    requested_period = c2.selectbox("Requested period", ["1y", "2y", "5y"], index=1)
    refresh = c3.checkbox("Force refresh", value=False)

    with st.spinner(f"Building validation report for {ticker}..."):
        try:
            report = load_validation_report(
                ticker,
                settings.database_file,
                settings.market_data_provider,
                requested_period,
                refresh=refresh,
                min_price=settings.scanner.min_price,
                min_avg_dollar_volume=settings.scanner.min_avg_dollar_volume,
            )
        except Exception as exc:
            st.error(f"Validation failed for {ticker}: {exc}")
            return

    metadata = report["metadata"]
    score_result = report["score_result"]
    warnings = report["warnings"]

    st.subheader("Final Score")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Ticker", report["ticker"])
    c2.metric("Final Score", f"{score_result.get('score', 0)}/100")
    c3.metric("Final Label", score_result.get("label", "n/a"))
    c4.metric("Regime", report["regime"].get("regime", "n/a"))

    if warnings:
        st.subheader("Data Quality Warnings")
        for warning in warnings:
            message = f"{warning['name']}: {warning['message']}"
            if warning["severity"] == "high":
                st.error(message)
            elif warning["severity"] == "medium":
                st.warning(message)
            else:
                st.info(message)
    else:
        st.success("No validation warnings detected for the selected ticker and period.")

    st.subheader("Raw OHLCV Metadata")
    st.caption(
        "Latest expected U.S. trading day: "
        f"{metadata.get('latest_expected_trading_day', 'n/a')} | "
        f"Calendar source: {metadata.get('calendar_source', 'n/a')}"
    )
    st.dataframe(
        pd.DataFrame([metadata]),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Latest 5 OHLCV Rows")
    st.dataframe(report["latest_rows"], width="stretch", hide_index=True)

    st.subheader("Missing-Value Counts")
    st.dataframe(
        pd.DataFrame(
            [{"field": key, "missing_count": value} for key, value in report["missing_value_counts"].items()]
        ),
        width="stretch",
        hide_index=True,
    )

    st.subheader("Feature Values")
    st.dataframe(report["feature_table"], width="stretch", hide_index=True)

    st.subheader("Score Breakdown")
    st.dataframe(report["score_breakdown_table"], width="stretch", hide_index=True)
    st.plotly_chart(
        score_breakdown_chart(score_result.get("breakdown", {})),
        width="stretch",
        key=f"validation_score_breakdown_{report['ticker']}",
    )

    st.subheader("Catalyst Inputs")
    catalyst_features = report.get("catalyst_features", {})
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Catalyst Score", f"{float(catalyst_features.get('catalyst_score', 0) or 0):.1f}/10")
    c2.metric("Catalyst Penalty", f"{float(catalyst_features.get('catalyst_penalty', 0) or 0):.1f}")
    c3.metric("Events", int(catalyst_features.get("catalyst_events_count", 0) or 0))
    c4.metric("Manual Note?", "Yes" if catalyst_features.get("has_manual_catalyst") else "No")
    if report.get("catalyst_table") is not None and not report["catalyst_table"].empty:
        st.dataframe(report["catalyst_table"], width="stretch", hide_index=True)
    else:
        st.info("No stored catalyst events for this ticker.")
    if catalyst_features.get("catalyst_warnings"):
        for warning in catalyst_features["catalyst_warnings"]:
            st.info(warning)

    st.subheader("Source Document Inputs")
    c1, c2 = st.columns(2)
    c1.metric("Stored source documents", int(report.get("document_count", 0) or 0))
    c2.metric("Future extraction ready?", "Yes" if int(report.get("document_count", 0) or 0) > 0 else "No")
    if report.get("document_table") is not None and not report["document_table"].empty:
        st.dataframe(report["document_table"], width="stretch", hide_index=True)
    else:
        st.info("No stored source documents for this ticker.")
    for warning in report.get("document_warnings", []):
        st.info(f"{warning['name']}: {warning['message']}")

    st.subheader("LLM Review / Proposal Inputs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Catalyst proposals", int(report.get("proposal_count", 0) or 0))
    c2.metric("Extraction-catalyst links", int(report.get("extraction_link_count", 0) or 0))
    c3.metric("LLM proposal score contribution", int(report.get("llm_proposal_score_contribution", 0) or 0))
    c4, c5 = st.columns(2)
    c4.metric("Publication audit rows", int(report.get("publication_count", 0) or 0))
    active_publications = 0
    if report.get("publication_table") is not None and not report["publication_table"].empty:
        active_publications = int(report["publication_table"]["publication_status"].eq("published").sum())
    c5.metric("Active publications", active_publications)
    st.info(report.get("proposal_scoring_note", "LLM proposals are review-only."))
    if report.get("proposal_table") is not None and not report["proposal_table"].empty:
        st.write("Review-only catalyst proposals")
        st.dataframe(report["proposal_table"], width="stretch", hide_index=True)
    else:
        st.info("No review-only catalyst proposals for this ticker.")
    if report.get("extraction_link_table") is not None and not report["extraction_link_table"].empty:
        st.write("Extraction-catalyst audit links")
        st.dataframe(report["extraction_link_table"], width="stretch", hide_index=True)
    else:
        st.info("No extraction-catalyst audit links for this ticker.")
    if report.get("publication_table") is not None and not report["publication_table"].empty:
        st.write("Publication/reversal audit")
        st.dataframe(report["publication_table"], width="stretch", hide_index=True)
    else:
        st.info("No active/reverted publication audit rows for this ticker.")

    st.subheader("Penalties")
    penalties = report.get("penalties", [])
    if penalties:
        st.dataframe(pd.DataFrame(penalties), width="stretch", hide_index=True)
    else:
        st.success("No penalties applied.")

    st.subheader("Explanation Reasons")
    for reason in score_result.get("reasons", []):
        st.write(f"- {reason}")

    with st.expander("Raw Debug Payload"):
        st.subheader("Raw Feature Dictionary")
        st.json(report["features"])
        st.subheader("Raw Score Result")
        st.json(score_result)


def backtesting_page(settings: Settings) -> None:
    st.header("Backtesting")
    st.caption(
        "Simplified educational backtests. Signals are evaluated after the close and current v1 entries/exits use the next trading day's close. Results may not reflect live performance."
    )
    universe = load_universe(settings.universe_file)

    strategy = st.selectbox(
        "Strategy",
        [
            "Momentum breakout",
            "Mean reversion",
            "Scanner score weekly rebalance",
            "Moving average trend",
        ],
    )
    c1, c2, c3 = st.columns(3)
    initial_capital = c1.number_input("Initial capital", min_value=1_000.0, value=float(settings.backtest.initial_capital), step=10_000.0)
    slippage_bps = c2.number_input("Slippage bps per trade", min_value=0.0, value=float(settings.backtest.default_slippage_bps), step=1.0)
    commission = c3.number_input("Commission per trade", min_value=0.0, value=float(settings.backtest.commission_per_trade), step=1.0)
    slippage = slippage_bps / 10_000

    if strategy == "Scanner score weekly rebalance":
        st.caption(
            "Portfolio metrics are rebalance-period approximations. They summarize holding periods and selected legs, not exact broker-style fills."
        )
        c1, c2 = st.columns(2)
        top_n = c1.number_input("Top N", min_value=1, max_value=20, value=5)
        max_tickers = c2.number_input("Universe size", min_value=3, max_value=len(universe), value=min(15, len(universe)))
        run = st.button("Run Top-Score Backtest")
        if run:
            tickers = universe["ticker"].tolist()[: int(max_tickers)]
            with st.spinner("Loading histories and running top-score rebalance backtest..."):
                histories = market_data.get_histories(
                    sorted(set(tickers + ["SPY", "QQQ", "IWM"])),
                    settings.database_file,
                    settings.market_data_provider,
                    "5y",
                    refresh=False,
                )
                result = backtest_top_score_strategy(
                    histories,
                    histories.get("SPY", pd.DataFrame()),
                    top_n=int(top_n),
                    initial_capital=initial_capital,
                    slippage=slippage,
                    commission=commission,
                )
            _show_backtest_result(result)
        return

    ticker = st.text_input("Ticker", value="AAPL").strip().upper()
    if strategy == "Momentum breakout":
        volume_threshold = st.slider("Volume ratio threshold", min_value=0.5, max_value=3.0, value=1.2, step=0.1)
    elif strategy == "Mean reversion":
        pullback_pct = st.slider("Pullback below 20D MA", min_value=0.01, max_value=0.20, value=0.04, step=0.01)
    run = st.button("Run Backtest")
    if not run:
        return
    if not ticker:
        st.error("Enter a ticker.")
        return

    with st.spinner(f"Loading {ticker} and SPY history..."):
        try:
            ticker_df = market_data.get_history(ticker, settings.database_file, settings.market_data_provider, "5y", refresh=False)
            spy_df = market_data.get_history("SPY", settings.database_file, settings.market_data_provider, "5y", refresh=False)
        except Exception as exc:
            st.error(str(exc))
            return
    if ticker_df.empty:
        st.error(f"No data found for {ticker}.")
        return

    if strategy == "Momentum breakout":
        result = backtest_momentum_breakout_strategy(ticker_df, spy_df, volume_threshold, initial_capital, slippage, commission)
    elif strategy == "Mean reversion":
        result = backtest_mean_reversion_strategy(ticker_df, spy_df, pullback_pct, initial_capital, slippage, commission)
    else:
        result = backtest_moving_average_strategy(ticker_df, spy_df, initial_capital, slippage, commission)
    _show_backtest_result(result)


def _show_backtest_result(result: dict[str, Any]) -> None:
    equity = result.get("equity", pd.Series(dtype=float))
    if equity.empty:
        st.warning("Backtest did not produce an equity curve. Check data availability and strategy settings.")
        return
    reporting = result.get("reporting", {})
    if reporting.get("note"):
        st.info(reporting["note"])
    st.plotly_chart(
        equity_curve_chart(equity, result.get("benchmark"), "Strategy vs SPY"),
        width="stretch",
        key=f"backtest_equity_{reporting.get('mode', 'single_strategy')}",
    )
    _metrics_grid(result.get("metrics", {}), reporting)
    trades = result.get("trades", pd.DataFrame())
    if not trades.empty:
        st.subheader(reporting.get("trades_table_label", "Trades / Turnover"))
        st.dataframe(trades, width="stretch", hide_index=True)


def trade_journal_page(settings: Settings) -> None:
    st.header("Trade Journal")
    st.caption("Local paper-trading journal stored in SQLite. No broker execution.")
    with st.form("trade_journal_form"):
        c1, c2, c3 = st.columns(3)
        ticker = c1.text_input("Ticker", value="").strip().upper()
        direction = c2.selectbox("Direction", ["long", "short", "watch only"])
        entry_date = c3.date_input("Entry date", value=date.today())
        c4, c5, c6, c7 = st.columns(4)
        entry_price = c4.number_input("Entry price", min_value=0.0, value=0.0, step=0.5)
        stop_loss = c5.number_input("Stop loss", min_value=0.0, value=0.0, step=0.5)
        target_price = c6.number_input("Target price", min_value=0.0, value=0.0, step=0.5)
        position_size = c7.number_input("Position size", min_value=0.0, value=0.0, step=100.0)
        thesis = st.text_area("Thesis")
        c8, c9 = st.columns(2)
        exit_date = c8.date_input("Exit date", value=None)
        exit_price = c9.number_input("Exit price", min_value=0.0, value=0.0, step=0.5)
        result_text = st.text_input("Result")
        lessons = st.text_area("Lessons learned")
        submitted = st.form_submit_button("Add Paper Trade")

    if submitted:
        if not ticker:
            st.error("Ticker is required.")
        else:
            storage.add_trade(
                settings.database_file,
                {
                    "ticker": ticker,
                    "direction": direction,
                    "entry_date": entry_date.isoformat() if entry_date else None,
                    "entry_price": entry_price or None,
                    "stop_loss": stop_loss or None,
                    "target_price": target_price or None,
                    "position_size": position_size or None,
                    "thesis": thesis,
                    "exit_date": exit_date.isoformat() if exit_date else None,
                    "exit_price": exit_price or None,
                    "result": result_text,
                    "lessons_learned": lessons,
                },
            )
            st.success("Paper trade added.")

    trades = storage.load_trades(settings.database_file)
    if trades.empty:
        st.info("No journal entries yet.")
    else:
        st.dataframe(trades, width="stretch", hide_index=True)


def alert_preview_page(settings: Settings) -> None:
    st.header("Alert Preview")
    st.caption("Generates human-readable alerts only. Nothing is sent in v1.")
    if "last_scan_df" in st.session_state:
        scan_df = st.session_state["last_scan_df"]
    else:
        scan_df = storage.load_latest_scan_results(settings.database_file)

    if scan_df.empty:
        st.info("Run the Daily Opportunity Scanner first.")
        return

    top_n = st.slider("Top alerts", min_value=1, max_value=min(10, len(scan_df)), value=min(5, len(scan_df)))
    for _, row in scan_df.head(top_n).iterrows():
        result = row.get("full_result")
        if not isinstance(result, dict):
            continue
        st.code(format_alert(result), language="text")
