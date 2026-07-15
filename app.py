from __future__ import annotations

import streamlit as st

from src.config import load_settings
from src.data.storage import init_db
from src.ui.pages import (
    alert_preview_page,
    backtesting_page,
    catalyst_center_page,
    documents_text_page,
    dataset_lab_page,
    llm_review_page,
    market_regime_page,
    model_lab_page,
    options_research_page,
    scanner_page,
    shadow_research_page,
    ticker_research_page,
    trade_journal_page,
    validation_debug_page,
)
from src.utils.dates import market_session_label, now_in_user_tz
from src.utils.logging import setup_logging


def main() -> None:
    setup_logging()
    settings = load_settings()
    init_db(settings.database_file)

    st.set_page_config(
        page_title="Personal Alpha Lab",
        page_icon="PAL",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.sidebar.title("Personal Alpha Lab")
    st.sidebar.caption("U.S. equities alpha research assistant")
    st.sidebar.warning("For research and paper trading only. Not financial advice.")
    st.sidebar.write(f"Singapore time: {now_in_user_tz().strftime('%Y-%m-%d %H:%M')}")
    st.sidebar.write(f"Session: {market_session_label()}")
    st.sidebar.write(f"Data provider: {settings.market_data_provider}")
    st.sidebar.write(f"Database: {settings.database_path}")

    page = st.sidebar.radio(
        "Page",
        [
            "Market Regime",
            "Daily Scanner",
            "Ticker Research",
            "Catalyst Center",
            "Documents / Text",
            "LLM Review",
            "Dataset Lab",
            "Model Lab",
            "Shadow Research",
            "Options Research",
            "Backtesting",
            "Validation / Debug",
            "Trade Journal",
            "Alert Preview",
        ],
    )

    if page == "Market Regime":
        market_regime_page(settings)
    elif page == "Daily Scanner":
        scanner_page(settings)
    elif page == "Ticker Research":
        ticker_research_page(settings)
    elif page == "Catalyst Center":
        catalyst_center_page(settings)
    elif page == "Documents / Text":
        documents_text_page(settings)
    elif page == "LLM Review":
        llm_review_page(settings)
    elif page == "Dataset Lab":
        dataset_lab_page(settings)
    elif page == "Model Lab":
        model_lab_page(settings)
    elif page == "Shadow Research":
        shadow_research_page(settings)
    elif page == "Options Research":
        options_research_page(settings)
    elif page == "Backtesting":
        backtesting_page(settings)
    elif page == "Validation / Debug":
        validation_debug_page(settings)
    elif page == "Trade Journal":
        trade_journal_page(settings)
    else:
        alert_preview_page(settings)


if __name__ == "__main__":
    main()
