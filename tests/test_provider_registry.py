from __future__ import annotations

from datetime import date

import pandas as pd

from src.annotations.news_csv_provider import parse_candidate_import_frame
from src.annotations.provider_registry import (
    MANUAL_CSV_PROVIDER_NAME,
    REDDIT_PROVIDER_NAME,
    RSS_MANUAL_URL_PROVIDER_NAME,
    DisabledRedditApiProvider,
    DisabledRssManualUrlProvider,
    ManualCsvResearchEventProvider,
    build_provider_readiness_report,
    default_provider_registry,
)
from src.quality.harness import check_provider_readiness


def _candidate_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "event_date": "2024-01-03",
                "available_at": "2024-01-03T14:00:00Z",
                "event_type": "news",
                "title": "Manual CSV event",
                "summary": "Synthetic provider-readiness row.",
                "source": "manual",
                "source_url": "https://example.com/manual-csv-event",
                "evidence_text": "Company announced a synthetic event.",
                "sentiment_label": "positive",
                "strength": 6,
                "confidence": 0.75,
            }
        ]
    )


def test_default_provider_registry_is_compliance_safe_without_api_keys() -> None:
    statuses = {provider.provider_status().provider_name: provider.provider_status() for provider in default_provider_registry()}

    assert statuses[MANUAL_CSV_PROVIDER_NAME].enabled is True
    assert statuses[MANUAL_CSV_PROVIDER_NAME].requires_api_key is False
    assert statuses[MANUAL_CSV_PROVIDER_NAME].network_calls_would_occur is False
    assert statuses[RSS_MANUAL_URL_PROVIDER_NAME].enabled is False
    assert statuses[RSS_MANUAL_URL_PROVIDER_NAME].network_calls_would_occur is False
    assert statuses[REDDIT_PROVIDER_NAME].enabled is False
    assert statuses[REDDIT_PROVIDER_NAME].requires_api_key is True
    assert statuses[REDDIT_PROVIDER_NAME].network_calls_would_occur is False


def test_manual_csv_provider_adds_metadata_defaults_and_fetches_locally() -> None:
    frame = _candidate_frame()
    result = parse_candidate_import_frame(frame)
    provider = ManualCsvResearchEventProvider(frame)
    events = provider.fetch_candidates("AAA", date(2024, 1, 1), date(2024, 1, 31))

    assert not result.errors
    assert result.candidates[0].provider == MANUAL_CSV_PROVIDER_NAME
    assert result.candidates[0].provider_metadata["provider_name"] == MANUAL_CSV_PROVIDER_NAME
    assert result.candidates[0].provider_metadata["provider_type"] == "manual_csv"
    assert result.candidates[0].provider_metadata["network_calls_would_occur"] is False
    assert len(events) == 1
    assert events[0].ticker == "AAA"


def test_disabled_stub_providers_return_no_candidates_without_network_calls() -> None:
    rss = DisabledRssManualUrlProvider()
    reddit = DisabledRedditApiProvider()

    assert rss.fetch_candidates("AAA", date(2024, 1, 1), date(2024, 1, 31)) == []
    assert reddit.fetch_candidates("AAA", date(2024, 1, 1), date(2024, 1, 31)) == []
    assert rss.provider_status().network_calls_would_occur is False
    assert reddit.provider_status().network_calls_would_occur is False


def test_provider_readiness_report_and_harness_are_non_networking_and_non_scoring() -> None:
    report = build_provider_readiness_report()
    result = check_provider_readiness()

    assert report["network_calls_would_occur"] is False
    assert report["guardrails"]["scanner_scoring_effect"] == 0
    assert report["guardrails"]["active_catalyst_creation"] is False
    assert result.status == "passed"
    assert result.summary["network_calls_would_occur"] is False
