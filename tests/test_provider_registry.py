from __future__ import annotations

from datetime import date

import pandas as pd

from src.annotations.news_csv_provider import parse_candidate_import_frame
from src.annotations.provider_registry import (
    COMPANY_IR_PROVIDER_NAME,
    MANUAL_CSV_PROVIDER_NAME,
    REDDIT_PROVIDER_NAME,
    RSS_MANUAL_URL_PROVIDER_NAME,
    CompanyIrPressReleaseResearchEventProvider,
    DisabledRedditApiProvider,
    DisabledRssManualUrlProvider,
    ManualCsvResearchEventProvider,
    build_provider_readiness_report,
    default_provider_registry,
)
from src.annotations.news_repository import accept_candidate, import_accepted_candidates, list_candidates, stage_candidates
from src.annotations.repository import list_annotations
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
    assert statuses[COMPANY_IR_PROVIDER_NAME].enabled is True
    assert statuses[COMPANY_IR_PROVIDER_NAME].requires_api_key is False
    assert statuses[COMPANY_IR_PROVIDER_NAME].source_quality_default == "official_company"
    assert statuses[COMPANY_IR_PROVIDER_NAME].network_calls_would_occur is False
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


def test_company_ir_provider_requires_user_supplied_url_and_official_source_quality() -> None:
    valid = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "event_date": "2024-01-03",
                "available_at": "2024-01-03T14:00:00Z",
                "event_type": "product_launch",
                "title": "Company IR product event",
                "source": "company_ir",
                "source_url": "https://example.com/ir/product-event",
                "evidence_text": "Company announced a product event.",
                "sentiment_label": "positive",
                "strength": 6,
                "confidence": 0.75,
            }
        ]
    )
    missing_url = valid.drop(columns=["source_url"])
    wrong_quality = valid.assign(source_quality="credible_news")
    wrong_provider = valid.assign(provider_name="manual_csv")

    provider = CompanyIrPressReleaseResearchEventProvider(valid)
    events = provider.fetch_candidates("AAA", date(2024, 1, 1), date(2024, 1, 31))

    assert len(events) == 1
    assert events[0].provider == COMPANY_IR_PROVIDER_NAME
    assert events[0].provider_metadata["source_quality"] == "official_company"
    assert events[0].provider_metadata["network_calls_would_occur"] is False
    assert provider.provider_status().network_calls_would_occur is False
    assert CompanyIrPressReleaseResearchEventProvider(missing_url).fetch_candidates("AAA", date(2024, 1, 1), date(2024, 1, 31)) == []
    assert CompanyIrPressReleaseResearchEventProvider(wrong_quality).fetch_candidates("AAA", date(2024, 1, 1), date(2024, 1, 31)) == []
    assert CompanyIrPressReleaseResearchEventProvider(wrong_provider).fetch_candidates("AAA", date(2024, 1, 1), date(2024, 1, 31)) == []


def test_company_ir_candidates_import_only_as_research_annotations(tmp_path) -> None:
    db_path = tmp_path / "alpha_lab.db"
    frame = pd.DataFrame(
        [
            {
                "ticker": "AAA",
                "event_date": "2024-01-03",
                "available_at": "2024-01-03T14:00:00Z",
                "event_type": "partnership",
                "title": "Company IR partnership event",
                "source": "company_ir",
                "source_url": "https://example.com/ir/partnership",
                "evidence_text": "Company announced a partnership.",
                "sentiment_label": "positive",
                "strength": 6,
                "confidence": 0.8,
                "source_quality": "official_company",
                "provider_name": COMPANY_IR_PROVIDER_NAME,
            }
        ]
    )
    provider = CompanyIrPressReleaseResearchEventProvider(frame)
    staged = stage_candidates(db_path, provider.fetch_candidates("AAA", date(2024, 1, 1), date(2024, 1, 31)))
    accept_candidate(db_path, staged[0].candidate_id)
    summary = import_accepted_candidates(db_path)
    annotations = list_annotations(db_path)
    candidates = list_candidates(db_path, limit=None)

    assert summary.imported_count == 1
    assert int(annotations.iloc[0]["research_only"]) == 1
    assert int(annotations.iloc[0]["scanner_scoring_effect"]) == 0
    assert candidates.iloc[0]["provider"] == COMPANY_IR_PROVIDER_NAME
    assert candidates.iloc[0]["status"] == "imported"


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
    assert report["enabled_provider_count"] == 2
    assert result.status == "passed"
    assert result.summary["network_calls_would_occur"] is False
