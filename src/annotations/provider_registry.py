from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
from typing import Any, Protocol

import pandas as pd

from src.annotations.news_csv_provider import (
    COMPANY_IR_PRESS_RELEASE_PROVIDER_NAME,
    parse_candidate_import_frame,
    parse_company_ir_press_release_frame,
)
from src.annotations.news_events import ResearchEventAnnotationCandidate


PROVIDER_REGISTRY_VERSION = "research_event_provider_registry_v1"
MANUAL_CSV_PROVIDER_NAME = "manual_csv"
COMPANY_IR_PROVIDER_NAME = COMPANY_IR_PRESS_RELEASE_PROVIDER_NAME
RSS_MANUAL_URL_PROVIDER_NAME = "rss_manual_url_stub"
REDDIT_PROVIDER_NAME = "reddit_api_placeholder"


@dataclass(frozen=True)
class ResearchEventProviderConfig:
    provider_name: str
    provider_type: str
    enabled: bool
    requires_api_key: bool
    rate_limit_notes: str
    compliance_notes: str
    supports_point_in_time_available_at: bool
    supports_backfill: bool
    source_quality_default: str
    allowed_usage: str
    config_status: str
    next_action_required: str
    network_calls_enabled: bool = False
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ResearchEventProviderStatus:
    provider_name: str
    provider_type: str
    enabled: bool
    config_status: str
    requires_api_key: bool
    supports_point_in_time_available_at: bool
    supports_backfill: bool
    source_quality_default: str
    allowed_usage: str
    rate_limit_notes: str
    compliance_notes: str
    next_action_required: str
    network_calls_would_occur: bool
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ResearchEventSourceProvider(Protocol):
    config: ResearchEventProviderConfig

    def validate_config(self) -> ResearchEventProviderStatus:
        ...

    def fetch_candidates(self, ticker: str, start_date: date, end_date: date) -> list[ResearchEventAnnotationCandidate]:
        ...

    def normalize_candidate(self, raw_item: Any) -> ResearchEventAnnotationCandidate:
        ...

    def provider_status(self) -> ResearchEventProviderStatus:
        ...


class BaseResearchEventProvider:
    config: ResearchEventProviderConfig

    def validate_config(self) -> ResearchEventProviderStatus:
        config = self.config
        network_calls = bool(config.enabled and config.network_calls_enabled)
        warnings = list(config.warnings)
        if network_calls:
            warnings.append("This provider would make network calls if fetch_candidates is invoked.")
        return ResearchEventProviderStatus(
            provider_name=config.provider_name,
            provider_type=config.provider_type,
            enabled=config.enabled,
            config_status=config.config_status,
            requires_api_key=config.requires_api_key,
            supports_point_in_time_available_at=config.supports_point_in_time_available_at,
            supports_backfill=config.supports_backfill,
            source_quality_default=config.source_quality_default,
            allowed_usage=config.allowed_usage,
            rate_limit_notes=config.rate_limit_notes,
            compliance_notes=config.compliance_notes,
            next_action_required=config.next_action_required,
            network_calls_would_occur=network_calls,
            warnings=warnings,
        )

    def provider_status(self) -> ResearchEventProviderStatus:
        return self.validate_config()

    def fetch_candidates(self, ticker: str, start_date: date, end_date: date) -> list[ResearchEventAnnotationCandidate]:
        return []

    def normalize_candidate(self, raw_item: Any) -> ResearchEventAnnotationCandidate:
        raise NotImplementedError(f"{self.config.provider_name} does not normalize arbitrary raw provider rows yet.")


class ManualCsvResearchEventProvider(BaseResearchEventProvider):
    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.frame = frame
        self.config = ResearchEventProviderConfig(
            provider_name=MANUAL_CSV_PROVIDER_NAME,
            provider_type="manual_csv",
            enabled=True,
            requires_api_key=False,
            rate_limit_notes="No provider rate limit; user supplies local CSV rows.",
            compliance_notes=(
                "Manual/user-supplied CSV only. Rows are staged for review and import as research-only annotations; "
                "no active catalysts or scanner scoring changes are created."
            ),
            supports_point_in_time_available_at=True,
            supports_backfill=True,
            source_quality_default="manual_note",
            allowed_usage="Local CSV/manual candidate staging only.",
            config_status="ready",
            next_action_required="Upload a user-curated candidate CSV, then review staged rows before import.",
            network_calls_enabled=False,
        )

    def fetch_candidates(self, ticker: str, start_date: date, end_date: date) -> list[ResearchEventAnnotationCandidate]:
        if self.frame is None:
            return []
        result = parse_candidate_import_frame(self.frame, provider=self.config.provider_name)
        ticker_upper = ticker.strip().upper()
        return [
            candidate
            for candidate in result.candidates
            if candidate.ticker == ticker_upper and start_date <= candidate.event_date <= end_date
        ]

    def normalize_candidate(self, raw_item: Any) -> ResearchEventAnnotationCandidate:
        frame = pd.DataFrame([raw_item])
        result = parse_candidate_import_frame(frame, provider=self.config.provider_name)
        if result.errors or not result.candidates:
            message = "; ".join(error.message for error in result.errors) or "Candidate row could not be normalized."
            raise ValueError(message)
        return result.candidates[0]


class CompanyIrPressReleaseResearchEventProvider(BaseResearchEventProvider):
    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.frame = frame
        self.config = ResearchEventProviderConfig(
            provider_name=COMPANY_IR_PROVIDER_NAME,
            provider_type="company_ir_press_release",
            enabled=True,
            requires_api_key=False,
            rate_limit_notes="No provider rate limit; user supplies CSV rows or source URLs. No network calls are made.",
            compliance_notes=(
                "Strict user-supplied company IR / press-release workflow. Requires source_url, stages candidates "
                "for review, imports accepted rows only as research-only annotations, and never crawls or discovers sites."
            ),
            supports_point_in_time_available_at=True,
            supports_backfill=True,
            source_quality_default="official_company",
            allowed_usage="User-supplied company IR/newsroom/press-release rows only.",
            config_status="ready_local_only",
            next_action_required=(
                "Upload a company IR / press-release candidate CSV with source_url values, then review staged rows."
            ),
            network_calls_enabled=False,
        )

    def fetch_candidates(self, ticker: str, start_date: date, end_date: date) -> list[ResearchEventAnnotationCandidate]:
        if self.frame is None:
            return []
        result = parse_company_ir_press_release_frame(self.frame)
        ticker_upper = ticker.strip().upper()
        return [
            candidate
            for candidate in result.candidates
            if candidate.ticker == ticker_upper and start_date <= candidate.event_date <= end_date
        ]

    def normalize_candidate(self, raw_item: Any) -> ResearchEventAnnotationCandidate:
        frame = pd.DataFrame([raw_item])
        result = parse_company_ir_press_release_frame(frame)
        if result.errors or not result.candidates:
            message = "; ".join(error.message for error in result.errors) or "Company IR row could not be normalized."
            raise ValueError(message)
        return result.candidates[0]


class DisabledRssManualUrlProvider(BaseResearchEventProvider):
    def __init__(self) -> None:
        self.config = ResearchEventProviderConfig(
            provider_name=RSS_MANUAL_URL_PROVIDER_NAME,
            provider_type="rss_manual_url",
            enabled=False,
            requires_api_key=False,
            rate_limit_notes="Disabled. Future phase must use conservative request pacing and user-supplied feeds only.",
            compliance_notes=(
                "Stub only. Future implementation may use user-supplied RSS/manual URLs; no crawling, scraping, "
                "robots bypassing, or automatic discovery is allowed."
            ),
            supports_point_in_time_available_at=True,
            supports_backfill=False,
            source_quality_default="credible_news",
            allowed_usage="Disabled placeholder for future explicit user-supplied RSS/manual URL ingestion.",
            config_status="disabled",
            next_action_required="Design explicit allowlist, pacing, and review workflow before enabling.",
            network_calls_enabled=False,
            warnings=["Disabled by default; this phase intentionally makes no RSS/manual URL network calls."],
        )


class DisabledRedditApiProvider(BaseResearchEventProvider):
    def __init__(self) -> None:
        self.config = ResearchEventProviderConfig(
            provider_name=REDDIT_PROVIDER_NAME,
            provider_type="reddit_api",
            enabled=False,
            requires_api_key=True,
            rate_limit_notes="Disabled. Would require official Reddit API limits and credentials in a future phase.",
            compliance_notes=(
                "Placeholder only. Reddit scraping is not allowed; official API access and compliance review are required "
                "before any future implementation."
            ),
            supports_point_in_time_available_at=True,
            supports_backfill=False,
            source_quality_default="unknown",
            allowed_usage="Disabled placeholder; no calls and no scraping.",
            config_status="blocked_missing_official_api_access",
            next_action_required="Obtain and review official Reddit API access before considering this provider.",
            network_calls_enabled=False,
            warnings=["Reddit access is not configured and does not block manual/provider-ready workflows."],
        )


def default_provider_registry() -> list[ResearchEventSourceProvider]:
    return [
        ManualCsvResearchEventProvider(),
        CompanyIrPressReleaseResearchEventProvider(),
        DisabledRssManualUrlProvider(),
        DisabledRedditApiProvider(),
    ]


def provider_status_rows(providers: list[ResearchEventSourceProvider] | None = None) -> list[dict[str, Any]]:
    return [provider.provider_status().to_dict() for provider in (providers or default_provider_registry())]


def build_provider_readiness_report(providers: list[ResearchEventSourceProvider] | None = None) -> dict[str, Any]:
    rows = provider_status_rows(providers)
    enabled = [row for row in rows if row["enabled"]]
    blocked = [row for row in rows if not row["enabled"] or str(row["config_status"]).startswith("blocked")]
    network_calls = [row for row in rows if row["network_calls_would_occur"]]
    return {
        "schema_version": PROVIDER_REGISTRY_VERSION,
        "configured_provider_count": len(rows),
        "enabled_provider_count": len(enabled),
        "blocked_or_disabled_provider_count": len(blocked),
        "requires_api_key_count": sum(1 for row in rows if row["requires_api_key"]),
        "network_calls_would_occur": bool(network_calls),
        "network_calling_providers": [row["provider_name"] for row in network_calls],
        "providers": rows,
        "guardrails": {
            "scanner_scoring_effect": 0,
            "active_catalyst_creation": False,
            "model_runs_created": False,
            "dataset_50_evaluated": False,
            "external_network_calls": False,
        },
    }
