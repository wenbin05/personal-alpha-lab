# Annotation Source Provider Readiness

This roadmap is for research-only event/annotation coverage. It does not authorize scraping, live LLM calls, active catalyst creation, scanner scoring changes, or trading recommendations.

## Source-Quality Taxonomy

- `official_company`: company investor relations, newsroom, earnings releases, official press releases.
- `regulator`: DOJ, FTC, SEC press releases, FDA, CFTC, CFPB, and other regulator pages.
- `exchange_or_index_provider`: NYSE, Nasdaq, S&P Dow Jones Indices, index inclusion/removal announcements.
- `sec_archive`: SEC EDGAR filing metadata or archive links.
- `credible_news`: manually sourced reputable public news pages.
- `manual_note`: local/manual CSV notes, curated research notes, demo rows.
- `unknown`: source quality cannot be determined from the supplied metadata.

## Event-Informativeness Taxonomy

- `material_high`: likely material public event, such as legal/regulatory action, financing, corporate action, guidance, or other strong high-confidence event.
- `material_medium`: meaningful but less clearly decisive event, such as earnings, product launch, partnership, management change, analyst/news item.
- `routine_low`: routine metadata or filing, especially neutral SEC rows without material interpretation.
- `duplicate_theme`: duplicate or near-duplicate event theme already represented elsewhere.
- `low_specificity`: weak, vague, neutral/unknown, or insufficiently evidenced row.

## Provider Options

| Provider/source | Compliance risk | Effort | Likely signal value | PIT `available_at` support | Fits staging flow? | Status |
|---|---:|---:|---:|---:|---:|---|
| Manual CSV | Low | Low | Medium when protocol-driven | Strong, user supplies timestamp | Yes | Current best path |
| Company IR / press release manual import | Low | Medium | Medium-high for material company events | Strong if release timestamp is captured | Yes | Recommended next |
| RSS/manual URL import | Medium | Medium | Medium | Good if feed timestamp is retained | Yes, if manually supplied and rate-limited | Safe later phase |
| Reddit API | Medium-high | Medium-high | Unknown/noisy | Weak to medium, depends on post timestamp and API terms | Possible, but not available now | Deferred |
| Paid news provider | Low-medium | Medium | Medium-high depending provider | Strong | Yes | Deferred until budget/API approved |
| Stocktwits/social-style provider | Medium-high | Medium-high | Unknown/noisy | Medium | Possible, but needs compliance review | Deferred |

## Provider Registry Contract

Research-event sources are represented by a provider registry before any network integration is enabled. A provider entry should expose:

- `provider_name`
- `provider_type`
- `enabled`
- `requires_api_key`
- `rate_limit_notes`
- `compliance_notes`
- `supports_point_in_time_available_at`
- `supports_backfill`
- `source_quality_default`
- `allowed_usage`
- `config_status`
- `next_action_required`

The common provider interface is:

- `validate_config()`
- `fetch_candidates(ticker, start_date, end_date)`
- `normalize_candidate(raw_item)`
- `provider_status()`

Current registry state:

- `manual_csv`: enabled, local CSV/manual staging only, no API key, no network calls.
- `company_ir_press_release`: enabled, strict user-supplied company IR/newsroom/press-release CSV rows only, source URL required, no network calls.
- `rss_manual_url_stub`: disabled placeholder; future user-supplied RSS/manual URLs only, no crawling.
- `reddit_api_placeholder`: disabled placeholder; official API access required, no scraping.

Use the quality harness command below to audit provider readiness without making provider calls:

```bash
.venv/bin/python scripts/quality_harness.py provider-readiness
```

## Recommendation

Prioritize a protocol-driven manual/company-IR workflow before social data. The next useful source expansion should target non-SEC, material public events with clear timestamps and source URLs:

- official company product/customer/partnership announcements,
- regulator/legal actions,
- financing/corporate-action events,
- index-provider announcements,
- selectively curated credible news rows.

Reddit or social-style data should wait until API access, terms-of-use constraints, timestamp semantics, deduplication, and toxicity/noise controls are designed explicitly.

## Current Guardrails

- Imported rows remain research-only annotations.
- `scanner_scoring_effect` must stay `0`.
- Active catalysts are not created from source candidates.
- Dataset 49 comparisons are exploratory/dev only.
- Dataset 50 is an immature `holdout_candidate` and must not be modeled yet.
