# Phase 2D-6B-3 Annotation Sampling Protocol

This protocol was fixed before importing Phase 2D-6B-3 research annotations or running model comparisons.

## Scope

- Dataset: Dataset 49.
- Ticker universe: the 25 corporate equities in Dataset 49, excluding SPY.
- Date range: 2023-06-19 through 2026-06-18.
- Target: increase total research-only annotations from 70 into the 150-250 range.
- Minimum coverage: at least four annotations per ticker where local evidence exists.

## Source Pools

Use only existing local, auditable source pools:

- `earnings_events` rows already stored in SQLite.
- SEC archive-backed `catalysts` rows with `sec_filing_classifications`.

No new external provider calls, live LLM calls, social scraping, paid APIs, or price-move/hindsight screens are allowed.

## Selection Rules

For each ticker, select a deterministic sample:

- Up to three earnings events from `earnings_events`.
- Up to two SEC filing events from SEC archive-backed local catalyst records.
- Exclude rows that duplicate an existing annotation by ticker, event type, and event date.
- Order candidates chronologically, then prefer underrepresented deterministic categories when needed.
- Do not use model results, model errors, future returns, or later price movement to choose events.

## Labeling Rules

Earnings sentiment is derived only from provider EPS surprise fields available in `earnings_events`:

- `positive` if EPS surprise percent is at least 5%.
- `negative` if EPS surprise percent is at most -5%.
- `mixed` if EPS surprise percent is present but between -5% and 5%.
- `neutral` if EPS surprise percent is unavailable.

SEC filing annotations are conservative:

- Routine periodic/current-event/ownership filings are `neutral`.
- Structured-note or uncertain financing/prospectus activity is `mixed`, not automatically dilution.
- Unknown classifications stay low-strength and neutral/mixed.
- SEC filing type alone is not used to infer positive or negative alpha.

## Balance Targets

After import, target approximate total sentiment mix:

- Positive: 30-45%.
- Neutral/mixed: 25-40%.
- Negative/risk: 20-35%.

The protocol may miss the negative/risk target if the local source pool is earnings/SEC-heavy. Any miss must be reported rather than repaired using hindsight.

## Event Categories

Expected event-type coverage includes:

- `earnings`
- `sec_filing`
- Existing prior annotations may also cover product launches, corporate actions, legal/regulatory events, macro-sensitive events, financing, and news.

This phase does not create active catalysts and does not alter scanner scoring.

## Point-In-Time Rules

- `available_at` controls feature activation.
- `event_date` must not be later than the information availability timestamp unless the local source explicitly represents a future-scheduled event; this protocol does not use future-scheduled events.
- Research annotations must keep `research_only = true` and `scanner_scoring_effect = 0`.
- Future returns and outcome labels must not be used in annotation creation, selection, or sentiment.

## Acceptance Checks

- Import produces no parser errors.
- Future-availability violations equal zero in the quality harness.
- Scanner snapshot comparison passes with no score, label, risk, or catalyst-score diffs.
- Dataset 49 manifest check passes.
- Simple baselines are compared against model runs 145 and 283.
