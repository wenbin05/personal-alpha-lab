# Holdout-Safe Annotation Expansion Protocol

Use this workflow before adding more research-only annotations or comparing annotation-enhanced models.

## Holdout Status

- Dataset 49's final test period has been repeatedly inspected across Phase 2D model and annotation iterations.
- Future improvements on Dataset 49 are useful for development, debugging, and exploratory comparison only.
- Do not claim signal robustness from Dataset 49 final-test metrics after Phase 2D-6B-8.
- Before making a robustness claim, create a fresh untouched holdout dataset/window or a locked walk-forward confirmation protocol.

## Evaluation Regimes

Use these labels in reports and artifacts:

- `exploratory_dev`: used during iterative feature, annotation, target, or model research.
- `holdout_candidate`: locked before evaluation, but not yet accepted as a final untouched holdout.
- `final_holdout`: untouched until final confirmation, with no feature/annotation/model selection based on its metrics.

## Fresh Holdout Strategy

Preferred order:

1. Time-forward holdout after the exploratory dataset period.
2. Hybrid time-forward plus ticker-held-out confirmation.
3. Ticker-held-out split only when future market data is unavailable.

Dataset 49 covers through 2026-06-23 and has been repeatedly inspected. The first fresh setup after Phase 2D-6B-8 should therefore be a time-forward `holdout_candidate` using only later cached sessions. It is not a `final_holdout` until:

- the date range is locked before evaluation,
- enough future sessions exist for the target horizon labels,
- feature and annotation rules are frozen,
- no model or annotation choices are made from its metrics,
- the opening/evaluation event is recorded in an audit artifact.

If a time-forward candidate has incomplete 5-session or 20-session labels, use it for manifest/leakage/runtime checks only. Do not evaluate models on it and do not claim robustness.

## Holdout Maturity Thresholds

Use `python scripts/quality_harness.py holdout-status --dataset-id <id>` before running any model evaluation on a holdout candidate.

Default maturity gates:

- Protocol validation: at least 1 row, at least 1 ticker, and no manifest/leakage violations.
- Holdout-candidate sanity check: at least 250 rows, at least 10 tickers, at least 20 labeled 5-session dates, and at least 50% 5-session label coverage.
- Final 5-session holdout evaluation: dataset must be `holdout_candidate`, at least 1,000 rows, at least 20 tickers, at least 60 labeled 5-session dates, at least 80% 5-session label coverage, and no manifest/leakage violations.
- Final 20-session holdout evaluation: all final 5-session gates plus at least 60 labeled 20-session dates and at least 80% 20-session label coverage.

These thresholds are defaults, not proof of statistical power. They are intended to prevent accidental final-holdout use while labels are obviously incomplete.

## Safe Extension Workflow

Time-forward holdout candidates should be extended only through a new dataset build:

1. Run `holdout-status` and inspect the cache-only extension plan.
2. Keep provider fetching disabled unless the user explicitly requests a data-refresh phase.
3. Use only dates after the parent dataset's requested end date.
4. Preserve point-in-time feature rules, annotation `available_at` filtering, SEC acceptance timing, and earnings availability timing.
5. Store the new dataset with parent/reference metadata and a new version string.
6. Do not overwrite Dataset 49, Dataset 50, or prior holdout candidates.
7. Re-run manifest/leakage, scanner invariance, annotation coverage, and health checks.

Promotion from `holdout_candidate` to `final_holdout` requires explicit user confirmation and passing maturity gates. Promotion does not itself authorize repeated model evaluation.

## Final Holdout Access Rules

- A `final_holdout` can be opened only once per locked model/feature/annotation protocol.
- Opening must be documented with timestamp, dataset ID, hash, target, feature set, model configuration, and reviewer note.
- After opening, the result is final evidence for that protocol only.
- Any follow-up tuning creates a new development protocol and requires a different untouched holdout for the next final claim.
- Do not repeatedly compare candidate models against `final_holdout`.
- Do not select annotations based on `final_holdout` target behavior, errors, or coverage gaps.

Known Dataset 49 runs that are `exploratory_dev`:

- 145
- 283
- 301
- 319
- 337
- 370

## Development Workflow

1. Define annotation rules before collecting rows.
2. Use development folds for feature and annotation design.
3. Preserve purge and embargo settings for the target horizon.
4. Keep labels, audit columns, identifiers, timestamps, raw JSON, hashes, and workflow statuses out of model features.
5. Do not inspect final-holdout metrics while selecting annotations, feature sets, targets, or model settings.
6. Treat Dataset 49 model comparisons as exploratory once final-test metrics have influenced the research direction.

## Annotation Expansion Plan Template

Pre-register these fields before import:

- Ticker list.
- Date range.
- Event-type target mix.
- Sentiment target mix.
- Source rules.
- Maximum rows per ticker.
- Exclusion rules.
- Confirmation that future price movement and model results were not used.

Recommended event-type mix:

- Legal/regulatory: 20-30%.
- Financing or corporate action: 15-25%.
- Product, customer, or partnership events: 25-35%.
- Earnings or guidance: 10-20%.
- Neutral material updates: 10-20%.

Recommended sentiment mix:

- Positive: 30-45%.
- Neutral or mixed: 25-40%.
- Negative/risk: 20-35%.

## Source Rules

Allowed:

- Company investor relations/newsroom.
- Official press releases.
- SEC archive links.
- Regulator pages.
- Index-provider announcements.
- Manually verified credible public news.

Not allowed unless a future phase explicitly approves:

- Reddit, X/Twitter, forum, or social scraping.
- Website crawling.
- Paid provider data.
- Live LLM extraction.
- Rumor-only events.
- Hindsight price-move reasoning.

## Reporting Rules

- Mark Dataset 49 results as `exploratory_dev`.
- Report scanner invariance.
- Report active-catalyst invariance.
- Report future-availability violations.
- Report annotation coverage by ticker, event type, sentiment, source, and fold.
- Separate development evidence from final holdout evidence.
