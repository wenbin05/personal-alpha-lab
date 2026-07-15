# Current State Handoff

## Core System Concept

Personal Alpha Lab is a point-in-time U.S. equity research platform combining technical and regime features, SEC and earnings metadata, manually reviewed alternative/event data, LLM-assisted document extraction, and auditable modeling and holdout governance.

It is research and paper-trading software, not live execution.

## Current Status

- Dataset 49 is `exploratory_dev`; further annotation tuning is frozen.
- Dataset 50 is an immature `holdout_candidate` with 104 rows, one 5-session label, and zero 20-session labels.
- Do not model the holdout until its maturity thresholds pass.
- No deep learning or nonlinear modeling has been added.

## Best Exploratory Results

- Run 283 has the best RMSE, out-of-sample R2, and directional accuracy.
- Run 319 has the best Spearman IC.
- Later compact annotation features reduced some collinearity but did not improve robustness.
- All Dataset 49 results are exploratory development evidence only.

## Current Database Counts

- Research annotations: 424
- Research candidates: 293
- Source documents: 12
- Extractions: 12
- Active catalysts: 30,671

## Provider Status

- `manual_csv`: enabled, local only
- `company_ir_press_release`: enabled, local only
- `rss_manual_url_stub`: disabled
- `reddit_api_placeholder`: disabled
- No enabled provider makes network calls.

## Company IR Document Coverage

Phase 2E-5A audited the production database through read-only SQLite access:

- Company IR candidates: 38 across 8 tickers
- Candidates resolving to research annotations: 38 (100%)
- Candidates with linked SourceDocuments: 0 (0%)
- Missing documents: 38 (100%)
- Broken or reused document links: 0
- Manual enrichment queue: 38 rows at `data/processed/phase2e5a_company_ir_enrichment_queue.csv`
- Top missing-document tickers: NVDA 6; AAPL, AMD, COIN, META, and TSLA 5 each; AMZN 4; MSFT 3

The queue requires manually supplied source text. It does not fetch URLs, fabricate text, create documents, run extraction, or affect scanner scoring.

Phase 2E-5B1 added a dry-run-first company IR document-enrichment importer and workflow. It can validate a manually completed queue, plan document creation/reuse/linkage, and requires explicit `--apply` with a timestamped database backup. No production enrichment was applied in Phase 2E-5B1, so production coverage remains 0 of 38 linked documents.

## Frozen Shadow Artifact

Phase 3A-0B defines `shadow_ridge_technical_v1`, a new executable technical Ridge artifact derived from the accepted Dataset 49 contract. Runs 145 and 283 are design references only; the artifact is explicitly not equivalent to either historical experiment. Its required contract is Dataset 49 with 19,495 exact stored rows and hash `e9523e1134b7eb32b142cb628d51bde76d5a6d139f4be2aba2545f3ca4416184`, using only the ordered `technical_core` features and no Dataset 50 data.

Both reproducibility gates passed and the immutable local artifact was created:

- Artifact ID: `shadow_ridge_technical_v1_1ee8071db3f0`
- Artifact checksum: `4bb794ba7c9f1d9edb2b5430b58c9e6a9f692aad80ba16fab6959f298ac7da45`
- Feature manifest hash: `05d60960a2a6917fe3b2aa9acbe6167b351324860c4cb1e33bebf2c0a66c256d`
- Training rows: 19,339 through 2026-06-12; 156 missing-target rows excluded
- Replay: passed on 12 rows with maximum absolute difference `1.0408340855860843e-16`

The fit preserved the frozen no-scaling preprocessing contract and surfaced an ill-conditioned-matrix warning; no tuning or contract change was made. The artifact remains `frozen_exploratory` / `exploratory_shadow`. It does not create daily shadow predictions, change scanner scoring, or establish validated alpha. Phase 3A-1 may use this artifact only after retaining its strict feature-order, integrity, and prospective-research guardrails.

Phase 3A-1A adds a cache-only, dry-run-first path for immutable daily shadow predictions. It uses only the frozen artifact's ordered technical features, blocks duplicate date/artifact runs, stores deterministic ranks, and keeps shadow results separate from scanner scoring. The first controlled run is immutable run 1 for 2026-06-29 with 26 predictions and zero excluded tickers.

Phase 3A-1B adds append-only 1-, 5-, and 20-session outcome maturity tracking using the Dataset Lab next-session-close convention. The first controlled update appended two immutable SPY audit outcomes for run 1: one 1-session and one 5-session result. The remaining 76 prediction/horizon pairs are pending; 50 matured non-SPY pairs are blocked by missing post-2026-06-29 cached equity prices, and all 26 twenty-session pairs remain immature. SPY outcomes are excluded from cross-sectional model evidence, and one prediction date remains `insufficient_forward_sample`. No retraining, prediction changes, scoring integration, or Dataset 50 evaluation occurs.

Phase 3A-2A refreshed only the authorized missing OHLCV ranges through 2026-07-10, matured 25 non-SPY 1-session and 25 non-SPY 5-session outcomes for run 1, and recorded immutable run 2 for 2026-07-10 with 26 predictions. There are now 2 prospective prediction dates, 52 predictions, and 52 matured outcomes; 20-session outcomes remain pending and the sample remains `insufficient_forward_sample`.

Phase 3A-2B adds `scripts/run_daily_shadow_cycle.py`, a dry-run-first scheduler-ready coordinator for the existing refresh, maturity, prediction, and status contracts. Network refresh requires explicit `--apply --refresh-market-data`; mutating cycles use one lock and one backup, outcomes are appended before at most one current-session prediction, and repeat invocations safely no-op. No scheduler daemon is installed.

Phase 3A-2C corrected the cycle's completed-session refresh boundary. The provider already translated inclusive 2026-07-13 into yfinance's exclusive 2026-07-14 end; the defect was filtering the normalized response by its integer index instead of its `date` column. The controlled retry refreshed 26 missing symbols through 2026-07-13, retained no later rows, and created immutable run 3 with 26 predictions. A repeat cycle was a no-op. Shadow monitoring now has 3 prediction dates, 78 predictions, and 52 matured outcomes; the sample remains `insufficient_forward_sample`.

Phase 3B-1 adds a separate prospective options snapshot track for eight liquid equities. Daily yfinance chains are dry-run-first, immutable, duplicate-safe, and summarized with descriptive open-interest, volume, IV, concentration, liquidity, and missingness metrics. The track does not backfill history or affect datasets, shadow predictions, catalysts, scanner scoring, or recommendations. Use `scripts/collect_options_snapshots.py` for explicit collection and `scripts/quality_harness.py options-status` for read-only monitoring. The first controlled apply was safely blocked because the U.S. session was still open; production contains zero options snapshot runs and the first post-close collection remains pending.

Phase 3C-1 registered immutable research-only portfolio policy `top5_equal_weight_5session_v1` at `2026-07-15T15:25:51+00:00`. It is bound to `shadow_ridge_technical_v1_1ee8071db3f0`, excludes SPY, selects the top five stored equity ranks at 20% each, holds from the next-session close through five sessions after entry, and charges 10 basis points per side. The registration boundary is shadow run 4, so runs 1-4 are permanently excluded and the first eligible cohort may only use run 5 or later. No cohort or portfolio outcome exists yet; sample status is `insufficient_forward_sample`.

Phase 3C-2 adds `scripts/run_daily_research_cycle.py` as the permanent scheduler-facing entry point. It holds one master advisory lock and delegates, in order, to the existing shadow cycle, prospective portfolio cohort creation, portfolio outcome maturity, and eight-ticker options collection. Dry-run makes no database changes or network calls; yfinance access requires explicit `--apply --refresh-market-data`. Duplicate records safely skip, component failures remain isolated in a combined JSON report, and no model, policy, scanner, Dataset 49, or Dataset 50 contract changes. The first production dry-run resolved 2026-07-14, found shadow run 4 already recorded, planned no cohort, and made no mutation or network call.

## Latest Completed Work

- Phase 2E-1: compliant provider readiness
- Phase 2E-2: strict company IR provider workflow
- Phase 2E-3: real company IR candidate pilot
- Phase 2E-4: company IR `SourceDocument` bridge
- Phase 2E-5A: read-only company IR document coverage audit and manual enrichment queue
- Phase 2E-5B1: manual company IR document-enrichment backfill foundation; production dry-run only
- Phase 3A-1A: immutable cache-only daily shadow predictions
- Phase 3A-1B: immutable cache-only shadow outcome maturity tracking
- Phase 3A-2A: controlled OHLCV refresh, run 1 maturity update, and immutable run 2
- Phase 3A-2B: one-command guarded daily shadow cycle
- Phase 3A-2C: normalized-date OHLCV refresh fix and immutable run 3
- Phase 3B-1: prospective options snapshot storage, collection, monitoring, and read-only UI
- Phase 3C-1: immutable prospective shadow portfolio policy and cohort/outcome foundation
- Phase 3C-2: stable scheduler-facing daily research-cycle orchestrator
- Latest accepted pre-Phase 2E-5B1 commit: `12e02d8 Add company IR document coverage audit`

## Hard Constraints

- Do not model Dataset 50 until it is mature.
- Do not continue Dataset 49 annotation tuning.
- Do not change scanner scoring without explicit approval.
- Research annotations must retain `scanner_scoring_effect = 0`.
- Do not scrape or crawl websites; Reddit requires official API access.
- Do not automatically run LLM extraction or publish its output.
- Do not add deep learning until data quality and baseline evidence justify it.

## Recommended Future Directions

1. **A. Wait for and refresh Dataset 50 maturity.** Preserve the untouched holdout and extend it only under the existing cache-first governance workflow.
2. **B. Improve source-document coverage for existing company IR candidates.** Use the local, review-gated bridge without changing scanner scoring.
3. **C. Add controlled compliant RSS/manual-URL ingestion.** Require user-supplied sources, conservative pacing, staging, and review; never crawl.
4. **D. Build an options and microstructure data foundation.** Keep it separate from scoring until its point-in-time and provider contracts are validated.
5. **E. Try nonlinear modeling only after a mature holdout and stronger feature coverage.** Freeze the protocol before opening any final holdout.

## Fast-Start Checklist

Run from the repository root:

```bash
git status --short
.venv/bin/python scripts/quality_harness.py health-check
.venv/bin/python scripts/quality_harness.py provider-readiness
.venv/bin/python scripts/quality_harness.py holdout-status --dataset-id 50
```
