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

## Latest Completed Work

- Phase 2E-1: compliant provider readiness
- Phase 2E-2: strict company IR provider workflow
- Phase 2E-3: real company IR candidate pilot
- Phase 2E-4: company IR `SourceDocument` bridge
- Latest commit: `1a1359a Add company IR source document bridge`

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
