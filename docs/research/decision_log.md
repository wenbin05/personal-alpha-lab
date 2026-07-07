# Research Decision Log

Generated: 2026-07-07

This log records project-level research decisions that should guide future Codex phases.

## 2026-07-07: Freeze Dataset 49 Annotation Modeling

Decision: Pause further annotation-feature tuning and model experiments on Dataset 49.

Rationale:

- Dataset 49 has been repeatedly inspected across Phase 2D.
- The final-test window is now exploratory development evidence only.
- Compact lag/decay annotation features reduced some feature-design issues but did not improve robustness.
- Continued Dataset 49 tuning would increase holdout contamination risk.

Implications:

- Do not claim final signal robustness from Dataset 49 results.
- Do not continue Dataset 49 annotation-feature tweaks unless the phase is explicitly diagnostic and labeled `exploratory_dev`.
- Prefer better compliant source coverage and a mature untouched holdout before more modeling.

## 2026-07-07: No Nonlinear Models Yet

Decision: Do not add tree, boosting, nonlinear, or deep learning models yet.

Rationale:

- Current linear baselines are weak and source-quality sensitive.
- Dataset 50 is not mature enough for final confirmation.
- More complex models would make overfitting easier without resolving data coverage and holdout maturity.

Resume condition:

- A fresh holdout candidate passes maturity thresholds, and feature/source coverage improves under a pre-registered protocol.

## 2026-07-07: Dataset 50 Remains Holdout Candidate Only

Decision: Dataset 50 remains `holdout_candidate`; do not model it or promote it.

Known state:

- 104 rows.
- 1-session labels: 54.
- 5-session labels: 1.
- 20-session labels: 0.

Resume condition:

- `scripts/quality_harness.py holdout-status --dataset-id 50` or a successor holdout reports enough row and label coverage for the intended target horizon.

## 2026-07-07: Next Value Is Better Compliant Source Coverage

Decision: The primary next track should be compliant source coverage/provider readiness, not more manual feature tweaks.

Preferred direction:

- Improve research-only news/event candidate sourcing through compliant manual/RSS/provider interfaces.
- Keep all candidates staged and review-gated.
- Keep scanner scoring unchanged.
- Keep active catalysts unchanged unless a separate controlled catalyst phase explicitly requests publication.

Deferred:

- Reddit/X/social scraping.
- Paid provider dependence.
- Options or microstructure expansion.
- Micro/small-cap universe expansion.
- Scanner scoring changes.

## Standing Decisions

- No buy/sell/hold recommendations.
- No labels, audit fields, identifiers, metadata timestamps, hashes, raw JSON, or workflow statuses in model features.
- LLM outputs remain review-gated before affecting catalysts or scoring.
- Scanner invariance is required unless a phase explicitly changes scoring.
- Back up the active SQLite database before risky migrations, ingestion pilots, publication/reversal audits, or large backfills.
