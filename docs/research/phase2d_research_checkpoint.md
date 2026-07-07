# Phase 2D Research Checkpoint

Generated: 2026-07-07

This checkpoint freezes the current exploratory modeling state so future phases do not keep tuning against Dataset 49.

## Current Status

`personal-alpha-lab` has a point-in-time research dataset, safe model loader, chronological split workflow, research-only annotation layer, quality harness, and a maturing holdout protocol. The project remains a research and paper-trading decision-support tool. Model outputs do not affect the scanner, catalysts, alerts, or recommendations.

## Dataset Status

| Dataset | Role | Date range | Rows | Hash | Status |
|---:|---|---|---:|---|---|
| 49 | `exploratory_dev` | 2023-06-23 to 2026-06-23 | 19,495 | `e9523e1134b7eb32b142cb628d51bde76d5a6d139f4be2aba2545f3ca4416184` | Canonical development dataset. Its final test period has been repeatedly inspected. |
| 50 | `holdout_candidate` | 2026-06-24 to 2026-06-29 | 104 | `12155cb7ad5d84e7365e461edd3be181eb74e4bee516d0d588cad6a55fc01645` | Protocol-validation only. Immature; do not model or promote yet. |

Dataset 49 is useful for development diagnostics only. It should not be used to claim final signal robustness. Dataset 50 is not mature enough for final evaluation: it currently has only one 5-session labeled date and zero 20-session labels.

## Exploratory Run Summary

All runs below are Dataset 49 exploratory-development evidence only.

| Run | Phase | Feature set | Target | RMSE | OOS R2 | Spearman IC | Directional accuracy |
|---:|---|---|---|---:|---:|---:|---:|
| 145 | 2D-4 | `technical_core` | Winsorized 5-session SPY excess return | 0.06748 | 0.00218 | 0.04162 | 47.24% |
| 283 | 2D-6A | `technical_core_plus_annotations` | Winsorized 5-session SPY excess return | 0.06722 | 0.00990 | 0.06280 | 49.38% |
| 301 | 2D-6B-3 | `technical_core_plus_annotations` | Winsorized 5-session SPY excess return | 0.06740 | 0.00468 | 0.07235 | 48.68% |
| 319 | 2D-6B-5 | `technical_core_plus_annotations` | Winsorized 5-session SPY excess return | 0.06736 | 0.00588 | 0.07943 | 48.61% |
| 337 | 2D-6B-6 | `technical_core_plus_annotations` | Winsorized 5-session SPY excess return | 0.06731 | 0.00732 | 0.05533 | 49.48% |
| 370 | 2D-6B-7 | `technical_core_plus_annotations` | Winsorized 5-session SPY excess return | 0.06733 | 0.00670 | 0.05485 | 49.20% |
| 400 | 2D-6D-2 | `technical_core_plus_annotations` | Winsorized 5-session SPY excess return | 0.06733 | 0.00665 | 0.05729 | 49.12% |
| 405 | 2D-6D-3 | `technical_core_plus_annotations` | Winsorized 5-session SPY excess return | 0.06746 | 0.00278 | 0.03604 | 48.45% |
| 408 | 2D-6D-4 | `technical_core` | Winsorized 5-session SPY excess return | 0.06748 | 0.00218 | 0.04162 | 47.24% |
| 409 | 2D-6D-4 | `technical_core_plus_annotations` | Winsorized 5-session SPY excess return | 0.06746 | 0.00278 | 0.03604 | 48.45% |
| 410 | 2D-6D-4 | `technical_core_plus_annotation_compact_decay` | Winsorized 5-session SPY excess return | 0.06775 | -0.00564 | 0.01471 | 48.40% |
| 411 | 2D-6D-4 | `technical_core_plus_annotation_compact_weighted` | Winsorized 5-session SPY excess return | 0.06759 | -0.00105 | 0.03196 | 47.42% |

Best RMSE, OOS R2, and directional accuracy: run 283.

Best Spearman IC: run 319.

Latest compact-feature runs: 410 and 411. They reduced some annotation-feature collinearity but did not improve predictive robustness.

## What Was Tried

- Leakage-safe baseline modeling with Ridge regression and simple baselines.
- Winsorized, volatility-normalized, cross-sectional, and binary target variants.
- Technical-only, SEC, earnings, broad annotation, high-quality annotation, negative/mixed annotation, and compact annotation feature sets.
- Research-only annotation ingestion, candidate staging, candidate review, quality normalization, source-quality taxonomy, informativeness taxonomy, and holdout-safe workflow documentation.
- Compact annotation lag/decay features:
  - 0-5, 6-20, and 21-60 session event activation.
  - days since material and negative/mixed events.
  - exponential sentiment decay.
  - source-quality and informativeness weighting.

## What Worked

- The point-in-time dataset/model contract held up: labels and audit columns remain separated from model features.
- Chronological/walk-forward evaluation is working.
- The research-only annotation pipeline is auditable and scanner-neutral.
- Broad annotations improved several exploratory metrics versus technical-only in some runs, especially run 283 and run 319.
- Quality harness checks are useful for scanner invariance, manifest leakage, annotation coverage, and holdout maturity.

## What Did Not Work

- SEC and earnings metadata alone did not improve linear baselines robustly.
- Increasing annotation volume did not monotonically improve results.
- High-signal and negative/mixed filters were diagnostically useful but did not reliably beat broad annotation features.
- Compact lag/decay features did not beat prior broad annotation runs.
- Dataset 49 final-test metrics have been inspected too often to remain a credible final holdout.

## Why Annotation Modeling Is Paused

Annotation modeling is paused because Dataset 49 is now a development surface, not a valid final confirmation surface. Continued tuning against Dataset 49 risks overfitting to a repeatedly inspected final-test window. The latest compact-feature experiment improved representation cleanliness but worsened predictive metrics versus the better broad-annotation runs.

## Conditions To Resume Modeling

Resume modeling only when at least one of these is true:

- Dataset 50, or a later time-forward holdout candidate, passes maturity thresholds and is promoted under explicit holdout rules.
- A pre-registered source-coverage phase adds materially better compliant event coverage without using Dataset 49 model results or future price movement to select events.
- A new untouched holdout or walk-forward confirmation protocol is created before evaluating any new modeling claim.

## Recommended Next Track

Primary next track: improve compliant source coverage and provider readiness while Dataset 50 matures.

Do not start nonlinear models, tree/boosting models, deep learning, or more Dataset 49 annotation-feature tuning until a fresh holdout is mature enough for confirmation.

## Track Options

| Track | Value | Risk | Decision |
|---|---|---|---|
| A. Wait for Dataset 50 maturity | Preserves holdout integrity. | Passive; no new signal coverage. | Do in background. |
| B. Add compliant provider/manual RSS support | Improves event coverage and reduces manual bottleneck. | Must avoid scraping and point-in-time errors. | Primary next track. |
| C. Expand into options/microstructure foundation | Could add distinct signal families later. | New data complexity; not yet justified by current dataset maturity. | Defer. |
| D. Add micro/small-cap universe foundation | Potentially richer alpha surface. | Higher liquidity/data-quality risk; scanner guardrails need extension. | Defer until data controls are stronger. |
| E. Resume modeling after holdout maturity | Required for credible confirmation. | Blocked until labels mature. | Wait. |

## Guardrail

Future reports should state explicitly whether results are `exploratory_dev`, `holdout_candidate`, or `final_holdout`. Dataset 49 results must be labeled `exploratory_dev`.
