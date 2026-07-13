# Daily Shadow Prediction Workflow

## Purpose

The daily shadow pipeline records prospective research predictions from an immutable frozen artifact. It does not retrain models, modify scanner scoring, evaluate Dataset 50, create trading recommendations, or fetch market data.

The initial artifact is `shadow_ridge_technical_v1_1ee8071db3f0`. It remains `frozen_exploratory` / `exploratory_shadow` and is not validated alpha.

## Point-In-Time Contract

- Read OHLCV only from the local SQLite cache.
- Slice every ticker and benchmark history through the requested completed trading session.
- Require same-session cache bars for SPY, QQQ, IWM, and VIX.
- Assemble only the artifact's exact ordered `technical_core` features using the accepted Dataset 49 formulas.
- Never use annotation, SEC, earnings, catalyst, LLM, Dataset 50, or outcome-label inputs.
- Validate artifact files, checksums, registry metadata, and feature hashes before inference.
- Missing required columns fail closed; missing values are retained for the artifact's frozen imputer and recorded as data-quality flags.

## Dry Run

```bash
.venv/bin/python scripts/run_shadow_prediction.py \
  --artifact-id shadow_ridge_technical_v1_1ee8071db3f0 \
  --as-of YYYY-MM-DD \
  --dry-run
```

Dry-run uses read-only SQLite access and reports the cache-complete session, eligible and excluded tickers, feature-input hashes, rankings, warnings, and duplicate status. It creates no tables, rows, backup, or artifact changes.

## Apply

Review the dry-run, then apply explicitly:

```bash
.venv/bin/python scripts/run_shadow_prediction.py \
  --artifact-id shadow_ridge_technical_v1_1ee8071db3f0 \
  --as-of YYYY-MM-DD \
  --apply
```

Apply creates a timestamped SQLite backup, opens one transaction, creates the shadow tables if needed, and inserts one completed run plus all ticker predictions. A duplicate date/artifact run is rejected. Completed runs and predictions cannot be updated or deleted through SQLite.

## Ranking

Predictions are sorted by descending predicted value with ticker as the deterministic tie-break. Rank starts at 1. Percentile is 1.0 for the top row and 0.0 for the bottom row. These are research rankings, not buy/sell/hold signals.

## Monitoring

```bash
.venv/bin/python scripts/quality_harness.py shadow-status \
  --artifact-id shadow_ridge_technical_v1_1ee8071db3f0
```

The harness verifies run uniqueness, stored prediction counts, outcome uniqueness, artifact/hash consistency, and artifact integrity. Forward-sample language is based on distinct immutable prediction dates:

- fewer than 20: `insufficient_forward_sample`
- 20 through 59: `preliminary_only`
- 60 through 119: `developing_sample`
- 120 or more: `eligible_for_formal_review`

These thresholds govern monitoring language only. They do not establish validated alpha.

## Outcome Maturity

Preview newly matured outcomes without changing SQLite:

```bash
.venv/bin/python scripts/update_shadow_outcomes.py --run-id <run-id> --dry-run
```

After reviewing missing-cache and pending-horizon details, append the matured rows explicitly:

```bash
.venv/bin/python scripts/update_shadow_outcomes.py --run-id <run-id> --apply
```

Apply creates a timestamped database backup. Outcome rows are unique by prediction and horizon, immutable after insertion, and idempotent on repeat execution.

The timing contract matches Dataset Lab: prediction date `T` is known after its close, entry uses the next U.S. trading-session close, exit uses the close `N` sessions after entry, and `label_available_at` is after the exit close. Returns never include the prediction-date-to-entry move. Only cached OHLCV may be read; absent entry, exit, or benchmark bars remain pending with explicit data-quality reasons.

SPY is the benchmark for excess return. A SPY prediction is retained with its matured outcome for audit, but it is excluded from cross-sectional IC, rank, and equity directional metrics because SPY excess return versus itself is not useful model evidence.

## Safety

- Never refresh or download market data merely to create a shadow run.
- Never overwrite an existing run.
- Never combine shadow predictions with Daily Scanner scores.
- Never overwrite a matured outcome or fabricate a missing cached close.
- Preserve the artifact and its feature order exactly.
