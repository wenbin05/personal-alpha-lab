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

The harness verifies run uniqueness, stored prediction counts, artifact/hash consistency, and artifact integrity. It reports `insufficient_forward_sample` before 20 prediction dates, `preliminary_monitoring` from 20 through 59 dates, and `meaningful_review_sample` at 60 dates. These thresholds govern monitoring language only; outcome evaluation is a separate phase.

## Safety

- Never refresh or download market data merely to create a shadow run.
- Never overwrite an existing run.
- Never combine shadow predictions with Daily Scanner scores.
- Never create outcome rows in this phase.
- Preserve the artifact and its feature order exactly.
