# Prospective Shadow Portfolio Workflow

## Frozen Policy

`top5_equal_weight_5session_v1` is a research-only cohort policy tied to `shadow_ridge_technical_v1_1ee8071db3f0`.

- Exclude SPY from selection.
- Select the five eligible equities with the best stored prediction ranks.
- Assign exactly 20% to each constituent.
- Long-only, fully invested, and unlevered.
- Enter at the next U.S. trading-session close after the prediction date.
- Exit five trading sessions after entry at that session's close.
- Charge 10 basis points on entry notional and 10 basis points on exit proceeds.
- Compare the net cohort return with SPY over the identical entry/exit dates.

The policy is immutable. There are no alternative policies, tuning, or retrospective policy selection in this phase.

## Prospective Boundary

Policy registration stores `eligible_after_prediction_run_id`, the highest immutable shadow run that exists at registration time. That run and every earlier run remain permanently ineligible. Cohort creation begins only with a later shadow run from the frozen artifact.

Runs that predate policy registration may be used in isolated tests only; they must never be presented as prospective portfolio evidence.

## Commands

Preview and register the policy:

```bash
.venv/bin/python scripts/manage_shadow_portfolio.py register-policy --dry-run
.venv/bin/python scripts/manage_shadow_portfolio.py register-policy --apply
```

Preview and create one cohort after a new eligible shadow run exists:

```bash
.venv/bin/python scripts/manage_shadow_portfolio.py create-cohort \
  --prediction-run-id <new-run-id> \
  --dry-run

.venv/bin/python scripts/manage_shadow_portfolio.py create-cohort \
  --prediction-run-id <new-run-id> \
  --apply
```

Preview and append newly matured cohort outcomes:

```bash
.venv/bin/python scripts/manage_shadow_portfolio.py update-outcomes --dry-run
.venv/bin/python scripts/manage_shadow_portfolio.py update-outcomes --apply
```

Every mutating command creates a timestamped SQLite backup. Dry runs are read-only. Repeated cohort creation and outcome updates safely return the existing/no-change state.

## Outcome Contract

Maturity uses the same point-in-time convention as Dataset Lab and shadow outcomes. `label_available_at` is after the exit-session close. Only cached OHLCV is read; no provider call is made. Missing constituent or SPY entry/exit closes leave the cohort pending.

Gross cohort return is the sum of fixed constituent weights times constituent returns. Entry cost is 0.001 of initial notional. Exit cost is 0.001 of exit proceeds, so total cost in return units is `0.001 + 0.001 * (1 + gross_return)`. Net return equals gross return minus that cost. Excess return equals net return minus the uncapped SPY return over the same dates.

Selections, weights, and matured outcomes are immutable. No broker orders or portfolio rebalancing are performed.

## Monitoring

```bash
.venv/bin/python scripts/quality_harness.py portfolio-shadow-status
```

Sample language is based on matured cohorts:

- fewer than 20: `insufficient_forward_sample`
- 20 through 59: `preliminary_only`
- 60 through 119: `developing_sample`
- 120 or more: `eligible_for_formal_review`

These labels control reporting language only. They do not establish validated alpha or authorize scanner integration.
