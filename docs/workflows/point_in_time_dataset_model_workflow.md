# Point-In-Time Dataset And Model Workflow

Use this workflow for dataset, feature, label, baseline-model, and walk-forward evaluation phases.

## Non-Negotiables

- Preserve point-in-time semantics for every snapshot.
- Use only information available at or before the snapshot `as_of_timestamp`.
- Use publication/availability time for catalysts, SEC filings, earnings events, and LLM-supported signals.
- Never use ingestion time as a substitute for public availability time.
- Outcome labels must stay outside model features.
- Do not include IDs, timestamps, hashes, policy versions, raw JSON, workflow statuses, audit columns, or unselected labels in default model inputs.
- Use chronological or walk-forward splits only.
- Do not train models in ingestion, audit, or dataset-foundation phases unless explicitly requested.

## Dataset Build Checklist

- Confirm the active database path.
- Back up the database before risky migrations, large backfills, or destructive cleanup.
- Record requested tickers, date range, feature version, policy version, and provider/cache mode.
- Use cached OHLCV/provider data first when requested.
- Do not download or refresh unrelated data during scanner-invariance checks.
- Build deterministic snapshots ordered by ticker and trading date.
- Generate labels separately from features using the documented next-session convention.
- Store dataset metadata, feature manifest, label definitions, row count, hash, warnings, and export path.
- Never silently overwrite a previous dataset build.

## Feature Contract Checklist

- Assign every flattened dataset column one role: `identifier`, `metadata`, `model_feature`, `audit`, or `label`.
- Put only model-ready inputs in `feature_columns_json`.
- Keep audit columns available for inspection but excluded from default training loaders.
- Keep raw filing/event counts audit-only when they can create issuer-volume dominance.
- Keep unknown or insufficient classifications audit-only unless a phase explicitly promotes them.
- Verify feature definitions remain unchanged before claiming a hash should match a prior dataset.

## Label And Split Checklist

- Label horizons should be explicit, for example `1_session`, `5_session`, and `20_session`.
- Labels should use the documented entry and exit convention.
- `label_available_at` must be after the exit information is available.
- Recent rows without enough future data should have unavailable labels, not fabricated outcomes.
- Chronological train/validation/test windows must not overlap.
- Optional gaps between train, validation, and test periods should be measured in trading sessions.

## Model-Evaluation Guardrails

- Start with simple baselines before advanced models.
- Compare against transparent benchmarks and naive baselines.
- Use walk-forward or strictly chronological evaluation.
- Report out-of-sample metrics separately from in-sample metrics.
- Treat feature importance and model output as research signals, not trading instructions.
- Do not add buy/sell/hold recommendations.
- Do not promote model outputs into scanner scoring without an explicit scoring-integration phase.

## Required Validation

- Run `.venv/bin/pytest -q`.
- Run `python3 -m compileall -q personal-alpha-lab` after Python code changes.
- Confirm no label or audit columns leak into `X`.
- Confirm scanner scores are unchanged unless scoring changes are in scope.
- For UI changes, smoke-test Dataset Lab and other affected Streamlit pages.
- Report dataset IDs, hashes, row counts, warnings, and any known coverage limitations.
