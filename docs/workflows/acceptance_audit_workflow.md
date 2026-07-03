# Acceptance Audit Workflow

Use this workflow for final validation passes before moving to a new phase.

## Safety First

- Identify the active SQLite database.
- Back up the database before risky migrations, ingestion pilots, publication/reversal workflows, or large dataset rebuilds.
- Record baseline table counts, dataset IDs, scanner scores, or other state needed for reconciliation.
- Use clearly labeled synthetic records for destructive or reversible workflow tests.
- Do not modify genuine user-created catalysts, documents, extractions, publications, or journal records unless explicitly requested.
- Do not expose secrets, `.env`, API keys, or full provider user-agent strings.

## Scope Discipline

- Do not add new features during an acceptance audit.
- Do not make live API calls unless explicitly requested and bounded.
- Do not change scanner scoring unless the audit is for a scoring phase.
- Do not train models during data-ingestion or dataset-foundation audits.
- Only fix code when the audit reveals a reproducible defect.

## Core Checks

- Runtime: app starts cleanly and relevant pages load.
- Tests: `.venv/bin/pytest -q` passes.
- Compile: `python3 -m compileall -q personal-alpha-lab` passes after Python changes.
- UI: browser smoke-test affected Streamlit pages when UI changes.
- Data: no duplicate snapshots, labels, catalysts, filings, earnings events, or publications unless duplicates are expected and audited.
- Scoring: scanner outputs remain unchanged unless scoring changes are explicitly in scope.
- PIT: future data, future reversals, pending records, proposal-only rows, and labels do not leak into earlier features.
- Audit: provenance, timestamps, snapshots, warnings, and reversal records remain readable.

## Dataset Acceptance Checks

- Confirm dataset row count, ticker coverage, date range, feature manifest, label definitions, and hash.
- Compare repeated builds for identical hashes when source data and feature policy are unchanged.
- Verify labels are absent from model features.
- Verify audit columns are excluded from default training inputs.
- Inspect early, middle, recent, and label-unavailable snapshots.
- Confirm no provider refresh or network call occurred when a cache-only audit is requested.

## LLM/Catalyst Acceptance Checks

- LLM outputs must start as `pending_review`.
- Approved outputs should not affect catalysts or scoring unless a controlled publication phase explicitly does so.
- Proposals are not active catalysts.
- Published catalysts affect scoring only through the existing catalyst layer.
- Reverted publications contribute zero after reversal.
- Evidence snippets must be exact, grounded quotations when required by the phase.

## Final Report Template

- Runtime status.
- Tests and compile status.
- Browser smoke-test status, if applicable.
- Files changed.
- Database backup path, if applicable.
- Dataset IDs/hashes or synthetic record IDs, if applicable.
- Defects found and fixes made.
- Known limitations.
- Explicit decision: accepted, not accepted, or accepted with stated limitations.
- Recommended next milestone.
