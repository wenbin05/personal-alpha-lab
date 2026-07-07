# Phase 2E-3 Company IR / Press-Release Candidate Pilot

Date: 2026-07-07

## Scope

Phase 2E-3 validated the strict `company_ir_press_release` provider workflow using manually sourced official-company newsroom, IR, and press-release URLs only.

No scraping, crawling, website discovery, LLM calls, model runs, Dataset 50 evaluation, active catalyst publication, or scanner scoring changes were performed.

## Artifacts

- Candidate CSV: `data/processed/phase2e3_company_ir_candidates_20260707.csv`
- Import report: `data/processed/phase2e3_company_ir_import_report_20260707.json`
- Database backup: `data/alpha_lab_backup_phase2e3_company_ir_20260707_233547.db`
- Scanner snapshots:
  - `data/processed/phase2e3_scanner_before.json`
  - `data/processed/phase2e3_scanner_after.json`

These are local run artifacts and remain ignored by git.

## Candidate Workflow Result

- Input rows: 38
- Parsed candidates: 38
- Parse errors: 0
- Staged as new: 19
- Marked duplicate: 19
- Accepted: 19
- Rejected: 0
- Imported as research-only annotations: 19
- Skipped: 0
- Research-only / scanner-score guardrail violations: 0

Duplicate reasons:

- Existing annotation source URL: 18
- Existing annotation title: 1

Annotation counts:

- Before: 405
- After: 424

Candidate counts:

- Before: 255
- After: 293

Catalyst table count remained unchanged at 30,671.

## Imported Mix

Imported annotations by ticker:

- AAPL: 3
- AMD: 2
- AMZN: 2
- COIN: 2
- META: 3
- MSFT: 1
- NVDA: 4
- TSLA: 2

Imported annotations by event type:

- Corporate action: 3
- Earnings: 3
- Financing: 1
- Legal/regulatory: 3
- Management change: 1
- News: 2
- Partnership: 1
- Product launch: 5

Imported annotations by sentiment:

- Positive: 12
- Mixed: 5
- Negative: 2

Source quality and informativeness:

- `official_company`: 19
- `material_high`: 9
- `material_medium`: 10

## Validation

- `pytest -q`: 219 passed
- `python3 -m compileall -q personal-alpha-lab`: passed
- Streamlit health check: passed
- Provider readiness: passed; no network-calling providers enabled
- Dataset 49 manifest/leakage check: passed
- Dataset 50 holdout status: passed; remains immature `holdout_candidate`
- Scanner invariance: passed with 0 diffs across 35 tickers
- Annotation coverage: passed
  - Annotation rows: 424
  - Dataset 49 labeled rows: 19,339
  - Future availability violations: 0
  - Research-only: true
  - Scanner scoring effect: 0

## Notes

The pilot deliberately accepted only non-duplicate strict company IR / press-release rows. Duplicate rows were preserved in `research_event_candidates` with duplicate status and were not imported.

The annotation coverage harness completed successfully but remains slow and memory-heavy as annotation history grows. This did not block the pilot, but future phases should consider optimizing the coverage audit path before scaling much further.
